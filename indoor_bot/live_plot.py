
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import os
import glob
import numpy as np
import yaml

BASE_PATH = "/home/gonazza/container_ws/catkin_ws/src/indoor_bot/indoor_bot/data"   # oppure la cartella condivisa host
WAYPOINTS_FILE = "/home/gonazza/container_ws/catkin_ws/src/indoor_bot/config/waypoints.yaml"
UPDATE_INTERVAL = 500  # ms

# trova automaticamente i robot
def get_robot_files():
    pos_files = glob.glob(BASE_PATH + "/*_position.csv")
    robots = []

    for f in pos_files:
        name = os.path.basename(f).replace("_position.csv", "")
        robots.append(name)
    
    return robots
# legge i waypoints dal file YAML
def read_waypoints(yaml_file):
    waypoints = []
    if os.path.exists(yaml_file):
        with open(yaml_file, 'r') as f:
            try:
                data = yaml.safe_load(f)
                if "wp" in data:
                    for wp in data["wp"]:
                        waypoints.append((wp["x"], wp["y"]))
            except yaml.YAMLError as e:
                print("Errore lettura YAML:", e)
    return waypoints

waypoints = read_waypoints(WAYPOINTS_FILE)
# crea figure
fig1, ax1 = plt.subplots()
fig2, axs2 = plt.subplots(3, 1)  # metti un numero massimo robot
axs2 = list(axs2)

wp_colors = plt.cm.tab10(range(len(waypoints) + 1))
priority_lines = {}  # robot -> lista di linee
robots = get_robot_files()
for idx, robot in enumerate(robots):
    pr_file = f"{BASE_PATH}/{robot}_priority.csv"
    if os.path.exists(pr_file):
        pr = pd.read_csv(pr_file, on_bad_lines='skip')
        if not pr.empty and pr.shape[1] > 1:
            lines = []
            for i, col in enumerate(pr.columns[1:]):
                line, = axs2[idx].plot([], [], label=col, color=wp_colors[i % len(wp_colors)])
                lines.append(line)
            axs2[idx].set_title(f"Priorità {robot}")
            axs2[idx].set_xlabel("Time")
            
            axs2[idx].set_ylabel("Priorità")
            axs2[idx].legend()
            priority_lines[robot] = lines
ax1.set_ylim(-5, 5)
ax1.set_xlim(-5, 5)
plt.tight_layout()

# ===== FIGURA 3: MEMORY =====
fig3, ax3 = plt.subplots()

memory_file = os.path.join(BASE_PATH, "memory_log.csv")
memory_lines = []

if os.path.exists(memory_file):
    mem = pd.read_csv(memory_file, on_bad_lines='skip')
    if not mem.empty and mem.shape[1] > 1:
        for i, col in enumerate(mem.columns[1:]):
            line, = ax3.plot([], [], label=col, color=wp_colors[i % len(wp_colors)])
            memory_lines.append(line)

        ax3.set_title("Memory")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Valore")
        ax3.legend()

# ===== FIGURA 4: BATTERY =====
fig4, axs4 = plt.subplots(max(len(robots), 1), 2, figsize=(10, 4 * max(len(robots), 1)))

# Se c'è un solo robot, axs4 non è una matrice 2D automaticamente
if len(robots) == 1:
    axs4 = np.array([axs4])

battery_lines = {}  # robot -> {"percentage": line, "voltage": line}

for idx, robot in enumerate(robots):
    battery_file = f"{BASE_PATH}/{robot}_battery.csv"

    if os.path.exists(battery_file):
        batt = pd.read_csv(battery_file, on_bad_lines='skip')

        if not batt.empty:
            # subplot percentuale
            perc_line, = axs4[idx, 0].plot([], [], label=f"{robot} %")
            axs4[idx, 0].set_title(f"Battery Percentage {robot}")
            axs4[idx, 0].set_xlabel("Time")
            axs4[idx, 0].set_ylabel("Percentage [%]")
            axs4[idx, 0].set_ylim(0, 100)
            axs4[idx, 0].legend()

            # subplot voltaggio
            volt_line, = axs4[idx, 1].plot([], [], label=f"{robot} Voltage")
            axs4[idx, 1].set_title(f"Battery Voltage {robot}")
            axs4[idx, 1].set_xlabel("Time")
            axs4[idx, 1].set_ylabel("Voltage [V]")
            axs4[idx, 1].legend()

            battery_lines[robot] = {
                "percentage": perc_line,
                "voltage": volt_line
            }

plt.tight_layout()




def animate(i):
    global fig2, axs2
    
    robots = get_robot_files()
    print("Robot trovati:", robots)
    ax1.clear()
    ax1.set_title("Swarm Traiettorie + Direzione")
    ax1.set_xlabel("x")
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    ax1.set_ylabel("y")

    # colori automatici
    colors = plt.cm.tab10(range(len(robots)))
    
    # ===== FIGURA 1 =====

        # plottiamo i waypoints
    if waypoints:
        for i, (wx, wy) in enumerate(waypoints):
            ax1.scatter(wx, wy, color=wp_colors[i], marker='*', s=120)

    # legenda waypoint (opzionale)
    for i in range(len(waypoints)):
        ax1.plot([], [], marker='*', linestyle='', color=wp_colors[i], label=f'WP{i}')

    for idx, robot in enumerate(robots):
        try:
            pos_file = f"{BASE_PATH}/{robot}_position.csv"
            dir_file = f"{BASE_PATH}/{robot}_direction.csv"
            print(f"{robot} -> file:", os.path.exists(pos_file))
            if not os.path.exists(pos_file):
                continue

            pos = pd.read_csv(pos_file, on_bad_lines='skip')

            # rimuove tutte le righe con x=0 e y=0
            pos = pos[(pos["x"] != 0) | (pos["y"] != 0)]

            if pos.shape[0] < 2:
                continue
            if pos.empty:
                continue

            x = pos["x"].to_numpy()
            y = pos["y"].to_numpy()

            ax1.plot(x, y, color=colors[idx], label=robot)

            # direzione (freccia ultimo punto)
            if os.path.exists(dir_file):
                dr = pd.read_csv(dir_file, on_bad_lines='skip')
                if not dr.empty:
                    last = dr.iloc[-1]

                    ax1.arrow(
                        x[-1], y[-1],
                        float(last["dir_x"]), float(last["dir_y"]),
                        
                        head_width=0.1,
                        color=colors[idx]
                    )

        except Exception as e:
            print(f"Errore robot {robot}:", e)

    ax1.legend()
    ax1.axis("equal")

    # ===== FIGURA 2 =====
    if robots:

        for idx, robot in enumerate(robots):
            if robot in priority_lines:
                pr_file = f"{BASE_PATH}/{robot}_priority.csv"
                
                if os.path.exists(pr_file):
                    pr = pd.read_csv(pr_file, on_bad_lines='skip')
                    t = pr["time"].to_numpy()
                    data = pr.iloc[:, 1:].to_numpy()  # tutte le colonne di priorità
                    
                    if data.shape[0] > 0:
                        x = np.arange(data.shape[0])
                        for i, line in enumerate(priority_lines[robot]):
                            y = data[:, i]
                            line.set_data(t, y)

                        axs2[idx].set_ylim(-0.1, 1.1)
                        axs2[idx].set_xlim(t.min(), t.max() if len(t) > 1 else t.min() + 1) # aggiorna l'asse x se cresce
                        axs2[idx].figure.canvas.draw_idle()
        # ===== FIGURA 3 (MEMORY) =====
    if os.path.exists(memory_file):
        try:
            mem = pd.read_csv(memory_file, on_bad_lines='skip')
            t = mem["time"].to_numpy()
            if not mem.empty and mem.shape[1] > 1:
                data = mem.iloc[:, 1:].to_numpy()

                if data.shape[0] > 0:
                    x = np.arange(data.shape[0])

                    for i, line in enumerate(memory_lines):
                        if i < data.shape[1]:
                            y = data[:, i]
                            line.set_data(t, y)

                    ax3.set_xlim(t.min(), t.max() if len(t) > 1 else t.min() + 1)
                    ax3.relim()
                    ax3.autoscale_view()
                    ax3.figure.canvas.draw_idle()

        except Exception as e:
            print("Errore memory:", e)
        # ===== FIGURA 4 (BATTERY) =====
    for idx, robot in enumerate(robots):
        if robot in battery_lines:
            battery_file = f"{BASE_PATH}/{robot}_battery.csv"

            if os.path.exists(battery_file):
                try:
                    batt = pd.read_csv(battery_file, on_bad_lines='skip')

                    if not batt.empty and all(col in batt.columns for col in ["time", "voltage", "percentage"]):
                        t = batt["time"].to_numpy()
                        voltage = batt["voltage"].to_numpy()
                        percentage = batt["percentage"].to_numpy()

                        # aggiorna percentage
                        battery_lines[robot]["percentage"].set_data(t, percentage)
                        axs4[idx, 0].set_xlim(t.min(), t.max() if len(t) > 1 else t.min() + 1)
                        axs4[idx, 0].set_ylim(0, 100)
                        axs4[idx, 0].figure.canvas.draw_idle()

                        # aggiorna voltage
                        battery_lines[robot]["voltage"].set_data(t, voltage)
                        axs4[idx, 1].set_xlim(t.min(), t.max() if len(t) > 1 else t.min() + 1)
                        axs4[idx, 1].relim()
                        axs4[idx, 1].autoscale_view()
                        axs4[idx, 1].figure.canvas.draw_idle()

                except Exception as e:
                    print(f"Errore battery {robot}:", e)
                        



# animazione
ani = animation.FuncAnimation(fig1, animate, interval=UPDATE_INTERVAL)

plt.show()