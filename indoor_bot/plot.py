import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--experiment",
    required=True,
    help="Cartella dell'esperimento"
)
parser.add_argument(
    "--noshow",
    action="store_false",
    help="Nasconde le figure"
)

args = parser.parse_args()

EXPERIMENT_FOLDER = args.experiment

BASE_PATH = f"{EXPERIMENT_FOLDER}"
# OUTPUT_FOLDER = f"./results/{EXPERIMENT_FOLDER}"
MISSION_TIME_FILES = [
    "rosbot_1_0_mission_times.csv",
    "rosbot_2_1_mission_times.csv",
    "rosbot_3_2_mission_times.csv",
]
# BASE_PATH = "/home/gonazza/container_ws/catkin_ws/src/indoor_bot/indoor_bot/exp_equal2"
WAYPOINTS_FILE = "/home/gonazza/container_ws/catkin_ws/src/indoor_bot/config/waypoints.yaml"
OUTPUT_DIR = os.path.join(BASE_PATH, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# LOAD DATA
# ==================================================

def get_robot_files():
    pos_files = glob.glob(BASE_PATH + "/*_position.csv")
    return [os.path.basename(f).replace("_position.csv", "") for f in pos_files]


def read_waypoints(yaml_file):
    waypoints = []
    if os.path.exists(yaml_file):
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            if data and "wp" in data:
                for wp in data["wp"]:
                    waypoints.append((wp["x"], wp["y"]))
    return waypoints

def get_max_mission_time(base_path):

    mission_times = []

    for file in MISSION_TIME_FILES:

        path = os.path.join(base_path, file)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing mission time file: {path}"
            )

        df = pd.read_csv(path)

        # usa il completamento missione, NON la ricarica
        max_time = df["charging_completion_time"].max()

        mission_times.append(max_time)

    return max(mission_times)

robots = get_robot_files()
waypoints = read_waypoints(WAYPOINTS_FILE)

MAX_TIME = get_max_mission_time(BASE_PATH)

print(f"MAX_TIME used: {MAX_TIME:.2f} s")

wp_colors = plt.cm.tab10(range(len(waypoints) + 1))
robot_colors = plt.cm.tab10(range(len(robots)))


# ==================================================
# FIGURE 1: TRAIETTORIE
# ==================================================

fig1, ax1 = plt.subplots(figsize=(7, 7))

ax1.set_title("Swarm Trajectories")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_aspect("equal")

for i, (wx, wy) in enumerate(waypoints):
    ax1.scatter(wx, wy, color=wp_colors[i], marker='*', s=120)

for idx, robot in enumerate(robots):

    pos_file = f"{BASE_PATH}/{robot}_position.csv"
    dir_file = f"{BASE_PATH}/{robot}_direction.csv"

    if not os.path.exists(pos_file):
        continue

    pos = pd.read_csv(pos_file)

    pos = pos[pos["time"] <= MAX_TIME]

    pos = pos.iloc[1:]

    pos = pos[(pos["x"] != 0) | (pos["y"] != 0)]

    if pos.empty:
        continue

    # =========================
    # SAFE NUMPY CONVERSION
    # =========================
    x = np.asarray(pos["x"].values, dtype=float)
    y = np.asarray(pos["y"].values, dtype=float)

    ax1.plot(x, y, label=robot, color=robot_colors[idx])

    if os.path.exists(dir_file):
        dr = pd.read_csv(dir_file).iloc[1:]
        if not dr.empty:
            last = dr.iloc[-1]

            ax1.arrow(
                x[-1],
                y[-1],
                float(last["dir_x"]),
                float(last["dir_y"]),
                head_width=0.1,
                color=robot_colors[idx]
            )

ax1.legend()
fig1.savefig(os.path.join(OUTPUT_DIR, "trajectories.png"), dpi=300)

# ==================================================
# FIGURE 2: PRIORITY
# ==================================================

fig2, axes2 = plt.subplots(len(robots), 1, figsize=(10, 3 * len(robots)))

if len(robots) == 1:
    axes2 = [axes2]

for idx, robot in enumerate(robots):

    pr_file = f"{BASE_PATH}/{robot}_priority.csv"

    ax = axes2[idx]
    ax.set_title(f"Priority - {robot}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")

    if not os.path.exists(pr_file):
        continue

    pr = pd.read_csv(pr_file)

    pr = pr[pr["time"] <= MAX_TIME]

    pr = pr.iloc[1:]

    if pr.empty:
        continue

    # =========================
    # SAFE NUMPY CONVERSION
    # =========================
    t_raw = np.asarray(pr["time"].values, dtype=float)
    t = t_raw - t_raw[0]

    data = pr.iloc[:, 1:].apply(
        pd.to_numeric,
        errors="coerce"
    ).to_numpy(dtype=float)

    for i in range(data.shape[1]):
        ax.plot(t, data[:, i], label=f"p{i}")

    ax.legend()
fig2.savefig(os.path.join(OUTPUT_DIR, "priority.png"), dpi=300)

# ==================================================
# FIGURE 3: MEMORY
# ==================================================

fig3, ax3 = plt.subplots(figsize=(10, 4))

memory_file = os.path.join(BASE_PATH, "photosynthesis_log.csv")

if os.path.exists(memory_file):

    mem = pd.read_csv(memory_file)

    #mem = mem[mem["time"] <= MAX_TIME]
    mem = mem.iloc[2:]

    if not mem.empty:

        t_raw = np.asarray(mem["time"].values, dtype=float)
        t = t_raw - t_raw[2]

        data = mem.iloc[:, 1:].apply(
            pd.to_numeric,
            errors="coerce"
        ).to_numpy(dtype=float)

        for i in range(data.shape[1]):
            ax3.plot(t, data[:, i])

ax3.set_title("Memory")
ax3.set_xlabel("Time")
ax3.set_ylabel("Value")
ax3.legend()
fig3.savefig(os.path.join(OUTPUT_DIR, "memory.png"), dpi=300)

# ==================================================
# FIGURE 4: BATTERY
# ==================================================

fig4, axes4 = plt.subplots(
    len(robots),
    2,
    figsize=(12, 4 * len(robots))
)

if len(robots) == 1:
    axes4 = np.array([axes4])

for idx, robot in enumerate(robots):

    batt_file = f"{BASE_PATH}/{robot}_battery.csv"

    axp = axes4[idx, 0]
    axv = axes4[idx, 1]

    axp.set_title(f"Battery % - {robot}")
    axv.set_title(f"Battery Voltage - {robot}")

    if not os.path.exists(batt_file):
        continue

    batt = pd.read_csv(batt_file)

    batt = batt[batt["time"] <= MAX_TIME]

    batt = batt.iloc[1:]

    if batt.empty:
        continue

    # =========================
    # SAFE NUMPY CONVERSION
    # =========================
    t_raw = np.asarray(batt["time"].values, dtype=float)
    t = t_raw - t_raw[0]

    if "percentage" in batt.columns:
        percentage = np.asarray(batt["percentage"].values, dtype=float)
        axp.plot(t, percentage, label="Battery %")
        axp.set_ylim(0, 100)
        axp.legend()

    if "voltage" in batt.columns:
        voltage = np.asarray(batt["voltage"].values, dtype=float)
        axv.plot(t, voltage, label="Voltage")
        axv.legend()

fig4.savefig(os.path.join(OUTPUT_DIR, "battery.png"), dpi=300)
plt.tight_layout()
if args.noshow:
    plt.show()