import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURAZIONE
# ============================================================
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

parser.add_argument(
    "--ral",
    action="store_true",
    help="Use RAL/IEEE paper plot formatting"
)

args = parser.parse_args()

EXPERIMENT_FOLDER = args.experiment

DATA_FOLDER = f"{EXPERIMENT_FOLDER}"
OUTPUT_FOLDER = f"{EXPERIMENT_FOLDER}/plots"

# ============================================================
# RAL / IEEE PLOT STYLE
# ============================================================

if args.ral:

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,

        "axes.titlesize": 9,
        "axes.labelsize": 8,

        "xtick.labelsize": 7,
        "ytick.labelsize": 7,

        "legend.fontsize": 7,

        "lines.linewidth": 1.2,
        "lines.markersize": 0.1,

        "axes.linewidth": 0.8,

        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    FIGSIZE = (4.0, 3.0)

else:

    FIGSIZE = None


# ============================================================
# FILES
# ============================================================

POSITION_FILES = [
    "rosbot_1_0_position.csv",
    "rosbot_2_1_position.csv",
    "rosbot_3_2_position.csv",
]

MISSION_TIME_FILES = [
    "rosbot_1_0_mission_times.csv",
    "rosbot_2_1_mission_times.csv",
    "rosbot_3_2_mission_times.csv",
]

WAYPOINT_FILE = (
    "/home/gonazza/container_ws/catkin_ws/src/"
    "indoor_bot/config/waypoints.yaml"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Salva anche in PDF
SAVE_PDF = False


# ============================================================
# Calcolo automatico MAX_TIME dai file mission_times
# ============================================================

mission_times = []

for file in MISSION_TIME_FILES:

    path = os.path.join(DATA_FOLDER, file)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File mission time non trovato: {path}"
        )

    df_time = pd.read_csv(path)

    # prende il valore massimo della colonna tempo
    max_t = df_time["charging_completion_time"].max()

    mission_times.append(max_t)


MAX_TIME = max(mission_times)

print(f"MAX_TIME utilizzato: {MAX_TIME:.2f} s")


# Robot considerato "sul waypoint" se entro questa distanza
WAYPOINT_RADIUS = 2.5

# Heading considerato corretto se entro questa soglia
HEADING_THRESHOLD_DEG = 45


# ============================================================
# UTILITY
# ============================================================

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


# ============================================================
# Caricamento waypoints
# ============================================================

with open(WAYPOINT_FILE, "r") as f:
    wp_yaml = yaml.safe_load(f)

waypoints = np.array(
    [
        [w["x"], w["y"]]
        for w in wp_yaml["wp"]
    ]
)


# ============================================================
# Caricamento robot
# ============================================================

robots = []

for file in POSITION_FILES:

    df = pd.read_csv(
        os.path.join(DATA_FOLDER, file)
    )

    if MAX_TIME is not None:
        df = df[
            df["time"] <= MAX_TIME
        ]

    robots.append(
        df.reset_index(drop=True)
    )


# ============================================================
# Timeline comune
# ============================================================

common_time = robots[0]["time"].values

if MAX_TIME is not None:
    common_time = common_time[
        common_time <= MAX_TIME
    ]

interp_data = []

for df in robots:

    interp = {
        "x": np.interp(
            common_time,
            df["time"],
            df["x"]
        ),

        "y": np.interp(
            common_time,
            df["time"],
            df["y"]
        ),

        "yaw": np.interp(
            common_time,
            df["time"],
            df["yaw"]
        ),
    }

    interp_data.append(interp)


dt = np.mean(
    np.diff(common_time)
)

total_time = (
    common_time[-1]
    - common_time[0]
)


# ============================================================
# 1) Distanza dal centro dello sciame
# ============================================================

sum_distances = []
distances = []

for i in range(len(common_time)):

    positions = np.array([
        [
            interp_data[0]["x"][i],
            interp_data[0]["y"][i]
        ],
        [
            interp_data[1]["x"][i],
            interp_data[1]["y"][i]
        ],
        [
            interp_data[2]["x"][i],
            interp_data[2]["y"][i]
        ]
    ])

    center = positions.mean(axis=0)

    dist = np.linalg.norm(
        positions - center,
        axis=1
    )

    distances.append(dist)
    sum_distances.append(
        dist.sum()
    )


# ============================================================
# Swarm compactness
# ============================================================

if args.ral:
    plt.figure(figsize=FIGSIZE)
else:
    plt.figure(figsize=(8, 4))

plt.plot(
    common_time,
    sum_distances
)

plt.grid(True)

plt.xlabel(
    "Time [s]"
)

plt.ylabel(
    "Sum distance from swarm center [m]"
)

plt.title(
    "Swarm compactness"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "swarm_compactness.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

if SAVE_PDF:
    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "swarm_compactness.pdf"
        ),
        bbox_inches="tight"
    )


# ============================================================
# Distanza individuale dal centro dello sciame
# ============================================================

if args.ral:
    plt.figure(figsize=FIGSIZE)
else:
    plt.figure(figsize=(8, 4))

distances = np.array(distances)

for j in range(3):

    plt.plot(
        common_time,
        distances[:, j],
        label=f"UAV {j+1}"
    )

plt.grid(True)

plt.xlabel(
    "Time [s]"
)

plt.ylabel(
    "Distance from swarm center [m]"
)

plt.title(
    "Distance of each UAV from swarm center"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "individual_swarm_distance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

if SAVE_PDF:
    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "individual_swarm_distance.pdf"
        ),
        bbox_inches="tight"
    )


# ============================================================
# 2) Tempo entro 2.5 m da ciascun waypoint
# ============================================================

n_wp = len(waypoints)

percent_near_wp = np.zeros(
    (3, n_wp)
)

for r, robot in enumerate(interp_data):

    x = robot["x"]
    y = robot["y"]

    for w, wp in enumerate(waypoints):

        d = np.sqrt(
            (x - wp[0])**2
            + (y - wp[1])**2
        )

        near = d < WAYPOINT_RADIUS

        time_near = (
            near.sum() * dt
        )

        # percent_near_wp[r, w] = (
        #     100 * time_near / total_time
        # )

        percent_near_wp[r, w] = time_near


# ============================================================
# Waypoint distance plot
# ============================================================

if args.ral:
    plt.figure(figsize=FIGSIZE)
else:
    plt.figure(figsize=(8, 5))

robots_name = [
    "Robot1",
    "Robot2",
    "Robot3"
]

bottom = np.zeros(3)

for w in range(n_wp):

    # plt.bar(
    #     robots_name,
    #     percent_near_wp[:, w],
    #     bottom=bottom,
    #     label=f"WP {w+1}"
    # )

    bottom += percent_near_wp[:, w]


plt.bar(
    robots_name,
    bottom
)

plt.ylabel(
    "Time [s]"
)

plt.title(
    "Tempo trascorso entro 2.5 m dai waypoint"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "wpdist.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

if SAVE_PDF:
    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "wpdist.pdf"
        ),
        bbox_inches="tight"
    )


# ============================================================
# 3) Heading verso waypoint
# ============================================================

heading_percent = []

threshold = np.deg2rad(
    HEADING_THRESHOLD_DEG
)

for robot in interp_data:

    x = robot["x"]
    y = robot["y"]
    yaw = robot["yaw"]

    good_heading = np.zeros(
        len(common_time),
        dtype=bool
    )

    for i in range(len(common_time)):

        pos = np.array([
            x[i],
            y[i]
        ])

        d = np.linalg.norm(
            waypoints - pos,
            axis=1
        )

        nearest = np.argmin(d)

        vec = (
            waypoints[nearest]
            - pos
        )

        desired = np.arctan2(
            vec[1],
            vec[0]
        )

        error = abs(
            wrap_angle(
                desired - yaw[i]
            )
        )

        if error < threshold:
            good_heading[i] = True

    heading_percent.append(
        100
        * good_heading.sum()
        * dt
        / total_time
    )


# ============================================================
# Heading plot
# ============================================================

if args.ral:
    plt.figure(figsize=FIGSIZE)
else:
    plt.figure(figsize=(6, 4))

plt.bar(
    [
        "Robot1",
        "Robot2",
        "Robot3"
    ],
    heading_percent
)

plt.ylabel(
    "% mission"
)

plt.title(
    "Heading toward nearest waypoint (<45°)"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "heading.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

if SAVE_PDF:
    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "heading.pdf"
        ),
        bbox_inches="tight"
    )


# ============================================================
# Stampa risultati
# ============================================================

log_file = os.path.join(
    OUTPUT_FOLDER,
    "results1.log"
)

with open(log_file, "w") as f:

    text = (
        "========== RESULTS ==========\n\n"
    )

    print(
        text,
        end=""
    )

    f.write(text)

    for i in range(3):

        text = (
            f"Robot {i+1}\n"
            f"Time near waypoint : "
            f"{percent_near_wp[i].sum():.2f}%\n"
            f"Heading to waypoint: "
            f"{heading_percent[i]:.2f}%\n\n"
        )

        print(
            text,
            end=""
        )

        f.write(text)


# ============================================================
# Summary
# ============================================================

summary = pd.DataFrame({
    "Robot": [
        "Robot1",
        "Robot2",
        "Robot3"
    ],

    "NearWaypoint_%":
        percent_near_wp.sum(axis=1),

    "Heading_%":
        heading_percent
})

summary.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "summary.csv"
    ),
    index=False
)


# ============================================================
# Dettaglio waypoint
# ============================================================

columns = [
    f"WP{i+1}"
    for i in range(len(waypoints))
]

detail = pd.DataFrame(
    percent_near_wp,
    columns=columns,
    index=[
        "Robot1",
        "Robot2",
        "Robot3"
    ]
)

detail.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "waypoint_statistics.csv"
    )
)


# ============================================================
# SHOW
# ============================================================

if args.noshow:
    plt.show()