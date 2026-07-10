import os
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ==================================================
# CONFIGURATION
# ==================================================
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

CSV_FOLDER = f"{EXPERIMENT_FOLDER}"

MISSION_TIME_FILES = [
    "rosbot_1_0_mission_times.csv",
    "rosbot_2_1_mission_times.csv",
    "rosbot_3_2_mission_times.csv",
]
# OUTPUT_FOLDER = f"./results/{EXPERIMENT_FOLDER}"
# CSV_FOLDER = "/home/gonazza/container_ws/catkin_ws/src/indoor_bot/indoor_bot/exp_equal2"
OUTPUT_DIR = os.path.join(CSV_FOLDER, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
GRID_RESOLUTION = 0.2      # meters
ROBOT_RADIUS = 2.5         # meters

# --------------------------------------------------
# Manual world bounds
# --------------------------------------------------
XMIN = -14.0
XMAX = 7.0
YMIN = -6.0
YMAX = 4.0

# --------------------------------------------------
# Rectangular areas to exclude
# Format:
# (xmin, xmax, ymin, ymax)
# --------------------------------------------------
EMPTY_AREAS = [
    (-4.0, -1.0, -6.0, -1.0),
    (-14.0, -8.0, 0.0, 0.1),
    (2.0, 5.0, -0.1, 0.0),
    (1.8, 2.0, 2.0, 4.0),
    (-1.0, 1.0, -6.0, -5.0),
    (4.0, 7.0, -6.0, -5.0),
    (5.0, 7.0, -6.0, -4.0),
]


LOG_FILE = os.path.join(OUTPUT_DIR, "results.log")


def log_print(text):
    
    line = f"{text}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ==================================================
# GRID UTILITIES
# ==================================================

def build_grid(xmin, xmax, ymin, ymax, resolution):
    width = int(np.ceil((xmax - xmin) / resolution))
    height = int(np.ceil((ymax - ymin) / resolution))

    visited = np.zeros((height, width), dtype=bool)
    excluded = np.zeros((height, width), dtype=bool)

    return visited, excluded


def world_to_grid(x, y, xmin, ymin, resolution):
    gx = int((x - xmin) / resolution)
    gy = int((y - ymin) / resolution)

    return gx, gy


# ==================================================
# EXCLUDED AREAS
# ==================================================

def mark_empty_areas(excluded_grid,
                     empty_areas,
                     xmin,
                     ymin,
                     resolution):

    height, width = excluded_grid.shape

    for x0, x1, y0, y1 in empty_areas:

        gx0, gy0 = world_to_grid(
            x0,
            y0,
            xmin,
            ymin,
            resolution
        )

        gx1, gy1 = world_to_grid(
            x1,
            y1,
            xmin,
            ymin,
            resolution
        )

        gx0 = max(0, min(width - 1, gx0))
        gx1 = max(0, min(width - 1, gx1))

        gy0 = max(0, min(height - 1, gy0))
        gy1 = max(0, min(height - 1, gy1))

        excluded_grid[gy0:gy1 + 1, gx0:gx1 + 1] = True


# ==================================================
# VISITED CELLS
# ==================================================
def mark_visited_cells(
        visited_grid,
        csv_files,
        xmin,
        ymin,
        resolution,
        robot_radius,
        max_time):

    height, width = visited_grid.shape

    radius_cells = int(
        np.ceil(robot_radius / resolution)
    )

    for csv_file in csv_files:

        print(f"Processing {csv_file}")

        df = pd.read_csv(csv_file)

        # taglio temporale
        df = df[df["time"] <= max_time]

        # mantiene il comportamento precedente
        df = df.iloc[2:]

        for _, row in df.iterrows():

            x = row["x"]
            y = row["y"]

            gx, gy = world_to_grid(
                x,
                y,
                xmin,
                ymin,
                resolution
            )

            for dx in range(
                    -radius_cells,
                    radius_cells + 1):

                for dy in range(
                        -radius_cells,
                        radius_cells + 1):

                    nx = gx + dx
                    ny = gy + dy

                    if not (
                        0 <= nx < width and
                        0 <= ny < height
                    ):
                        continue

                    if (
                        dx * dx +
                        dy * dy
                    ) <= (
                        radius_cells *
                        radius_cells
                    ):
                        visited_grid[ny, nx] = True

# ==================================================
# MISSION TIME
# ==================================================

def get_max_mission_time(csv_folder):

    mission_times = []

    for file in MISSION_TIME_FILES:

        path = os.path.join(
            csv_folder,
            file
        )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Mission time file not found: {path}"
            )

        df = pd.read_csv(path)

        # Usa il completamento della ricarica come tempo massimo
        max_time = df["charging_completion_time"].max()

        mission_times.append(max_time)

        print(
            f"{file}: mission end = {max_time:.2f} s"
        )

    # Tempo massimo globale tra i tre robot
    return max(mission_times)

# ==================================================
# MAIN
# ==================================================

def main():

    xmin = XMIN
    xmax = XMAX
    ymin = YMIN
    ymax = YMAX

    print("===== WORLD BOUNDS =====")
    print(f"xmin = {xmin}")
    print(f"xmax = {xmax}")
    print(f"ymin = {ymin}")
    print(f"ymax = {ymax}")

    visited_grid, excluded_grid = build_grid(
        xmin,
        xmax,
        ymin,
        ymax,
        GRID_RESOLUTION
    )

    # Mark manually excluded areas
    mark_empty_areas(
        excluded_grid,
        EMPTY_AREAS,
        xmin,
        ymin,
        GRID_RESOLUTION
    )
    MAX_TIME = get_max_mission_time(
    CSV_FOLDER
    )

    log_print(
        f"MAX_TIME used: {MAX_TIME:.2f} s"
    )
    csv_files = glob.glob(
        os.path.join(
            CSV_FOLDER,
            "rosbot*_position.csv"
        )
    )

    if len(csv_files) == 0:
        raise FileNotFoundError(
            "No CSV files found"
        )

    mark_visited_cells(
    visited_grid,
    csv_files,
    xmin,
    ymin,
    GRID_RESOLUTION,
    ROBOT_RADIUS,
    MAX_TIME
    )

    # ==================================================
    # Statistics
    # ==================================================

    total_cells = visited_grid.size

    excluded_cells = np.sum(excluded_grid)

    valid_cells = (
        total_cells -
        excluded_cells
    )

    visited_valid_cells = np.sum(
        np.logical_and(
            visited_grid,
            ~excluded_grid
        )
    )

    occupancy_percentage = (
        visited_valid_cells /
        valid_cells
    ) * 100.0

    log_print("===== OCCUPANCY RESULTS =====")
    log_print(f"Total cells: {total_cells}")
    log_print(f"Excluded cells: {excluded_cells}")
    log_print(f"Valid cells: {valid_cells}")
    log_print(f"Visited cells: {visited_valid_cells}")
    log_print(f"Occupancy %: {occupancy_percentage:.2f}%")

    # ==================================================
    # Visualization
    # ==================================================

    display_grid = np.zeros_like(
        visited_grid,
        dtype=int
    )

    # 0 = free
    # 1 = excluded
    # 2 = visited

    display_grid[excluded_grid] = 1

    display_grid[
        np.logical_and(
            visited_grid,
            ~excluded_grid
        )
    ] = 2

    plt.figure(figsize=(12, 8))

    plt.imshow(
        display_grid,
        origin="lower",
        interpolation="nearest"
    )

    plt.title("Coverage Map")

    plt.xlabel("Grid X")
    plt.ylabel("Grid Y")

    plt.colorbar(
        label="0=free, 1=excluded, 2=visited"
    )
    # ==================================================
    # ROBOT TRAJECTORIES
    # ==================================================

    csv_files = sorted(csv_files)

    colors = ["red", "blue", "green"]

    for i, csv_file in enumerate(csv_files[:3]):

        df = pd.read_csv(csv_file)

        df = df[df["time"] <= MAX_TIME]

        df = df.iloc[2:]
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()

        plt.plot(
            (x - xmin) / GRID_RESOLUTION,
            (y - ymin) / GRID_RESOLUTION,
            color=colors[i % len(colors)],
            linewidth=1.5,
            label=f"Robot {i+1}"
        )

        plt.scatter(
        (x[0] - xmin) / GRID_RESOLUTION,
        (y[0] - ymin) / GRID_RESOLUTION,
        color=colors[i % len(colors)],
        marker="o",
        s=40
        )
    plt.savefig(os.path.join(OUTPUT_DIR, "occupancy_map.png"), dpi=300)
    plt.tight_layout()
    if args.noshow:
        plt.show()


if __name__ == "__main__":
    main()