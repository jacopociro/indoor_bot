import os
import glob
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml

from shapely.geometry import LineString, box
from shapely.ops import unary_union


# ==================================================
# CONFIGURATION
# ==================================================

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
CSV_FOLDER = EXPERIMENT_FOLDER


# ==================================================
# MISSION TIME FILES
# ==================================================

MISSION_TIME_FILES = [
    "rosbot_1_0_mission_times.csv",
    "rosbot_2_1_mission_times.csv",
    "rosbot_3_2_mission_times.csv",
]


# ==================================================
# OUTPUT
# ==================================================

OUTPUT_DIR = os.path.join(
    CSV_FOLDER,
    "plots"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


LOG_FILE = os.path.join(
    OUTPUT_DIR,
    "results.log"
)


# ==================================================
# COVERAGE PARAMETERS
# ==================================================

ROBOT_RADIUS = 2.5       # meters


# ==================================================
# COLORS
# ==================================================
#
# Puoi modificare liberamente questi colori.
#
# Esempi:
# "lightgreen"
# "green"
# "lime"
# "#90EE90"
# "#00FF00"
#
# oppure colori matplotlib:
# "tab:green"
# "tab:blue"
# etc.
#

COVERED_COLOR = "green"
UNCOVERED_COLOR = "white"
EXCLUDED_COLOR = "black"

TRAJECTORY_COLORS = [
    "red",
    "blue",
    "orange",
]

WAYPOINT_COLOR = "purple"
START_COLOR = "black"


# ==================================================
# WORLD BOUNDS
# ==================================================

XMIN = -14.0
XMAX = 7.0
YMIN = -6.0
YMAX = 4.0


# ==================================================
# AREAS TO EXCLUDE
# ==================================================
#
# Format:
# (xmin, xmax, ymin, ymax)
#

EMPTY_AREAS = [
    (-4.0, -1.0, -6.0, -1.0),
    (-14.0, -8.0, 0.0, 0.1),
    (2.0, 5.0, -0.1, 0.0),
    (1.8, 2.0, 2.0, 4.0),
    (-1.0, 1.0, -6.0, -5.0),
    (4.0, 7.0, -6.0, -5.0),
    (5.0, 7.0, -6.0, -4.0),
]


# ==================================================
# WAYPOINTS
# ==================================================

WAYPOINTS_FILE = (
    "/home/gonazza/container_ws/"
    "catkin_ws/src/indoor_bot/config/waypoints.yaml"
)


def read_waypoints(yaml_file):

    waypoints = []

    if os.path.exists(yaml_file):

        with open(yaml_file, "r") as f:

            data = yaml.safe_load(f)

            if data and "wp" in data:

                for wp in data["wp"]:

                    waypoints.append(
                        (
                            wp["x"],
                            wp["y"]
                        )
                    )

    return waypoints


waypoints = read_waypoints(
    WAYPOINTS_FILE
)


# ==================================================
# LOGGING
# ==================================================

def log_print(text):

    line = f"{text}"

    print(line)

    with open(LOG_FILE, "a") as f:

        f.write(
            line + "\n"
        )


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

        max_time = df[
            "charging_completion_time"
        ].max()

        mission_times.append(
            max_time
        )

        print(
            f"{file}: mission end = "
            f"{max_time:.2f} s"
        )

    return max(mission_times)


# ==================================================
# READ TRAJECTORY
# ==================================================

def read_trajectory(
        csv_file,
        max_time):

    df = pd.read_csv(
        csv_file
    )

    # Limit to mission time
    df = df[
        df["time"] <= max_time
    ]

    # Keep previous behavior
    df = df.iloc[2:]

    # Remove invalid values
    df = df.dropna(
        subset=["x", "y"]
    )

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    return x, y


# ==================================================
# BUILD WORLD GEOMETRY
# ==================================================

def build_world():

    return box(
        XMIN,
        YMIN,
        XMAX,
        YMAX
    )


# ==================================================
# BUILD EXCLUDED AREAS
# ==================================================

def build_excluded_geometry():

    excluded = []

    for x0, x1, y0, y1 in EMPTY_AREAS:

        excluded.append(
            box(
                x0,
                y0,
                x1,
                y1
            )
        )

    if not excluded:

        return None

    return unary_union(
        excluded
    )


# ==================================================
# COMPUTE ROBOT COVERAGE
# ==================================================

def compute_robot_coverage(
        csv_files,
        max_time,
        robot_radius):

    robot_areas = []

    trajectories = []

    for csv_file in csv_files:

        print(
            f"Processing {csv_file}"
        )

        x, y = read_trajectory(
            csv_file,
            max_time
        )

        if len(x) < 2:

            print(
                f"WARNING: not enough "
                f"points in {csv_file}"
            )

            continue

        trajectories.append(
            (x, y)
        )

        # ------------------------------------------
        # Create trajectory
        # ------------------------------------------

        trajectory = LineString(
            np.column_stack(
                (x, y)
            )
        )

        # ------------------------------------------
        # Robot coverage
        #
        # The robot covers everything within
        # robot_radius from its trajectory.
        # ------------------------------------------

        coverage = trajectory.buffer(
            robot_radius,
            resolution=64
        )

        robot_areas.append(
            coverage
        )

    # ----------------------------------------------
    # Union of all robot coverage areas
    # ----------------------------------------------

    if robot_areas:

        total_coverage = unary_union(
            robot_areas
        )

    else:

        total_coverage = None

    return (
        total_coverage,
        trajectories
    )


# ==================================================
# COMPUTE COVERAGE STATISTICS
# ==================================================

def compute_statistics(
        world,
        excluded,
        coverage):

    # ----------------------------------------------
    # Valid world area
    # ----------------------------------------------

    if excluded is not None:

        valid_area = world.difference(
            excluded
        )

    else:

        valid_area = world

    total_valid_area = valid_area.area

    # ----------------------------------------------
    # Coverage inside valid world
    # ----------------------------------------------

    if coverage is not None:

        covered_area = coverage.intersection(
            valid_area
        )

    else:

        covered_area = None

    if covered_area is not None:

        covered_area_value = (
            covered_area.area
        )

    else:

        covered_area_value = 0.0

    # ----------------------------------------------
    # Percentage
    # ----------------------------------------------

    if total_valid_area > 0:

        coverage_percentage = (
            covered_area_value /
            total_valid_area
        ) * 100.0

    else:

        coverage_percentage = 0.0

    # ----------------------------------------------
    # Uncovered
    # ----------------------------------------------

    if covered_area is not None:

        uncovered_area = valid_area.difference(
            covered_area
        )

    else:

        uncovered_area = valid_area

    uncovered_area_value = (
        uncovered_area.area
    )

    return (
        total_valid_area,
        covered_area,
        covered_area_value,
        uncovered_area,
        uncovered_area_value,
        coverage_percentage
    )


# ==================================================
# PLOT POLYGON
# ==================================================

def plot_geometry(
        ax,
        geometry,
        color,
        alpha=1.0,
        edgecolor=None,
        linewidth=1.0):

    if geometry is None:
        return

    if geometry.is_empty:
        return

    # ----------------------------------------------
    # MultiPolygon
    # ----------------------------------------------

    if geometry.geom_type == "MultiPolygon":

        for polygon in geometry.geoms:

            x, y = polygon.exterior.xy

            ax.fill(
                x,
                y,
                facecolor=color,
                alpha=alpha,
                edgecolor=edgecolor,
                linewidth=linewidth
            )

    # ----------------------------------------------
    # Polygon
    # ----------------------------------------------

    elif geometry.geom_type == "Polygon":

        x, y = geometry.exterior.xy

        ax.fill(
            x,
            y,
            facecolor=color,
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=linewidth
        )


# ==================================================
# PLOT COVERAGE MAP
# ==================================================

def plot_coverage_map(
        world,
        excluded,
        covered_area,
        uncovered_area,
        trajectories):

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    # ----------------------------------------------
    # Uncovered area
    # ----------------------------------------------

    plot_geometry(
        ax,
        uncovered_area,
        UNCOVERED_COLOR
    )

    # ----------------------------------------------
    # Covered area
    # ----------------------------------------------

    plot_geometry(
        ax,
        covered_area,
        COVERED_COLOR
    )

    # ----------------------------------------------
    # Excluded areas
    # ----------------------------------------------

    plot_geometry(
        ax,
        excluded,
        EXCLUDED_COLOR,
        edgecolor="black",
        linewidth=1.0
    )

    # ----------------------------------------------
    # Trajectories
    # ----------------------------------------------

    for i, (x, y) in enumerate(
            trajectories):

        color = TRAJECTORY_COLORS[
            i % len(TRAJECTORY_COLORS)
        ]

        ax.plot(
            x,
            y,
            color=color,
            linewidth=1.5,
            label=f"Robot {i + 1}"
        )

        # Starting point
        ax.scatter(
            x[0],
            y[0],
            color=START_COLOR,
            marker="o",
            s=40,
            zorder=10
        )

    # ----------------------------------------------
    # Waypoints
    # ----------------------------------------------

    for i, (wx, wy) in enumerate(
            waypoints):

        ax.scatter(
            wx,
            wy,
            color=WAYPOINT_COLOR,
            marker="*",
            s=120,
            zorder=10
        )

        ax.annotate(
            f"WP{i + 1}",
            (wx, wy),
            xytext=(5, 5),
            textcoords="offset points"
        )

    # ----------------------------------------------
    # Plot settings
    # ----------------------------------------------

    ax.set_xlim(
        XMIN,
        XMAX
    )

    ax.set_ylim(
        YMIN,
        YMAX
    )

    ax.set_aspect(
        "equal"
    )

    ax.set_xlabel(
        "X [m]"
    )

    ax.set_ylabel(
        "Y [m]"
    )

    ax.set_title(
        "Coverage Map"
    )

    ax.grid(
        True,
        alpha=0.2
    )

    ax.legend()

    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_DIR,
        "occupancy_map.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    print(
        f"Coverage map saved to: "
        f"{output_file}"
    )

    if args.noshow:

        plt.show()

    plt.close()


# ==================================================
# MAIN
# ==================================================

def main():

    print(
        "=========================================="
    )
    print(
        " ACCURATE GEOMETRIC COVERAGE CALCULATION"
    )
    print(
        "=========================================="
    )

    # ----------------------------------------------
    # World
    # ----------------------------------------------

    world = build_world()

    excluded = build_excluded_geometry()

    # ----------------------------------------------
    # Mission time
    # ----------------------------------------------

    max_time = get_max_mission_time(
        CSV_FOLDER
    )

    log_print(
        f"MAX_TIME used: "
        f"{max_time:.2f} s"
    )

    # ----------------------------------------------
    # Find trajectories
    # ----------------------------------------------

    csv_files = glob.glob(
        os.path.join(
            CSV_FOLDER,
            "rosbot*_position.csv"
        )
    )

    csv_files = sorted(
        csv_files
    )

    if len(csv_files) == 0:

        raise FileNotFoundError(
            "No CSV files found"
        )

    print(
        f"Found {len(csv_files)} "
        f"trajectory files"
    )

    # ----------------------------------------------
    # Compute coverage
    # ----------------------------------------------

    coverage, trajectories = (
        compute_robot_coverage(
            csv_files,
            max_time,
            ROBOT_RADIUS
        )
    )

    # ----------------------------------------------
    # Statistics
    # ----------------------------------------------

    (
        total_valid_area,
        covered_area,
        covered_area_value,
        uncovered_area,
        uncovered_area_value,
        coverage_percentage
    ) = compute_statistics(
        world,
        excluded,
        coverage
    )

    # ----------------------------------------------
    # Results
    # ----------------------------------------------

    excluded_area = (
        excluded.area
        if excluded is not None
        else 0.0
    )

    total_world_area = world.area

    log_print(
        "===== GEOMETRIC COVERAGE RESULTS ====="
    )

    log_print(
        f"Total world area: "
        f"{total_world_area:.4f} m²"
    )

    log_print(
        f"Excluded area: "
        f"{excluded_area:.4f} m²"
    )

    log_print(
        f"Valid area: "
        f"{total_valid_area:.4f} m²"
    )

    log_print(
        f"Covered area: "
        f"{covered_area_value:.4f} m²"
    )

    log_print(
        f"Uncovered area: "
        f"{uncovered_area_value:.4f} m²"
    )

    log_print(
        f"Coverage: "
        f"{coverage_percentage:.4f}%"
    )

    print()
    print(
        f"Valid area      : "
        f"{total_valid_area:.4f} m²"
    )

    print(
        f"Covered area    : "
        f"{covered_area_value:.4f} m²"
    )

    print(
        f"Uncovered area  : "
        f"{uncovered_area_value:.4f} m²"
    )

    print(
        f"Coverage        : "
        f"{coverage_percentage:.4f}%"
    )

    # ----------------------------------------------
    # Plot
    # ----------------------------------------------

    plot_coverage_map(
        world,
        excluded,
        covered_area,
        uncovered_area,
        trajectories
    )


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    main()