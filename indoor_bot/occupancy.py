import os
import glob
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import LineString, box
from shapely.ops import unary_union


# ==================================================
# RAL FIGURE FORMAT
# ==================================================

FIGSIZE = (4.0, 3.0)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,

    "axes.titlesize": 9,
    "axes.labelsize": 8,

    "xtick.labelsize": 7,
    "ytick.labelsize": 7,

    "legend.fontsize": 7,

    "axes.linewidth": 0.8,

    "lines.linewidth": 1.2,

    "figure.dpi": 150,
    "savefig.dpi": 300,

    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,

    "xtick.major.size": 3,
    "ytick.major.size": 3,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ==================================================
# CONFIGURATION
# ==================================================

parser = argparse.ArgumentParser()

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

RAL_FIGSIZE = (4.0, 3.0)

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


def get_figsize(default):
    """Return the RAL figure size when --ral is enabled."""
    return RAL_FIGSIZE if args.ral else default


# ==================================================
# BASE PATH
# ==================================================

BASE_PATH = (
    "/home/gonazza/container_ws/"
    "catkin_ws/src/indoor_bot/indoor_bot"
)


CONFIGURATIONS = [
    "config1",
    "config2",
    "config3",
]


# ==================================================
# MISSION TIME FILES
# ==================================================

MISSION_TIME_FILES = [
    "rosbot_1_0_mission_times.csv",
    "rosbot_2_1_mission_times.csv",
    "rosbot_3_2_mission_times.csv",
]


# ==================================================
# COVERAGE PARAMETERS
# ==================================================

ROBOT_RADIUS = 2.5


# ==================================================
# COLORS
# ==================================================

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

WAYPOINTS = {

    "config1": [
        (-6.5, -1.0),
        (3.0, 2.0),
        (3.5, -3.5),
        (-10.0, 2.0),
        (4.0, -2.5),
    ],

    "config2": [
        (-3.0, 0.0),
        (5.0, 2.5),
        (2.5, -4.5),
        (-10.0, 0.0),
        (5.0, -1.5),
    ],

    "config3": [
        (-5.0, 2.0),
        (4.0, 2.0),
        (0.0, -3.0),
        (-9.0, 3.0),
        (-8.0, -1.0),
    ],
}


# ==================================================
# GET WAYPOINTS
# ==================================================

def get_waypoints(config_name):

    if config_name not in WAYPOINTS:

        raise ValueError(
            f"Unknown configuration '{config_name}'. "
            f"Available configurations: "
            f"{list(WAYPOINTS.keys())}"
        )

    waypoints = WAYPOINTS[config_name]

    print()
    print(
        f"Using waypoints for {config_name}:"
    )

    for i, (x, y) in enumerate(waypoints):

        print(
            f"  WP{i + 1}: "
            f"x = {x:.2f}, "
            f"y = {y:.2f}"
        )

    return waypoints


# ==================================================
# GET EXPERIMENTS
# ==================================================

def get_experiments():

    experiments = []

    for config_name in CONFIGURATIONS:

        config_path = os.path.join(
            BASE_PATH,
            config_name
        )

        if not os.path.isdir(config_path):

            print(
                f"WARNING: configuration folder "
                f"not found: {config_path}"
            )

            continue

        # ------------------------------------------
        # Equal experiments
        # ------------------------------------------

        equal_experiments = sorted(
            glob.glob(
                os.path.join(
                    config_path,
                    "exp_equal*"
                )
            )
        )

        # ------------------------------------------
        # Reward experiments
        # ------------------------------------------

        reward_experiments = sorted(
            glob.glob(
                os.path.join(
                    config_path,
                    "exp_reward*"
                )
            )
        )

        config_experiments = (
            equal_experiments +
            reward_experiments
        )

        print()
        print(
            f"{config_name}: "
            f"{len(equal_experiments)} equal + "
            f"{len(reward_experiments)} reward"
        )

        for experiment in config_experiments:

            if os.path.isdir(experiment):

                experiments.append(
                    (
                        config_name,
                        experiment
                    )
                )

    return experiments


# ==================================================
# LOGGING
# ==================================================

def log_print(
        text,
        log_file):

    line = f"{text}"

    print(line)

    with open(
        log_file,
        "a"
    ) as f:

        f.write(
            line + "\n"
        )


# ==================================================
# MISSION TIME
# ==================================================

def get_max_mission_time(
        csv_folder):

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

    return max(
        mission_times
    )


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
        subset=[
            "x",
            "y"
        ]
    )

    x = df[
        "x"
    ].to_numpy()

    y = df[
        "y"
    ].to_numpy()

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

    total_valid_area = (
        valid_area.area
    )

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

        uncovered_area = (
            valid_area.difference(
                covered_area
            )
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
        trajectories,
        waypoints,
        output_dir):

    fig, ax = plt.subplots(
        figsize=get_figsize(FIGSIZE)
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
        linewidth=0.8
    )

    # ----------------------------------------------
    # Trajectories
    # ----------------------------------------------

    for i, (x, y) in enumerate(
        trajectories
    ):

        color = TRAJECTORY_COLORS[
            i % len(
                TRAJECTORY_COLORS
            )
        ]

        ax.plot(
            x,
            y,
            color=color,
            linewidth=1.2,
            label=f"Robot {i + 1}",
            zorder=5
        )

        # Starting point
        ax.scatter(
            x[0],
            y[0],
            color=START_COLOR,
            marker="o",
            s=20,
            zorder=10
        )

    # ----------------------------------------------
    # Waypoints
    # ----------------------------------------------

    for i, (wx, wy) in enumerate(
        waypoints
    ):

        ax.scatter(
            wx,
            wy,
            color=WAYPOINT_COLOR,
            marker="*",
            s=55,
            zorder=10
        )

        ax.annotate(
            f"WP{i + 1}",
            (wx, wy),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
            zorder=11
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

    # ----------------------------------------------
    # Grid
    # ----------------------------------------------

    ax.grid(
        True,
        linewidth=0.5,
        alpha=0.25
    )

    ax.set_axisbelow(
        True
    )

    # ----------------------------------------------
    # Spines
    # ----------------------------------------------

    for spine in ax.spines.values():

        spine.set_linewidth(
            0.8
        )

    # ----------------------------------------------
    # Legend
    # ----------------------------------------------

    ax.legend(
        loc="best",
        frameon=True,
        framealpha=0.9,
        edgecolor="black"
    )

    # ----------------------------------------------
    # Layout
    # ----------------------------------------------

    fig.tight_layout()

    # ----------------------------------------------
    # PNG
    # ----------------------------------------------

    output_png = os.path.join(
        output_dir,
        "occupancy_map.png"
    )

    fig.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight"
    )

    # ----------------------------------------------
    # PDF
    # ----------------------------------------------

    output_pdf = os.path.join(
        output_dir,
        "occupancy_map.pdf"
    )

    fig.savefig(
        output_pdf,
        bbox_inches="tight"
    )

    print(
        f"Coverage map saved to: "
        f"{output_png}"
    )

    print(
        f"Coverage map saved to: "
        f"{output_pdf}"
    )

    # ----------------------------------------------
    # Show
    # ----------------------------------------------

    # if args.noshow:

    #     plt.show()

    plt.close(fig)


# ==================================================
# PROCESS ONE EXPERIMENT
# ==================================================

def process_experiment(
        experiment_folder,
        config_name):

    print()
    print(
        "=========================================="
    )
    print(
        f"PROCESSING: {config_name}"
    )
    print(
        f"EXPERIMENT: "
        f"{os.path.basename(experiment_folder)}"
    )
    print(
        "=========================================="
    )

    csv_folder = experiment_folder

    # ----------------------------------------------
    # Output
    # ----------------------------------------------

    output_dir = os.path.join(
        csv_folder,
        "plots"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    log_file = os.path.join(
        output_dir,
        "results.log"
    )

    # ----------------------------------------------
    # Waypoints
    # ----------------------------------------------

    waypoints = get_waypoints(
        config_name
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
        csv_folder
    )

    log_print(
        f"MAX_TIME used: "
        f"{max_time:.2f} s",
        log_file
    )

    # ----------------------------------------------
    # Find trajectories
    # ----------------------------------------------

    csv_files = glob.glob(
        os.path.join(
            csv_folder,
            "rosbot*_position.csv"
        )
    )

    csv_files = sorted(
        csv_files
    )

    if len(csv_files) == 0:

        print(
            "WARNING: No CSV files found. "
            "Skipping experiment."
        )

        return

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
        "===== GEOMETRIC COVERAGE RESULTS =====",
        log_file
    )

    log_print(
        f"Configuration: {config_name}",
        log_file
    )

    log_print(
        f"Experiment: "
        f"{os.path.basename(experiment_folder)}",
        log_file
    )

    log_print(
        f"Total world area: "
        f"{total_world_area:.4f} m²",
        log_file
    )

    log_print(
        f"Excluded area: "
        f"{excluded_area:.4f} m²",
        log_file
    )

    log_print(
        f"Valid area: "
        f"{total_valid_area:.4f} m²",
        log_file
    )

    log_print(
        f"Covered area: "
        f"{covered_area_value:.4f} m²",
        log_file
    )

    log_print(
        f"Uncovered area: "
        f"{uncovered_area_value:.4f} m²",
        log_file
    )

    log_print(
        f"Coverage: "
        f"{coverage_percentage:.4f}%",
        log_file
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
        trajectories,
        waypoints,
        output_dir
    )


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

    print()
    print(
        f"Base path: {BASE_PATH}"
    )

    print(
        f"Configurations: "
        f"{', '.join(CONFIGURATIONS)}"
    )

    # ----------------------------------------------
    # Find all experiments
    # ----------------------------------------------

    experiments = get_experiments()

    if len(experiments) == 0:

        raise RuntimeError(
            "No experiments found."
        )

    print()
    print(
        f"Total experiments found: "
        f"{len(experiments)}"
    )

    # ----------------------------------------------
    # Process all experiments
    # ----------------------------------------------

    processed = 0
    failed = 0

    for config_name, experiment_folder in experiments:

        try:

            process_experiment(
                experiment_folder,
                config_name
            )

            processed += 1

        except Exception as e:

            failed += 1

            print()
            print(
                "ERROR processing experiment:"
            )

            print(
                f"  Configuration: "
                f"{config_name}"
            )

            print(
                f"  Experiment: "
                f"{experiment_folder}"
            )

            print(
                f"  Error: {e}"
            )

    # ----------------------------------------------
    # Summary
    # ----------------------------------------------

    print()
    print(
        "=========================================="
    )
    print(
        " PROCESSING COMPLETE"
    )
    print(
        "=========================================="
    )

    print(
        f"Experiments processed: {processed}"
    )

    print(
        f"Experiments failed:    {failed}"
    )

    print(
        f"Total experiments:     "
        f"{len(experiments)}"
    )


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    main()