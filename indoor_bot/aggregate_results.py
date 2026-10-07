import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Genera i risultati per tutti gli esperimenti "
        "delle configurazioni config1/config2/config3."
    )
)

parser.add_argument(
    "--base-path",
    default="/home/gonazza/container_ws/catkin_ws/src/indoor_bot/indoor_bot",
    help="Percorso base degli esperimenti."
)

parser.add_argument(
    "--ral",
    action="store_true",
    help="Usa il formato grafico IEEE RA-L."
)

args = parser.parse_args()

BASE_PATH = args.base_path
RAL_MODE = args.ral


# ============================================================
# RAL STYLE
# ============================================================

RAL_FIGSIZE = (4.0, 3.0)

if RAL_MODE:
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
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def get_figsize(default=(8, 5)):
    if RAL_MODE:
        return RAL_FIGSIZE
    return default


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG_NAMES = [
    "config1",
    "config2",
    "config3",
]

GROUPS = [
    "equal",
    "reward",
]

ROBOTS = [
    "rosbot_1_0",
    "rosbot_2_1",
    "rosbot_3_2",
]

ROBOT_LABELS = {
    "rosbot_1_0": "Robot 1",
    "rosbot_2_1": "Robot 2",
    "rosbot_3_2": "Robot 3",
}

ROBOT_COLORS = {
    "rosbot_1_0": "tab:blue",
    "rosbot_2_1": "tab:orange",
    "rosbot_3_2": "tab:green",
}


# ============================================================
# WAYPOINTS
# ============================================================

CONFIG_WAYPOINTS = {

    "config1": np.array([
        [-6.5, -1.0],
        [3.0, 2.0],
        [3.5, -3.5],
        [-10.0, 2.0],
        [4.0, -2.5],
    ]),

    "config2": np.array([
        [-3.0, 0.0],
        [5.0, 2.5],
        [2.5, -4.5],
        [-10.0, 0.0],
        [5.0, -1.5],
    ]),

    "config3": np.array([
        [-5.0, 2.0],
        [4.0, 2.0],
        [0.0, -3.0],
        [-9.0, 3.0],
        [-8.0, -1.0],
    ]),
}

WAYPOINT_VISIT_THRESHOLD = 2.5
N_WAYPOINTS = 5


# ============================================================
# GENERAL PARAMETERS
# ============================================================

DT = 0.05
TIME_SAMPLES = 500

MISSION_TIME_FILES = [
    "rosbot_1_0_mission_times.csv",
    "rosbot_2_1_mission_times.csv",
    "rosbot_3_2_mission_times.csv",
]

CONFIG_COMPARISON_DIR = os.path.join(
    BASE_PATH,
    "configuration_comparison"
)

# Nuova cartella:
# Equal vs Reward mediato tra config1/config2/config3
ALL_CONFIG_EQUAL_REWARD_DIR = os.path.join(
    CONFIG_COMPARISON_DIR,
    "equal_reward_all_configurations"
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def save(fig, folder, filename):

    os.makedirs(folder, exist_ok=True)

    path = os.path.join(
        folder,
        filename
    )

    fig.savefig(path)
    plt.close(fig)

    print(f"Saved: {path}")


def mean_std(curves):

    if len(curves) == 0:
        return None, None

    min_length = min(
        len(curve)
        for curve in curves
    )

    trimmed = np.array([
        curve[:min_length]
        for curve in curves
    ])

    mean = np.mean(
        trimmed,
        axis=0
    )

    std = np.std(
        trimmed,
        axis=0
    )

    return mean, std


def safe_float(value):

    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


# ============================================================
# MISSION TIME
# ============================================================

def get_max_time(experiment_path):

    times = []

    for filename in MISSION_TIME_FILES:

        path = os.path.join(
            experiment_path,
            filename
        )

        if not os.path.exists(path):
            continue

        try:

            df = pd.read_csv(path)

            if "charging_completion_time" in df.columns:

                values = pd.to_numeric(
                    df["charging_completion_time"],
                    errors="coerce"
                ).dropna()

                if len(values) > 0:
                    times.append(
                        values.iloc[-1]
                    )

        except Exception as exc:

            print(
                f"Warning reading {path}: {exc}"
            )

    if len(times) == 0:
        return None

    return max(times)


# ============================================================
# CSV LOADING / INTERPOLATION
# ============================================================

def load_interpolated_csv(
    path,
    common_time
):

    if not os.path.exists(path):
        return None

    try:

        df = pd.read_csv(path)

    except Exception as exc:

        print(
            f"Warning reading {path}: {exc}"
        )

        return None

    if "time" not in df.columns:
        return None

    df = df.copy()

    df["time"] = pd.to_numeric(
        df["time"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["time"]
    )

    if len(df) == 0:
        return None

    relative_time = (
        df["time"].values
        - df["time"].values[0]
    )

    result = pd.DataFrame({
        "time": common_time
    })

    for column in df.columns:

        if column == "time":
            continue

        numeric = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        valid = numeric.notna()

        if valid.sum() == 0:
            continue

        x = relative_time[valid.values]
        y = numeric[valid].values

        if len(x) == 1:

            result[column] = np.full(
                len(common_time),
                y[0]
            )

        else:

            result[column] = np.interp(
                common_time,
                x,
                y
            )

    return result


# ============================================================
# EXPERIMENT LOADING
# ============================================================

def load_experiment(
    experiment_path,
    common_time
):

    experiment_data = {}

    for robot in ROBOTS:

        position_path = os.path.join(
            experiment_path,
            f"{robot}_position.csv"
        )

        battery_path = os.path.join(
            experiment_path,
            f"{robot}_battery.csv"
        )

        priority_path = os.path.join(
            experiment_path,
            f"{robot}_priority.csv"
        )

        memory_path = os.path.join(
            experiment_path,
            f"{robot}_memory.csv"
        )

        experiment_data[robot] = {

            "position": load_interpolated_csv(
                position_path,
                common_time
            ),

            "battery": load_interpolated_csv(
                battery_path,
                common_time
            ),

            "priority": load_interpolated_csv(
                priority_path,
                common_time
            ),

            "memory": load_interpolated_csv(
                memory_path,
                common_time
            ),
        }

    # --------------------------------------------------------
    # Photosynthesis
    # --------------------------------------------------------

    photosynthesis_path = os.path.join(
        experiment_path,
        "photosynthesis.csv"
    )

    experiment_data["photosynthesis"] = (
        load_interpolated_csv(
            photosynthesis_path,
            common_time
        )
    )

    # --------------------------------------------------------
    # Mission times
    # --------------------------------------------------------

    experiment_data["mission_times"] = {}

    for robot in ROBOTS:

        path = os.path.join(
            experiment_path,
            f"{robot}_mission_times.csv"
        )

        if os.path.exists(path):

            try:

                experiment_data[
                    "mission_times"
                ][robot] = pd.read_csv(path)

            except Exception:

                experiment_data[
                    "mission_times"
                ][robot] = None

        else:

            experiment_data[
                "mission_times"
            ][robot] = None

    return experiment_data


# ============================================================
# LOAD GROUP
# ============================================================

def load_group(
    experiment_list
):

    max_times = []

    valid_experiments = []

    for experiment_path in experiment_list:

        max_time = get_max_time(
            experiment_path
        )

        if max_time is None:
            continue

        max_times.append(
            max_time
        )

        valid_experiments.append(
            experiment_path
        )

    if len(max_times) == 0:

        return {
            "time": np.array([]),
            "data": []
        }

    max_group_time = max(
        max_times
    )

    common_time = np.arange(
        0,
        max_group_time,
        0.1
    )

    data = []

    for experiment_path in valid_experiments:

        experiment_data = load_experiment(
            experiment_path,
            common_time
        )

        data.append(
            experiment_data
        )

    return {
        "time": common_time,
        "data": data
    }


# ============================================================
# EXPERIMENT DISCOVERY
# ============================================================

def get_experiment_list(
    config_name,
    group
):

    config_path = os.path.join(
        BASE_PATH,
        config_name
    )

    experiments = []

    for i in range(1, 11):

        experiment_path = os.path.join(
            config_path,
            f"exp_{group}{i}"
        )

        if os.path.isdir(
            experiment_path
        ):
            experiments.append(
                experiment_path
            )

    return experiments


# ============================================================
# AGGREGATE TRAJECTORIES
# ============================================================

def aggregate_trajectories(
    group_data,
    output_folder
):

    if len(group_data["data"]) == 0:
        return

    fig, ax = plt.subplots(
        figsize=get_figsize()
    )

    for robot in ROBOTS:

        trajectories = []

        for experiment in group_data["data"]:

            df = experiment[
                robot
            ]["position"]

            if df is None:
                continue

            if (
                "x" not in df.columns
                or "y" not in df.columns
            ):
                continue

            trajectories.append(
                (
                    df["x"].values,
                    df["y"].values
                )
            )

        for x, y in trajectories:

            ax.plot(
                x,
                y,
                alpha=0.25,
                linewidth=0.8,
                color=ROBOT_COLORS[robot]
            )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Aggregated trajectories")

    ax.grid(
        alpha=0.25
    )

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    save(
        fig,
        output_folder,
        "trajectories.png"
    )


# ============================================================
# AGGREGATE PRIORITY
# ============================================================

def aggregate_priority(
    group_data,
    output_folder
):

    if len(group_data["data"]) == 0:
        return

    fig, ax = plt.subplots(
        figsize=get_figsize()
    )

    time = group_data["time"]

    for robot in ROBOTS:

        curves = []

        for experiment in group_data["data"]:

            df = experiment[
                robot
            ]["priority"]

            if (
                df is None
                or "priority" not in df.columns
            ):
                continue

            curves.append(
                df["priority"].values
            )

        if len(curves) == 0:
            continue

        mean, std = mean_std(
            curves
        )

        ax.plot(
            time[:len(mean)],
            mean,
            label=ROBOT_LABELS[robot]
        )

        ax.fill_between(
            time[:len(mean)],
            mean - std,
            mean + std,
            alpha=0.15
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Priority")
    ax.set_title("Priority")

    ax.legend()
    ax.grid(alpha=0.25)

    save(
        fig,
        output_folder,
        "priority.png"
    )


# ============================================================
# AGGREGATE BATTERY
# ============================================================

def aggregate_battery(
    group_data,
    output_folder
):

    if len(group_data["data"]) == 0:
        return

    fig, ax = plt.subplots(
        figsize=get_figsize()
    )

    time = group_data["time"]

    for robot in ROBOTS:

        curves = []

        for experiment in group_data["data"]:

            df = experiment[
                robot
            ]["battery"]

            if df is None:
                continue

            numeric = df.select_dtypes(
                include=[np.number]
            )

            if numeric.shape[1] == 0:
                continue

            curves.append(
                numeric.iloc[:, 0].values
            )

        if len(curves) == 0:
            continue

        mean, std = mean_std(
            curves
        )

        ax.plot(
            time[:len(mean)],
            mean,
            label=ROBOT_LABELS[robot]
        )

        ax.fill_between(
            time[:len(mean)],
            mean - std,
            mean + std,
            alpha=0.15
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Battery")
    ax.set_title("Battery")

    ax.legend()
    ax.grid(alpha=0.25)

    save(
        fig,
        output_folder,
        "battery.png"
    )


# ============================================================
# AGGREGATE MEMORY
# ============================================================

def aggregate_memory(
    group_data,
    output_folder
):

    if len(group_data["data"]) == 0:
        return

    fig, ax = plt.subplots(
        figsize=get_figsize()
    )

    time = group_data["time"]

    curves = []

    for experiment in group_data["data"]:

        for robot in ROBOTS:

            df = experiment[
                robot
            ]["memory"]

            if df is None:
                continue

            numeric = df.select_dtypes(
                include=[np.number]
            )

            if numeric.shape[1] == 0:
                continue

            curves.append(
                numeric.iloc[:, 0].values
            )

    if len(curves) > 0:

        mean, std = mean_std(
            curves
        )

        ax.plot(
            time[:len(mean)],
            mean,
            label="Memory"
        )

        ax.fill_between(
            time[:len(mean)],
            mean - std,
            mean + std,
            alpha=0.15
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Memory")
    ax.set_title("Memory")

    ax.legend()
    ax.grid(alpha=0.25)

    save(
        fig,
        output_folder,
        "memory.png"
    )


# ============================================================
# COMPACTNESS
# ============================================================

def compute_compactness_curves(
    group_data
):

    time = group_data["time"]

    curves = []

    for experiment in group_data["data"]:

        robot_positions = []

        for robot in ROBOTS:

            df = experiment[
                robot
            ]["position"]

            if df is None:
                continue

            if (
                "x" not in df.columns
                or "y" not in df.columns
            ):
                continue

            robot_positions.append(
                np.column_stack([
                    df["x"].values,
                    df["y"].values
                ])
            )

        if len(robot_positions) < 2:
            continue

        min_length = min(
            len(pos)
            for pos in robot_positions
        )

        compactness = []

        for i in range(min_length):

            positions = np.array([
                pos[i]
                for pos in robot_positions
            ])

            centroid = np.mean(
                positions,
                axis=0
            )

            distances = np.linalg.norm(
                positions - centroid,
                axis=1
            )

            compactness.append(
                np.mean(distances)
            )

        curves.append(
            np.array(compactness)
        )

    if len(curves) == 0:
        return time, None

    mean, std = mean_std(
        curves
    )

    return time[:len(mean)], (
        mean,
        std
    )


def aggregate_compactness(
    group_data,
    output_folder
):

    time, result = compute_compactness_curves(
        group_data
    )

    if result is None:
        return

    mean, std = result

    fig, ax = plt.subplots(
        figsize=get_figsize()
    )

    ax.plot(
        time,
        mean,
        label="Compactness"
    )

    ax.fill_between(
        time,
        mean - std,
        mean + std,
        alpha=0.15
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mean distance [m]")
    ax.set_title("Swarm compactness")

    ax.legend()
    ax.grid(alpha=0.25)

    save(
        fig,
        output_folder,
        "compactness.png"
    )


# ============================================================
# WAYPOINT VISIT COUNT
# ============================================================

def count_waypoint_visits(
    position_df,
    waypoint
):

    if position_df is None:
        return 0

    if (
        "x" not in position_df.columns
        or "y" not in position_df.columns
    ):
        return 0

    x = pd.to_numeric(
        position_df["x"],
        errors="coerce"
    ).values

    y = pd.to_numeric(
        position_df["y"],
        errors="coerce"
    ).values

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return 0

    distances = np.sqrt(
        (x - waypoint[0]) ** 2
        + (y - waypoint[1]) ** 2
    )

    inside = (
        distances
        <= WAYPOINT_VISIT_THRESHOLD
    )

    if inside[0]:
        visits = 1
    else:
        visits = 0

    for i in range(1, len(inside)):

        if (
            inside[i]
            and not inside[i - 1]
        ):
            visits += 1

    return visits


def aggregate_waypoint_visits(
    experiment_list,
    config_name,
    group,
    output_folder
):

    waypoints = CONFIG_WAYPOINTS[
        config_name
    ]

    rows = []

    for experiment_index, experiment_path in enumerate(
        experiment_list,
        start=1
    ):

        row = {
            "experiment": experiment_index
        }

        total_visits = 0

        for robot in ROBOTS:

            position_path = os.path.join(
                experiment_path,
                f"{robot}_position.csv"
            )

            if not os.path.exists(
                position_path
            ):

                for waypoint_index in range(
                    N_WAYPOINTS
                ):
                    row[
                        f"{robot}_waypoint_{waypoint_index + 1}"
                    ] = 0

                row[
                    f"{robot}_total"
                ] = 0

                continue

            try:

                position_df = pd.read_csv(
                    position_path
                )

            except Exception:

                position_df = None

            robot_total = 0

            for waypoint_index, waypoint in enumerate(
                waypoints
            ):

                visits = count_waypoint_visits(
                    position_df,
                    waypoint
                )

                row[
                    f"{robot}_waypoint_{waypoint_index + 1}"
                ] = visits

                robot_total += visits

            row[
                f"{robot}_total"
            ] = robot_total

            total_visits += robot_total

        row["total_visits"] = total_visits

        rows.append(row)

    if len(rows) == 0:
        return

    statistics_df = pd.DataFrame(
        rows
    )

    csv_path = os.path.join(
        output_folder,
        "waypoint_visit_statistics.csv"
    )

    statistics_df.to_csv(
        csv_path,
        index=False
    )

    print(
        f"Saved: {csv_path}"
    )

    # --------------------------------------------------------
    # Mean visits per waypoint
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=get_figsize()
    )

    for robot in ROBOTS:

        means = []

        for waypoint_index in range(
            N_WAYPOINTS
        ):

            column = (
                f"{robot}_waypoint_"
                f"{waypoint_index + 1}"
            )

            if column in statistics_df:

                means.append(
                    statistics_df[column].mean()
                )

            else:

                means.append(0)

        ax.plot(
            range(
                1,
                N_WAYPOINTS + 1
            ),
            means,
            marker="o",
            label=ROBOT_LABELS[robot]
        )

    ax.set_xlabel("Waypoint")
    ax.set_ylabel("Mean number of visits")
    ax.set_title(
        f"Waypoint visits - {config_name} - {group}"
    )

    ax.set_xticks(
        range(
            1,
            N_WAYPOINTS + 1
        )
    )

    ax.grid(alpha=0.25)
    ax.legend()

    save(
        fig,
        output_folder,
        "waypoint_visits.png"
    )


# ============================================================
# PROCESS CONFIGURATION
# ============================================================

def process_configuration(
    config_name
):

    config_results = {}

    for group in GROUPS:

        experiments = get_experiment_list(
            config_name,
            group
        )

        print(
            f"\n{config_name} - {group}: "
            f"{len(experiments)} experiments"
        )

        group_data = load_group(
            experiments
        )

        config_results[group] = {
            "experiments": experiments,
            "data": group_data
        }

        output_folder = os.path.join(
            BASE_PATH,
            config_name,
            "aggregated_results",
            group
        )

        os.makedirs(
            output_folder,
            exist_ok=True
        )

        aggregate_trajectories(
            group_data,
            output_folder
        )

        aggregate_priority(
            group_data,
            output_folder
        )

        aggregate_battery(
            group_data,
            output_folder
        )

        aggregate_memory(
            group_data,
            output_folder
        )

        aggregate_compactness(
            group_data,
            output_folder
        )

        aggregate_waypoint_visits(
            experiments,
            config_name,
            group,
            output_folder
        )

    return config_results


# ============================================================
# CONFIGURATION COMPARISON
# ============================================================

def compare_time_series(
    configuration_results,
    group,
    metric,
    output_folder
):

    fig, ax = plt.subplots(
        figsize=get_figsize()
    )

    found = False

    for config_index, config_name in enumerate(
        CONFIG_NAMES
    ):

        if config_name not in configuration_results:
            continue

        if group not in configuration_results[
            config_name
        ]:
            continue

        group_result = configuration_results[
            config_name
        ][group]

        group_data = group_result["data"]

        time = group_data["time"]

        curves_by_robot = {}

        for robot in ROBOTS:
            curves_by_robot[robot] = []

        for experiment in group_data["data"]:

            for robot in ROBOTS:

                if metric == "priority":

                    df = experiment[
                        robot
                    ]["priority"]

                    if (
                        df is None
                        or "priority" not in df.columns
                    ):
                        continue

                    curves_by_robot[
                        robot
                    ].append(
                        df["priority"].values
                    )

                elif metric == "battery":

                    df = experiment[
                        robot
                    ]["battery"]

                    if df is None:
                        continue

                    numeric = df.select_dtypes(
                        include=[np.number]
                    )

                    if numeric.shape[1] == 0:
                        continue

                    curves_by_robot[
                        robot
                    ].append(
                        numeric.iloc[:, 0].values
                    )

        for robot in ROBOTS:

            curves = curves_by_robot[
                robot
            ]

            if len(curves) == 0:
                continue

            mean, std = mean_std(
                curves
            )

            label = (
                f"{config_name.capitalize()} "
                f"{ROBOT_LABELS[robot]}"
            )

            ax.plot(
                time[:len(mean)],
                mean,
                label=label
            )

            ax.fill_between(
                time[:len(mean)],
                mean - std,
                mean + std,
                alpha=0.08
            )

            found = True

    if not found:

        plt.close(fig)
        return

    ax.set_xlabel(
        "Time [s]"
    )

    if metric == "priority":
        ax.set_ylabel("Priority")

    elif metric == "battery":
        ax.set_ylabel("Battery")

    ax.set_title(
        f"{metric.capitalize()} - "
        f"{group.capitalize()}"
    )

    ax.legend(
        ncol=2
    )

    ax.grid(
        alpha=0.25
    )

    save(
        fig,
        output_folder,
        f"{metric}_configurations.png"
    )


# ============================================================
# CONFIGURATION WAYPOINT COMPARISON
# ============================================================

def compare_waypoint_visits_configurations(
    configuration_results,
    group,
    output_folder
):

    fig, ax = plt.subplots(
        figsize=get_figsize()
    )

    found = False

    x = np.arange(
        1,
        N_WAYPOINTS + 1
    )

    for config_index, config_name in enumerate(
        CONFIG_NAMES
    ):

        if config_name not in configuration_results:
            continue

        if group not in configuration_results[
            config_name
        ]:
            continue

        experiments = configuration_results[
            config_name
        ][group]["experiments"]

        waypoints = CONFIG_WAYPOINTS[
            config_name
        ]

        all_robot_means = []

        for robot in ROBOTS:

            waypoint_visits = []

            for waypoint_index, waypoint in enumerate(
                waypoints
            ):

                values = []

                for experiment_path in experiments:

                    position_path = os.path.join(
                        experiment_path,
                        f"{robot}_position.csv"
                    )

                    if not os.path.exists(
                        position_path
                    ):
                        continue

                    try:

                        df = pd.read_csv(
                            position_path
                        )

                    except Exception:

                        continue

                    values.append(
                        count_waypoint_visits(
                            df,
                            waypoint
                        )
                    )

                if len(values) > 0:
                    waypoint_visits.append(
                        np.mean(values)
                    )
                else:
                    waypoint_visits.append(0)

            all_robot_means.append(
                waypoint_visits
            )

        for robot_index, robot in enumerate(
            ROBOTS
        ):

            ax.plot(
                x,
                all_robot_means[
                    robot_index
                ],
                marker="o",
                label=(
                    f"{config_name.capitalize()} "
                    f"{ROBOT_LABELS[robot]}"
                )
            )

            found = True

    if not found:

        plt.close(fig)
        return

    ax.set_xlabel(
        "Waypoint"
    )

    ax.set_ylabel(
        "Mean number of visits"
    )

    ax.set_title(
        f"Waypoint visits - "
        f"{group.capitalize()}"
    )

    ax.set_xticks(
        x
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        ncol=2
    )

    save(
        fig,
        output_folder,
        f"waypoint_visits_{group}_configurations.png"
    )


# ============================================================
# EQUAL VS REWARD TIME SERIES
# ============================================================

def compare_equal_reward_time_series(
    configuration_results,
    config_name,
    metric,
    output_folder
):

    fig, ax = plt.subplots(
        figsize=get_figsize()
    )

    found = False

    for group in [
        "equal",
        "reward"
    ]:

        if group not in configuration_results[
            config_name
        ]:
            continue

        group_result = configuration_results[
            config_name
        ][group]

        group_data = group_result["data"]

        time = group_data["time"]

        if metric == "memory":

            curves = []

            for experiment in group_data["data"]:

                for robot in ROBOTS:

                    df = experiment[
                        robot
                    ]["memory"]

                    if df is None:
                        continue

                    numeric = df.select_dtypes(
                        include=[np.number]
                    )

                    if numeric.shape[1] == 0:
                        continue

                    curves.append(
                        numeric.iloc[:, 0].values
                    )

            if len(curves) == 0:
                continue

            mean, std = mean_std(
                curves
            )

            ax.plot(
                time[:len(mean)],
                mean,
                label=group.capitalize()
            )

            ax.fill_between(
                time[:len(mean)],
                mean - std,
                mean + std,
                alpha=0.08
            )

            found = True

        else:

            for robot in ROBOTS:

                curves = []

                for experiment in group_data["data"]:

                    if metric == "priority":

                        df = experiment[
                            robot
                        ]["priority"]

                        if (
                            df is None
                            or "priority" not in df.columns
                        ):
                            continue

                        curves.append(
                            df["priority"].values
                        )

                    elif metric == "battery":

                        df = experiment[
                            robot
                        ]["battery"]

                        if df is None:
                            continue

                        numeric = df.select_dtypes(
                            include=[np.number]
                        )

                        if numeric.shape[1] == 0:
                            continue

                        curves.append(
                            numeric.iloc[:, 0].values
                        )

                if len(curves) == 0:
                    continue

                mean, std = mean_std(
                    curves
                )

                ax.plot(
                    time[:len(mean)],
                    mean,
                    label=(
                        f"{group.capitalize()} "
                        f"{ROBOT_LABELS[robot]}"
                    )
                )

                ax.fill_between(
                    time[:len(mean)],
                    mean - std,
                    mean + std,
                    alpha=0.08
                )

                found = True

    if not found:

        plt.close(fig)
        return

    ax.set_xlabel(
        "Time [s]"
    )

    ax.set_ylabel(
        metric.capitalize()
    )

    ax.set_title(
        f"{config_name.capitalize()} - "
        f"{metric.capitalize()}"
    )

    ax.legend(
        ncol=2
    )

    ax.grid(
        alpha=0.25
    )

    save(
        fig,
        output_folder,
        f"{metric}_equal_reward.png"
    )


# ============================================================
# EQUAL VS REWARD PHOTOSYNTHESIS
# ============================================================

def compare_equal_reward_photosynthesis(
    configuration_results,
    config_name,
    output_folder
):

    fig, ax = plt.subplots(
        figsize=get_figsize()
    )

    found = False

    for group in [
        "equal",
        "reward"
    ]:

        if group not in configuration_results[
            config_name
        ]:
            continue

        group_data = configuration_results[
            config_name
        ][group]["data"]

        time = group_data["time"]

        curves = []

        for experiment in group_data["data"]:

            df = experiment[
                "photosynthesis"
            ]

            if df is None:
                continue

            numeric_columns = [
                column
                for column in df.columns
                if column != "time"
            ]

            if len(numeric_columns) == 0:
                continue

            values = pd.to_numeric(
                df[numeric_columns[0]],
                errors="coerce"
            ).values

            if len(values) == 0:
                continue

            curves.append(values)

        if len(curves) == 0:
            continue

        mean, std = mean_std(
            curves
        )

        ax.plot(
            time[:len(mean)],
            mean,
            label=group.capitalize()
        )

        ax.fill_between(
            time[:len(mean)],
            mean - std,
            mean + std,
            alpha=0.08
        )

        found = True

    if not found:

        plt.close(fig)
        return

    ax.set_xlabel(
        "Time [s]"
    )

    ax.set_ylabel(
        "Photosynthesis"
    )

    ax.set_title(
        f"{config_name.capitalize()} - "
        "Photosynthesis"
    )

    ax.legend()
    ax.grid(alpha=0.25)

    save(
        fig,
        output_folder,
        "photosynthesis_equal_reward.png"
    )


# ============================================================
# SCALAR METRIC COMPARISON
# ============================================================

def compare_equal_reward_scalar_metric(
    configuration_results,
    config_name,
    metric,
    output_folder
):

    rows = []

    for group in [
        "equal",
        "reward"
    ]:

        if group not in configuration_results[
            config_name
        ]:
            continue

        experiments = configuration_results[
            config_name
        ][group]["experiments"]

        values = []

        for experiment_path in experiments:

            path = os.path.join(
                experiment_path,
                f"{metric}.csv"
            )

            if not os.path.exists(path):
                continue

            try:

                df = pd.read_csv(path)

            except Exception:

                continue

            numeric = df.select_dtypes(
                include=[np.number]
            )

            if numeric.shape[1] == 0:
                continue

            values.extend(
                numeric.iloc[:, 0]
                .dropna()
                .values
                .tolist()
            )

        for value in values:

            rows.append({
                "group": group,
                "value": value
            })

    if len(rows) == 0:
        return

    result = pd.DataFrame(
        rows
    )

    fig, ax = plt.subplots(
        figsize=get_figsize()
    )

    groups = [
        "equal",
        "reward"
    ]

    data = []

    labels = []

    for group in groups:

        values = result[
            result["group"] == group
        ]["value"].values

        if len(values) == 0:
            continue

        data.append(values)
        labels.append(
            group.capitalize()
        )

    if len(data) == 0:

        plt.close(fig)
        return

    ax.boxplot(
        data,
        labels=labels
    )

    ax.set_ylabel(
        metric.capitalize()
    )

    ax.set_title(
        f"{config_name.capitalize()} - "
        f"{metric.capitalize()}"
    )

    ax.grid(
        alpha=0.25
    )

    # IMPORTANT:
    # This used to have the same filename as the
    # photosynthesis time-series plot.
    save(
        fig,
        output_folder,
        f"{metric}_scalar_equal_reward.png"
    )


# ============================================================
# MISSION TIME COMPARISON
# ============================================================

def compare_equal_reward_mission_time(
    configuration_results,
    config_name,
    output_folder
):

    rows = []

    for group in [
        "equal",
        "reward"
    ]:

        if group not in configuration_results[
            config_name
        ]:
            continue

        experiments = configuration_results[
            config_name
        ][group]["experiments"]

        for experiment_index, experiment_path in enumerate(
            experiments,
            start=1
        ):

            for robot in ROBOTS:

                path = os.path.join(
                    experiment_path,
                    f"{robot}_mission_times.csv"
                )

                if not os.path.exists(path):
                    continue

                try:

                    df = pd.read_csv(
                        path
                    )

                except Exception:

                    continue

                if (
                    "charging_completion_time"
                    not in df.columns
                ):
                    continue

                values = pd.to_numeric(
                    df[
                        "charging_completion_time"
                    ],
                    errors="coerce"
                ).dropna()

                if len(values) == 0:
                    continue

                rows.append({
                    "group": group,
                    "robot": robot,
                    "experiment": experiment_index,
                    "time": values.iloc[-1]
                })

    if len(rows) == 0:
        return

    result = pd.DataFrame(
        rows
    )

    fig, ax = plt.subplots(
        figsize=get_figsize()
    )

    positions = []
    data = []
    labels = []

    position = 1

    for group in [
        "equal",
        "reward"
    ]:

        for robot in ROBOTS:

            values = result[
                (
                    result["group"] == group
                )
                & (
                    result["robot"] == robot
                )
            ]["time"].values

            if len(values) == 0:
                continue

            data.append(values)
            positions.append(position)

            labels.append(
                f"{group.capitalize()}\n"
                f"{ROBOT_LABELS[robot]}"
            )

            position += 1

    if len(data) == 0:

        plt.close(fig)
        return

    ax.boxplot(
        data,
        positions=positions
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_ylabel(
        "Mission time [s]"
    )

    ax.set_title(
        f"{config_name.capitalize()} - "
        "Mission time"
    )

    ax.grid(
        alpha=0.25
    )

    save(
        fig,
        output_folder,
        "mission_time_equal_reward.png"
    )


# ============================================================
# WAYPOINT TIME COMPARISON
# ============================================================

def compare_equal_reward_waypoint_time(
    configuration_results,
    config_name,
    output_folder
):

    rows = []

    for group in [
        "equal",
        "reward"
    ]:

        if group not in configuration_results[
            config_name
        ]:
            continue

        experiments = configuration_results[
            config_name
        ][group]["experiments"]

        waypoints = CONFIG_WAYPOINTS[
            config_name
        ]

        for experiment_index, experiment_path in enumerate(
            experiments,
            start=1
        ):

            for robot in ROBOTS:

                path = os.path.join(
                    experiment_path,
                    f"{robot}_position.csv"
                )

                if not os.path.exists(path):
                    continue

                try:

                    df = pd.read_csv(
                        path
                    )

                except Exception:

                    continue

                if "time" not in df.columns:
                    continue

                if (
                    "x" not in df.columns
                    or "y" not in df.columns
                ):
                    continue

                time = pd.to_numeric(
                    df["time"],
                    errors="coerce"
                ).values

                x = pd.to_numeric(
                    df["x"],
                    errors="coerce"
                ).values

                y = pd.to_numeric(
                    df["y"],
                    errors="coerce"
                ).values

                valid = (
                    np.isfinite(time)
                    & np.isfinite(x)
                    & np.isfinite(y)
                )

                time = time[valid]
                x = x[valid]
                y = y[valid]

                if len(time) == 0:
                    continue

                for waypoint_index, waypoint in enumerate(
                    waypoints
                ):

                    distance = np.sqrt(
                        (
                            x - waypoint[0]
                        ) ** 2
                        + (
                            y - waypoint[1]
                        ) ** 2
                    )

                    inside = (
                        distance
                        <= WAYPOINT_VISIT_THRESHOLD
                    )

                    indices = np.where(
                        inside
                    )[0]

                    if len(indices) == 0:
                        continue

                    first_time = time[
                        indices[0]
                    ]

                    rows.append({
                        "group": group,
                        "robot": robot,
                        "waypoint": waypoint_index + 1,
                        "time": (
                            first_time
                            - time[0]
                        )
                    })

    if len(rows) == 0:
        return

    result = pd.DataFrame(
        rows
    )

    fig, ax = plt.subplots(
        figsize=get_figsize()
    )

    for group in [
        "equal",
        "reward"
    ]:

        values = []

        for waypoint_index in range(
            1,
            N_WAYPOINTS + 1
        ):

            subset = result[
                (
                    result["group"] == group
                )
                & (
                    result["waypoint"]
                    == waypoint_index
                )
            ]

            if len(subset) == 0:
                values.append(
                    np.nan
                )
            else:
                values.append(
                    subset["time"].mean()
                )

        ax.plot(
            range(
                1,
                N_WAYPOINTS + 1
            ),
            values,
            marker="o",
            label=group.capitalize()
        )

    ax.set_xlabel(
        "Waypoint"
    )

    ax.set_ylabel(
        "Time to waypoint [s]"
    )

    ax.set_title(
        f"{config_name.capitalize()} - "
        "Waypoint time"
    )

    ax.set_xticks(
        range(
            1,
            N_WAYPOINTS + 1
        )
    )

    ax.grid(alpha=0.25)
    ax.legend()

    save(
        fig,
        output_folder,
        "waypoint_time_equal_reward.png"
    )
def compare_equal_reward_mission_time_all_configurations(
        configuration_results,
        output_folder
):

    robots = [
        "rosbot_1_0",
        "rosbot_2_1",
        "rosbot_3_2"
    ]

    robot_labels = [
        "Robot 1",
        "Robot 2",
        "Robot 3",
        "Swarm"
    ]

    equal_config_values = {
        robot: []
        for robot in robots
    }

    reward_config_values = {
        robot: []
        for robot in robots
    }

    equal_swarm_values = []
    reward_swarm_values = []

    # ============================================================
    # LETTURA MISSION TIME
    # ============================================================

    def read_experiment_mission_times(experiment):

        mission_times = {}

        for robot in robots:

            filename = (
                f"{robot}_mission_times.csv"
            )

            filepath = os.path.join(
                experiment,
                filename
            )

            if not os.path.exists(filepath):
                continue

            df = pd.read_csv(filepath)

            if (
                "charging_completion_time"
                not in df.columns
            ):
                continue

            values = (
                pd.to_numeric(
                    df["charging_completion_time"],
                    errors="coerce"
                )
                .dropna()
                .values
            )

            if len(values) == 0:
                continue

            # Ultimo charging completion time
            mission_times[robot] = float(
                values[-1]
            )

        return mission_times

    # ============================================================
    # CONFIGURAZIONE PER CONFIGURAZIONE
    # ============================================================

    for config_name in CONFIG_NAMES:

        if config_name not in configuration_results:
            continue

        for group in ["equal", "reward"]:

            if group not in configuration_results[config_name]:
                continue

            experiments = (
                configuration_results[
                    config_name
                ][group]["experiments"]
            )

            group_values = {
                robot: []
                for robot in robots
            }

            swarm_values = []

            # ----------------------------------------------------
            # Esperimenti della configurazione
            # ----------------------------------------------------

            for experiment in experiments:

                # experiment è un PATH/stringa
                mission_times = (
                    read_experiment_mission_times(
                        experiment
                    )
                )

                robot_times = []

                for robot in robots:

                    if robot not in mission_times:
                        continue

                    value = mission_times[robot]

                    if np.isfinite(value):

                        group_values[robot].append(
                            value
                        )

                        robot_times.append(
                            value
                        )

                # ------------------------------------------------
                # Swarm = ultimo robot che termina
                # ------------------------------------------------

                if len(robot_times) == len(robots):

                    swarm_values.append(
                        max(robot_times)
                    )

            # ----------------------------------------------------
            # Media degli esperimenti della configurazione
            # ----------------------------------------------------

            target_values = (
                equal_config_values
                if group == "equal"
                else reward_config_values
            )

            target_swarm_values = (
                equal_swarm_values
                if group == "equal"
                else reward_swarm_values
            )

            for robot in robots:

                if len(group_values[robot]) > 0:

                    target_values[robot].append(
                        np.mean(
                            group_values[robot]
                        )
                    )

            if len(swarm_values) > 0:

                target_swarm_values.append(
                    np.mean(swarm_values)
                )

    # ============================================================
    # MEDIA TRA CONFIGURAZIONI
    # ============================================================

    equal_means = []
    equal_stds = []

    reward_means = []
    reward_stds = []

    for robot in robots:

        # --------------------------------------------------------
        # Equal
        # --------------------------------------------------------

        equal_values = [
            value
            for value in equal_config_values[robot]
            if np.isfinite(value)
        ]

        if len(equal_values) > 0:

            equal_means.append(
                np.mean(equal_values)
            )

            equal_stds.append(
                np.std(equal_values)
            )

        else:

            equal_means.append(
                np.nan
            )

            equal_stds.append(
                0
            )

        # --------------------------------------------------------
        # Reward
        # --------------------------------------------------------

        reward_values = [
            value
            for value in reward_config_values[robot]
            if np.isfinite(value)
        ]

        if len(reward_values) > 0:

            reward_means.append(
                np.mean(reward_values)
            )

            reward_stds.append(
                np.std(reward_values)
            )

        else:

            reward_means.append(
                np.nan
            )

            reward_stds.append(
                0
            )

    # ============================================================
    # SWARM
    # ============================================================

    equal_swarm_values = [
        value
        for value in equal_swarm_values
        if np.isfinite(value)
    ]

    reward_swarm_values = [
        value
        for value in reward_swarm_values
        if np.isfinite(value)
    ]

    if len(equal_swarm_values) > 0:

        equal_means.append(
            np.mean(equal_swarm_values)
        )

        equal_stds.append(
            np.std(equal_swarm_values)
        )

    else:

        equal_means.append(
            np.nan
        )

        equal_stds.append(
            0
        )

    if len(reward_swarm_values) > 0:

        reward_means.append(
            np.mean(reward_swarm_values)
        )

        reward_stds.append(
            np.std(reward_swarm_values)
        )

    else:

        reward_means.append(
            np.nan
        )

        reward_stds.append(
            0
        )

    # ============================================================
    # CSV
    # ============================================================

    rows = []

    for i, label in enumerate(robot_labels):

        rows.append({
            "Metric": label,
            "Equal_mean": equal_means[i],
            "Equal_std": equal_stds[i],
            "Reward_mean": reward_means[i],
            "Reward_std": reward_stds[i]
        })

    pd.DataFrame(rows).to_csv(
        os.path.join(
            output_folder,
            "mission_time_equal_reward_all_configurations.csv"
        ),
        index=False
    )

    # ============================================================
    # PLOT
    # ============================================================

    x = np.arange(
        len(robot_labels)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=get_figsize()
    )

    ax.bar(
        x - width / 2,
        equal_means,
        width,
        yerr=equal_stds,
        capsize=3,
        label="Equal"
    )

    ax.bar(
        x + width / 2,
        reward_means,
        width,
        yerr=reward_stds,
        capsize=3,
        label="Reward"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        robot_labels
    )

    ax.set_ylabel(
        "Mission completion time [s]"
    )

    ax.set_title(
        "Mission completion time - Equal vs Reward\n"
        "Mean across configurations"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25
    )

    save(
        fig,
        output_folder,
        "mission_time_equal_reward_all_configurations"
    )

    plt.close(fig)

    # ============================================================
    # Print numerical results
    # ============================================================

    print()
    print("=" * 75)
    print(
        "GLOBAL MISSION COMPLETION TIME - "
        "EQUAL VS REWARD"
    )
    print(
        "Average across config1, config2 and config3"
    )
    print("=" * 75)

    for i, label in enumerate(robot_labels):

        print(
            f"{label:8s} | "
            f"Equal: "
            f"{equal_means[i]:.3f} ± "
            f"{equal_stds[i]:.3f} s | "
            f"Reward: "
            f"{reward_means[i]:.3f} ± "
            f"{reward_stds[i]:.3f} s"
        )

    print("=" * 75)
# ============================================================
# WAYPOINT PERCENTAGE
# ============================================================

def compare_equal_reward_waypoint_percentage(
    configuration_results,
    config_name,
    output_folder
):

    rows = []

    for group in [
        "equal",
        "reward"
    ]:

        if group not in configuration_results[
            config_name
        ]:
            continue

        experiments = configuration_results[
            config_name
        ][group]["experiments"]

        waypoints = CONFIG_WAYPOINTS[
            config_name
        ]

        for experiment_path in experiments:

            for robot in ROBOTS:

                path = os.path.join(
                    experiment_path,
                    f"{robot}_position.csv"
                )

                if not os.path.exists(path):
                    continue

                try:

                    df = pd.read_csv(
                        path
                    )

                except Exception:

                    continue

                visited = 0

                for waypoint in waypoints:

                    if (
                        count_waypoint_visits(
                            df,
                            waypoint
                        ) > 0
                    ):
                        visited += 1

                percentage = (
                    100.0
                    * visited
                    / len(waypoints)
                )

                rows.append({
                    "group": group,
                    "robot": robot,
                    "percentage": percentage
                })

    if len(rows) == 0:
        return

    result = pd.DataFrame(
        rows
    )

    fig, ax = plt.subplots(
        figsize=get_figsize()
    )

    for robot in ROBOTS:

        values = []

        for group in [
            "equal",
            "reward"
        ]:

            subset = result[
                (
                    result["robot"] == robot
                )
                & (
                    result["group"] == group
                )
            ]

            if len(subset) == 0:
                values.append(
                    np.nan
                )
            else:
                values.append(
                    subset["percentage"].mean()
                )

        ax.plot(
            [0, 1],
            values,
            marker="o",
            label=ROBOT_LABELS[robot]
        )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_xticklabels(
        [
            "Equal",
            "Reward"
        ]
    )

    ax.set_ylabel(
        "Waypoints visited [%]"
    )

    ax.set_title(
        f"{config_name.capitalize()} - "
        "Waypoint coverage"
    )

    ax.set_ylim(
        0,
        100
    )

    ax.grid(alpha=0.25)
    ax.legend()

    save(
        fig,
        output_folder,
        "waypoint_percentage_equal_reward.png"
    )


# ============================================================
# EQUAL VS REWARD COMPACTNESS
# ============================================================

def compare_equal_reward_compactness(
    configuration_results,
    config_name,
    output_folder
):

    fig, ax = plt.subplots(
        figsize=get_figsize()
    )

    found = False

    for group in [
        "equal",
        "reward"
    ]:

        if group not in configuration_results[
            config_name
        ]:
            continue

        group_result = configuration_results[
            config_name
        ][group]

        group_data = group_result["data"]

        time, result = compute_compactness_curves(
            group_data
        )

        if result is None:
            continue

        mean, std = result

        ax.plot(
            time,
            mean,
            label=group.capitalize()
        )

        ax.fill_between(
            time,
            mean - std,
            mean + std,
            alpha=0.08
        )

        found = True

    if not found:

        plt.close(fig)
        return

    ax.set_xlabel(
        "Time [s]"
    )

    ax.set_ylabel(
        "Mean distance [m]"
    )

    ax.set_title(
        f"{config_name.capitalize()} - "
        "Swarm compactness"
    )

    ax.legend()
    ax.grid(alpha=0.25)

    save(
        fig,
        output_folder,
        "compactness_equal_reward.png"
    )


# ============================================================
# EQUAL VS REWARD WAYPOINT VISITS
# ============================================================

def compare_waypoint_visits_equal_reward(
    configuration_results,
    config_name,
    output_folder
):

    fig, ax = plt.subplots(
        figsize=get_figsize()
    )

    x = np.arange(
        1,
        N_WAYPOINTS + 1
    )

    found = False

    for group in [
        "equal",
        "reward"
    ]:

        if group not in configuration_results[
            config_name
        ]:
            continue

        experiments = configuration_results[
            config_name
        ][group]["experiments"]

        waypoints = CONFIG_WAYPOINTS[
            config_name
        ]

        means = []

        for waypoint in waypoints:

            values = []

            for experiment_path in experiments:

                for robot in ROBOTS:

                    path = os.path.join(
                        experiment_path,
                        f"{robot}_position.csv"
                    )

                    if not os.path.exists(path):
                        continue

                    try:

                        df = pd.read_csv(
                            path
                        )

                    except Exception:

                        continue

                    values.append(
                        count_waypoint_visits(
                            df,
                            waypoint
                        )
                    )

            if len(values) == 0:
                means.append(0)
            else:
                means.append(
                    np.mean(values)
                )

        ax.plot(
            x,
            means,
            marker="o",
            label=group.capitalize()
        )

        found = True

    if not found:

        plt.close(fig)
        return

    ax.set_xlabel(
        "Waypoint"
    )

    ax.set_ylabel(
        "Mean number of visits"
    )

    ax.set_title(
        f"{config_name.capitalize()} - "
        "Equal vs Reward"
    )

    ax.set_xticks(
        x
    )

    ax.grid(alpha=0.25)
    ax.legend()

    save(
        fig,
        output_folder,
        "waypoint_visits_equal_reward.png"
    )


# ============================================================
# ============================================================
# GLOBAL EQUAL VS REWARD
# AVERAGE ACROSS CONFIG1 / CONFIG2 / CONFIG3
# ============================================================
# ============================================================

def get_group_data(
    configuration_results,
    config_name,
    group
):

    if config_name not in configuration_results:
        return None

    if group not in configuration_results[
        config_name
    ]:
        return None

    return configuration_results[
        config_name
    ][group]["data"]


# ============================================================
# GLOBAL TIME AXIS
# ============================================================

def interpolate_curve(
    source_time,
    values,
    target_time
):

    source_time = np.asarray(
        source_time,
        dtype=float
    )

    values = np.asarray(
        values,
        dtype=float
    )

    n = min(
        len(source_time),
        len(values)
    )

    if n == 0:
        return None

    source_time = source_time[:n]
    values = values[:n]

    valid = (
        np.isfinite(source_time)
        & np.isfinite(values)
    )

    source_time = source_time[valid]
    values = values[valid]

    if len(source_time) == 0:
        return None

    if len(source_time) == 1:

        return np.full(
            len(target_time),
            values[0]
        )

    # Remove duplicated time samples.
    unique_time, unique_indices = np.unique(
        source_time,
        return_index=True
    )

    unique_values = values[
        unique_indices
    ]

    if len(unique_time) == 1:

        return np.full(
            len(target_time),
            unique_values[0]
        )

    return np.interp(
        target_time,
        unique_time,
        unique_values
    )


def get_common_global_time(
    configuration_results,
    group_curves
):

    max_times = []

    for config_name in CONFIG_NAMES:

        if config_name not in group_curves:
            continue

        if group_curves[
            config_name
        ] is None:
            continue

        time, curve = group_curves[
            config_name
        ]

        if (
            time is None
            or curve is None
            or len(time) == 0
            or len(curve) == 0
        ):
            continue

        max_times.append(
            min(
                time[-1],
                time[len(curve) - 1]
            )
        )

    if len(max_times) == 0:
        return None

    max_common_time = min(
        max_times
    )

    if max_common_time <= 0:
        return None

    return np.arange(
        0,
        max_common_time + 1e-9,
        0.1
    )


# ============================================================
# GLOBAL CONFIGURATION CURVE
# ============================================================

def get_configuration_curve(
    group_data,
    metric,
    robot=None
):

    if group_data is None:
        return None, None

    data = group_data["data"]
    time = group_data["time"]

    if len(data) == 0:
        return None, None

    curves = []

    # --------------------------------------------------------
    # Priority
    # --------------------------------------------------------

    if metric == "priority":

        if robot is None:
            return None, None

        for experiment in data:

            df = experiment[
                robot
            ]["priority"]

            if (
                df is None
                or "priority" not in df.columns
            ):
                continue

            curves.append(
                df["priority"].values
            )

    # --------------------------------------------------------
    # Battery
    # --------------------------------------------------------

    elif metric == "battery":

        if robot is None:
            return None, None

        for experiment in data:

            df = experiment[
                robot
            ]["battery"]

            if df is None:
                continue

            numeric = df.select_dtypes(
                include=[np.number]
            )

            if numeric.shape[1] == 0:
                continue

            curves.append(
                numeric.iloc[:, 0].values
            )

    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------

    elif metric == "memory":

        for experiment in data:

            robot_curves = []

            for current_robot in ROBOTS:

                df = experiment[
                    current_robot
                ]["memory"]

                if df is None:
                    continue

                numeric = df.select_dtypes(
                    include=[np.number]
                )

                if numeric.shape[1] == 0:
                    continue

                robot_curves.append(
                    numeric.iloc[:, 0].values
                )

            if len(robot_curves) == 0:
                continue

            min_length = min(
                len(curve)
                for curve in robot_curves
            )

            experiment_curve = np.mean(
                np.array([
                    curve[:min_length]
                    for curve in robot_curves
                ]),
                axis=0
            )

            curves.append(
                experiment_curve
            )

    # --------------------------------------------------------
    # Compactness
    # --------------------------------------------------------

    elif metric == "compactness":

        compactness_time, result = (
            compute_compactness_curves(
                group_data
            )
        )

        if result is None:
            return None, None

        mean, std = result

        return (
            compactness_time,
            mean
        )

    # --------------------------------------------------------
    # Photosynthesis
    # --------------------------------------------------------

    elif metric == "photosynthesis":

        for experiment in data:

            df = experiment[
                "photosynthesis"
            ]

            if df is None:
                continue

            numeric_columns = [
                column
                for column in df.columns
                if column != "time"
            ]

            if len(numeric_columns) == 0:
                continue

            values = pd.to_numeric(
                df[numeric_columns[0]],
                errors="coerce"
            ).values

            if len(values) == 0:
                continue

            curves.append(
                values
            )

    else:

        return None, None

    if len(curves) == 0:
        return None, None

    mean, std = mean_std(
        curves
    )

    return (
        time[:len(mean)],
        mean
    )


# ============================================================
# GLOBAL EQUAL VS REWARD TIME SERIES
# ============================================================

def compare_equal_reward_all_configurations_time_series(
    configuration_results,
    metric,
    output_folder
):

    fig, ax = plt.subplots(
        figsize=get_figsize()
    )

    found = False

    # --------------------------------------------------------
    # Priority and Battery:
    # keep the distinction between robots.
    #
    # For every robot:
    #
    #   experiment mean
    #       -> configuration mean
    #       -> mean across configurations
    #
    # This gives equal weight to config1/config2/config3.
    # --------------------------------------------------------

    if metric in [
        "priority",
        "battery"
    ]:

        for robot in ROBOTS:

            for group in [
                "equal",
                "reward"
            ]:

                configuration_curves = {}

                for config_name in CONFIG_NAMES:

                    group_data = get_group_data(
                        configuration_results,
                        config_name,
                        group
                    )

                    if group_data is None:
                        continue

                    time, curve = (
                        get_configuration_curve(
                            group_data,
                            metric,
                            robot
                        )
                    )

                    if (
                        time is None
                        or curve is None
                    ):
                        continue

                    configuration_curves[
                        config_name
                    ] = (
                        time,
                        curve
                    )

                if len(configuration_curves) == 0:
                    continue

                common_time = get_common_global_time(
                    configuration_results,
                    configuration_curves
                )

                if common_time is None:
                    continue

                aligned_curves = []

                for config_name in CONFIG_NAMES:

                    if config_name not in configuration_curves:
                        continue

                    time, curve = (
                        configuration_curves[
                            config_name
                        ]
                    )

                    interpolated = interpolate_curve(
                        time,
                        curve,
                        common_time
                    )

                    if interpolated is not None:
                        aligned_curves.append(
                            interpolated
                        )

                if len(aligned_curves) == 0:
                    continue

                aligned_curves = np.array(
                    aligned_curves
                )

                global_mean = np.mean(
                    aligned_curves,
                    axis=0
                )

                global_std = np.std(
                    aligned_curves,
                    axis=0
                )

                ax.plot(
                    common_time,
                    global_mean,
                    label=(
                        f"{group.capitalize()} "
                        f"{ROBOT_LABELS[robot]}"
                    )
                )

                ax.fill_between(
                    common_time,
                    global_mean - global_std,
                    global_mean + global_std,
                    alpha=0.08
                )

                found = True

    # --------------------------------------------------------
    # Memory / Compactness / Photosynthesis:
    # one curve for each group.
    # --------------------------------------------------------

    else:

        for group in [
            "equal",
            "reward"
        ]:

            configuration_curves = {}

            for config_name in CONFIG_NAMES:

                group_data = get_group_data(
                    configuration_results,
                    config_name,
                    group
                )

                if group_data is None:
                    continue

                time, curve = (
                    get_configuration_curve(
                        group_data,
                        metric
                    )
                )

                if (
                    time is None
                    or curve is None
                ):
                    continue

                configuration_curves[
                    config_name
                ] = (
                    time,
                    curve
                )

            if len(configuration_curves) == 0:
                continue

            common_time = get_common_global_time(
                configuration_results,
                configuration_curves
            )

            if common_time is None:
                continue

            aligned_curves = []

            for config_name in CONFIG_NAMES:

                if config_name not in configuration_curves:
                    continue

                time, curve = (
                    configuration_curves[
                        config_name
                    ]
                )

                interpolated = interpolate_curve(
                    time,
                    curve,
                    common_time
                )

                if interpolated is not None:
                    aligned_curves.append(
                        interpolated
                    )

            if len(aligned_curves) == 0:
                continue

            aligned_curves = np.array(
                aligned_curves
            )

            global_mean = np.mean(
                aligned_curves,
                axis=0
            )

            global_std = np.std(
                aligned_curves,
                axis=0
            )

            ax.plot(
                common_time,
                global_mean,
                label=group.capitalize()
            )

            ax.fill_between(
                common_time,
                global_mean - global_std,
                global_mean + global_std,
                alpha=0.08
            )

            found = True

    if not found:

        plt.close(fig)
        return

    ax.set_xlabel(
        "Time [s]"
    )

    if metric == "priority":
        ax.set_ylabel("Priority")

    elif metric == "battery":
        ax.set_ylabel("Battery")

    elif metric == "memory":
        ax.set_ylabel("Memory")

    elif metric == "compactness":
        ax.set_ylabel("Mean distance [m]")

    elif metric == "photosynthesis":
        ax.set_ylabel("Photosynthesis")

    ax.set_title(
        f"{metric.capitalize()} - "
        "Equal vs Reward\n"
        "Mean across configurations"
    )

    ax.legend(
        ncol=2
    )

    ax.grid(
        alpha=0.25
    )

    save(
        fig,
        output_folder,
        f"{metric}_equal_reward_all_configurations.png"
    )


# ============================================================
# GLOBAL PHOTOSYNTHESIS SCALAR
# ============================================================

def get_photosynthesis_scalar_configuration_mean(
    configuration_results,
    config_name,
    group
):

    group_data = get_group_data(
        configuration_results,
        config_name,
        group
    )

    if group_data is None:
        return None

    experiment_values = []

    for experiment in group_data["data"]:

        df = experiment[
            "photosynthesis"
        ]

        if df is None:
            continue

        numeric_columns = [
            column
            for column in df.columns
            if column != "time"
        ]

        if len(numeric_columns) == 0:
            continue

        values = pd.to_numeric(
            df[numeric_columns[0]],
            errors="coerce"
        ).dropna()

        if len(values) == 0:
            continue

        experiment_values.append(
            values.mean()
        )

    if len(experiment_values) == 0:
        return None

    return np.mean(
        experiment_values
    )


def compare_equal_reward_all_configurations_photosynthesis_scalar(
    configuration_results,
    output_folder
):

    group_means = {
        "equal": [],
        "reward": []
    }

    for group in [
        "equal",
        "reward"
    ]:

        for config_name in CONFIG_NAMES:

            value = (
                get_photosynthesis_scalar_configuration_mean(
                    configuration_results,
                    config_name,
                    group
                )
            )

            if value is not None:
                group_means[group].append(
                    value
                )

    if (
        len(group_means["equal"]) == 0
        and len(group_means["reward"]) == 0
    ):
        return

    fig, ax = plt.subplots(
        figsize=get_figsize()
    )

    labels = []
    means = []
    stds = []

    for group in [
        "equal",
        "reward"
    ]:

        values = group_means[group]

        if len(values) == 0:
            continue

        labels.append(
            group.capitalize()
        )

        means.append(
            np.mean(values)
        )

        stds.append(
            np.std(values)
        )

    if len(means) == 0:

        plt.close(fig)
        return

    x = np.arange(
        len(means)
    )

    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=3
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_ylabel(
        "Mean photosynthesis"
    )

    ax.set_title(
        "Photosynthesis - Equal vs Reward\n"
        "Mean across configurations"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    save(
        fig,
        output_folder,
        "photosynthesis_scalar_equal_reward_all_configurations.png"
    )


# ============================================================
# GLOBAL MISSION TIME
# ============================================================


def get_swarm_mission_time(experiment_path):
    """
    Return the swarm mission completion time defined as the
    first instant at which the photosynthesis log has ph != 0.

    The returned value is relative to the first valid time t0,
    following the same definition used in the standalone
    photosynthesis-based mission-time analysis.
    """

    file_path = os.path.join(
        experiment_path,
        "photosynthesis_log.csv"
    )

    if not os.path.exists(file_path):
        return None

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        print(
            f"Warning reading {file_path}: {exc}"
        )
        return None

    if len(df) == 0:
        return None

    if "time" not in df.columns or "ph" not in df.columns:
        return None

    time = pd.to_numeric(
        df["time"],
        errors="coerce"
    )

    ph = pd.to_numeric(
        df["ph"],
        errors="coerce"
    )

    valid = time.notna() & ph.notna()
    df_valid = df.loc[valid].copy()

    if len(df_valid) == 0:
        return None

    df_valid["time"] = pd.to_numeric(
        df_valid["time"],
        errors="coerce"
    )

    df_valid["ph"] = pd.to_numeric(
        df_valid["ph"],
        errors="coerce"
    )

    t0 = float(
        df_valid["time"].iloc[0]
    )

    # Preserve the definition used in the reference analysis:
    # if the first timestamp is zero, use the following sample
    # as t0.
    if t0 == 0.0 and len(df_valid) > 1:
        t0 = float(
            df_valid["time"].iloc[1]
        )

    # First value of ph different from zero.
    completed = df_valid[
        df_valid["ph"] != 0
    ]

    if completed.empty:
        return None

    t_end = float(
        completed["time"].iloc[0]
    )

    return t_end - t0


def get_configuration_mission_means(
    configuration_results,
    config_name,
    group
):

    if config_name not in configuration_results:
        return {}

    if group not in configuration_results[
        config_name
    ]:
        return {}

    experiments = configuration_results[
        config_name
    ][group]["experiments"]

    robot_values = {
        robot: []
        for robot in ROBOTS
    }

    for experiment_path in experiments:

        for robot in ROBOTS:

            path = os.path.join(
                experiment_path,
                f"{robot}_mission_times.csv"
            )

            if not os.path.exists(path):
                continue

            try:

                df = pd.read_csv(
                    path
                )

            except Exception:

                continue

            if (
                "charging_completion_time"
                not in df.columns
            ):
                continue

            values = pd.to_numeric(
                df[
                    "charging_completion_time"
                ],
                errors="coerce"
            ).dropna()

            if len(values) == 0:
                continue

            robot_values[robot].append(
                values.iloc[-1]
            )

    result = {}

    for robot in ROBOTS:

        values = robot_values[robot]

        if len(values) > 0:

            result[robot] = np.mean(
                values
            )

    return result

# ============================================================
# GLOBAL WAYPOINT TIME
# ============================================================

def get_configuration_waypoint_time_means(
    configuration_results,
    config_name,
    group
):

    if config_name not in configuration_results:
        return None

    if group not in configuration_results[
        config_name
    ]:
        return None

    experiments = configuration_results[
        config_name
    ][group]["experiments"]

    waypoints = CONFIG_WAYPOINTS[
        config_name
    ]

    values_by_waypoint = {
        i: []
        for i in range(N_WAYPOINTS)
    }

    for experiment_path in experiments:

        for robot in ROBOTS:

            path = os.path.join(
                experiment_path,
                f"{robot}_position.csv"
            )

            if not os.path.exists(path):
                continue

            try:

                df = pd.read_csv(
                    path
                )

            except Exception:

                continue

            if "time" not in df.columns:
                continue

            if (
                "x" not in df.columns
                or "y" not in df.columns
            ):
                continue

            time = pd.to_numeric(
                df["time"],
                errors="coerce"
            ).values

            x = pd.to_numeric(
                df["x"],
                errors="coerce"
            ).values

            y = pd.to_numeric(
                df["y"],
                errors="coerce"
            ).values

            valid = (
                np.isfinite(time)
                & np.isfinite(x)
                & np.isfinite(y)
            )

            time = time[valid]
            x = x[valid]
            y = y[valid]

            if len(time) == 0:
                continue

            for waypoint_index, waypoint in enumerate(
                waypoints
            ):

                distance = np.sqrt(
                    (
                        x - waypoint[0]
                    ) ** 2
                    + (
                        y - waypoint[1]
                    ) ** 2
                )

                indices = np.where(
                    distance <= WAYPOINT_VISIT_THRESHOLD
                )[0]

                if len(indices) == 0:
                    continue

                first_time = time[
                    indices[0]
                ]

                values_by_waypoint[
                    waypoint_index
                ].append(
                    first_time - time[0]
                )

    result = []

    for waypoint_index in range(
        N_WAYPOINTS
    ):

        values = values_by_waypoint[
            waypoint_index
        ]

        if len(values) == 0:
            result.append(
                np.nan
            )
        else:
            result.append(
                np.mean(values)
            )

    return np.array(
        result
    )


def compare_equal_reward_all_configurations_mission_time(
    configuration_results,
    output_folder
):
    """
    Compare robot and swarm mission completion times averaged
    across config1/config2/config3.

    Robot mission time:
        last charging_completion_time in *_mission_times.csv.

    Swarm mission time:
        first time at which ph != 0 in photosynthesis_log.csv,
        relative to t0.

    The swarm time is therefore NOT the sum or the maximum of
    the three robot mission times.
    """

    labels = [
        "Robot 1",
        "Robot 2",
        "Robot 3",
        "Swarm"
    ]

    equal_means = []
    equal_stds = []
    reward_means = []
    reward_stds = []

    # ========================================================
    # ROBOT 1 / ROBOT 2 / ROBOT 3
    # ========================================================

    for robot in ROBOTS:

        equal_values = []
        reward_values = []

        for config_name in CONFIG_NAMES:

            equal_result = get_configuration_mission_means(
                configuration_results,
                config_name,
                "equal"
            )

            reward_result = get_configuration_mission_means(
                configuration_results,
                config_name,
                "reward"
            )

            if robot in equal_result:
                value = equal_result[robot]
                if np.isfinite(value):
                    equal_values.append(value)

            if robot in reward_result:
                value = reward_result[robot]
                if np.isfinite(value):
                    reward_values.append(value)

        if len(equal_values) > 0:
            equal_means.append(np.mean(equal_values))
            equal_stds.append(np.std(equal_values))
        else:
            equal_means.append(np.nan)
            equal_stds.append(0)

        if len(reward_values) > 0:
            reward_means.append(np.mean(reward_values))
            reward_stds.append(np.std(reward_values))
        else:
            reward_means.append(np.nan)
            reward_stds.append(0)

    # ========================================================
    # SWARM
    # ========================================================
    #
    # For EACH experiment:
    #
    #     swarm_time = first(ph != 0) - t0
    #
    # Then:
    #
    #     experiment swarm times
    #             -> mean per configuration
    #             -> mean across config1/config2/config3
    #
    # This gives equal weight to the three configurations and
    # never combines the three robot completion times.
    # ========================================================

    equal_configuration_means = []
    reward_configuration_means = []

    for config_name in CONFIG_NAMES:

        if config_name not in configuration_results:
            continue

        for group, destination in [
            ("equal", equal_configuration_means),
            ("reward", reward_configuration_means)
        ]:

            if group not in configuration_results[config_name]:
                continue

            experiments = configuration_results[
                config_name
            ][group]["experiments"]

            experiment_swarm_times = []

            for experiment_path in experiments:

                swarm_time = get_swarm_mission_time(
                    experiment_path
                )

                if (
                    swarm_time is not None
                    and np.isfinite(swarm_time)
                ):
                    experiment_swarm_times.append(
                        swarm_time
                    )

            if len(experiment_swarm_times) > 0:
                configuration_mean = np.mean(
                    experiment_swarm_times
                )

                destination.append(
                    configuration_mean
                )

    # Mean across configurations.
    equal_swarm_values = [
        value
        for value in equal_configuration_means
        if np.isfinite(value)
    ]

    reward_swarm_values = [
        value
        for value in reward_configuration_means
        if np.isfinite(value)
    ]

    if len(equal_swarm_values) > 0:
        equal_means.append(
            np.mean(equal_swarm_values)
        )
        equal_stds.append(
            np.std(equal_swarm_values)
        )
    else:
        equal_means.append(np.nan)
        equal_stds.append(0)

    if len(reward_swarm_values) > 0:
        reward_means.append(
            np.mean(reward_swarm_values)
        )
        reward_stds.append(
            np.std(reward_swarm_values)
        )
    else:
        reward_means.append(np.nan)
        reward_stds.append(0)

    # ========================================================
    # CSV
    # ========================================================

    rows = []

    for i, label in enumerate(labels):
        rows.append({
            "Metric": label,
            "Equal_mean": equal_means[i],
            "Equal_std": equal_stds[i],
            "Reward_mean": reward_means[i],
            "Reward_std": reward_stds[i]
        })

    csv_path = os.path.join(
        output_folder,
        "mission_time_equal_reward_all_configurations.csv"
    )

    pd.DataFrame(rows).to_csv(
        csv_path,
        index=False
    )

    print(
        f"Saved: {csv_path}"
    )

    # ========================================================
    # PLOT
    # ========================================================

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(
        figsize=get_figsize()
    )

    ax.bar(
        x - width / 2,
        equal_means,
        width,
        yerr=equal_stds,
        capsize=3,
        label="Equal"
    )

    ax.bar(
        x + width / 2,
        reward_means,
        width,
        yerr=reward_stds,
        capsize=3,
        label="Reward"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylabel(
        "Mission completion time [s]"
    )

    ax.set_title(
        "Mission completion time - Equal vs Reward\n"
        "Mean across configurations"
    )

    ax.legend()
    ax.grid(
        axis="y",
        alpha=0.25
    )

    save(
        fig,
        output_folder,
        "mission_time_equal_reward_all_configurations.png"
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 75)
    print("GLOBAL MISSION COMPLETION TIME - EQUAL VS REWARD")
    print("Average across config1, config2 and config3")
    print("Swarm = first ph != 0 in photosynthesis_log.csv")
    print("=" * 75)

    for i, label in enumerate(labels):
        print(
            f"{label:8s} | "
            f"Equal: {equal_means[i]:.3f} ± "
            f"{equal_stds[i]:.3f} s | "
            f"Reward: {reward_means[i]:.3f} ± "
            f"{reward_stds[i]:.3f} s"
        )

    print("=" * 75)

# ============================================================
# GLOBAL WAYPOINT PERCENTAGE
# ============================================================

def get_configuration_waypoint_percentage_means(
    configuration_results,
    config_name,
    group
):

    if config_name not in configuration_results:
        return None

    if group not in configuration_results[
        config_name
    ]:
        return None

    experiments = configuration_results[
        config_name
    ][group]["experiments"]

    waypoints = CONFIG_WAYPOINTS[
        config_name
    ]

    robot_values = {
        robot: []
        for robot in ROBOTS
    }

    for experiment_path in experiments:

        for robot in ROBOTS:

            path = os.path.join(
                experiment_path,
                f"{robot}_position.csv"
            )

            if not os.path.exists(path):
                continue

            try:

                df = pd.read_csv(
                    path
                )

            except Exception:

                continue

            visited = 0

            for waypoint in waypoints:

                if (
                    count_waypoint_visits(
                        df,
                        waypoint
                    ) > 0
                ):
                    visited += 1

            percentage = (
                100.0
                * visited
                / len(waypoints)
            )

            robot_values[
                robot
            ].append(
                percentage
            )

    result = {}

    for robot in ROBOTS:

        values = robot_values[
            robot
        ]

        if len(values) > 0:

            result[robot] = np.mean(
                values
            )

    return result

# ============================================================
# GLOBAL WAYPOINT TIME - EQUAL VS REWARD
# ============================================================

def compare_equal_reward_all_configurations_waypoint_time(
    configuration_results,
    output_folder
):

    fig, ax = plt.subplots(
        figsize=get_figsize()
    )

    width = 0.35

    equal_means = []
    equal_stds = []

    reward_means = []
    reward_stds = []

    waypoint_labels = [
        f"WP {i + 1}"
        for i in range(N_WAYPOINTS)
    ]

    # ========================================================
    # Equal vs Reward
    # ========================================================

    for waypoint_index in range(N_WAYPOINTS):

        equal_values = []
        reward_values = []

        for config_name in CONFIG_NAMES:

            # ------------------------------------------------
            # Equal
            # ------------------------------------------------

            equal_result = (
                get_configuration_waypoint_time_means(
                    configuration_results,
                    config_name,
                    "equal"
                )
            )

            if equal_result is not None:

                value = equal_result[waypoint_index]

                if np.isfinite(value):

                    equal_values.append(
                        value
                    )

            # ------------------------------------------------
            # Reward
            # ------------------------------------------------

            reward_result = (
                get_configuration_waypoint_time_means(
                    configuration_results,
                    config_name,
                    "reward"
                )
            )

            if reward_result is not None:

                value = reward_result[waypoint_index]

                if np.isfinite(value):

                    reward_values.append(
                        value
                    )

        # ----------------------------------------------------
        # Equal statistics
        # ----------------------------------------------------

        if len(equal_values) > 0:

            equal_means.append(
                np.mean(equal_values)
            )

            equal_stds.append(
                np.std(equal_values)
            )

        else:

            equal_means.append(
                np.nan
            )

            equal_stds.append(
                0
            )

        # ----------------------------------------------------
        # Reward statistics
        # ----------------------------------------------------

        if len(reward_values) > 0:

            reward_means.append(
                np.mean(reward_values)
            )

            reward_stds.append(
                np.std(reward_values)
            )

        else:

            reward_means.append(
                np.nan
            )

            reward_stds.append(
                0
            )

    # ========================================================
    # Plot
    # ========================================================

    x = np.arange(
        N_WAYPOINTS
    )

    ax.bar(
        x - width / 2,
        equal_means,
        width,
        yerr=equal_stds,
        capsize=3,
        label="Equal"
    )

    ax.bar(
        x + width / 2,
        reward_means,
        width,
        yerr=reward_stds,
        capsize=3,
        label="Reward"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        waypoint_labels
    )

    ax.set_ylabel(
        "First visit time [s]"
    )

    ax.set_title(
        "Waypoint first visit time - Equal vs Reward\n"
        "Mean across configurations"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25
    )

    save(
        fig,
        output_folder,
        "waypoint_time_equal_reward_all_configurations.png"
    )

def compare_equal_reward_all_configurations_waypoint_percentage(
    configuration_results,
    output_folder
):

    fig, ax = plt.subplots(
        figsize=get_figsize()
    )

    equal_values = []
    reward_values = []

    valid_robots = []

    for robot in ROBOTS:

        equal_config_values = []
        reward_config_values = []

        for config_name in CONFIG_NAMES:

            equal_result = (
                get_configuration_waypoint_percentage_means(
                    configuration_results,
                    config_name,
                    "equal"
                )
            )

            reward_result = (
                get_configuration_waypoint_percentage_means(
                    configuration_results,
                    config_name,
                    "reward"
                )
            )

            if robot in equal_result:
                equal_config_values.append(
                    equal_result[robot]
                )

            if robot in reward_result:
                reward_config_values.append(
                    reward_result[robot]
                )

        if (
            len(equal_config_values) == 0
            and len(reward_config_values) == 0
        ):
            continue

        valid_robots.append(
            robot
        )

        equal_values.append(
            np.mean(equal_config_values)
            if len(equal_config_values) > 0
            else np.nan
        )

        reward_values.append(
            np.mean(reward_config_values)
            if len(reward_config_values) > 0
            else np.nan
        )

    if len(valid_robots) == 0:

        plt.close(fig)
        return

    x = np.arange(
        len(valid_robots)
    )

    width = 0.35

    ax.bar(
        x - width / 2,
        equal_values,
        width,
        label="Equal"
    )

    ax.bar(
        x + width / 2,
        reward_values,
        width,
        label="Reward"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels([
        ROBOT_LABELS[robot]
        for robot in valid_robots
    ])

    ax.set_ylabel(
        "Waypoints visited [%]"
    )

    ax.set_ylim(
        0,
        100
    )

    ax.set_title(
        "Waypoint coverage - Equal vs Reward\n"
        "Mean across configurations"
    )

    ax.legend()
    ax.grid(
        axis="y",
        alpha=0.25
    )

    save(
        fig,
        output_folder,
        "waypoint_percentage_equal_reward_all_configurations.png"
    )


# ============================================================
# GLOBAL WAYPOINT VISITS
# ============================================================

def get_configuration_waypoint_visit_means(
    configuration_results,
    config_name,
    group
):

    if config_name not in configuration_results:
        return None

    if group not in configuration_results[
        config_name
    ]:
        return None

    experiments = configuration_results[
        config_name
    ][group]["experiments"]

    waypoints = CONFIG_WAYPOINTS[
        config_name
    ]

    values_by_waypoint = {
        i: []
        for i in range(N_WAYPOINTS)
    }

    for waypoint_index, waypoint in enumerate(
        waypoints
    ):

        for experiment_path in experiments:

            for robot in ROBOTS:

                path = os.path.join(
                    experiment_path,
                    f"{robot}_position.csv"
                )

                if not os.path.exists(path):
                    continue

                try:

                    df = pd.read_csv(
                        path
                    )

                except Exception:

                    continue

                values_by_waypoint[
                    waypoint_index
                ].append(
                    count_waypoint_visits(
                        df,
                        waypoint
                    )
                )

    result = []

    for waypoint_index in range(
        N_WAYPOINTS
    ):

        values = values_by_waypoint[
            waypoint_index
        ]

        if len(values) == 0:
            result.append(
                np.nan
            )
        else:
            result.append(
                np.mean(values)
            )

    return np.array(
        result
    )


def compare_equal_reward_all_configurations_waypoint_visits(
    configuration_results,
    output_folder
):

    fig, ax = plt.subplots(
        figsize=get_figsize()
    )

    x = np.arange(
        1,
        N_WAYPOINTS + 1
    )

    found = False

    for group in [
        "equal",
        "reward"
    ]:

        configuration_curves = []

        for config_name in CONFIG_NAMES:

            curve = (
                get_configuration_waypoint_visit_means(
                    configuration_results,
                    config_name,
                    group
                )
            )

            if curve is not None:
                configuration_curves.append(
                    curve
                )

        if len(configuration_curves) == 0:
            continue

        matrix = np.array(
            configuration_curves,
            dtype=float
        )

        mean = np.nanmean(
            matrix,
            axis=0
        )

        std = np.nanstd(
            matrix,
            axis=0
        )

        ax.plot(
            x,
            mean,
            marker="o",
            label=group.capitalize()
        )

        ax.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.08
        )

        found = True

    if not found:

        plt.close(fig)
        return

    ax.set_xlabel(
        "Waypoint"
    )

    ax.set_ylabel(
        "Mean number of visits"
    )

    ax.set_title(
        "Waypoint visits - Equal vs Reward\n"
        "Mean across configurations"
    )

    ax.set_xticks(
        x
    )

    ax.legend()
    ax.grid(alpha=0.25)

    save(
        fig,
        output_folder,
        "waypoint_visits_equal_reward_all_configurations.png"
    )


# ============================================================
# GLOBAL SUMMARY CSV
# ============================================================

def create_global_waypoint_summary_csv(
    configuration_results,
    output_folder
):

    rows = []

    for group in [
        "equal",
        "reward"
    ]:

        for config_name in CONFIG_NAMES:

            curve = (
                get_configuration_waypoint_visit_means(
                    configuration_results,
                    config_name,
                    group
                )
            )

            if curve is None:
                continue

            for waypoint_index, value in enumerate(
                curve,
                start=1
            ):

                rows.append({
                    "configuration": config_name,
                    "group": group,
                    "waypoint": waypoint_index,
                    "mean_visits": value
                })

    if len(rows) == 0:
        return

    df = pd.DataFrame(
        rows
    )

    path = os.path.join(
        output_folder,
        "waypoint_visits_all_configurations.csv"
    )

    df.to_csv(
        path,
        index=False
    )

    print(
        f"Saved: {path}"
    )


# ============================================================
# ALL CONFIGURATION COMPARISONS
# ============================================================

def compare_all_configurations(
    configuration_results
):

    os.makedirs(
        CONFIG_COMPARISON_DIR,
        exist_ok=True
    )

    for group in GROUPS:

        compare_time_series(
            configuration_results,
            group,
            "priority",
            CONFIG_COMPARISON_DIR
        )

        compare_time_series(
            configuration_results,
            group,
            "battery",
            CONFIG_COMPARISON_DIR
        )

        compare_waypoint_visits_configurations(
            configuration_results,
            group,
            CONFIG_COMPARISON_DIR
        )


# ============================================================
# ALL EQUAL VS REWARD COMPARISONS
# ============================================================

def compare_equal_reward_all(
    configuration_results
):

    for config_name in CONFIG_NAMES:

        if config_name not in configuration_results:
            continue

        output_folder = os.path.join(
            CONFIG_COMPARISON_DIR,
            config_name
        )

        os.makedirs(
            output_folder,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Time series
        # ----------------------------------------------------

        compare_equal_reward_time_series(
            configuration_results,
            config_name,
            "priority",
            output_folder
        )

        compare_equal_reward_time_series(
            configuration_results,
            config_name,
            "battery",
            output_folder
        )

        compare_equal_reward_time_series(
            configuration_results,
            config_name,
            "memory",
            output_folder
        )

        # ----------------------------------------------------
        # Photosynthesis
        # ----------------------------------------------------

        compare_equal_reward_photosynthesis(
            configuration_results,
            config_name,
            output_folder
        )

        # ----------------------------------------------------
        # Scalar metrics
        # ----------------------------------------------------

        compare_equal_reward_scalar_metric(
            configuration_results,
            config_name,
            "photosynthesis",
            output_folder
        )

        # ----------------------------------------------------
        # Mission time
        # ----------------------------------------------------

        compare_equal_reward_mission_time(
            configuration_results,
            config_name,
            output_folder
        )

        # ----------------------------------------------------
        # Waypoint time
        # ----------------------------------------------------

        compare_equal_reward_waypoint_time(
            configuration_results,
            config_name,
            output_folder
        )

        # ----------------------------------------------------
        # Waypoint percentage
        # ----------------------------------------------------

        compare_equal_reward_waypoint_percentage(
            configuration_results,
            config_name,
            output_folder
        )

        # ----------------------------------------------------
        # Compactness
        # ----------------------------------------------------

        compare_equal_reward_compactness(
            configuration_results,
            config_name,
            output_folder
        )

        # ----------------------------------------------------
        # Waypoint visits
        # ----------------------------------------------------

        compare_waypoint_visits_equal_reward(
            configuration_results,
            config_name,
            output_folder
        )


# ============================================================
# NEW:
# EQUAL VS REWARD AVERAGED OVER ALL CONFIGURATIONS
# ============================================================

def compare_equal_reward_all_configurations(
    configuration_results
):

    os.makedirs(
        ALL_CONFIG_EQUAL_REWARD_DIR,
        exist_ok=True
    )

    print(
        "\n"
        "------------------------------------------------------------\n"
        " Equal vs Reward averaged across configurations\n"
        "------------------------------------------------------------"
    )

    # --------------------------------------------------------
    # Time-series metrics
    # --------------------------------------------------------

    for metric in [
        "priority",
        "battery",
        "memory",
        "compactness",
        "photosynthesis"
    ]:

        compare_equal_reward_all_configurations_time_series(
            configuration_results,
            metric,
            ALL_CONFIG_EQUAL_REWARD_DIR
        )

    # --------------------------------------------------------
    # Scalar photosynthesis
    # --------------------------------------------------------

    compare_equal_reward_all_configurations_photosynthesis_scalar(
        configuration_results,
        ALL_CONFIG_EQUAL_REWARD_DIR
    )

    # --------------------------------------------------------
    # Mission time
    # --------------------------------------------------------

    compare_equal_reward_all_configurations_mission_time(
        configuration_results,
        ALL_CONFIG_EQUAL_REWARD_DIR
    )

    # --------------------------------------------------------
    # Waypoint time
    # --------------------------------------------------------

    compare_equal_reward_all_configurations_waypoint_time(
        configuration_results,
        ALL_CONFIG_EQUAL_REWARD_DIR
    )

    # --------------------------------------------------------
    # Waypoint percentage
    # --------------------------------------------------------

    compare_equal_reward_all_configurations_waypoint_percentage(
        configuration_results,
        ALL_CONFIG_EQUAL_REWARD_DIR
    )

    # --------------------------------------------------------
    # Waypoint visits
    # --------------------------------------------------------

    compare_equal_reward_all_configurations_waypoint_visits(
        configuration_results,
        ALL_CONFIG_EQUAL_REWARD_DIR
    )

    # --------------------------------------------------------
    # CSV with configuration-level waypoint statistics
    # --------------------------------------------------------

    create_global_waypoint_summary_csv(
        configuration_results,
        ALL_CONFIG_EQUAL_REWARD_DIR
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # No Equal-vs-Reward averaged trajectory plot is created.
    #
    # config1/config2/config3 have different waypoint
    # positions and therefore different spatial layouts.
    # Averaging x/y trajectories across configurations would
    # produce a trajectory without a physically meaningful
    # interpretation.
    #
    # The trajectory plots remain available separately inside
    # each configuration.
    # --------------------------------------------------------

    print(
        "\n"
        "NOTE: Equal-vs-Reward averaged trajectories were NOT "
        "generated because config1/config2/config3 have "
        "different spatial layouts and waypoint coordinates."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        "============================================================\n"
        " AGGREGATE RESULTS\n"
        "============================================================\n"
    )

    print(
        f"Base path: {BASE_PATH}"
    )

    print(
        f"RAL mode: {RAL_MODE}"
    )

    print(
        f"Waypoint visit threshold: "
        f"{WAYPOINT_VISIT_THRESHOLD} m"
    )

    configuration_results = {}

    # --------------------------------------------------------
    # Process all configurations
    # --------------------------------------------------------

    for config_name in CONFIG_NAMES:

        config_path = os.path.join(
            BASE_PATH,
            config_name
        )

        if not os.path.isdir(
            config_path
        ):

            print(
                f"\nSkipping {config_name}: "
                f"folder not found."
            )

            continue

        print(
            f"\nProcessing {config_name}..."
        )

        configuration_results[
            config_name
        ] = process_configuration(
            config_name
        )

    # --------------------------------------------------------
    # Configuration comparisons
    # --------------------------------------------------------

    print(
        "\n"
        "============================================================\n"
        " CONFIGURATION COMPARISONS\n"
        "============================================================\n"
    )

    compare_all_configurations(
        configuration_results
    )

    # --------------------------------------------------------
    # Equal vs Reward comparisons
    # --------------------------------------------------------

    print(
        "\n"
        "============================================================\n"
        " EQUAL VS REWARD COMPARISONS\n"
        "============================================================\n"
    )

    compare_equal_reward_all(
        configuration_results
    )

    # --------------------------------------------------------
    # Equal vs Reward averaged across configurations
    # --------------------------------------------------------

    print(
        "\n"
        "============================================================\n"
        " EQUAL VS REWARD - AVERAGE ACROSS CONFIGURATIONS\n"
        "============================================================\n"
    )

    compare_equal_reward_all_configurations(
        configuration_results
    )

    print(
        "\n"
        "============================================================\n"
        " DONE\n"
        "============================================================\n"
    )

    print(
        "\nGenerated global Equal-vs-Reward results in:"
    )

    print(
        ALL_CONFIG_EQUAL_REWARD_DIR
    )



if __name__ == "__main__":
    main()