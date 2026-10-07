import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse


# ==================================================
# CONFIGURATION
# ==================================================

BASE_PATH = (
    "/home/gonazza/container_ws/"
    "catkin_ws/src/indoor_bot/indoor_bot"
)

CONFIGURATIONS = [
    "config1",
    "config2",
    "config3"
]

MISSION_TIME_FILES = [
    "rosbot_1_0_mission_times.csv",
    "rosbot_2_1_mission_times.csv",
    "rosbot_3_2_mission_times.csv",
]


# ============================================================
# WAYPOINTS PER CONFIGURAZIONE
# ============================================================

WAYPOINTS = {

    # --------------------------------------------------------
    # CONFIG 1
    # --------------------------------------------------------
    "config1": np.array([
        [-6.5, -1.0],
        [ 3.0,  2.0],
        [ 3.5, -3.5],
        [-10.0, 2.0],
        [ 4.0, -2.5],
    ]),

    # --------------------------------------------------------
    # CONFIG 2
    # --------------------------------------------------------
    "config2": np.array([
        [-3.0,  0.0],
        [ 5.0,  2.5],
        [ 2.5, -4.5],
        [-10.0, 0.0],
        [ 5.0, -1.5],
    ]),

    # --------------------------------------------------------
    # CONFIG 3
    # --------------------------------------------------------
    "config3": np.array([
        [-5.0,  2.0],
        [ 4.0,  2.0],
        [ 0.0, -3.0],
        [-9.0,  3.0],
        [-8.0, -1.0],
    ]),
}


# ==================================================
# ARGUMENTS
# ==================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--noshow",
    action="store_true",
    help="Non mostra le figure"
)

parser.add_argument(
    "--ral",
    action="store_true",
    help="Use RAL/IEEE paper plot formatting"
)

args = parser.parse_args()


# ==================================================
# FIGURE FORMAT
# ==================================================

FIGSIZE = (8.0, 6.0)

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

        "figure.dpi": 150,
        "savefig.dpi": 300,

        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        "xtick.major.size": 3,
        "ytick.major.size": 3,

        "savefig.bbox": "tight",

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def get_figsize(default):
    """
    Return the RAL figure size when --ral is enabled.
    Otherwise return the default figure size.
    """

    return RAL_FIGSIZE if args.ral else default


# ==================================================
# LOAD DATA
# ==================================================

def get_robot_files(base_path):

    pos_files = glob.glob(
        os.path.join(
            base_path,
            "*_position.csv"
        )
    )

    return [
        os.path.basename(f).replace(
            "_position.csv",
            ""
        )
        for f in pos_files
    ]


def get_max_mission_time(base_path):

    mission_times = []

    for file in MISSION_TIME_FILES:

        path = os.path.join(
            base_path,
            file
        )

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Missing mission time file: {path}"
            )

        df = pd.read_csv(
            path
        )

        max_time = df[
            "charging_completion_time"
        ].max()

        mission_times.append(
            max_time
        )

    return max(mission_times)


# ==================================================
# PROCESS ONE EXPERIMENT
# ==================================================

def process_experiment(
        experiment_folder,
        config_name):

    print()
    print(
        "=" * 60
    )
    print(
        f"PROCESSING: {experiment_folder}"
    )
    print(
        f"CONFIGURATION: {config_name}"
    )
    print(
        "=" * 60
    )

    BASE_PATH_EXPERIMENT = experiment_folder

    # --------------------------------------------------
    # Output
    # --------------------------------------------------

    OUTPUT_DIR = os.path.join(
        BASE_PATH_EXPERIMENT,
        "plots"
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------
    # Load robots
    # --------------------------------------------------

    robots = get_robot_files(
        BASE_PATH_EXPERIMENT
    )

    if len(robots) == 0:

        print(
            "[WARNING] No robot position files found."
        )

        return

    # --------------------------------------------------
    # Load waypoints
    # --------------------------------------------------

    if config_name not in WAYPOINTS:

        print(
            f"[WARNING] No waypoints defined for "
            f"configuration '{config_name}'."
        )

        return

    waypoints = WAYPOINTS[config_name]

    print(
        f"Waypoints loaded from WAYPOINTS: "
        f"{len(waypoints)} points"
    )

    # --------------------------------------------------
    # Mission time
    # --------------------------------------------------

    try:

        MAX_TIME = get_max_mission_time(
            BASE_PATH_EXPERIMENT
        )

    except FileNotFoundError as e:

        print(
            f"[WARNING] {e}"
        )

        return

    print(
        f"MAX_TIME used: "
        f"{MAX_TIME:.2f} s"
    )

    # ==================================================
    # COLORS
    # ==================================================

    wp_colors = plt.cm.tab10(
        range(len(waypoints))
    )

    robot_colors = plt.cm.tab10(
        range(len(robots))
    )


    # ==================================================
    # FIGURE 1: TRAJECTORIES
    # ==================================================

    fig1, ax1 = plt.subplots(
        figsize=get_figsize(FIGSIZE)
    )

    ax1.set_xlabel(
        "x [m]"
    )

    ax1.set_ylabel(
        "y [m]"
    )

    ax1.set_aspect(
        "equal"
    )

    # ----------------------------------------------
    # Waypoints
    # ----------------------------------------------

    for i, (wx, wy) in enumerate(waypoints):

        ax1.scatter(
            wx,
            wy,
            color=wp_colors[i],
            marker="*",
            s=55,
            zorder=10
        )

    # ----------------------------------------------
    # Robot trajectories
    # ----------------------------------------------

    for idx, robot in enumerate(robots):

        pos_file = os.path.join(
            BASE_PATH_EXPERIMENT,
            f"{robot}_position.csv"
        )

        dir_file = os.path.join(
            BASE_PATH_EXPERIMENT,
            f"{robot}_direction.csv"
        )

        if not os.path.exists(pos_file):
            continue

        pos = pd.read_csv(
            pos_file
        )

        pos = pos[
            pos["time"] <= MAX_TIME
        ]

        pos = pos.iloc[1:]

        pos = pos[
            (pos["x"] != 0) |
            (pos["y"] != 0)
        ]

        if pos.empty:
            continue

        x = np.asarray(
            pos["x"].values,
            dtype=float
        )

        y = np.asarray(
            pos["y"].values,
            dtype=float
        )

        ax1.plot(
            x,
            y,
            label=robot,
            color=robot_colors[idx],
            linewidth=1.2
        )

        if os.path.exists(dir_file):

            dr = pd.read_csv(
                dir_file
            ).iloc[1:]

            if not dr.empty:

                last = dr.iloc[-1]

                ax1.arrow(
                    x[-1],
                    y[-1],
                    float(last["dir_x"]),
                    float(last["dir_y"]),
                    head_width=0.08,
                    linewidth=0.8,
                    color=robot_colors[idx],
                    length_includes_head=True
                )

    ax1.legend(
        loc="best",
        frameon=True
    )

    ax1.grid(
        True,
        linewidth=0.5,
        alpha=0.25
    )

    ax1.set_axisbelow(
        True
    )

    fig1.tight_layout()

    fig1.savefig(
        os.path.join(
            OUTPUT_DIR,
            "trajectories.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    fig1.savefig(
        os.path.join(
            OUTPUT_DIR,
            "trajectories.pdf"
        ),
        bbox_inches="tight"
    )

    # if not args.noshow:
    #     plt.show()

    plt.close(fig1)


    # ==================================================
    # FIGURE 2: PRIORITY
    # ==================================================

    fig2, axes2 = plt.subplots(
        len(robots),
        1,
        figsize=(
            get_figsize(FIGSIZE)[0],
            get_figsize(FIGSIZE)[1] * len(robots)
        ),
        squeeze=False
    )

    axes2 = axes2.flatten()

    for idx, robot in enumerate(robots):

        pr_file = os.path.join(
            BASE_PATH_EXPERIMENT,
            f"{robot}_priority.csv"
        )

        ax = axes2[idx]

        ax.set_xlabel(
            "Time [s]"
        )

        ax.set_ylabel(
            "Value"
        )

        if not os.path.exists(pr_file):
            continue

        pr = pd.read_csv(
            pr_file
        )

        pr = pr[
            pr["time"] <= MAX_TIME
        ]

        pr = pr.iloc[1:]

        if pr.empty:
            continue

        t_raw = np.asarray(
            pr["time"].values,
            dtype=float
        )

        t = (
            t_raw -
            t_raw[0]
        )

        data = (
            pr.iloc[:, 1:]
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .to_numpy(
                dtype=float
            )
        )

        for j in range(
            data.shape[1]
        ):

            ax.plot(
                t,
                data[:, j],
                label=f"p{j}",
                linewidth=1.2
            )

        ax.legend(
            loc="best",
            frameon=True
        )

        ax.grid(
            True,
            linewidth=0.5,
            alpha=0.25
        )

        ax.set_axisbelow(
            True
        )

    fig2.tight_layout()

    fig2.savefig(
        os.path.join(
            OUTPUT_DIR,
            "priority.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    fig2.savefig(
        os.path.join(
            OUTPUT_DIR,
            "priority.pdf"
        ),
        bbox_inches="tight"
    )

    # if not args.noshow:
    #     plt.show()

    plt.close(fig2)


    # ==================================================
    # FIGURE 3: MEMORY
    # ==================================================

    fig3, ax3 = plt.subplots(
        figsize=get_figsize(FIGSIZE)
    )

    memory_file = os.path.join(
        BASE_PATH_EXPERIMENT,
        "memory_log.csv"
    )

    if os.path.exists(memory_file):

        mem = pd.read_csv(
            memory_file
        )

        mem = mem.iloc[2:]

        if not mem.empty:

            t_raw = np.asarray(
                mem["time"].values,
                dtype=float
            )

            t = (
                t_raw -
                t_raw[0]
            )

            data = (
                mem.iloc[:, 1:]
                .apply(
                    pd.to_numeric,
                    errors="coerce"
                )
                .to_numpy(
                    dtype=float
                )
            )

            for i in range(
                data.shape[1]
            ):

                ax3.plot(
                    t,
                    data[:, i],
                    linewidth=1.2,
                    label=f"M{i}"
                )

    ax3.set_xlabel(
        "Time [s]"
    )

    ax3.set_ylabel(
        "Value"
    )

    ax3.grid(
        True,
        linewidth=0.5,
        alpha=0.25
    )

    ax3.set_axisbelow(
        True
    )

    ax3.legend(
        loc="best",
        frameon=True
    )

    fig3.tight_layout()

    fig3.savefig(
        os.path.join(
            OUTPUT_DIR,
            "memory.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    fig3.savefig(
        os.path.join(
            OUTPUT_DIR,
            "memory.pdf"
        ),
        bbox_inches="tight"
    )

    # if not args.noshow:
    #     plt.show()

    plt.close(fig3)


    # ==================================================
    # FIGURE 4: BATTERY
    # ==================================================

    fig4, axes4 = plt.subplots(
        len(robots),
        2,
        figsize=(
            get_figsize(FIGSIZE)[0] * 2,
            get_figsize(FIGSIZE)[1] * len(robots)
        ),
        squeeze=False
    )

    for idx, robot in enumerate(robots):

        batt_file = os.path.join(
            BASE_PATH_EXPERIMENT,
            f"{robot}_battery.csv"
        )

        axp = axes4[idx, 0]
        axv = axes4[idx, 1]

        axp.set_xlabel(
            "Time [s]"
        )

        axp.set_ylabel(
            "Battery [%]"
        )

        axv.set_xlabel(
            "Time [s]"
        )

        axv.set_ylabel(
            "Voltage [V]"
        )

        if not os.path.exists(batt_file):
            continue

        batt = pd.read_csv(
            batt_file
        )

        batt = batt[
            batt["time"] <= MAX_TIME
        ]

        batt = batt.iloc[1:]

        if batt.empty:
            continue

        t_raw = np.asarray(
            batt["time"].values,
            dtype=float
        )

        t = (
            t_raw -
            t_raw[0]
        )

        if "percentage" in batt.columns:

            percentage = np.asarray(
                batt["percentage"].values,
                dtype=float
            )

            axp.plot(
                t,
                percentage,
                label="Battery %",
                linewidth=1.2
            )

            axp.set_ylim(
                0,
                100
            )

            axp.legend(
                loc="best"
            )

        if "voltage" in batt.columns:

            voltage = np.asarray(
                batt["voltage"].values,
                dtype=float
            )

            axv.plot(
                t,
                voltage,
                label="Voltage",
                linewidth=1.2
            )

            axv.legend(
                loc="best"
            )

        axp.grid(
            True,
            linewidth=0.5,
            alpha=0.25
        )

        axv.grid(
            True,
            linewidth=0.5,
            alpha=0.25
        )

        axp.set_axisbelow(
            True
        )

        axv.set_axisbelow(
            True
        )

    fig4.tight_layout()

    fig4.savefig(
        os.path.join(
            OUTPUT_DIR,
            "battery.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    fig4.savefig(
        os.path.join(
            OUTPUT_DIR,
            "battery.pdf"
        ),
        bbox_inches="tight"
    )

    # if not args.noshow:
    #     plt.show()

    plt.close(fig4)


    # ==================================================
    # FIGURE 5: PHOTOSYNTHESIS
    # ==================================================

    fig5, ax5 = plt.subplots(
        figsize=get_figsize(FIGSIZE)
    )

    photosynthesis_file = os.path.join(
        BASE_PATH_EXPERIMENT,
        "photosynthesis_log.csv"
    )

    if os.path.exists(
        photosynthesis_file
    ):

        photo = pd.read_csv(
            photosynthesis_file
        )

        photo = photo.iloc[2:]

        if not photo.empty:

            t_raw = np.asarray(
                photo["time"].values,
                dtype=float
            )

            t = (
                t_raw -
                t_raw[0]
            )

            data = (
                photo.iloc[:, 1:]
                .apply(
                    pd.to_numeric,
                    errors="coerce"
                )
                .to_numpy(
                    dtype=float
                )
            )

            for i in range(
                data.shape[1]
            ):

                ax5.plot(
                    t,
                    data[:, i],
                    linewidth=1.2,
                    label=f"PH{i}"
                )

    ax5.set_xlabel(
        "Time [s]"
    )

    ax5.set_ylabel(
        "Value"
    )

    ax5.grid(
        True,
        linewidth=0.5,
        alpha=0.25
    )

    ax5.set_axisbelow(
        True
    )

    ax5.legend(
        loc="best",
        frameon=True
    )

    fig5.tight_layout()

    fig5.savefig(
        os.path.join(
            OUTPUT_DIR,
            "photosynthesis.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    fig5.savefig(
        os.path.join(
            OUTPUT_DIR,
            "photosynthesis.pdf"
        ),
        bbox_inches="tight"
    )

    # if not args.noshow:
    #     plt.show()

    plt.close(fig5)

    print(
        f"Plots saved in: {OUTPUT_DIR}"
    )


# ==================================================
# PROCESS CONFIGURATION
# ==================================================

def process_configuration(
        config_name):

    print()
    print(
        "#" * 60
    )
    print(
        f"PROCESSING CONFIGURATION: {config_name}"
    )
    print(
        "#" * 60
    )

    config_path = os.path.join(
        BASE_PATH,
        config_name
    )

    if not os.path.isdir(config_path):

        print(
            f"[WARNING] Configuration not found: "
            f"{config_path}"
        )

        return

    # --------------------------------------------------
    # Check waypoints
    # --------------------------------------------------

    if config_name not in WAYPOINTS:

        print(
            f"[WARNING] No WAYPOINTS defined for "
            f"configuration '{config_name}'."
        )

        return

    print(
        f"Waypoints for {config_name}:"
    )

    for i, (x, y) in enumerate(
        WAYPOINTS[config_name]
    ):

        print(
            f"  WP{i}: ({x:.2f}, {y:.2f})"
        )

    # --------------------------------------------------
    # Equal experiments
    # --------------------------------------------------

    equal_experiments = [
        os.path.join(
            config_path,
            f"exp_equal{i}"
        )
        for i in range(1, 11)
    ]

    equal_experiments = [
        exp
        for exp in equal_experiments
        if os.path.isdir(exp)
    ]

    # --------------------------------------------------
    # Reward experiments
    # --------------------------------------------------

    reward_experiments = [
        os.path.join(
            config_path,
            f"exp_reward{i}"
        )
        for i in range(1, 11)
    ]

    reward_experiments = [
        exp
        for exp in reward_experiments
        if os.path.isdir(exp)
    ]

    print(
        f"Equal experiments: "
        f"{len(equal_experiments)}"
    )

    print(
        f"Reward experiments: "
        f"{len(reward_experiments)}"
    )

    # --------------------------------------------------
    # Process equal
    # --------------------------------------------------

    for experiment in equal_experiments:

        process_experiment(
            experiment,
            config_name
        )

    # --------------------------------------------------
    # Process reward
    # --------------------------------------------------

    for experiment in reward_experiments:

        process_experiment(
            experiment,
            config_name
        )


# ==================================================
# MAIN
# ==================================================

def main():

    print(
        "\n"
        "==========================================\n"
        " ROBOT DATA PLOTS - ALL CONFIGURATIONS\n"
        "=========================================="
    )

    for config_name in CONFIGURATIONS:

        process_configuration(
            config_name
        )

    print()
    print(
        "=========================================="
    )
    print(
        "Plot generation complete for all "
        "configurations."
    )
    print(
        "=========================================="
    )


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    main()