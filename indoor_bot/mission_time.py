import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# CONFIGURATION
# ==================================================

BASE_PATH = "/home/gonazza/container_ws/catkin_ws/src/indoor_bot/indoor_bot"

EQUAL_EXPERIMENTS = [
    os.path.join(BASE_PATH, f"exp_equal{i}")
    for i in range(1, 11)
]

REWARD_EXPERIMENTS = [
    os.path.join(BASE_PATH, f"exp_reward{i}")
    for i in range(1, 11)
]

OUTPUT_DIR = os.path.join(BASE_PATH, "aggregated_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROBOT_FILES = [
    "rosbot_1_0_mission_times.csv",
    "rosbot_2_1_mission_times.csv",
    "rosbot_3_2_mission_times.csv",
]


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
# ROBOT MISSION TIME
# ==================================================

def get_robot_mission_time(exp_path, filename):

    file_path = os.path.join(
        exp_path,
        filename
    )

    if not os.path.exists(file_path):
        return None

    df = pd.read_csv(file_path)

    return float(
        df["mission_completion_time"].iloc[0]
    )


# ==================================================
# SWARM MISSION TIME
# ==================================================

def get_swarm_mission_time(exp_path):

    file_path = os.path.join(
        exp_path,
        "photosynthesis_log.csv"
    )

    if not os.path.exists(file_path):
        return None

    df = pd.read_csv(file_path)

    if len(df) == 0:
        return None

    t0 = float(
        df["time"].iloc[0]
    )

    if t0 == 0.0:
        t0 = float(
            df["time"].iloc[1]
        )

    # primo valore ph diverso da zero
    completed = df[
        df["ph"] != 0
    ]

    if completed.empty:
        return None

    t_end = float(
        completed["time"].iloc[0]
    )

    for i in range(1, 11):

        if exp_path == (
            f"/home/gonazza/container_ws/catkin_ws/"
            f"src/indoor_bot/indoor_bot/exp_reward{i}"
        ):

            print(
                f"exp_reward{i} - "
                f"t_end: {t_end - t0:.2f} s"
            )

    return t_end - t0


# ==================================================
# LOAD EXPERIMENT GROUP
# ==================================================

def analyze_group(experiments, name):

    results = {
        "Robot1": [],
        "Robot2": [],
        "Robot3": [],
        "Swarm": []
    }

    for exp in experiments:

        print(
            f"Processing {exp}"
        )

        # ==============================================
        # SINGLE EXPERIMENT RESULTS
        # ==============================================

        experiment_results = {
            "Robot1": None,
            "Robot2": None,
            "Robot3": None,
            "Swarm": None
        }

        # ==============================================
        # ROBOT MISSION TIMES
        # ==============================================

        for i, file in enumerate(ROBOT_FILES):

            value = get_robot_mission_time(
                exp,
                file
            )

            if value is not None:

                experiment_results[
                    f"Robot{i+1}"
                ] = value

                results[
                    f"Robot{i+1}"
                ].append(value)

        # ==============================================
        # SWARM MISSION TIME
        # ==============================================

        swarm_time = get_swarm_mission_time(exp)

        if swarm_time is not None:

            experiment_results["Swarm"] = swarm_time

            results["Swarm"].append(
                swarm_time
            )

        # ==============================================
        # SAVE SINGLE EXPERIMENT DATA
        # ==============================================

        experiment_df = pd.DataFrame({

            "Agent": [
                "Robot1",
                "Robot2",
                "Robot3",
                "Swarm"
            ],

            "Mission time [s]": [
                experiment_results["Robot1"],
                experiment_results["Robot2"],
                experiment_results["Robot3"],
                experiment_results["Swarm"]
            ]
        })

        experiment_df.to_csv(
            os.path.join(
                exp,
                "mission_time_statistics.csv"
            ),
            index=False
        )

    # ==================================================
    # GROUP STATISTICS
    # ==================================================

    statistics = []

    for key, values in results.items():

        values = np.asarray(
            values,
            dtype=float
        )

        statistics.append({

            "Group": name,

            "Agent": key,

            "Samples": len(values),

            "Mean [s]": np.mean(values),

            "Variance [s²]": np.var(values),

            "Std [s]": np.std(values)

        })

    return pd.DataFrame(statistics)


# ==================================================
# MAIN
# ==================================================

equal_results = analyze_group(
    EQUAL_EXPERIMENTS,
    "equal"
)

reward_results = analyze_group(
    REWARD_EXPERIMENTS,
    "reward"
)

all_results = pd.concat(
    [
        equal_results,
        reward_results
    ],
    ignore_index=True
)

print("\n==============================")
print(all_results)
print("==============================")


# ==================================================
# SAVE AGGREGATED RESULTS
# ==================================================

for group_name, group_results in [
    ("equal", equal_results),
    ("reward", reward_results)
]:

    aggregated = group_results[
        [
            "Agent",
            "Mean [s]"
        ]
    ].copy()

    aggregated.columns = [
        "Agent",
        "Mission time [s]"
    ]

    aggregated.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{group_name}_mission_time.csv"
        ),
        index=False
    )


# ==================================================
# PLOT
# ==================================================

for group in ["equal", "reward"]:

    df = all_results[
        all_results["Group"] == group
    ]

    fig, ax = plt.subplots(
        figsize=FIGSIZE
    )

    x = np.arange(len(df))

    ax.bar(
        x,
        df["Mean [s]"],
        yerr=df["Std [s]"],
        capsize=3,
        linewidth=0.8
    )

    # ----------------------------------------------
    # AXIS
    # ----------------------------------------------

    ax.set_xticks(x)
    ax.set_xticklabels(
        df["Agent"]
    )

    ax.set_ylabel(
        "Mission time [s]"
    )

    # ----------------------------------------------
    # GRID
    # ----------------------------------------------

    ax.grid(
        axis="y",
        linewidth=0.6,
        alpha=0.4
    )

    ax.set_axisbelow(True)

    # ----------------------------------------------
    # SPINES
    # ----------------------------------------------

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # ----------------------------------------------
    # LAYOUT
    # ----------------------------------------------

    fig.tight_layout()

    # ----------------------------------------------
    # SAVE
    # ----------------------------------------------

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{group}_mission_time.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    # Optional vector version for the paper
    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{group}_mission_time.pdf"
        ),
        bbox_inches="tight"
    )

    plt.close(fig)


print("\nAggregation complete.")