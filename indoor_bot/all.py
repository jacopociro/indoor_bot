import subprocess

# Esperimenti da analizzare
experiments = []

# exp_equal1 ... exp_equal10
experiments += [f"/home/gonazza/container_ws/catkin_ws/src/indoor_bot/indoor_bot/exp_equal{i}" for i in range(1, 11)]

# exp_reward1 ... exp_reward10
experiments += [f"/home/gonazza/container_ws/catkin_ws/src/indoor_bot/indoor_bot/exp_reward{i}" for i in range(1, 11)]

scripts = [
    "/home/gonazza/container_ws/catkin_ws/src/indoor_bot/indoor_bot/data.py",
    "/home/gonazza/container_ws/catkin_ws/src/indoor_bot/indoor_bot/plot.py",
    "/home/gonazza/container_ws/catkin_ws/src/indoor_bot/indoor_bot/occupancy.py"
]

for exp in experiments:

    print("=" * 60)
    print(f"Processing {exp}")
    print("=" * 60)

    for script in scripts:

        print(f"Running {script}")

        subprocess.run(
            [
                "python3",
                script,
                "--experiment",
                exp,
                "--noshow"
            ],
            check=True
        )

print("\nAll experiments completed.")
print("Starting statistics analysis...")
scripts = ["/home/gonazza/container_ws/catkin_ws/src/indoor_bot/indoor_bot/aggregate_results.py",
           "/home/gonazza/container_ws/catkin_ws/src/indoor_bot/indoor_bot/mission_time.py"]
print(f"Running {scripts}")
for script in scripts:
    subprocess.run(
        [
            "python3",
            script
        ],
        check=True
    )
# ==========================================================
# STATISTICA ESPERIMENTI
# ==========================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


RESULT_DIR = "/home/gonazza/container_ws/catkin_ws/src/indoor_bot/indoor_bot/aggregated_results"


def extract_occupancy(log_file):

    occupancy = None

    with open(log_file, "r") as f:
        for line in f:

            if "Occupancy %" in line:
                value = line.split(":")[1]

                value = value.replace("%", "").strip()

                occupancy = float(value)

    return occupancy


def collect_results(exp_list):

    data = []

    for exp in exp_list:

        name = os.path.basename(exp)

        plots = os.path.join(
            exp,
            "plots"
        )

        row = {
            "experiment": name
        }


        # --------------------------
        # waypoint + heading
        # --------------------------

        summary_file = os.path.join(
            plots,
            "summary.csv"
        )

        if os.path.exists(summary_file):

            df = pd.read_csv(summary_file)

            row["NearWaypoint"] = (
                df["NearWaypoint_%"].mean()
            )

            row["Heading"] = (
                df["Heading_%"].mean()
            )


        # --------------------------
        # occupancy
        # --------------------------

        log_file = os.path.join(
            plots,
            "results.log"
        )

        if os.path.exists(log_file):

            row["Occupancy"] = (
                extract_occupancy(log_file)
            )


        data.append(row)


    return pd.DataFrame(data)



# ==========================================================
# separazione equal / reward
# ==========================================================

equal_exp = [
    e for e in experiments
    if "equal" in e
]

reward_exp = [
    e for e in experiments
    if "reward" in e
]


equal_results = collect_results(equal_exp)
reward_results = collect_results(reward_exp)



# ==========================================================
# funzione statistiche
# ==========================================================

def statistics(df, name):

    numeric = df.select_dtypes(
        include=np.number
    )

    stats = pd.DataFrame({
        "mean": numeric.mean(),
        "variance": numeric.var(),
        "std": numeric.std()
    })


    out = os.path.join(
        RESULT_DIR,
        f"{name}_statistics.csv"
    )

    stats.to_csv(out)

    print("\n", name)
    print(stats)

    return stats



equal_stats = statistics(
    equal_results,
    "equal"
)


reward_stats = statistics(
    reward_results,
    "reward"
)



# ==========================================================
# PLOT MEDIE + VARIANZA
# ==========================================================

def plot_statistics(
        equal_stats,
        reward_stats):


    metrics = equal_stats.index


    for metric in metrics:

        plt.figure(figsize=(6,4))


        means = [
            equal_stats.loc[metric,"mean"],
            reward_stats.loc[metric,"mean"]
        ]

        errors = [
            equal_stats.loc[metric,"std"],
            reward_stats.loc[metric,"std"]
        ]


        plt.bar(
            [
                "equal",
                "reward"
            ],
            means,
            yerr=errors,
            capsize=5
        )


        plt.ylabel(metric)

        plt.title(
            f"{metric}: mean ± std"
        )

        plt.tight_layout()


        plt.savefig(
            os.path.join(
                RESULT_DIR,
                f"{metric}_comparison.png"
            ),
            dpi=300
        )

        plt.close()



plot_statistics(
    equal_stats,
    reward_stats
)


print("\nStatistics completed.")
