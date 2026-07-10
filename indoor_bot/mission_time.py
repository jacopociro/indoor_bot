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


    # tempo missione completata
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
    for i in range(1,11):
        if exp_path == f"/home/gonazza/container_ws/catkin_ws/src/indoor_bot/indoor_bot/exp_reward{i}":
            print(f"exp_reward{i} - t_end: {t_end - t0:.2f} s")

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


        for i,file in enumerate(ROBOT_FILES):

            value = get_robot_mission_time(
                exp,
                file
            )

            if value is not None:

                results[f"Robot{i+1}"].append(
                    value
                )


        swarm_time = get_swarm_mission_time(exp)

        if swarm_time is not None:

            results["Swarm"].append(
                swarm_time
            )


    statistics = []


    for key,values in results.items():

        values = np.asarray(values)


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



all_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "mission_time_statistics.csv"
    ),
    index=False
)



# ==================================================
# PLOT
# ==================================================

for group in ["equal","reward"]:


    df = all_results[
        all_results["Group"] == group
    ]


    plt.figure(
        figsize=(8,5)
    )


    plt.bar(
        df["Agent"],
        df["Mean [s]"],
        yerr=df["Std [s]"],
        capsize=5
    )


    plt.ylabel(
        "Mission time [s]"
    )

    plt.title(
        f"{group} - Mission time mean ± std"
    )


    plt.grid(
        axis="y"
    )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{group}_mission_time.png"
        ),
        dpi=300
    )


    plt.close()



print("\nAggregation complete.")