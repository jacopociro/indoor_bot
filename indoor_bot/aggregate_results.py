import os
import glob
import yaml
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict

# ==========================================================
# PLOT CONFIGURATION
# ==========================================================

parser = argparse.ArgumentParser()
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

# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_PATH = "/home/gonazza/container_ws/catkin_ws/src/indoor_bot/indoor_bot"
BASE_PATHS_EQUAL = [
    os.path.join(BASE_PATH, f"exp_equal{i}")
    for i in range(1, 11)
]

BASE_PATHS_REWARD = [
    os.path.join(BASE_PATH, f"exp_reward{i}")
    for i in range(1, 11)
]
OUTPUT_DIR = os.path.join(BASE_PATH, "aggregated_results")
AGGREGATE_DIR = "/home/gonazza/container_ws/catkin_ws/src/indoor_bot/indoor_bot/aggregated_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WAYPOINT_FILE = "/home/gonazza/container_ws/catkin_ws/src/indoor_bot/config/waypoints.yaml"

MISSION_TIME_FILES = [
    "rosbot_1_0_mission_times.csv",
    "rosbot_2_1_mission_times.csv",
    "rosbot_3_2_mission_times.csv",
]

POSITION_FILES = [
    "rosbot_1_0_position.csv",
    "rosbot_2_1_position.csv",
    "rosbot_3_2_position.csv",
]

ROBOTS = [
    "rosbot_1_0",
    "rosbot_2_1",
    "rosbot_3_2"
]

DT = 0.05


# ==========================================================
# EXPERIMENTS
# ==========================================================

equal_experiments = sorted(
    glob.glob(os.path.join(BASE_PATH, "exp_equal*"))
)

reward_experiments = sorted(
    glob.glob(os.path.join(BASE_PATH, "exp_reward*"))
)

groups = {
    "equal": equal_experiments,
    "reward": reward_experiments
}

print("===================================")
print("Experiments found")
print("===================================")

for k,v in groups.items():
    print(k, len(v))

# ==========================================================
# UTILITIES
# ==========================================================

def read_waypoints():

    with open(WAYPOINT_FILE,"r") as f:
        wp = yaml.safe_load(f)

    return np.array(
        [
            [w["x"], w["y"]]
            for w in wp["wp"]
        ]
    )

waypoints = read_waypoints()

# ==========================================================
# MAX TIME
# ==========================================================

def get_max_time(exp_folder):

    times = []

    for file in MISSION_TIME_FILES:

        df = pd.read_csv(
            os.path.join(exp_folder,file)
        )

        times.append(
            df["charging_completion_time"].max()
        )

    return max(times)

# ==========================================================
# COMMON TIME
# ==========================================================

def common_time(exp_folder):

    Tmax = get_max_time(exp_folder)

    return np.arange(
        0,
        Tmax,
        DT
    )

# ==========================================================
# CSV
# ==========================================================

def load_csv(csv_file,max_time):

    df = pd.read_csv(csv_file)

    if "time" in df.columns:
        df = df[df.time<=max_time]

    return df.reset_index(drop=True)

# ==========================================================
# INTERPOLATION
# ==========================================================

def interpolate_signal(
        time,
        signal_time,
        signal):

    return np.interp(
        time,
        signal_time,
        signal
    )

# ==========================================================
# POSITION
# ==========================================================

def load_position(
        experiment,
        robot, 
        common_time):

    Tmax = get_max_time(experiment)
    t = common_time
    file = os.path.join(
        experiment,
        robot+"_position.csv"
    )


    if not os.path.exists(file):
        return {}


    mem = pd.read_csv(file)


    if mem.empty:
        return {}


    # ============================================
    # gestione tempo assoluto
    # ============================================

    t0 = mem["time"].iloc[0]

    t_end = t0 + Tmax


    # taglio missione
    mem = mem[
        mem["time"] <= t_end
    ]


    if len(mem) > 2:
        mem = mem.iloc[2:]


    if mem.empty:
        return {}


    # tempo relativo alla missione

    time = (
        mem["time"].values.astype(float)
        - t0
    )


    output = {}


    for col in mem.columns:

        if col == "time":
            continue


        values = (
            mem[col]
            .values
            .astype(float)
        )


        interp = interpolate_signal(
            t,
            time,
            values
        )


        if interp is not None:
            output[col] = interp


    return output

# ==========================================================
# BATTERY
# ==========================================================

def load_battery(
        experiment,
        robot, 
        common_time):

    Tmax = get_max_time(experiment)
    t = common_time
    file = os.path.join(
        experiment,
        robot+"_battery.csv"
    )



    if not os.path.exists(file):
        return {}


    mem = pd.read_csv(file)


    if mem.empty:
        return {}


    # ============================================
    # gestione tempo assoluto
    # ============================================

    t0 = mem["time"].iloc[0]

    t_end = t0 + Tmax


    # taglio missione
    mem = mem[
        mem["time"] <= t_end
    ]


    if len(mem) > 2:
        mem = mem.iloc[2:]


    if mem.empty:
        return {}


    # tempo relativo alla missione

    time = (
        mem["time"].values.astype(float)
        - t0
    )


    output = {}


    for col in mem.columns:

        if col == "time":
            continue


        values = (
            mem[col]
            .values
            .astype(float)
        )


        interp = interpolate_signal(
            t,
            time,
            values
        )


        if interp is not None:
            output[col] = interp


    return output

# ==========================================================
# PRIORITY
# ==========================================================

def load_priority(
        experiment,
        robot,
        common_time):

    Tmax=get_max_time(experiment)
    t = common_time
    file=os.path.join(
        experiment,
        robot+"_priority.csv"
    )

    if not os.path.exists(file):
        return {}


    mem = pd.read_csv(file)


    if mem.empty:
        return {}


    # ============================================
    # gestione tempo assoluto
    # ============================================

    t0 = mem["time"].iloc[0]

    t_end = t0 + Tmax


    # taglio missione
    mem = mem[
        mem["time"] <= t_end
    ]


    if len(mem) > 2:
        mem = mem.iloc[2:]


    if mem.empty:
        return {}


    # tempo relativo alla missione

    time = (
        mem["time"].values.astype(float)
        - t0
    )


    output = {}


    for col in mem.columns:

        if col == "time":
            continue


        values = (
            mem[col]
            .values
            .astype(float)
        )


        interp = interpolate_signal(
            t,
            time,
            values
        )


        if interp is not None:
            output[col] = interp


    return output

# ==========================================================
# MEMORY
# ==========================================================

def load_memory(experiment, common_time):

    Tmax=get_max_time(experiment)
    t = common_time
    memory_file = os.path.join(
        experiment,
        "memory_log.csv"
    )

    if not os.path.exists(memory_file):
        return {}


    mem = pd.read_csv(memory_file)


    if mem.empty:
        return {}


    # ============================================
    # gestione tempo assoluto
    # ============================================

    t0 = mem["time"].iloc[0]

    t_end = t0 + Tmax


    # taglio missione
    mem = mem[
        mem["time"] <= t_end
    ]


    if len(mem) > 2:
        mem = mem.iloc[2:]


    if mem.empty:
        return {}


    # tempo relativo alla missione

    time = (
        mem["time"].values.astype(float)
        - t0
    )


    output = {}


    for col in mem.columns:

        if col == "time":
            continue


        values = (
            mem[col]
            .values
            .astype(float)
        )


        interp = interpolate_signal(
            t,
            time,
            values
        )


        if interp is not None:
            output[col] = interp


    return output

# ==========================================================
# LOAD COMPLETE EXPERIMENT
# ==========================================================

def load_experiment(exp, c_time):

    experiment={}

    experiment["time"]=common_time(exp)

    experiment["robots"]={}

    for robot in ROBOTS:

        experiment["robots"][robot]={}

        experiment["robots"][robot]["position"]=\
            load_position(exp,robot, c_time)

        experiment["robots"][robot]["battery"]=\
            load_battery(exp,robot, c_time)

        experiment["robots"][robot]["priority"]=\
            load_priority(exp,robot,c_time)

    experiment["memory"]=load_memory(exp,c_time)

    return experiment

# ==========================================================
# LOAD GROUP
# ==========================================================

def load_group(exp_list):
    # trova il massimo tempo del gruppo
    max_group_time = max(
        get_max_time(exp)
        for exp in exp_list
    )

    # crea timeline comune
    common_time = np.arange(
        0,
        max_group_time,
        0.1
    )
    data=[]

    for exp in exp_list:

        print("Loading",exp)

        data.append(
            load_experiment(exp, common_time)
        )

    return data, common_time

# ==========================================================
# MEAN STD
# ==========================================================

def mean_std(data):
    lengths = [len(x) for x in data]

    if len(set(lengths)) != 1:
        print("ERROR: different lengths:")
        print(lengths)
        raise ValueError(
            "Signals have different lengths"
        )
    data=np.asarray(data)

    return np.mean(data,axis=0),\
           np.std(data,axis=0)

# ==========================================================
# SAVE FIGURE
# ==========================================================

def save(fig,folder,name):

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            folder,
            name+".png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

# ==========================================================
# CREATE OUTPUT FOLDERS
# ==========================================================

equal_output=os.path.join(
    OUTPUT_DIR,
    "equal"
)

reward_output=os.path.join(
    OUTPUT_DIR,
    "reward"
)

os.makedirs(equal_output,exist_ok=True)
os.makedirs(reward_output,exist_ok=True)

print("\nLoading Equal experiments...\n")
equal_data, equal_time=load_group(equal_experiments)

print("\nLoading Reward experiments...\n")
reward_data, reward_time=load_group(reward_experiments)

print("\nData loading completed.")

# ==========================================================
# PART 2
# TIME SERIES AGGREGATION
# ==========================================================


# ==========================================================
# TRAJECTORIES
# ==========================================================

def aggregate_trajectories(
        experiments,
        output_folder,
        time,
        group_name):


    for robot in ROBOTS:

        x_data=[]
        y_data=[]


        for exp in experiments:

            pos = exp["robots"][robot]["position"]

            x_data.append(
                pos["x"]
            )

            y_data.append(
                pos["y"]
            )


        x_mean,x_std = mean_std(x_data)
        y_mean,y_std = mean_std(y_data)


        fig,ax=plt.subplots(
            figsize=get_figsize((7,7))
        )

        ax.set_title(
            f"{group_name} - {robot} mean trajectory"
        )

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        ax.set_aspect("equal")


        # trajectories individuali leggere
        for x,y in zip(x_data,y_data):

            ax.plot(
                x,
                y,
                alpha=0.15
            )


        # traiettoria media

        ax.plot(
            x_mean,
            y_mean,
            linewidth=3,
            label="mean"
        )


        # deviazione standard come banda
        ax.fill_between(
            x_mean,
            y_mean-y_std,
            y_mean+y_std,
            alpha=0.25,
            label="std"
        )


        for i,wp in enumerate(waypoints):

            ax.scatter(
                wp[0],
                wp[1],
                marker="*",
                s=120,
                label=f"WP{i+1}"
            )


        ax.legend()

        save(
            fig,
            output_folder,
            f"trajectory_{robot}"
        )

        plt.close(fig)



# ==========================================================
# PRIORITY
# ==========================================================


def aggregate_priority(
        experiments,
        output_folder,
        time,
        group_name):


    for robot in ROBOTS:


        priority_names = list(
            experiments[0]
            ["robots"]
            [robot]
            ["priority"]
            .keys()
        )

        if "time" in priority_names:
            priority_names.remove("time")


        fig,ax=plt.subplots(
            figsize=get_figsize((10,5))
        )


        t = time


        for p in priority_names:


            values=[]


            for exp in experiments:

                values.append(
                    exp["robots"]
                    [robot]
                    ["priority"]
                    [p]
                )


            mean,std=mean_std(values)


            ax.plot(
                t,
                mean,
                label=p
            )


            ax.fill_between(
                t,
                mean-std,
                mean+std,
                alpha=0.2
            )


        ax.set_title(
            f"{group_name} - {robot} priority"
        )

        ax.set_xlabel(
            "Time [s]"
        )

        ax.set_ylabel(
            "Priority"
        )


        ax.legend()

        save(
            fig,
            output_folder,
            f"priority_{robot}"
        )

        plt.close(fig)



# ==========================================================
# MEMORY
# ==========================================================


def aggregate_memory(
        experiments,
        output_folder,
        time,
        group_name):


    memory_names=list(
        experiments[0]
        ["memory"]
        .keys()
    )

    if "time" in memory_names:
        memory_names.remove("time")


    fig,ax=plt.subplots(
        figsize=get_figsize((10,5))
    )


    t=time


    for m in memory_names:


        values=[]


        for exp in experiments:

            values.append(
                exp["memory"][m]
            )


        mean,std=mean_std(values)


        ax.plot(
            t,
            mean,
            label=m
        )


        ax.fill_between(
            t,
            mean-std,
            mean+std,
            alpha=0.2
        )


    ax.set_title(
        f"{group_name} - Memory"
    )

    ax.set_xlabel(
        "Time [s]"
    )

    ax.set_ylabel(
        "Memory value"
    )


    ax.legend()


    save(
        fig,
        output_folder,
        "memory"
    )

    plt.close(fig)



# ==========================================================
# BATTERY
# ==========================================================


def aggregate_battery(
        experiments,
        output_folder,
        time,
        group_name):


    for robot in ROBOTS:


        t=time


        # --------------------
        # Percentage
        # --------------------

        percentage=[]


        voltage=[]


        for exp in experiments:

            batt=exp["robots"][robot]["battery"]


            if "percentage" in batt:

                percentage.append(
                    batt["percentage"]
                )


            if "voltage" in batt:

                voltage.append(
                    batt["voltage"]
                )



        fig,axes=plt.subplots(
            1,
            2,
            figsize=get_figsize((12,4))
        )


        if len(percentage)>0:


            mean,std=mean_std(
                percentage
            )


            axes[0].plot(
                t,
                mean,
                label="mean"
            )


            axes[0].fill_between(
                t,
                mean-std,
                mean+std,
                alpha=0.25
            )


        axes[0].set_title(
            f"{group_name} {robot} battery %"
        )

        axes[0].set_xlabel(
            "Time [s]"
        )

        axes[0].set_ylabel(
            "%"
        )

        axes[0].set_ylim(
            0,
            100
        )


        # --------------------
        # Voltage
        # --------------------


        if len(voltage)>0:


            mean,std=mean_std(
                voltage
            )


            axes[1].plot(
                t,
                mean,
                label="mean"
            )


            axes[1].fill_between(
                t,
                mean-std,
                mean+std,
                alpha=0.25
            )


        axes[1].set_title(
            f"{group_name} {robot} voltage"
        )

        axes[1].set_xlabel(
            "Time [s]"
        )

        axes[1].set_ylabel(
            "Voltage [V]"
        )


        save(
            fig,
            output_folder,
            f"battery_{robot}"
        )


        plt.close(fig)



# ==========================================================
# EXECUTION PART 2
# ==========================================================


print("\nGenerating Equal time-series plots")

aggregate_trajectories(
    equal_data,
    equal_output,
    equal_time,
    "equal"
)

aggregate_priority(
    equal_data,
    equal_output,
    equal_time,
    "equal"
)

aggregate_memory(
    equal_data,
    equal_output,
    equal_time,
    "equal"
)

aggregate_battery(
    equal_data,
    equal_output,
    equal_time,
    "equal"
)



print("\nGenerating Reward time-series plots")


aggregate_trajectories(
    reward_data,
    reward_output,
    reward_time,
    "reward"
)

aggregate_priority(
    reward_data,
    reward_output,
    reward_time,
    "reward"
)

aggregate_memory(
    reward_data,
    reward_output,
    reward_time,
    "reward"
)

aggregate_battery(
    reward_data,
    reward_output,
    reward_time,
    "reward"
)


print("\nTime series aggregation completed.")
# ==========================================================
# PART 3
# FINAL METRICS AGGREGATION
# ==========================================================


import re


# ==========================================================
# LOAD CSV RESULTS
# ==========================================================


def read_summary_csv(experiment):

    file = os.path.join(
        experiment,
        "plots",
        "summary.csv"
    )

    if not os.path.exists(file):
        return None

    return pd.read_csv(file)



def read_occupancy(experiment):

    file = os.path.join(
        experiment,
        "plots",
        "results.log"
    )


    if not os.path.exists(file):
        return np.nan


    with open(file,"r") as f:

        for line in f:

            if "Occupancy %" in line:

                value=re.findall(
                    r"[-+]?\d*\.\d+",
                    line
                )

                if value:
                    return float(value[0])


    return np.nan



# ==========================================================
# OCCUPANCY
# ==========================================================


def aggregate_occupancy(
        experiments,
        output_folder,
        name):


    values=[]


    for exp in experiments:

        values.append(
            read_occupancy(exp)
        )


    values=np.array(
        values,
        dtype=float
    )


    mean=np.nanmean(values)
    std=np.nanstd(values)


    df=pd.DataFrame({

        "metric":[
            "occupancy"
        ],

        "mean":[
            mean
        ],

        "std":[
            std
        ]

    })


    df.to_csv(
        os.path.join(
            output_folder,
            "occupancy_statistics.csv"
        ),
        index=False
    )


    fig,ax=plt.subplots(
        figsize=get_figsize((4,4))
    )


    ax.bar(
        [name],
        [mean],
        yerr=[std],
        capsize=5
    )


    ax.set_ylabel(
        "Occupancy [%]"
    )


    ax.set_title(
        f"{name} occupancy"
    )


    save(
        fig,
        output_folder,
        "occupancy_mean"
    )


    plt.close(fig)




# ==========================================================
# WAYPOINT STATISTICS
# ==========================================================


def aggregate_waypoints(
        experiments,
        output_folder,
        name):


    data=[]


    for exp in experiments:


        file=os.path.join(
            exp,
            "plots",
            "waypoint_statistics.csv"
        )


        if os.path.exists(file):

            df=pd.read_csv(
                file,
                index_col=0
            )

            data.append(df)



    if len(data)==0:
        return



    stack=np.array(
        [
            d.values
            for d in data
        ]
    )


    mean=np.mean(
        stack,
        axis=0
    )


    std=np.std(
        stack,
        axis=0
    )


    robots=[
        "Robot1",
        "Robot2",
        "Robot3"
    ]


    wp_labels=[
        f"WP{i+1}"
        for i in range(mean.shape[1])
    ]


    result=[]


    for r in range(3):

        for w in range(mean.shape[1]):

            result.append({

                "robot":robots[r],
                "waypoint":wp_labels[w],
                "mean":mean[r,w],
                "std":std[r,w]

            })


    pd.DataFrame(result).to_csv(
        os.path.join(
            output_folder,
            "waypoint_statistics_mean.csv"
        ),
        index=False
    )



    # stacked bar


    fig,ax=plt.subplots(
        figsize=get_figsize((8,5))
    )


    bottom=np.zeros(3)


    for w in range(mean.shape[1]):


        # ax.bar(
        #     robots,
        #     mean[:,w],
        #     bottom=bottom,
        #     yerr=std[:,w],
        #     capsize=3,
        #     label=wp_labels[w]
        # )


        bottom+=mean[:,w]

    ax.bar(
        robots,
        bottom,
        yerr=np.sqrt(np.sum(std**2,axis=1)),
        capsize=5,
        
    )
    ax.set_ylabel(
        "Time [s]"
    )

    ax.set_title(
        f"{name} waypoint time"
    )


    ax.legend()


    save(
        fig,
        output_folder,
        "waypoint_mean"
    )


    plt.close(fig)



# ==========================================================
# HEADING
# ==========================================================


def aggregate_heading(
        experiments,
        output_folder,
        name):


    values=[]


    for exp in experiments:


        file=os.path.join(
            exp,
            "plots",
            "summary.csv"
        )


        if os.path.exists(file):

            df=pd.read_csv(file)

            values.append(
                df["Heading_%"].values
            )



    values=np.array(values)


    mean=np.mean(
        values,
        axis=0
    )


    std=np.std(
        values,
        axis=0
    )


    pd.DataFrame({

        "Robot":[
            "Robot1",
            "Robot2",
            "Robot3"
        ],

        "mean":mean,

        "std":std

    }).to_csv(

        os.path.join(
            output_folder,
            "heading_statistics.csv"
        ),

        index=False
    )



    fig,ax=plt.subplots(
        figsize=get_figsize((5,4))
    )


    ax.bar(
        [
            "Robot1",
            "Robot2",
            "Robot3"
        ],
        mean,
        yerr=std,
        capsize=5
    )


    ax.set_ylabel(
        "% mission"
    )


    ax.set_title(
        f"{name} heading"
    )


    save(
        fig,
        output_folder,
        "heading_mean"
    )

    plt.close(fig)



# ==========================================================
# SWARM COMPACTNESS
# ==========================================================
def aggregate_compactness(data, time, output_folder, name):

    robots = [
        "rosbot_1_0",
        "rosbot_2_1",
        "rosbot_3_2"
    ]

    robot_labels = [
        "Robot 1",
        "Robot 2",
        "Robot 3"
    ]

    distances = {
        robot: []
        for robot in robots
    }

    # ==================================================
    # COMPUTE DISTANCE FROM INSTANTANEOUS SWARM CENTER
    # ==================================================

    for experiment in data:

        positions = []

        for robot in robots:

            p = experiment["robots"][robot]["position"]

            xy = np.column_stack((
                p["x"],
                p["y"]
            ))

            positions.append(xy)

        # Make sure all robots have the same length
        min_len = min(
            len(position)
            for position in positions
        )

        positions = np.array([
            position[:min_len]
            for position in positions
        ])

        # Shape:
        # (3 robots, N samples, 2 coordinates)

        center = np.mean(
            positions,
            axis=0
        )

        # Distance of every robot from
        # instantaneous swarm center
        dist = np.linalg.norm(
            positions - center,
            axis=2
        )

        for i, robot in enumerate(robots):
            distances[robot].append(
                dist[i]
            )

    # ==================================================
    # MEAN ACROSS EXPERIMENTS
    # ==================================================

    mean_distances = {}

    for robot in robots:

        if len(distances[robot]) == 0:
            continue

        min_len = min(
            len(d)
            for d in distances[robot]
        )

        values = np.array([
            d[:min_len]
            for d in distances[robot]
        ])

        mean_distances[robot] = np.mean(
            values,
            axis=0
        )

    # ==================================================
    # CREATE TIME VECTOR FROM DATA
    # ==================================================

    if len(mean_distances) == 0:
        print(
            f"WARNING: no compactness data available for {name}"
        )
        return

    n_samples = max(
        len(values)
        for values in mean_distances.values()
    )

    # Use DT = 0.05 s
    compactness_time = np.arange(n_samples) * DT

    # ==================================================
    # PLOT
    # ==================================================

    fig, ax = plt.subplots(
        figsize=get_figsize((10, 5))
    )

    for robot, label in zip(
        robots,
        robot_labels
    ):

        if robot not in mean_distances:
            continue

        values = mean_distances[robot]

        local_time = np.arange(
            len(values)
        ) * DT

        ax.plot(
            local_time,
            values,
            label=label
        )

    ax.set_xlabel("Time [s]")

    ax.set_ylabel(
        "Distance from swarm center [m]"
    )

    ax.set_title(
        f"{name} - distance from swarm center"
    )

    ax.legend()

    ax.grid(
        axis="both",
        alpha=0.3
    )

    save(
        fig,
        output_folder,
        "swarm_compactness"
    )

    plt.close(fig)

def compare_swarm_compactness(
        equal_data,
        reward_data,
        output_folder):

    robots = [
        "rosbot_1_0",
        "rosbot_2_1",
        "rosbot_3_2"
    ]

    robot_labels = [
        "Robot 1",
        "Robot 2",
        "Robot 3"
    ]

    def compute_mean_distances(data):

        distances = {
            robot: []
            for robot in robots
        }

        for experiment in data:

            positions = []

            for robot in robots:

                p = experiment["robots"][robot]["position"]

                xy = np.column_stack((
                    p["x"],
                    p["y"]
                ))

                positions.append(xy)

            # Same number of samples for all robots
            min_len = min(
                len(position)
                for position in positions
            )

            positions = np.array([
                position[:min_len]
                for position in positions
            ])

            # Instantaneous swarm center
            center = np.mean(
                positions,
                axis=0
            )

            # Distance of each robot from swarm center
            dist = np.linalg.norm(
                positions - center,
                axis=2
            )

            for i, robot in enumerate(robots):
                distances[robot].append(
                    dist[i]
                )

        # Mean across experiments
        mean_distances = {}

        for robot in robots:

            if len(distances[robot]) == 0:
                continue

            min_len = min(
                len(d)
                for d in distances[robot]
            )

            values = np.array([
                d[:min_len]
                for d in distances[robot]
            ])

            mean_distances[robot] = np.mean(
                values,
                axis=0
            )

        return mean_distances

    # ==================================================
    # COMPUTE EQUAL AND REWARD
    # ==================================================

    equal_distances = compute_mean_distances(
        equal_data
    )

    reward_distances = compute_mean_distances(
        reward_data
    )

    # ==================================================
    # PLOT
    # ==================================================

    fig, ax = plt.subplots(
        figsize=get_figsize((12, 6))
    )

    for robot, label in zip(
        robots,
        robot_labels
    ):

        # Equal
        if robot in equal_distances:

            values = equal_distances[robot]

            time = np.arange(
                len(values)
            ) * DT

            ax.plot(
                time,
                values,
                label=f"Equal - {label}"
            )

        # Reward
        if robot in reward_distances:

            values = reward_distances[robot]

            time = np.arange(
                len(values)
            ) * DT

            ax.plot(
                time,
                values,
                linestyle="--",
                label=f"Reward - {label}"
            )

    ax.set_xlabel("Time [s]")

    ax.set_ylabel(
        "Distance from swarm center [m]"
    )

    ax.set_title(
        "Swarm compactness comparison"
    )

    ax.legend()

    ax.grid(
        axis="both",
        alpha=0.3
    )

    save(
        fig,
        output_folder,
        "swarm_compactness_comparison"
    )

    plt.close(fig)

def aggregate_mission_time(experiments, output_folder, name):

    agents = ["Robot1", "Robot2", "Robot3", "Swarm"]

    values = {
        agent: []
        for agent in agents
    }

    # ==================================================
    # READ DIRECTLY FROM mission_time_statistics.csv
    # ==================================================

    for exp in experiments:

        file_path = os.path.join(
            exp,
            "mission_time_statistics.csv"
        )

        if not os.path.exists(file_path):
            print(f"WARNING: file not found: {file_path}")
            continue

        df = pd.read_csv(file_path)

        for agent in agents:

            row = df[df["Agent"] == agent]

            if not row.empty:

                value = row["Mission time [s]"].iloc[0]

                if pd.notna(value):
                    values[agent].append(float(value))

    # ==================================================
    # COMPUTE MEAN
    # ==================================================

    result = []

    for agent in agents:

        if len(values[agent]) > 0:
            mean_value = np.mean(values[agent])
        else:
            mean_value = np.nan

        result.append({
            "Agent": agent,
            "Mission time [s]": mean_value
        })

    result = pd.DataFrame(result)

    # ==================================================
    # SAVE AGGREGATED FILE
    # ==================================================

    output_file = os.path.join(
        output_folder,
        "mission_time_statistics.csv"
    )

    result.to_csv(
        output_file,
        index=False
    )

    print(f"Saved: {output_file}")

    # ==================================================
    # PLOT
    # ==================================================

    std = []

    for agent in agents:

        if len(values[agent]) > 0:
            std.append(np.std(values[agent]))
        else:
            std.append(np.nan)

    fig, ax = plt.subplots(figsize=get_figsize((8, 5)))

    ax.bar(
        agents,
        result["Mission time [s]"],
        yerr=std,
        capsize=5
    )

    ax.set_xlabel("Agent")
    ax.set_ylabel("Mission time [s]")
    ax.set_title(f"{name} - Mission time")

    ax.grid(
        axis="y",
        alpha=0.3
    )

    save(
        fig,
        output_folder,
        "mission_time_mean"
    )

    plt.close(fig)

def compare_mission_time(
        equal_experiments,
        reward_experiments,
        output_folder):

    agents = [
        "Robot1",
        "Robot2",
        "Robot3",
        "Swarm"
    ]

    # ==================================================
    # LOAD DIRECTLY FROM mission_time_statistics.csv
    # ==================================================

    def load_values(experiments):

        data = {
            agent: []
            for agent in agents
        }

        for exp in experiments:

            file_path = os.path.join(
                exp,
                "mission_time_statistics.csv"
            )

            if not os.path.exists(file_path):
                print(f"WARNING: file not found: {file_path}")
                continue

            df = pd.read_csv(file_path)

            for agent in agents:

                row = df[df["Agent"] == agent]

                if not row.empty:

                    value = row["Mission time [s]"].iloc[0]

                    if pd.notna(value):
                        data[agent].append(float(value))

        return data

    equal_values = load_values(equal_experiments)
    reward_values = load_values(reward_experiments)

    # ==================================================
    # MEAN AND STD
    # ==================================================

    equal_mean = np.array([
        np.mean(equal_values[agent])
        for agent in agents
    ])

    equal_std = np.array([
        np.std(equal_values[agent])
        for agent in agents
    ])

    reward_mean = np.array([
        np.mean(reward_values[agent])
        for agent in agents
    ])

    reward_std = np.array([
        np.std(reward_values[agent])
        for agent in agents
    ])

    # ==================================================
    # SAVE COMPARISON DATA
    # ==================================================

    result = pd.DataFrame({
        "Agent": agents,
        "Equal_mean": equal_mean,
        "Equal_std": equal_std,
        "Reward_mean": reward_mean,
        "Reward_std": reward_std
    })

    result.to_csv(
        os.path.join(
            output_folder,
            "comparison_mission_time.csv"
        ),
        index=False
    )

    # ==================================================
    # PLOT
    # ==================================================

    fig, ax = plt.subplots(figsize=get_figsize((8, 5)))

    x = np.arange(len(agents))
    width = 0.35

    ax.bar(
        x - width / 2,
        equal_mean,
        width,
        yerr=equal_std,
        capsize=5,
        label="Equal"
    )

    ax.bar(
        x + width / 2,
        reward_mean,
        width,
        yerr=reward_std,
        capsize=5,
        label="Reward"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(agents)

    ax.set_ylabel("Mission time [s]")
    ax.set_title("Mission time comparison")

    ax.legend()
    ax.grid(
        axis="y",
        alpha=0.3
    )

    save(
        fig,
        output_folder,
        "comparison_mission_time"
    )

    plt.close(fig)
# ==========================================================
# EXECUTION PART 3
# ==========================================================


for data,out,name in [

    (equal_data,equal_output,"equal"),

    (reward_data,reward_output,"reward")

]:


    print(
        f"\nProcessing final statistics {name}"
    )


    aggregate_occupancy(
        groups[name],
        out,
        name
    )


    aggregate_waypoints(
        groups[name],
        out,
        name
    )


    aggregate_heading(
        groups[name],
        out,
        name
    )
    aggregate_mission_time(
        groups[name],
        out,
        name
    )
    if name == "equal":
        time = equal_time
    elif name == "reward":
        time = reward_time
    else:
        print("Unknown group name:", name)
    aggregate_compactness(
        data,
        time,
        out,
        name
    )




print(
    "\n================================"
)

print(
    "ALL AGGREGATION COMPLETED"
)

print(
    "================================"
)

# ============================================================
# PART 4
# TEMPORAL DATA AGGREGATION
# Priority / Memory / Battery
# ============================================================


TIME_SAMPLES = 500


def interpolate_curve(time, values, max_time, samples=TIME_SAMPLES):

    """
    Porta una curva temporale su una griglia comune
    """

    if len(time) < 2:
        return None, None
    t_rel = time - time[0]

    t_norm = np.linspace(
        0,
        max_time,
        samples
    )

    values_interp = np.interp(
        t_norm,
        t_rel,
        values
    )

    return t_norm, values_interp



def aggregate_time_series(
        experiment_paths,
        filename,
        columns,
        output_name):

    """
    Aggrega file csv temporali uguali
    esempio:
        robot_priority.csv
        battery.csv
        memory_log.csv
    """


    data = {
        col: []
        for col in columns
    }

    time_vectors = []


    for exp in experiment_paths:


        file_path = os.path.join(
            exp,
            filename
        )


        if not os.path.exists(file_path):
            continue


        df = pd.read_csv(file_path)



        if "time" not in df.columns:
            continue



        df = df.sort_values("time")



        time = df["time"].values.astype(float)

        time = time - time[0]



        for col in columns:

            if col not in df.columns:
                continue


            values = df[col].values.astype(float)



            t_new, values_new = interpolate_curve(
                time,
                values, 
                get_max_time(exp)
            )


            if values_new is not None:

                data[col].append(values_new)

            if values_new is not None:
                time_vectors.append(t_new)



    if len(time_vectors)==0:
        return



    common_time = time_vectors[0]



    for col in columns:


        if len(data[col]) == 0:
            continue



        array = np.vstack(
            data[col]
        )


        mean = np.mean(
            array,
            axis=0
        )


        std = np.std(
            array,
            axis=0
        )


        plt.figure(
            figsize=get_figsize((8,4))
        )


        plt.plot(
            common_time,
            mean,
            label="mean"
        )


        plt.fill_between(
            common_time,
            mean-std,
            mean+std,
            alpha=0.3,
            label="± std"
        )


        plt.xlabel(
            "Normalized time [s]"
        )

        plt.ylabel(
            col
        )


        plt.title(
            f"{output_name} - {col}"
        )


        plt.grid(True)

        plt.legend()


        plt.tight_layout()



        plt.savefig(
            os.path.join(
                AGGREGATE_DIR,
                f"{output_name}_{col}.png"
            ),
            dpi=300,
        bbox_inches="tight"
        )


        plt.close()




# ============================================================
# PRIORITY
# ============================================================


for tag, experiments in [
    ("equal", BASE_PATHS_EQUAL),
    ("reward", BASE_PATHS_REWARD)
]:


    robots = [
        "rosbot_1_0",
        "rosbot_2_1",
        "rosbot_3_2"
    ]


    for robot in robots:


        aggregate_time_series(

            experiments,

            f"{robot}_priority.csv",

            [
                "priority0",
                "priority1",
                "priority2"
            ],

            f"{tag}_{robot}_priority"

        )




# ============================================================
# MEMORY
# ============================================================


for tag, experiments in [
    ("equal", BASE_PATHS_EQUAL),
    ("reward", BASE_PATHS_REWARD)
]:


    aggregate_time_series(

        experiments,

        "memory_log.csv",

        [
            "memory0",
            "memory1",
            "memory2"
        ],

        f"{tag}_memory"

    )




# ============================================================
# BATTERY
# ============================================================


for tag, experiments in [
    ("equal", BASE_PATHS_EQUAL),
    ("reward", BASE_PATHS_REWARD)
]:


    robots = [
        "rosbot_1_0",
        "rosbot_2_1",
        "rosbot_3_2"
    ]


    for robot in robots:


        aggregate_time_series(

            experiments,

            f"{robot}_battery.csv",

            [
                "percentage",
                "voltage"
            ],

            f"{tag}_{robot}_battery"

        )



print(
    "Temporal aggregation completed"
)

# ==========================================================
# COMPARISON: OCCUPANCY
# ==========================================================

def compare_occupancy(
        equal_experiments,
        reward_experiments,
        output_folder):

    equal_values = np.array([
        read_occupancy(exp)
        for exp in equal_experiments
    ], dtype=float)

    reward_values = np.array([
        read_occupancy(exp)
        for exp in reward_experiments
    ], dtype=float)

    equal_mean = np.nanmean(equal_values)
    equal_std = np.nanstd(equal_values)

    reward_mean = np.nanmean(reward_values)
    reward_std = np.nanstd(reward_values)

    fig, ax = plt.subplots(figsize=get_figsize((6, 4)))

    x = np.arange(2)

    means = [
        equal_mean,
        reward_mean
    ]

    stds = [
        equal_std,
        reward_std
    ]

    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        tick_label=["Equal", "Reward"]
    )

    ax.set_ylabel("Occupancy [%]")
    ax.set_title("Occupancy comparison")

    save(
        fig,
        output_folder,
        "comparison_occupancy"
    )

    plt.close(fig)

# ==========================================================
# PART 4
# EQUAL vs REWARD COMPARISON
# ==========================================================

# ==========================================================
# COMPARISON: HEADING
# ==========================================================

def compare_heading(
        equal_experiments,
        reward_experiments,
        output_folder):

    def load_heading(experiments):

        values = []

        for exp in experiments:

            file = os.path.join(
                exp,
                "plots",
                "summary.csv"
            )

            if os.path.exists(file):

                df = pd.read_csv(file)

                values.append(
                    df["Heading_%"].values
                )

        return np.array(values)


    equal_values = load_heading(
        equal_experiments
    )

    reward_values = load_heading(
        reward_experiments
    )


    equal_mean = np.mean(
        equal_values,
        axis=0
    )

    equal_std = np.std(
        equal_values,
        axis=0
    )


    reward_mean = np.mean(
        reward_values,
        axis=0
    )

    reward_std = np.std(
        reward_values,
        axis=0
    )


    robots = [
        "Robot1",
        "Robot2",
        "Robot3"
    ]

    x = np.arange(len(robots))

    width = 0.35


    fig, ax = plt.subplots(
        figsize=get_figsize((7, 4))
    )


    ax.bar(
        x - width/2,
        equal_mean,
        width,
        yerr=equal_std,
        capsize=5,
        label="Equal"
    )


    ax.bar(
        x + width/2,
        reward_mean,
        width,
        yerr=reward_std,
        capsize=5,
        label="Reward"
    )


    ax.set_xticks(x)
    ax.set_xticklabels(robots)

    ax.set_ylabel("% mission")

    ax.set_title(
        "Heading comparison"
    )


    save(
        fig,
        output_folder,
        "comparison_heading"
    )

    plt.close(fig)
        # ==========================================================
    # WAYPOINT TIME AS PERCENTAGE OF TOTAL MISSION TIME
    # ==========================================================

    def load_mission_times(experiments):

        data = {
            "Robot1": [],
            "Robot2": [],
            "Robot3": []
        }

        for exp in experiments:

            file = os.path.join(
                exp,
                "mission_time_statistics.csv"
            )

            if not os.path.exists(file):
                print(f"WARNING: missing {file}")
                continue

            df = pd.read_csv(file)

            for robot in data:

                row = df[
                    df["Agent"] == robot
                ]

                if not row.empty:

                    value = row[
                        "Mission time [s]"
                    ].iloc[0]

                    if pd.notna(value):
                        data[robot].append(
                            float(value)
                        )

        return data


    equal_mission = load_mission_times(
        equal_experiments
    )

    reward_mission = load_mission_times(
        reward_experiments
    )


    # ------------------------------------------
    # Mean total mission time
    # ------------------------------------------

    equal_mission_mean = np.array([
        np.mean(equal_mission[robot])
        for robot in robots
    ])

    reward_mission_mean = np.array([
        np.mean(reward_mission[robot])
        for robot in robots
    ])


# ==========================================================
# COMPARISON: WAYPOINT
# ==========================================================

def compare_waypoints(
        equal_experiments,
        reward_experiments,
        output_folder):

    robots = [
        "Robot1",
        "Robot2",
        "Robot3"
    ]

    # ==========================================================
    # LOAD WAYPOINT DATA
    # ==========================================================

    def load_waypoint_data(experiments):

        data = []

        for exp in experiments:

            file = os.path.join(
                exp,
                "plots",
                "waypoint_statistics.csv"
            )

            if os.path.exists(file):

                df = pd.read_csv(
                    file,
                    index_col=0
                )

                data.append(
                    df.values
                )

        return np.array(data)

    equal_data = load_waypoint_data(
        equal_experiments
    )

    reward_data = load_waypoint_data(
        reward_experiments
    )

    # ==========================================================
    # MEAN AND STD FOR EACH WAYPOINT
    # ==========================================================

    equal_mean = np.mean(
        equal_data,
        axis=0
    )

    equal_std = np.std(
        equal_data,
        axis=0
    )

    reward_mean = np.mean(
        reward_data,
        axis=0
    )

    reward_std = np.std(
        reward_data,
        axis=0
    )

    # ==========================================================
    # TOTAL WAYPOINT TIME FOR EACH ROBOT
    # ==========================================================

    equal_mean_total = np.sum(
        equal_mean,
        axis=1
    )

    reward_mean_total = np.sum(
        reward_mean,
        axis=1
    )

    equal_std_total = np.sqrt(
        np.sum(
            equal_std ** 2,
            axis=1
        )
    )

    reward_std_total = np.sqrt(
        np.sum(
            reward_std ** 2,
            axis=1
        )
    )

    # ==========================================================
    # ABSOLUTE WAYPOINT TIME COMPARISON
    # ==========================================================

    x = np.arange(
        len(robots)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=get_figsize((7, 4))
    )

    ax.bar(
        x - width / 2,
        equal_mean_total,
        width,
        yerr=equal_std_total,
        capsize=5,
        label="Equal"
    )

    ax.bar(
        x + width / 2,
        reward_mean_total,
        width,
        yerr=reward_std_total,
        capsize=5,
        label="Reward"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(robots)

    ax.set_ylabel(
        "Time [s]"
    )

    ax.set_title(
        "Waypoint time comparison"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    save(
        fig,
        output_folder,
        "comparison_waypoint"
    )

    plt.close(fig)

    # ==========================================================
    # LOAD MISSION TIMES
    # ==========================================================

    def load_mission_times(experiments):

        data = {
            "Robot1": [],
            "Robot2": [],
            "Robot3": []
        }

        for exp in experiments:

            file = os.path.join(
                exp,
                "mission_time_statistics.csv"
            )

            if not os.path.exists(file):

                print(
                    f"WARNING: missing {file}"
                )

                continue

            df = pd.read_csv(file)

            for robot in robots:

                row = df[
                    df["Agent"] == robot
                ]

                if not row.empty:

                    value = row[
                        "Mission time [s]"
                    ].iloc[0]

                    if pd.notna(value):

                        data[robot].append(
                            float(value)
                        )

        return data

    equal_mission = load_mission_times(
        equal_experiments
    )

    reward_mission = load_mission_times(
        reward_experiments
    )

    # ==========================================================
    # MEAN TOTAL MISSION TIME
    # ==========================================================

    equal_mission_mean = np.array([
        np.mean(
            equal_mission[robot]
        )
        for robot in robots
    ])

    reward_mission_mean = np.array([
        np.mean(
            reward_mission[robot]
        )
        for robot in robots
    ])
    print("Equal mission 0:", np.mean(equal_mission[robots[0]]))
    print("Equal mean:", equal_mean_total[0])
    # ==========================================================
    # WAYPOINT TIME AS % OF MISSION TIME
    # ==========================================================

    equal_percentage = np.array([
        equal_mean_total[0]/ np.mean(equal_mission[robots[0]])* 100,
        equal_mean_total[1]/ np.mean(equal_mission[robots[1]])* 100,
        equal_mean_total[2]/ np.mean(equal_mission[robots[2]])* 100
    ])

    reward_percentage = np.array([
        reward_mean_total[0]/ np.mean(reward_mission[robots[0]])* 100,
        reward_mean_total[1]/ np.mean(reward_mission[robots[1]])* 100,
        reward_mean_total[2]/ np.mean(reward_mission[robots[2]])* 100
    ])

    # ==========================================================
    # PERCENTAGE COMPARISON
    # ==========================================================

    fig, ax = plt.subplots(
        figsize=get_figsize((7, 4))
    )

    ax.bar(
        x - width / 2,
        equal_percentage,
        width,
        label="Equal"
    )

    ax.bar(
        x + width / 2,
        reward_percentage,
        width,
        label="Reward"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(robots)

    ax.set_ylabel(
        "Waypoint time / Mission time [%]"
    )

    ax.set_title(
        "Waypoint time as percentage of mission time"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    save(
        fig,
        output_folder,
        "comparison_waypoint_percentage"
    )

    plt.close(fig)


comparison_output = os.path.join(
    OUTPUT_DIR,
    "comparison"
)

os.makedirs(
    comparison_output,
    exist_ok=True
)


print("\nGenerating Equal vs Reward comparisons...")


compare_occupancy(
    equal_experiments,
    reward_experiments,
    comparison_output
)


compare_heading(
    equal_experiments,
    reward_experiments,
    comparison_output
)


compare_waypoints(
    equal_experiments,
    reward_experiments,
    comparison_output
)
compare_mission_time(
    equal_experiments,
    reward_experiments,
    comparison_output
)
compare_swarm_compactness(
    equal_data,
    reward_data,
    comparison_output
)

print(
    "\nComparison plots generated."
)