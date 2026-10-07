#!/usr/bin/env python3

import os
import subprocess
import argparse


# ==========================================================
# ARGUMENTS
# ==========================================================

parser = argparse.ArgumentParser(
    description="Run all analysis scripts."
)

parser.add_argument(
    "--base-path",
    default="/home/gonazza/container_ws/catkin_ws/src/indoor_bot/indoor_bot",
    help="Root folder containing config1, config2 and config3"
)

parser.add_argument(
    "--ral",
    action="store_true",
    help="Use RAL/IEEE paper plot formatting"
)

parser.add_argument(
    "--noshow",
    action="store_true",
    help="Do not show figures"
)

args = parser.parse_args()

BASE_PATH = args.base_path


# ==========================================================
# CONFIGURATIONS
# ==========================================================

CONFIGURATIONS = [
    "config1",
    "config2",
    "config3"
]


# ==========================================================
# ARGUMENTS FOR SCRIPTS
# ==========================================================

def get_plot_args():

    command_args = []

    if args.ral:
        command_args.append("--ral")

    if args.noshow:
        command_args.append("--noshow")

    return command_args


def get_aggregate_args():

    command_args = [
        "--base-path",
        BASE_PATH
    ]

    if args.ral:
        command_args.append("--ral")

    return command_args


# ==========================================================
# HELPERS
# ==========================================================

def script_path(script):

    return os.path.join(
        BASE_PATH,
        script
    )


def run_script(script, script_args):

    path = script_path(script)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Script not found: {path}"
        )

    print("\n" + "=" * 70)
    print(f"Running: {script}")
    print("=" * 70)

    command = [
        "python3",
        path,
        *script_args
    ]

    print(
        "Command:",
        " ".join(command)
    )

    subprocess.run(
        command,
        check=True
    )


def check_configurations():

    print("\n" + "=" * 70)
    print("CHECKING CONFIGURATIONS")
    print("=" * 70)

    found = {}

    for config in CONFIGURATIONS:

        config_path = os.path.join(
            BASE_PATH,
            config
        )

        exists = os.path.isdir(config_path)

        found[config] = exists

        if exists:
            print(f"{config}: OK")
        else:
            print(f"{config}: NOT FOUND")

    return found


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("\n" + "=" * 70)
    print("STARTING COMPLETE ANALYSIS")
    print("=" * 70)

    print(
        f"\nBase path:\n{BASE_PATH}"
    )

    print("\nConfigurations:")

    for config in CONFIGURATIONS:
        print(f"  - {config}")

    # ------------------------------------------------------
    # Check configuration folders
    # ------------------------------------------------------

    found = check_configurations()

    missing = [
        config
        for config in CONFIGURATIONS
        if not found[config]
    ]

    if missing:

        raise RuntimeError(
            "Missing configuration folders: "
            + ", ".join(missing)
        )

    # ======================================================
    # EXPERIMENT ANALYSIS
    # ======================================================

    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT ANALYSIS")
    print("=" * 70)

    # ------------------------------------------------------
    # data.py
    # ------------------------------------------------------

    run_script(
        "data.py",
        get_plot_args()
    )

    # ------------------------------------------------------
    # plot.py
    # ------------------------------------------------------

    run_script(
        "plot.py",
        get_plot_args()
    )

    # ------------------------------------------------------
    # occupancy.py
    # ------------------------------------------------------

    run_script(
        "occupancy.py",
        get_plot_args()
    )

    print("\n" + "=" * 70)
    print("ALL EXPERIMENT ANALYSIS COMPLETED")
    print("=" * 70)

    # ======================================================
    # AGGREGATION
    # ======================================================

    print("\n" + "=" * 70)
    print("RUNNING AGGREGATION AND STATISTICS")
    print("=" * 70)

    # ------------------------------------------------------
    # aggregate_results.py
    # ------------------------------------------------------

    run_script(
        "aggregate_results.py",
        get_aggregate_args()
    )

    # ------------------------------------------------------
    # mission_time.py
    # ------------------------------------------------------
    #
    # Per ora NON passiamo --base-path, perché non abbiamo
    # ancora verificato gli argomenti accettati da mission_time.py.
    #

    mission_args = []

    if args.ral:
        mission_args.append("--ral")

    run_script(
        "mission_time.py",
        mission_args
    )

    print("\n" + "=" * 70)
    print("ALL PROCESSING COMPLETED")
    print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()