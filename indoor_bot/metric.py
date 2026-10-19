# -*- coding: utf-8 -*-
"""Aggregate and compare Equal vs Reward across config1/config2/config3.
a
For every experiment, computes:
- area covered / occupancy [%]
- time near WP1..WP5 and total waypoint time [s]
- heading-towards-WP1..WP5 and total heading time [% of mission time]
- total waypoint time / mission time [%], for each robot
- mission time [s], for each robot
- photosynthesis produced

Results are reported separately for Equal and Reward for:
    config1, config2, config3, and All configurations.

The script automatically scans exp_equal* and exp_reward* under each
configuration and keeps the waypoint definitions configuration-specific.
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, box
from shapely.ops import unary_union

BASE_PATH = "/home/gonazza/container_ws/catkin_ws/src/indoor_bot/indoor_bot"
OUTPUT_DIR = os.path.join(BASE_PATH, "metrics_comparison")
CONFIGURATIONS = ["config1", "config2", "config3"]
GROUPS = ["equal", "reward"]
ROBOTS = ["rosbot_1_0", "rosbot_2_1", "rosbot_3_2"]
ROBOT_LABELS = {"rosbot_1_0": "Robot 1", "rosbot_2_1": "Robot 2", "rosbot_3_2": "Robot 3"}
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
WAYPOINT_RADIUS = 2.5
WAYPOINT_REENTRY_DELAY_S = 15.0
HEADING_THRESHOLD_DEG = 45.0
ROBOT_RADIUS = 2.5
XMIN, XMAX, YMIN, YMAX = -14.0, 7.0, -6.0, 4.0
EMPTY_AREAS = [
    (-4.0, -1.0, -6.0, -1.0),
    (-14.0, -8.0, 0.0, 0.1),
    (2.0, 5.0, -0.1, 0.0),
    (1.8, 2.0, 2.0, 4.0),
    (-1.0, 1.0, -6.0, -5.0),
    (4.0, 7.0, -6.0, -5.0),
    (5.0, 7.0, -6.0, -4.0),
]
MISSION_TIME_COLUMN = "charging_completion_time"


def read_csv(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"WARNING: cannot read {path}: {e}")
        return None


def find_experiments():
    out = []
    for config in CONFIGURATIONS:
        path = os.path.join(BASE_PATH, config)
        if not os.path.isdir(path):
            continue
        for name in sorted(os.listdir(path)):
            exp_path = os.path.join(path, name)
            if not os.path.isdir(exp_path):
                continue
            if name.startswith("exp_equal"):
                group = "equal"
            elif name.startswith("exp_reward"):
                group = "reward"
            else:
                continue
            out.append({
                "configuration": config,
                "group": group,
                "experiment": name,
                "path": exp_path,
            })
    return out


def mission_time(path, robot):
    df = read_csv(os.path.join(path, f"{robot}_mission_times.csv"))
    if df is None or MISSION_TIME_COLUMN not in df.columns:
        return np.nan
    values = pd.to_numeric(df[MISSION_TIME_COLUMN], errors="coerce").dropna()
    return float(values.max()) if len(values) else np.nan


def position(path, robot, limit=None):
    df = read_csv(os.path.join(path, f"{robot}_position.csv"))
    if df is None or not all(c in df.columns for c in ["time", "x", "y"]):
        return None

    df = df.copy()
    for c in ["time", "x", "y"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time", "x", "y"]).sort_values("time")

    if limit is not None and np.isfinite(limit):
        df = df[df.time <= limit]

    # Preserve the convention used in the previous analysis scripts.
    if len(df) > 2:
        df = df.iloc[2:].reset_index(drop=True)

    return df if len(df) >= 2 else None


def dt_intervals(t):
    t = np.asarray(t, dtype=float)
    if len(t) < 2:
        return np.zeros(len(t))
    return np.r_[np.maximum(np.diff(t), 0.0), 0.0]


def wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def yaw_data(path, robot, pos):
    if "yaw" in pos.columns:
        values = pd.to_numeric(pos["yaw"], errors="coerce").to_numpy()
        if np.isfinite(values).sum() >= 2:
            return values

    df = read_csv(os.path.join(path, f"{robot}_direction.csv"))
    if df is None or "time" not in df.columns:
        return None

    df = df.copy()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")

    column = next(
        (c for c in ["yaw", "theta", "angle", "direction"] if c in df.columns),
        None,
    )
    if column is None:
        return None

    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[column])
    if len(df) < 2:
        return None

    values = df[column].to_numpy(float)
    if np.nanmax(np.abs(values)) > 2.0 * np.pi + 0.5:
        values = np.deg2rad(values)

    return np.interp(
        pos["time"].to_numpy(float),
        df["time"].to_numpy(float),
        values,
    )


def waypoint_time(pos, waypoints):
    points = pos[["x", "y"]].to_numpy(float)
    times = pos["time"].to_numpy(float)
    dt = dt_intervals(times)

    distances = np.linalg.norm(
        points[:, None, :] - waypoints[None, :, :], axis=2
    )
    nearest = np.argmin(distances, axis=1)
    minimum_distance = distances[np.arange(len(distances)), nearest]

    result = np.zeros(len(waypoints))
    for i in range(len(waypoints)):
        mask = (nearest == i) & (minimum_distance <= WAYPOINT_RADIUS)
        result[i] = dt[mask].sum()
    return result



def waypoint_visits(pos, waypoints):
    """
    Count entries into each waypoint zone independently.

    A visit starts when the robot enters the radius (distance <= WAYPOINT_RADIUS).
    If the first valid sample is already inside, it counts as the initial visit.
    After leaving a zone, the robot must remain continuously outside it for at
    least WAYPOINT_REENTRY_DELAY_S before a subsequent entry can count as a new
    visit. Brief exits shorter than this delay remain part of the same visit.
    """
    if pos is None or pos.empty:
        return np.full(len(waypoints), np.nan)

    points = pos[["x", "y"]].to_numpy(dtype=float)
    times = pos["time"].to_numpy(dtype=float)
    distances = np.linalg.norm(
        points[:, None, :] - waypoints[None, :, :], axis=2
    )
    inside = distances <= WAYPOINT_RADIUS
    visits = np.zeros(len(waypoints), dtype=float)

    for wp_idx in range(len(waypoints)):
        in_zone = inside[:, wp_idx]
        if len(in_zone) == 0:
            continue

        # The first sample inside is counted as an initial visit.
        active_visit = bool(in_zone[0])
        if active_visit:
            visits[wp_idx] = 1.0

        outside_since = None
        for sample_idx in range(1, len(in_zone)):
            t = times[sample_idx]

            if in_zone[sample_idx]:
                if not active_visit:
                    # The first entry always starts the initial visit. Later
                    # entries count only after the required outside delay.
                    if visits[wp_idx] == 0.0:
                        visits[wp_idx] = 1.0
                    elif outside_since is not None and t - outside_since >= WAYPOINT_REENTRY_DELAY_S:
                        visits[wp_idx] += 1.0
                    active_visit = True
                # If the robot returned before the delay elapsed, it is still
                # considered the same visit.
                outside_since = None

            else:
                if active_visit:
                    active_visit = False
                    outside_since = t
                elif outside_since is None:
                    outside_since = t

    return visits


def heading_percent(pos, yaw, waypoints, mission_time_value):
    if yaw is None or not np.isfinite(mission_time_value) or mission_time_value <= 0:
        return np.full(len(waypoints), np.nan), np.nan

    points = pos[["x", "y"]].to_numpy(float)
    times = pos["time"].to_numpy(float)
    dt = dt_intervals(times)
    yaw = np.asarray(yaw)

    if len(yaw) != len(times):
        return np.full(len(waypoints), np.nan), np.nan

    distances = np.linalg.norm(
        points[:, None, :] - waypoints[None, :, :], axis=2
    )
    nearest = np.argmin(distances, axis=1)
    threshold = np.deg2rad(HEADING_THRESHOLD_DEG)

    waypoint_times = np.zeros(len(waypoints))
    valid_yaw = np.isfinite(yaw)

    for i, waypoint in enumerate(waypoints):
        mask = (nearest == i) & valid_yaw
        if not np.any(mask):
            continue

        desired = np.arctan2(
            (waypoint - points[mask])[:, 1],
            (waypoint - points[mask])[:, 0],
        )
        error = np.abs(wrap(desired - yaw[mask]))
        waypoint_times[i] = dt[mask][error <= threshold].sum()

    return 100.0 * waypoint_times / mission_time_value, 100.0 * waypoint_times.sum() / mission_time_value


def coverage(path, mission_times):
    world = box(XMIN, YMIN, XMAX, YMAX)
    excluded = unary_union([
        box(x0, y0, x1, y1)
        for x0, x1, y0, y1 in EMPTY_AREAS
    ])
    valid = world.difference(excluded)

    values = [v for v in mission_times.values() if np.isfinite(v)]
    if not values:
        return np.nan

    # Preserve the previous occupancy convention: use the maximum mission
    # time among the three robots as the common trajectory cutoff.
    limit = max(values)
    geometries = []

    for robot in ROBOTS:
        df = position(path, robot, limit)
        if df is None:
            continue
        geometries.append(
            LineString(df[["x", "y"]].to_numpy(float)).buffer(
                ROBOT_RADIUS, resolution=64
            )
        )

    if not geometries:
        return np.nan

    covered = unary_union(geometries).intersection(valid).area
    return 100.0 * covered / valid.area


def photosynthesis(path):
    df = read_csv(os.path.join(path, "photosynthesis_log.csv"))
    if df is None:
        return np.nan

    column = next(
        (c for c in ["photosynthesis", "photosynthesis_produced", "ph", "PH"] if c in df.columns),
        None,
    )

    if column is None:
        for c in df.columns:
            if c.lower() != "time" and pd.to_numeric(df[c], errors="coerce").notna().any():
                column = c
                break

    if column is None:
        return np.nan

    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(values.iloc[-1]) if len(values) else np.nan


def photosynthesis_timeseries(path):
    """Read cumulative photosynthesis production as a function of elapsed time."""
    df = read_csv(os.path.join(path, "photosynthesis_log.csv"))
    if df is None or df.empty or "time" not in df.columns:
        return None

    column = next(
        (c for c in ["photosynthesis", "photosynthesis_produced", "ph", "PH"]
         if c in df.columns),
        None,
    )
    if column is None:
        for c in df.columns:
            if c.lower() != "time" and pd.to_numeric(df[c], errors="coerce").notna().any():
                column = c
                break
    if column is None:
        return None

    data = pd.DataFrame({
        "time": pd.to_numeric(df["time"], errors="coerce"),
        "production": pd.to_numeric(df[column], errors="coerce"),
    }).dropna()
    data = data.sort_values("time")
    if data.empty:
        return None

    # Combine duplicate timestamps and express time relative to the first sample.
    data = data.groupby("time", as_index=False)["production"].mean()
    data["time"] -= data["time"].iloc[0]
    if len(data) < 2 or data["time"].max() <= 0:
        return None

    return data[["time", "production"]].reset_index(drop=True)


def plot_photosynthesis_over_time(experiments):
    """
    Plot mean cumulative photosynthesis over elapsed mission time for Equal and
    Reward, separately for each configuration. Curves use a shared time horizon
    within each configuration so that the two approaches are compared fairly.
    """
    for config in CONFIGURATIONS:
        series_by_group = {"equal": [], "reward": []}
        for info in experiments:
            if info["configuration"] != config:
                continue
            series = photosynthesis_timeseries(info["path"])
            if series is not None:
                series_by_group[info["group"]].append(series)

        available = series_by_group["equal"] + series_by_group["reward"]
        if not available or not series_by_group["equal"] or not series_by_group["reward"]:
            print(
                f"WARNING: cannot plot photosynthesis time series for {config}: "
                "missing valid Equal or Reward logs."
            )
            continue

        # Use the shortest experiment duration in this configuration to avoid
        # extrapolating either approach beyond observed data.
        common_duration = min(float(ts["time"].max()) for ts in available)
        if common_duration <= 0:
            continue
        time_grid = np.linspace(0.0, common_duration, 250)

        fig, ax = plt.subplots(figsize=(4.0, 3.0))
        csv_rows = []
        for group, label in [("equal", "Equal"), ("reward", "Reward")]:
            interpolated = []
            for ts in series_by_group[group]:
                t = ts["time"].to_numpy(dtype=float)
                y = ts["production"].to_numpy(dtype=float)
                valid = (t >= 0.0) & (t <= common_duration)
                if valid.sum() < 2:
                    continue
                interpolated.append(np.interp(time_grid, t[valid], y[valid]))

            if not interpolated:
                continue

            values = np.vstack(interpolated)
            mean_values = np.nanmean(values, axis=0)
            std_values = (
                np.nanstd(values, axis=0, ddof=1)
                if values.shape[0] > 1
                else np.zeros(values.shape[1])
            )
            ax.plot(time_grid, mean_values, label=f"{label} (n={values.shape[0]})")
            if values.shape[0] > 1:
                ax.fill_between(
                    time_grid,
                    np.maximum(mean_values - std_values, 0.0),
                    mean_values + std_values,
                    alpha=0.18,
                )

            for t, mean, std in zip(time_grid, mean_values, std_values):
                csv_rows.append({
                    "configuration": config,
                    "group": group,
                    "time_s": t,
                    "photosynthesis_mean": mean,
                    "photosynthesis_std": std,
                    "n_experiments": values.shape[0],
                })

        ax.set_xlabel("Elapsed time [s]")
        ax.set_ylabel("Mean photosynthesis produced")
        ax.set_title(f"Photosynthesis production over time — {config.replace('config', 'Config ')}")
        ax.grid(alpha=0.25)
        ax.legend()
        savefig(fig, f"photosynthesis_over_time_{config}_equal_reward")

        if csv_rows:
            csv_path = os.path.join(
                OUTPUT_DIR, f"photosynthesis_over_time_{config}_mean_std.csv"
            )
            pd.DataFrame(csv_rows).to_csv(csv_path, index=False)



def plot_photosynthesis_over_time_all_configurations(experiments):
    """Plot all six mean time-series: Equal and Reward for each configuration."""
    styles = {
        ("config1", "equal"): ("tab:blue", "-"),
        ("config1", "reward"): ("tab:blue", "--"),
        ("config2", "equal"): ("tab:orange", "-"),
        ("config2", "reward"): ("tab:orange", "--"),
        ("config3", "equal"): ("tab:green", "-"),
        ("config3", "reward"): ("tab:green", "--"),
    }
    series_by_pair = {(config, group): [] for config in CONFIGURATIONS for group in GROUPS}
    for info in experiments:
        series = photosynthesis_timeseries(info["path"])
        if series is not None:
            series_by_pair[(info["configuration"], info["group"])].append(series)

    valid_series = [ts for items in series_by_pair.values() for ts in items]
    if not valid_series:
        print("WARNING: cannot plot combined photosynthesis time series: no valid logs.")
        return

    # One shared horizon ensures the six curves are compared over the same time interval.
    common_duration = min(float(ts["time"].max()) for ts in valid_series)
    if common_duration <= 0:
        print("WARNING: cannot plot combined photosynthesis time series: invalid duration.")
        return
    time_grid = np.linspace(0.0, common_duration, 250)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    csv_rows = []

    for config in CONFIGURATIONS:
        for group in GROUPS:
            series_list = series_by_pair[(config, group)]
            if not series_list:
                print(f"WARNING: no valid photosynthesis logs for {config}/{group}.")
                continue
            interpolated = []
            for ts in series_list:
                t = ts["time"].to_numpy(dtype=float)
                y = ts["production"].to_numpy(dtype=float)
                valid = (t >= 0.0) & (t <= common_duration)
                if valid.sum() >= 2:
                    interpolated.append(np.interp(time_grid, t[valid], y[valid]))
            if not interpolated:
                continue

            values = np.vstack(interpolated)
            mean_values = np.nanmean(values, axis=0)
            std_values = (np.nanstd(values, axis=0, ddof=1)
                          if values.shape[0] > 1 else np.zeros(values.shape[1]))
            color, linestyle = styles[(config, group)]
            label = f"{config.replace('config', 'Config ')} — {group.capitalize()} (n={values.shape[0]})"
            ax.plot(time_grid, mean_values, color=color, linestyle=linestyle, label=label)
            if values.shape[0] > 1:
                ax.fill_between(time_grid,
                                np.maximum(mean_values - std_values, 0.0),
                                mean_values + std_values,
                                color=color, alpha=0.08)
            for t, mean, std in zip(time_grid, mean_values, std_values):
                csv_rows.append({
                    "configuration": config, "group": group, "time_s": t,
                    "photosynthesis_mean": mean, "photosynthesis_std": std,
                    "n_experiments": values.shape[0],
                })

    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel("Mean photosynthesis produced")
    ax.set_title("Mean photosynthesis production over time")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6, ncol=2)
    savefig(fig, "photosynthesis_over_time_all_configurations_equal_reward")
    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(
            os.path.join(OUTPUT_DIR, "photosynthesis_over_time_all_configurations_mean_std.csv"),
            index=False,
        )


def plot_waypoint_configurations_map():
    """Plot all waypoint layouts on the same environment map, colored by configuration."""
    colors = {"config1": "tab:blue", "config2": "tab:orange", "config3": "tab:green"}
    labels = {"config1": "Config 1", "config2": "Config 2", "config3": "Config 3"}
    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    # Environment bounds and excluded/empty regions used by the coverage metric.
    ax.add_patch(plt.Rectangle((XMIN, YMIN), XMAX - XMIN, YMAX - YMIN,
                               fill=False, edgecolor="black", linewidth=1.0,
                               label="Environment bounds"))
    for i, (x0, x1, y0, y1) in enumerate(EMPTY_AREAS):
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   facecolor="silver", edgecolor="grey",
                                   alpha=0.35, linewidth=0.5,
                                   label="Excluded area" if i == 0 else None))

    for config in CONFIGURATIONS:
        points = np.asarray(WAYPOINTS[config], dtype=float)
        ax.scatter(points[:, 0], points[:, 1], s=48, marker="o",
                   color=colors[config], edgecolors="black", linewidths=0.5,
                   label=labels[config], zorder=3)
        for wp_idx, (x, y) in enumerate(points, start=1):
            ax.annotate(f"WP{wp_idx}", (x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=6,
                        color=colors[config])

    ax.set_xlim(XMIN - 0.3, XMAX + 0.3)
    ax.set_ylim(YMIN - 0.3, YMAX + 0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Waypoint layouts for the three configurations")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=2, loc="best")
    savefig(fig, "waypoint_map_all_configurations")


def process(info):
    config = info["configuration"]
    path = info["path"]
    waypoints = np.asarray(WAYPOINTS[config], dtype=float)

    row = {
        "configuration": config,
        "group": info["group"],
        "experiment": info["experiment"],
    }

    mission_times = {}
    for robot in ROBOTS:
        mission_times[robot] = mission_time(path, robot)
        row[f"{robot}_mission_time_s"] = mission_times[robot]

    row["area_covered_percent"] = coverage(path, mission_times)

    for robot in ROBOTS:
        mt = mission_times[robot]
        pos = position(path, robot, mt)
        waypoint_times = np.full(5, np.nan)
        waypoint_visit_counts = np.full(5, np.nan)
        heading_values = np.full(5, np.nan)
        heading_total = np.nan

        if pos is not None:
            waypoint_times = waypoint_time(pos, waypoints)
            waypoint_visit_counts = waypoint_visits(pos, waypoints)
            heading_values, heading_total = heading_percent(
                pos, yaw_data(path, robot, pos), waypoints, mt
            )

        for i in range(5):
            row[f"{robot}_WP{i + 1}_time_s"] = waypoint_times[i]
            row[f"{robot}_WP{i + 1}_visits"] = waypoint_visit_counts[i]
            row[f"{robot}_WP{i + 1}_heading_percent"] = heading_values[i]

        total_waypoint_time = (
            np.nan if np.all(np.isnan(waypoint_times)) else np.nansum(waypoint_times)
        )
        row[f"{robot}_waypoint_time_total_s"] = total_waypoint_time
        row[f"{robot}_waypoint_time_mission_percent"] = (
            100.0 * total_waypoint_time / mt
            if np.isfinite(total_waypoint_time) and np.isfinite(mt) and mt > 0
            else np.nan
        )
        row[f"{robot}_heading_waypoint_total_percent"] = heading_total

    # Swarm-level metrics: combine the three robots within each experiment.
    # Time is robot-time (the sum of time spent by all robots near a WP).
    for i in range(1, 6):
        visit_values = [
            row.get(f"{robot}_WP{i}_visits", np.nan) for robot in ROBOTS
        ]
        time_values = [
            row.get(f"{robot}_WP{i}_time_s", np.nan) for robot in ROBOTS
        ]
        row[f"swarm_WP{i}_visits"] = (
            float(np.nansum(visit_values))
            if np.isfinite(np.asarray(visit_values, dtype=float)).any()
            else np.nan
        )
        row[f"swarm_WP{i}_time_s"] = (
            float(np.nansum(time_values))
            if np.isfinite(np.asarray(time_values, dtype=float)).any()
            else np.nan
        )

    swarm_time_values = [row[f"swarm_WP{i}_time_s"] for i in range(1, 6)]
    row["swarm_waypoint_time_total_s"] = (
        float(np.nansum(swarm_time_values))
        if np.isfinite(np.asarray(swarm_time_values, dtype=float)).any()
        else np.nan
    )

    row["photosynthesis_produced"] = photosynthesis(path)
    return row


def metric_columns(df):
    identifiers = {"configuration", "group", "experiment"}
    return [c for c in df.columns if c not in identifiers]


def aggregate_equal_reward(df):
    """Return means/std for every config-group pair plus All-group rows."""
    metrics = metric_columns(df)
    rows = []

    # Configuration-specific Equal/Reward means.
    for config in CONFIGURATIONS:
        for group in GROUPS:
            subset = df[(df["configuration"] == config) & (df["group"] == group)]
            row = {
                "scope": config,
                "configuration": config,
                "group": group,
                "n_experiments": len(subset),
            }
            for metric in metrics:
                values = pd.to_numeric(subset[metric], errors="coerce").dropna()
                row[f"{metric}_mean"] = values.mean() if len(values) else np.nan
                row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
            rows.append(row)

    # Overall Equal/Reward means across all three configurations.
    for group in GROUPS:
        subset = df[df["group"] == group]
        row = {
            "scope": "all_configurations",
            "configuration": "All",
            "group": group,
            "n_experiments": len(subset),
        }
        for metric in metrics:
            values = pd.to_numeric(subset[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = values.mean() if len(values) else np.nan
            row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def get_metric_values(aggregated, metric, scope):
    result = {}
    for group in GROUPS:
        subset = aggregated[
            (aggregated["scope"] == scope) & (aggregated["group"] == group)
        ]
        if subset.empty:
            result[group] = (np.nan, np.nan)
        else:
            result[group] = (
                float(subset[f"{metric}_mean"].iloc[0]),
                float(subset[f"{metric}_std"].iloc[0]),
            )
    return result


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, name + ".png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(OUTPUT_DIR, name + ".pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)


def grouped_equal_reward_plot(aggregated, metric, ylabel, filename, title=None, ylim=None):
    scopes = ["config1", "config2", "config3", "all_configurations"]
    labels = ["Config 1", "Config 2", "Config 3", "All"]
    x = np.arange(len(scopes))
    width = 0.34

    equal_means, equal_stds = [], []
    reward_means, reward_stds = [], []
    for scope in scopes:
        values = get_metric_values(aggregated, metric, scope)
        equal_means.append(values["equal"][0])
        equal_stds.append(values["equal"][1])
        reward_means.append(values["reward"][0])
        reward_stds.append(values["reward"][1])

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.bar(
        x - width / 2,
        equal_means,
        width,
        yerr=equal_stds,
        capsize=2.5,
        label="Equal",
    )
    ax.bar(
        x + width / 2,
        reward_means,
        width,
        yerr=reward_stds,
        capsize=2.5,
        label="Reward",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    savefig(fig, filename)


def plot_waypoint_times(aggregated, robot):
    """Create one figure for each configuration (plus All) showing WP1..WP5."""
    scopes = ["config1", "config2", "config3", "all_configurations"]
    labels = ["Config 1", "Config 2", "Config 3", "All"]
    metrics = [f"{robot}_WP{i}_time_s" for i in range(1, 6)]
    metric_labels = ["WP1", "WP2", "WP3", "WP4", "WP5"]
    for scope, scope_label in zip(scopes, labels):
        x = np.arange(5); width = 0.32
        em=[]; es=[]; rm=[]; rs=[]
        for metric in metrics:
            v=get_metric_values(aggregated,metric,scope); em.append(v["equal"][0]); es.append(v["equal"][1]); rm.append(v["reward"][0]); rs.append(v["reward"][1])
        fig,ax=plt.subplots(figsize=(4.0,3.0))
        ax.bar(x-width/2,em,width,yerr=es,capsize=2.5,label="Equal")
        ax.bar(x+width/2,rm,width,yerr=rs,capsize=2.5,hatch="//",label="Reward")
        ax.set_xticks(x); ax.set_xticklabels(metric_labels); ax.set_ylabel("Time near waypoint [s]")
        ax.set_title(f"{ROBOT_LABELS[robot]} — {scope_label}"); ax.grid(axis="y",alpha=0.25); ax.legend(ncol=2)
        savefig(fig,f"{robot}_time_near_waypoints_{scope}_equal_reward")


def plot_waypoint_time_sum(aggregated, robot):
    """Compare the sum of WP1..WP5 time across the three configurations and All."""
    scopes=["config1","config2","config3","all_configurations"]; labels=["Config 1","Config 2","Config 3","All"]
    metric=f"{robot}_waypoint_time_total_s"; x=np.arange(4); width=0.34
    em=[]; es=[]; rm=[]; rs=[]
    for scope in scopes:
        v=get_metric_values(aggregated,metric,scope); em.append(v["equal"][0]); es.append(v["equal"][1]); rm.append(v["reward"][0]); rs.append(v["reward"][1])
    fig,ax=plt.subplots(figsize=(4.0,3.0))
    ax.bar(x-width/2,em,width,yerr=es,capsize=2.5,label="Equal")
    ax.bar(x+width/2,rm,width,yerr=rs,capsize=2.5,hatch="//",label="Reward")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("Total time near WPs [s]")
    ax.set_title(ROBOT_LABELS[robot]); ax.grid(axis="y",alpha=0.25); ax.legend(ncol=2)
    savefig(fig,f"{robot}_waypoint_time_sum_equal_reward")

def plot_waypoint_visits(aggregated, robot):
    """Plot mean number of visits to WP1..WP5 for each configuration and All."""
    scopes = ["config1", "config2", "config3", "all_configurations"]
    labels = ["Config 1", "Config 2", "Config 3", "All"]
    metrics = [f"{robot}_WP{i}_visits" for i in range(1, 6)]
    metric_labels = ["WP1", "WP2", "WP3", "WP4", "WP5"]

    for scope, scope_label in zip(scopes, labels):
        x = np.arange(5)
        width = 0.32
        equal_means, equal_stds, reward_means, reward_stds = [], [], [], []
        for metric in metrics:
            values = get_metric_values(aggregated, metric, scope)
            equal_means.append(values["equal"][0])
            equal_stds.append(values["equal"][1])
            reward_means.append(values["reward"][0])
            reward_stds.append(values["reward"][1])

        fig, ax = plt.subplots(figsize=(4.0, 3.0))
        ax.bar(x - width / 2, equal_means, width, yerr=equal_stds,
               capsize=2.5, label="Equal")
        ax.bar(x + width / 2, reward_means, width, yerr=reward_stds,
               capsize=2.5, label="Reward", hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylabel("Mean number of visits")
        ax.set_title(f"{ROBOT_LABELS[robot]} — {scope_label}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=2)
        savefig(fig, f"{robot}_mean_number_of_visits_{scope}_equal_reward")


def plot_swarm_waypoint_metric(aggregated, metric_template, ylabel, filename_prefix):
    """Compare waypoint metrics aggregated across all three robots."""
    scopes = ["config1", "config2", "config3", "all_configurations"]
    labels = ["Config 1", "Config 2", "Config 3", "All"]
    metric_labels = ["WP1", "WP2", "WP3", "WP4", "WP5"]

    for scope, scope_label in zip(scopes, labels):
        x = np.arange(5)
        width = 0.32
        equal_means, equal_stds, reward_means, reward_stds = [], [], [], []
        for i in range(1, 6):
            values = get_metric_values(aggregated, metric_template.format(i), scope)
            equal_means.append(values["equal"][0])
            equal_stds.append(values["equal"][1])
            reward_means.append(values["reward"][0])
            reward_stds.append(values["reward"][1])

        fig, ax = plt.subplots(figsize=(4.0, 3.0))
        ax.bar(x - width / 2, equal_means, width, yerr=equal_stds,
               capsize=2.5, label="Equal")
        ax.bar(x + width / 2, reward_means, width, yerr=reward_stds,
               capsize=2.5, label="Reward", hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(f"All three robots — {scope_label}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=2)
        savefig(fig, f"{filename_prefix}_{scope}_equal_reward")


def plot_swarm_waypoint_time_sum(aggregated):
    """Compare total time spent near all waypoints, summed over the three robots."""
    scopes = ["config1", "config2", "config3", "all_configurations"]
    labels = ["Config 1", "Config 2", "Config 3", "All"]
    x = np.arange(len(scopes))
    width = 0.34
    equal_means, equal_stds, reward_means, reward_stds = [], [], [], []
    for scope in scopes:
        values = get_metric_values(aggregated, "swarm_waypoint_time_total_s", scope)
        equal_means.append(values["equal"][0])
        equal_stds.append(values["equal"][1])
        reward_means.append(values["reward"][0])
        reward_stds.append(values["reward"][1])

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.bar(x - width / 2, equal_means, width, yerr=equal_stds,
           capsize=2.5, label="Equal")
    ax.bar(x + width / 2, reward_means, width, yerr=reward_stds,
           capsize=2.5, label="Reward", hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Total robot-time near WPs [s]")
    ax.set_title("All three robots")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    savefig(fig, "swarm_waypoint_time_sum_equal_reward")


def plot_heading(aggregated, robot):
    """Create one figure for each configuration (plus All) showing WP1..WP5 heading %."""
    scopes=["config1","config2","config3","all_configurations"]; labels=["Config 1","Config 2","Config 3","All"]
    metrics=[f"{robot}_WP{i}_heading_percent" for i in range(1,6)]; metric_labels=["WP1","WP2","WP3","WP4","WP5"]
    for scope,scope_label in zip(scopes,labels):
        x=np.arange(5); width=0.32; em=[]; es=[]; rm=[]; rs=[]
        for metric in metrics:
            v=get_metric_values(aggregated,metric,scope); em.append(v["equal"][0]); es.append(v["equal"][1]); rm.append(v["reward"][0]); rs.append(v["reward"][1])
        fig,ax=plt.subplots(figsize=(4.0,3.0))
        ax.bar(x-width/2,em,width,yerr=es,capsize=2.5,label="Equal")
        ax.bar(x+width/2,rm,width,yerr=rs,capsize=2.5,hatch="//",label="Reward")
        ax.set_xticks(x); ax.set_xticklabels(metric_labels); ax.set_ylabel("Heading towards waypoint [%]"); ax.set_ylim(0,100)
        ax.set_title(f"{ROBOT_LABELS[robot]} — {scope_label}"); ax.grid(axis="y",alpha=0.25); ax.legend(ncol=2)
        savefig(fig,f"{robot}_heading_towards_waypoints_{scope}_equal_reward")

def plots(aggregated):
    # 1. Occupancy / area covered.
    grouped_equal_reward_plot(
        aggregated,
        "area_covered_percent",
        "Area covered [%]",
        "area_covered_percent_equal_reward",
        ylim=(0, 100),
    )

    # 2. Mission time for every robot.
    for robot in ROBOTS:
        grouped_equal_reward_plot(
            aggregated,
            f"{robot}_mission_time_s",
            "Mission time [s]",
            f"{robot}_mission_time_equal_reward",
            title=ROBOT_LABELS[robot],
        )

    # 3. Time near each waypoint + total, for every robot.
    for robot in ROBOTS:
        plot_waypoint_times(aggregated, robot)
        plot_waypoint_time_sum(aggregated, robot)
        plot_waypoint_visits(aggregated, robot)

    # 4. Swarm-level waypoint metrics: combine all three robots per experiment.
    # Visit counts are summed across robots; time is summed in robot-seconds.
    plot_swarm_waypoint_metric(
        aggregated,
        "swarm_WP{}_visits",
        "Mean number of visits (3 robots)",
        "swarm_mean_number_of_visits",
    )
    plot_swarm_waypoint_metric(
        aggregated,
        "swarm_WP{}_time_s",
        "Time near waypoint [robot-s]",
        "swarm_time_near_waypoints",
    )
    plot_swarm_waypoint_time_sum(aggregated)

    # 5. Heading towards each waypoint, separately for each configuration and All.
    for robot in ROBOTS:
        plot_heading(aggregated, robot)

    # 6. Total waypoint time / mission time, separately for all robots.
    scopes = ["config1", "config2", "config3", "all_configurations"]
    labels = ["Config 1", "Config 2", "Config 3", "All"]
    x = np.arange(len(scopes))
    width = 0.13

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for j, robot in enumerate(ROBOTS):
        equal_means, equal_stds = [], []
        reward_means, reward_stds = [], []
        metric = f"{robot}_waypoint_time_mission_percent"
        for scope in scopes:
            values = get_metric_values(aggregated, metric, scope)
            equal_means.append(values["equal"][0])
            equal_stds.append(values["equal"][1])
            reward_means.append(values["reward"][0])
            reward_stds.append(values["reward"][1])

        center = x + (j - 1) * width
        ax.bar(
            center - width * 0.22,
            equal_means,
            width * 0.44,
            yerr=equal_stds,
            capsize=2,
            label=f"{ROBOT_LABELS[robot]} Equal",
        )
        ax.bar(
            center + width * 0.22,
            reward_means,
            width * 0.44,
            yerr=reward_stds,
            capsize=2,
            hatch="//",
            label=f"{ROBOT_LABELS[robot]} Reward",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Time near WPs / mission time [%]")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    savefig(fig, "waypoint_time_over_mission_time_equal_reward")

    # 7. Photosynthesis produced.
    grouped_equal_reward_plot(
        aggregated,
        "photosynthesis_produced",
        "Photosynthesis produced",
        "photosynthesis_produced_equal_reward",
    )

    # 8. Mean cumulative photosynthesis production over time, separately and combined.
    experiments = find_experiments()
    plot_photosynthesis_over_time(experiments)
    plot_photosynthesis_over_time_all_configurations(experiments)

    # 9. Single map showing waypoint locations for all three configurations.
    plot_waypoint_configurations_map()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ral", action="store_true", help="Use the RAL plotting style.")
    parser.add_argument("--noshow", action="store_true", help="Do not call plt.show().")
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    experiments = find_experiments()
    rows = []
    failed = []

    print(f"Found {len(experiments)} experiments.")

    for info in experiments:
        try:
            rows.append(process(info))
        except Exception as e:
            failed.append({
                "configuration": info["configuration"],
                "group": info["group"],
                "experiment": info["experiment"],
                "error": str(e),
            })
            print(
                "ERROR",
                info["configuration"],
                info["group"],
                info["experiment"],
                e,
            )

    if not rows:
        raise RuntimeError("No experiments processed.")

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(OUTPUT_DIR, "all_experiment_metrics.csv"),
        index=False,
    )

    if failed:
        pd.DataFrame(failed).to_csv(
            os.path.join(OUTPUT_DIR, "failed_experiments.csv"),
            index=False,
        )

    aggregated = aggregate_equal_reward(df)
    aggregated.to_csv(
        os.path.join(OUTPUT_DIR, "configuration_group_means.csv"),
        index=False,
    )

    # Separate, easier-to-use tables for Equal and Reward.
    for group in GROUPS:
        aggregated[aggregated["group"] == group].to_csv(
            os.path.join(OUTPUT_DIR, f"{group}_means_by_configuration.csv"),
            index=False,
        )

    plots(aggregated)

    print(f"Processed {len(rows)} experiments.")
    print(f"Equal experiments:  {sum(df['group'] == 'equal')}")
    print(f"Reward experiments: {sum(df['group'] == 'reward')}")
    if failed:
        print(f"Failed experiments: {len(failed)}")
    print(f"Results saved to: {OUTPUT_DIR}")

    if not args.noshow:
        plt.show()


if __name__ == "__main__":
    main()