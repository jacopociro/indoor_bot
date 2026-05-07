import os
import re
import glob
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


WORLD_FILE = "/home/gonazza/container_ws/catkin_ws/src/indoor_bot/worlds/big_apartment.world"
CSV_FOLDER = "/home/gonazza/container_ws/catkin_ws/src/indoor_bot/indoor_bot/data"
GRID_RESOLUTION = 0.2  # meters
ROBOT_RADIUS = 2.0     # meters used to mark nearby cells as visited

# da sistemare gli ostacoli che sono messi male. probabilmente da fare a mano.

# --------------------------------------------------
# Parse world extents from .world file
# --------------------------------------------------
def extract_world_bounds(world_file):
    tree = ET.parse(world_file)
    root = tree.getroot()

    x_values = []
    y_values = []

    for model in root.iter("model"):
        name = model.attrib.get("name", "")

        # Ignora elementi globali
        if any(k in name.lower() for k in ["sun", "ground"]):
            continue

        pose_tag = model.find("pose")
        if pose_tag is None or pose_tag.text is None:
            continue

        try:
            pose_values = [float(v) for v in pose_tag.text.strip().split()]
            x = pose_values[0]
            y = pose_values[1]
            x_values.append(x)
            y_values.append(y)

            size_tag = model.find(".//size")
            if size_tag is not None and size_tag.text is not None:
                size_values = [float(v) for v in size_tag.text.strip().split()]
                sx = size_values[0]
                sy = size_values[1]

                x_values.extend([x - sx / 2.0, x + sx / 2.0])
                y_values.extend([y - sy / 2.0, y + sy / 2.0])
        except:
            continue

    if len(x_values) == 0 or len(y_values) == 0:
        raise ValueError("Impossibile estrarre i bounds dal file .world")

    xmin = min(x_values)
    xmax = max(x_values)
    ymin = min(y_values)
    ymax = max(y_values)

    margin = 2.0
    xmin -= margin
    xmax += margin
    ymin -= margin
    ymax += margin

    return xmin, xmax, ymin, ymax


# --------------------------------------------------
# Parse obstacle positions from world file
# --------------------------------------------------
def extract_obstacles(world_file):
    tree = ET.parse(world_file)
    root = tree.getroot()

    obstacles = []

    for model in root.iter("model"):
        name = model.attrib.get("name", "")

        # Ignora robot, sole e ground plane
        if any(k in name.lower() for k in ["robot", "rosbot", "ground", "sun"]):
            continue

        model_pose = [0.0] * 6
        pose_tag = model.find("pose")
        if pose_tag is not None and pose_tag.text is not None:
            try:
                model_pose = [float(v) for v in pose_tag.text.strip().split()]
            except:
                pass

        model_x = model_pose[0]
        model_y = model_pose[1]

        # Cerca collision box e visual box nei link
        for link in model.findall("link"):
            link_pose = [0.0] * 6
            link_pose_tag = link.find("pose")
            if link_pose_tag is not None and link_pose_tag.text is not None:
                try:
                    link_pose = [float(v) for v in link_pose_tag.text.strip().split()]
                except:
                    pass

            link_x = model_x + link_pose[0]
            link_y = model_y + link_pose[1]

            collision_elements = link.findall("collision") + link.findall("visual")

            for element in collision_elements:
                local_pose = [0.0] * 6
                local_pose_tag = element.find("pose")
                if local_pose_tag is not None and local_pose_tag.text is not None:
                    try:
                        local_pose = [float(v) for v in local_pose_tag.text.strip().split()]
                    except:
                        pass

                obstacle_x = link_x + local_pose[0]
                obstacle_y = link_y + local_pose[1]

                size_x = 1.0
                size_y = 1.0

                box_size_tag = element.find("geometry/box/size")
                if box_size_tag is not None and box_size_tag.text is not None:
                    try:
                        size_values = [float(v) for v in box_size_tag.text.strip().split()]
                        size_x = size_values[0]
                        size_y = size_values[1]
                    except:
                        pass
                else:
                    cylinder_radius_tag = element.find("geometry/cylinder/radius")
                    if cylinder_radius_tag is not None and cylinder_radius_tag.text is not None:
                        try:
                            radius = float(cylinder_radius_tag.text.strip())
                            size_x = radius * 2.0
                            size_y = radius * 2.0
                        except:
                            pass

                obstacles.append({
                    "x": obstacle_x,
                    "y": obstacle_y,
                    "sx": size_x,
                    "sy": size_y
                })

    return obstacles



# --------------------------------------------------
# Build occupancy grid
# --------------------------------------------------
def build_grid(xmin, xmax, ymin, ymax, resolution):
    width = int(np.ceil((xmax - xmin) / resolution))
    height = int(np.ceil((ymax - ymin) / resolution))

    visited = np.zeros((height, width), dtype=bool)
    obstacle = np.zeros((height, width), dtype=bool)

    return visited, obstacle


# --------------------------------------------------
# Convert world coordinates to grid indices
# --------------------------------------------------
def world_to_grid(x, y, xmin, ymin, resolution):
    gx = int((x - xmin) / resolution)
    gy = int((y - ymin) / resolution)
    return gx, gy


# --------------------------------------------------
# Mark obstacles in occupancy grid
# --------------------------------------------------
def mark_obstacles(obstacle_grid, obstacles, xmin, ymin, resolution):
    height, width = obstacle_grid.shape

    for obs in obstacles:
        x0 = obs["x"] - obs["sx"] / 2.0
        x1 = obs["x"] + obs["sx"] / 2.0
        y0 = obs["y"] - obs["sy"] / 2.0
        y1 = obs["y"] + obs["sy"] / 2.0

        gx0, gy0 = world_to_grid(x0, y0, xmin, ymin, resolution)
        gx1, gy1 = world_to_grid(x1, y1, xmin, ymin, resolution)

        gx0 = max(0, min(width - 1, gx0))
        gx1 = max(0, min(width - 1, gx1))
        gy0 = max(0, min(height - 1, gy0))
        gy1 = max(0, min(height - 1, gy1))

        obstacle_grid[gy0:gy1 + 1, gx0:gx1 + 1] = True


# --------------------------------------------------
# Mark visited cells from CSV trajectory
# --------------------------------------------------
def mark_visited_cells(visited_grid, csv_files, xmin, ymin, resolution, robot_radius):
    height, width = visited_grid.shape
    radius_cells = int(np.ceil(robot_radius / resolution))

    for csv_file in csv_files:
        print(f"Processing {csv_file}")
        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            x = row["x"]
            y = row["y"]

            gx, gy = world_to_grid(x, y, xmin, ymin, resolution)

            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    nx = gx + dx
                    ny = gy + dy

                    if 0 <= nx < width and 0 <= ny < height:
                        if dx * dx + dy * dy <= radius_cells * radius_cells:
                            visited_grid[ny, nx] = True


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    xmin, xmax, ymin, ymax = extract_world_bounds(WORLD_FILE)

    print(f"World bounds:")
    print(f"xmin={xmin:.2f}, xmax={xmax:.2f}")
    print(f"ymin={ymin:.2f}, ymax={ymax:.2f}")

    visited_grid, obstacle_grid = build_grid(
        xmin, xmax, ymin, ymax, GRID_RESOLUTION
    )

    obstacles = extract_obstacles(WORLD_FILE)
    mark_obstacles(
        obstacle_grid,
        obstacles,
        xmin,
        ymin,
        GRID_RESOLUTION
    )

    csv_files = glob.glob(os.path.join(CSV_FOLDER, "rosbot*_position.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError("Nessun CSV trovato nella cartella specificata")

    mark_visited_cells(
        visited_grid,
        csv_files,
        xmin,
        ymin,
        GRID_RESOLUTION,
        ROBOT_RADIUS
    )

    total_cells = visited_grid.size
    obstacle_cells = np.sum(obstacle_grid)
    real_cells = total_cells - obstacle_cells

    visited_real_cells = np.sum(np.logical_and(visited_grid, ~obstacle_grid))

    occupancy_percentage = (visited_real_cells / real_cells) * 100.0

    print("\n===== RESULTS =====")
    print(f"Total cells      : {total_cells}")
    print(f"Obstacle cells   : {obstacle_cells}")
    print(f"Real free cells  : {real_cells}")
    print(f"Visited cells    : {visited_real_cells}")
    print(f"Occupancy %      : {occupancy_percentage:.2f}%")

        # --------------------------------------------------
    # Plot occupancy grid
    # --------------------------------------------------
    display_grid = np.zeros_like(visited_grid, dtype=int)

    # 0 = free
    # 1 = obstacle
    # 2 = visited
    display_grid[obstacle_grid] = 1
    display_grid[np.logical_and(visited_grid, ~obstacle_grid)] = 2

    plt.figure(figsize=(12, 8))
    plt.imshow(
        display_grid,
        origin="lower",
        interpolation="nearest"
    )
    plt.title("Occupancy Grid")
    plt.xlabel("Grid X")
    plt.ylabel("Grid Y")
    plt.colorbar(label="0=free, 1=obstacle, 2=visited")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
