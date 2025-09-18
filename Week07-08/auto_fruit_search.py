# M4 - Autonomous fruit searching (with semantic occupancy + 0.05 m inflation)

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time
import math
from math import atan2, hypot
import random
from TargetPoseEst import estimate_pose, merge_estimations
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation


# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "util")
from util.pibot import PenguinPi
import util.measure as measure
from YOLO.detector import Detector

# =========================
# Semantic occupancy (in-file)
# =========================

class Pose2D:
    def __init__(self, x: float, y: float, yaw: float = 0.0):
        self.x = float(x); self.y = float(y); self.yaw = float(yaw)

class OccupancyGridMap:
    """
    Semantic log-odds occupancy grid with freespace carving and inflation.

    Layers:
      - obstacles (distractions + ArUco + any non-target in L3/L4)
      - targets
    We inflate OBSTACLES ONLY for planning. Inflation radius fixed to 0.05 m, per request.
    """
    def __init__(self,
                 width_m: float,
                 height_m: float,
                 resolution: float = 0.05,
                 origin_x: float = -1.2,
                 origin_y: float = -1.2,
                 p_occ: float = 0.70,
                 p_free: float = 0.30,
                 max_logodds: float = 5.0,
                 inflation_radius_m: float = 0.12):  # <<< fixed 0.05 m
        self.res = resolution
        self.W = int(np.ceil(width_m / resolution))
        self.H = int(np.ceil(height_m / resolution))
        self.ox = origin_x
        self.oy = origin_y

        self.inflation_radius_m = float(inflation_radius_m)

        # log-odds helpers
        def l(p): return np.log(p/(1-p))
        self.l_occ = l(p_occ)
        self.l_free = l(p_free)
        self.max_l = max_logodds

        # layers (log-odds; 0 => unknown ~ 0.5 prob)
        self.L_obstacles = np.zeros((self.H, self.W), dtype=np.float32)
        self.L_targets   = np.zeros((self.H, self.W), dtype=np.float32)

        # cached inflated mask (binary)
        self.inflated_obstacles = None

    # --- coordinates ---
    def world_to_grid(self, x: float, y: float):
        gx = int(np.floor((x - self.ox) / self.res))
        gy = int(np.floor((y - self.oy) / self.res))
        return gx, gy

    def grid_in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.W and 0 <= gy < self.H

    # --- low-level update ---
    def _update_disc(self, L: np.ndarray, gx: int, gy: int, incr: float):
        if not self.grid_in_bounds(gx, gy): return
        L[gy, gx] = np.clip(L[gy, gx] + incr, -self.max_l, self.max_l)

    # --- freespace carving (Bresenham) ---
    def inflate_obstacles(self):
        """
        Inflate ONLY the obstacle layer (distractions, ArUco markers, etc.)
        Do NOT inflate targets - we want to be able to reach them!
        """
        occ = (self.L_obstacles > 0)  # Only obstacles, NOT targets
        if not np.any(occ):
            self.inflated_obstacles = occ.astype(np.uint8)
            return
        
        r_cells = max(1, int(np.ceil(self.inflation_radius_m / self.res)))
        out = np.zeros_like(occ, dtype=np.uint8)
        
        for j in range(occ.shape[0]):
            for i in range(occ.shape[1]):
                if occ[j, i]:  # if this cell is occupied BY AN OBSTACLE
                    # Inflate in a circular pattern
                    for dj in range(-r_cells, r_cells + 1):
                        for di in range(-r_cells, r_cells + 1):
                            if di*di + dj*dj <= r_cells*r_cells:  # circular check
                                ni, nj = i + di, j + dj
                                if 0 <= ni < occ.shape[1] and 0 <= nj < occ.shape[0]:
                                    out[nj, ni] = 1
        
        self.inflated_obstacles = out

    def get_obstacle_circles(self):
        """
        Export only INFLATED OBSTACLES (not targets) as collision circles for planning.
        Targets should be reachable goals, not obstacles to avoid.
        """
        if self.inflated_obstacles is None:
            self.inflate_obstacles()
        ys, xs = np.where(self.inflated_obstacles > 0)
        circles = []
        r = self.inflation_radius_m
        for gy, gx in zip(ys, xs):
            x = self.ox + (gx + 0.5) * self.res
            # Fix: flip the y-coordinate since array row 0 is at top, but world y increases upward
            y = self.oy + (self.H - gy - 0.5) * self.res  # Flip y-axis
            circles.append((x, y, r))
        return circles

    # --- object insertion (disc) ---
    def add_object(self, x: float, y: float, radius_m: float, is_target: bool):
        L = self.L_targets if is_target else self.L_obstacles
        gx, gy = self.world_to_grid(x, y)
        rad = int(np.ceil(radius_m / self.res))
        x0 = max(gx - rad, 0); x1 = min(gx + rad, self.W - 1)
        y0 = max(gy - rad, 0); y1 = min(gy + rad, self.H - 1)
        r2 = radius_m * radius_m
        for j in range(y0, y1 + 1):
            for i in range(x0, x1 + 1):
                dx = (i - gx) * self.res
                dy = (j - gy) * self.res
                if dx*dx + dy*dy <= r2:
                    self._update_disc(L, i, j, self.l_occ)

    
    def carve_freespace(self, pose: Pose2D, hit_x: float, hit_y: float, layer: str = "obstacles"):
        L = self.L_obstacles if layer == "obstacles" else self.L_targets
        sx, sy = self.world_to_grid(pose.x, pose.y)
        ex, ey = self.world_to_grid(hit_x, hit_y)

        dx = abs(ex - sx); dy = abs(ey - sy)
        xstep = 1 if sx < ex else -1
        ystep = 1 if sy < ey else -1
        err = dx - dy
        x0, y0 = sx, sy

        while True:
            self._update_disc(L, x0, y0, self.l_free)  # free evidence
            if x0 == ex and y0 == ey: break
            e2 = 2 * err
            if e2 > -dy: err -= dy; x0 += xstep
            if e2 <  dx: err += dx; y0 += ystep


class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.cost = 0.0

# =========================
# FruitSearch
# =========================

class FruitSearch:
    def __init__(self, ppi):
        """
        Autonomous fruit searching module for PenguinPi robot.

        ppi : PenguinPi instance (already connected)
        """
        self.ppi = ppi

        self.scale = np.loadtxt("calibration/param/scale.txt", delimiter=',')
        self.baseline = np.loadtxt("calibration/param/baseline.txt", delimiter=',')
        self.camera_matrix = np.loadtxt("calibration/param/intrinsic.txt", delimiter=',')
        self.dist_coeffs = np.loadtxt("calibration/param/distCoeffs.txt", delimiter=',')
        self.robot = Robot(self.baseline, self.scale, self.camera_matrix, self.dist_coeffs)
        self.ekf = EKF(self.robot)

        model_path = "YOLO\model\yolov8_model.pt"
        self.detector = Detector(model_path)

        self.pose = np.array([0.0, 0.0, 0.0])  # x, y, theta (rad)
        self.true_map = None
        self.targets = []      # list of np.array([x,y])
        self.obstacles = []    # list of np.array([x,y])
        self.path = []

        # --- arena/map config ---
        self.arena_size = (2.4, 2.4)  # width, height [m]
        ax, ay = self.arena_size
        # Grid origin centered so +/- coordinates map naturally
        self.map = OccupancyGridMap(width_m=ax, height_m=ay, resolution=0.05,
                                    origin_x=-ax/2, origin_y=-ay/2,
                                    inflation_radius_m=0.05)  # <<< fixed to 0.05 m
        self.level_number = None
        self.search_list = []

    ###################### Helper functions ############################
    def wrap_angle(self, angle):
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def get_robot_pose(self):
        return self.ekf.robot.state.flatten()

    def read_true_map(self,fname):
        """Read the ground truth map and output the pose of the ArUco markers and 5 target fruits&vegs to search for

        @param fname: filename of the map
        @return:
            1) list of targets, e.g. ['lemon', 'tomato', 'garlic']
            2) locations of the targets, [[x1, y1], ..... [xn, yn]]
            3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
        """
        with open(fname, 'r') as fd:
            gt_dict = json.load(fd)
            fruit_list = []
            fruit_true_pos = []
            aruco_true_pos = np.empty([10, 2])

            # remove unique id of targets of the same type
            for key in gt_dict:
                x = np.round(gt_dict[key]['x'], 1)
                y = np.round(gt_dict[key]['y'], 1)

                if key.startswith('aruco'):
                    if key.startswith('aruco10'):
                        aruco_true_pos[9][0] = x
                        aruco_true_pos[9][1] = y
                    else:
                        marker_id = int(key[5]) - 1
                        aruco_true_pos[marker_id][0] = x
                        aruco_true_pos[marker_id][1] = y
                else:
                    fruit_list.append(key[:-2])
                    if len(fruit_true_pos) == 0:
                        fruit_true_pos = np.array([[x, y]])
                    else:
                        fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

            return fruit_list, fruit_true_pos, aruco_true_pos

    def read_search_list(self,shopping_list = None):
        """Read the search order of the target fruits"""
        search_list = []
        with open(shopping_list, 'r') as fd:
            fruits = fd.readlines()
            for fruit in fruits:
                search_list.append(fruit.strip())
        return search_list

    def print_target_fruits_pos(self,search_list, fruit_list, fruit_true_pos):
        """Print out the target fruits' pos in the search order"""
        print("Search order:")
        n_fruit = 1
        if len(fruit_list)==0:
            for fruit in search_list:
                print('{}) {}'.format(n_fruit,fruit))
                n_fruit += 1
        else:
            for fruit in search_list:
                for i in range(len(fruit_list)): # there are 5 targets amongst 10 objects
                    if fruit == fruit_list[i]:
                        print('{}) {} at [{}, {}]'.format(n_fruit, fruit, np.round(fruit_true_pos[i][0], 1), np.round(fruit_true_pos[i][1], 1)))
                n_fruit+=1

    def dist(self,a, b):
        return hypot(a.x - b.x, a.y - b.y)

    def line_collision(self, p1, p2, obstacles, robot_radius=0.09, step=0.02):
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        dist_total = hypot(dx, dy)
        steps = max(2, int(dist_total / step))
        for i in range(steps+1):
            x = p1.x + dx*i/steps
            y = p1.y + dy*i/steps
            for obs in obstacles:
                ox, oy, r = obs
                if hypot(x-ox, y-oy) <= r + robot_radius:
                    return True
        return False

    def nearest_node(self, tree, sample):
        return min(tree, key=lambda node: hypot(node.x - sample[0], node.y - sample[1]))

    def rrt_star_sequential_targets(self, start, targets, obstacles, x_limits, y_limits,
                                max_iter=5000, step_size=0.15, goal_radius=0.15,
                                rewire_radius=0.4, robot_radius=0.09, goal_bias=0.15):
        """
        Alternative implementation: RRT* that explicitly handles sequential targets.
        This is a cleaner separation if you prefer to keep the multi-target logic separate.
        """
        if not targets:
            return None
            
        tree = [start]
        current_target_idx = 0
        
        for iteration in range(max_iter):
            if current_target_idx >= len(targets):
                # Success! Build path through all targets
                final_target = Node(targets[-1][0], targets[-1][1])
                closest = min(tree, key=lambda n: hypot(n.x - final_target.x, n.y - final_target.y))
                if hypot(closest.x - final_target.x, closest.y - final_target.y) <= goal_radius:
                    final_target.parent = closest
                    path = []
                    cur = final_target
                    while cur:
                        path.append([cur.x, cur.y])
                        cur = cur.parent
                    path.reverse()
                    return path
                break
                
            current_target = targets[current_target_idx]
            goal = Node(current_target[0], current_target[1])
            
            # Standard RRT* expansion
            if random.random() < goal_bias:
                sx, sy = goal.x, goal.y
            else:
                sx = random.uniform(*x_limits)
                sy = random.uniform(*y_limits)

            nearest = min(tree, key=lambda node: hypot(node.x - sx, node.y - sy))
            theta = atan2(sy - nearest.y, sx - nearest.x)
            new_x = nearest.x + step_size * np.cos(theta)
            new_y = nearest.y + step_size * np.sin(theta)
            new_node = Node(new_x, new_y, parent=nearest)
            new_node.cost = nearest.cost + step_size

            if self.line_collision(nearest, new_node, obstacles, robot_radius):
                continue

            # RRT* rewiring
            for node in tree:
                if node == nearest:
                    continue
                if hypot(node.x - new_node.x, node.y - new_node.y) <= rewire_radius:
                    if not self.line_collision(node, new_node, obstacles, robot_radius):
                        new_cost = node.cost + hypot(node.x - new_node.x, node.y - new_node.y)
                        if new_cost < new_node.cost:
                            new_node.parent = node
                            new_node.cost = new_cost

            tree.append(new_node)

            # Check if reached current target
            if hypot(new_node.x - goal.x, new_node.y - goal.y) <= goal_radius:
                current_target_idx += 1

    def return_to_start(self):
        """Drive straight back to (0,0)."""
        self.drive_to_point([0.0, 0.0])
    
    def get_target_aware_obstacles(self, current_target_coords):
        """
        Get obstacle circles but mark which ones might be targets we want to reach.
        Returns regular obstacle list but can be used with smart collision detection.
        """
        all_obstacles = self.map.get_obstacle_circles()
        
        # Separate obstacles that are near our target coordinates
        filtered_obstacles = []
        target_tolerance = 0.15  # 15cm tolerance
        
        for obs in all_obstacles:
            ox, oy, r = obs
            is_target_obstacle = False
            
            # Check if this obstacle is near any of our target coordinates
            if current_target_coords is not None:
                target_dist = hypot(ox - current_target_coords[0], oy - current_target_coords[1])
                if target_dist <= target_tolerance:
                    is_target_obstacle = True
            
            if not is_target_obstacle:
                filtered_obstacles.append(obs)
        
        return filtered_obstacles
    
    def line_collision_smart(self, p1, p2, obstacles, robot_radius=0.09, step=0.02, target_goal=None, tolerance=0.15):
        """
        Enhanced collision detection that ignores collisions with target locations when navigating toward them.
        
        Args:
            p1, p2: Start and end nodes
            obstacles: List of (x, y, r) obstacle circles
            robot_radius: Robot clearance radius
            step: Step size for line checking
            target_goal: Current target coordinates [x, y] or None
            tolerance: Distance within which to ignore target collisions
        """
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        dist_total = hypot(dx, dy)
        steps = max(2, int(dist_total / step))
        
        for i in range(steps + 1):
            x = p1.x + dx * i / steps
            y = p1.y + dy * i / steps
            
            for obs in obstacles:
                ox, oy, r = obs
                collision_dist = hypot(x - ox, y - oy)
                
                if collision_dist <= r + robot_radius:
                    # Check if this collision is with a target we're trying to reach
                    if target_goal is not None:
                        target_dist = hypot(ox - target_goal[0], oy - target_goal[1])
                        if target_dist <= tolerance:
                            # This obstacle is likely the target we're navigating to - ignore collision
                            continue
                    
                    # Real collision with non-target obstacle
                    return True
        return False


    ####################### Exploration Path ############################
    def generate_exploration_path(self, arena_size=(2.4, 2.4), step=0.6):
        """Generate a simple 'lawnmower' exploration path to cover the arena."""
        waypoints = []
        x_max, y_max = arena_size
        xs = np.arange(-x_max/2, x_max/2 + 1e-6, step)
        ys = np.arange(-y_max/2, y_max/2 + 1e-6, step)

        toggle = 1
        for y in ys:
            if toggle > 0:
                for x in xs:
                    waypoints.append([x, y])
            else:
                for x in reversed(xs):
                    waypoints.append([x, y])
            toggle *= -1
        return waypoints

    def detect_fruits(self):
        """Capture image and run YOLO detection."""
        img = self.ppi.get_image()
        if img is None:
            return [], None
        bboxes, img_out = self.detector.detect_single_image(img)
        return bboxes, img_out

    def estimate_fruit_position(self, detection):
        """
        Estimate fruit position using TargetPoseEst. 
        detection: tuple (class_label, [x, y, w, h], confidence)
        """
        robot_pose = self.get_robot_pose()  # EKF pose [x,y,theta]
        # Keep full detection tuple so your pose estimator can use class/conf if needed
        return estimate_pose(self.camera_matrix, detection, robot_pose)

    ##################### Occupancy Map ############################
    def create_occupancy_map(self,
                         level_number: int,
                         fruit_list: list,
                         fruit_true_pos: np.ndarray,
                         aruco_true_pos: np.ndarray,
                         search_list_path: str = "M3_prac_shopping_list.txt",
                         fruit_radius_m: float = 0.08,
                         marker_radius_m: float = 0.10):
        """
        Build/refresh the occupancy according to the level.
        - Level 2: use GT map. Targets → target layer. Distractions + ArUco → obstacles layer.
        - Level 3: GIVEN the 5 targets from GT; seed targets only. No distractions.
        - Level 4: start empty; exploration fills obstacles only.
        All obstacles are inflated to exactly 0.05 m radius for planning.
        """
        self.level_number = level_number
        self.search_list = self.read_search_list(search_list_path)

        # Reset map
        ax, ay = self.arena_size
        self.map = OccupancyGridMap(width_m=ax, height_m=ay, resolution=0.05,
                                    origin_x=-ax/2, origin_y=-ay/2,
                                    inflation_radius_m=0.05)

        self.targets = []
        self.obstacles = []

        if level_number == 2:
            # Split GT fruits into targets vs distractions by name
            targets_xy = []
            distractions_xy = []
            for i, name in enumerate(fruit_list):
                pos = fruit_true_pos[i]  # [x,y]
                if name in self.search_list:
                    targets_xy.append(pos)
                else:
                    distractions_xy.append(pos)

            # Add targets in **shopping list order**
            ordered_targets = []
            for fruit_name in self.search_list:
                for i, name in enumerate(fruit_list):
                    if name == fruit_name:
                        x, y = float(fruit_true_pos[i, 0]), float(fruit_true_pos[i, 1])
                        self.map.add_object(x, y, fruit_radius_m, is_target=True)
                        self.targets.append(np.array([x, y]))
                        ordered_targets.append(np.array([x, y]))
            self.targets = ordered_targets

            # Distractions + ArUco as obstacles
            for pos in distractions_xy:
                x, y = float(pos[0]), float(pos[1])
                self.map.add_object(x, y, fruit_radius_m, is_target=False)
                self.obstacles.append(np.array([x, y]))
            for k in range(aruco_true_pos.shape[0]):
                x, y = float(aruco_true_pos[k, 0]), float(aruco_true_pos[k, 1])
                self.map.add_object(x, y, marker_radius_m, is_target=False)
                self.obstacles.append(np.array([x, y]))

            self.map.inflate_obstacles()

        elif level_number == 3:
            # Seed targets from GT in shopping list order
            for fruit_name in self.search_list:
                for i, name in enumerate(fruit_list):
                    if name == fruit_name:
                        x, y = float(fruit_true_pos[i, 0]), float(fruit_true_pos[i, 1])
                        self.map.add_object(x, y, fruit_radius_m, is_target=True)
                        self.targets.append(np.array([x, y]))
            # Do not add distractions or ArUco; exploration will handle obstacles
            self.map.inflate_obstacles()

        elif level_number == 4:
            # Start empty; exploration will populate obstacles and targets
            self.map.inflate_obstacles()

        else:
            print(f"[WARN] Unknown level {level_number}; leaving empty map.")



        ##################### Path Planning ############################

    def plan_path(self):
        """
        Plan path to all current targets using RRT* with smart collision detection.
        Visits targets sequentially and concatenates paths properly.
        """
        if not self.targets:
            print("[INFO] No targets to plan for.")
            self.path = []
            return

        # Map extents
        x_limits = (self.map.ox, self.map.ox + self.map.W * self.map.res)
        y_limits = (self.map.oy, self.map.oy + self.map.H * self.map.res)
        robot_radius = 0.09
        map_obstacles = self.map.get_obstacle_circles()

        print(f"[DEBUG] Robot starting position: {[self.pose[0], self.pose[1]]}")
        print(f"[DEBUG] Number of targets: {len(self.targets)}")
        print(f"[DEBUG] First target: {self.targets[0]}")
        print(f"[DEBUG] Total obstacle circles: {len(map_obstacles)}")

        # RRT* parameters
        max_iter = 8000
        step_size = 0.025
        goal_radius = 0.25
        rewire_radius = 0.3
        goal_bias = 0.3

        tree = [Node(self.pose[0], self.pose[1])]
        successful_expansions = 0
        collision_count = 0

        current_target_idx = 0
        while current_target_idx < len(self.targets) and max_iter > 0:
            current_target = self.targets[current_target_idx]
            goal = Node(float(current_target[0]), float(current_target[1]))

            # Sample point with goal bias
            if random.random() < goal_bias:
                sx, sy = goal.x, goal.y
            else:
                sx = random.uniform(*x_limits)
                sy = random.uniform(*y_limits)

            # Find nearest node
            nearest = self.nearest_node(tree, (sx, sy))

            # Extend toward sample
            theta = atan2(sy - nearest.y, sx - nearest.x)
            new_x = nearest.x + step_size * np.cos(theta)
            new_y = nearest.y + step_size * np.sin(theta)
            new_node = Node(new_x, new_y, parent=nearest)
            new_node.cost = nearest.cost + step_size

            # Smart collision check
            if self.line_collision_smart(nearest, new_node, map_obstacles, robot_radius,
                                        target_goal=[current_target[0], current_target[1]], tolerance=0.2):
                collision_count += 1
                max_iter -= 1
                continue

            successful_expansions += 1

            # RRT* rewiring
            for node in tree:
                if node == nearest:
                    continue
                if self.dist(node, new_node) <= rewire_radius:
                    if not self.line_collision_smart(node, new_node, map_obstacles, robot_radius,
                                                    target_goal=[current_target[0], current_target[1]], tolerance=0.2):
                        new_cost = node.cost + self.dist(node, new_node)
                        if new_cost < new_node.cost:
                            new_node.parent = node
                            new_node.cost = new_cost

            tree.append(new_node)

            # Check if target reached
            if self.dist(new_node, goal) <= goal_radius:
                print(f"[SUCCESS] Reached target {current_target_idx + 1}/{len(self.targets)} at {current_target}")
                current_target_idx += 1

            if successful_expansions % 500 == 0:
                print(f"[DEBUG] Iterations remaining: {max_iter}, target index: {current_target_idx}, tree size: {len(tree)}")
            max_iter -= 1

        # Build full path sequentially through all targets
        full_path = [[self.pose[0], self.pose[1]]]
        last_node = Node(self.pose[0], self.pose[1])

        for target in self.targets:
            goal_node = Node(float(target[0]), float(target[1]))
            # Find closest node to previous target/starting point
            closest = min(tree, key=lambda n: self.dist(n, last_node))
            
            # Trace path from closest to goal
            segment = []
            cur = goal_node
            while cur is not None and cur != closest:
                segment.append([cur.x, cur.y])
                cur = cur.parent
            segment.append([closest.x, closest.y])
            segment.reverse()
            
            # Avoid duplicates at connection points
            if full_path[-1] == segment[0]:
                segment = segment[1:]
            
            full_path.extend(segment)
            last_node = goal_node

        self.path = full_path
        print(f"[INFO] Full path with {len(self.path)} waypoints ready")
        print(self.path)
        self.map_path()

        ###################### Path Following ############################
    def drive_to_point(self, waypoint):
        """
        Drive the robot to a specific waypoint by turning first and then driving straight.
        Updates self.pose after each movement.

        Args:
            waypoint : [x, y] target coordinates
        """
        x, y, theta = self.pose
        dx = waypoint[0] - x
        dy = waypoint[1] - y

        # --- Compute heading and distance ---
        desired_theta = math.atan2(dy, dx)
        heading_error = self.wrap_angle(desired_theta - theta)
        distance = math.hypot(dx, dy)

        if distance < 1e-3:
            print(f"[INFO] Already at waypoint [{waypoint[0]:.3f}, {waypoint[1]:.3f}]")
            return

        # --- Calibration parameters ---
        scale = self.scale
        baseline = self.baseline
        wheel_vel = 30  # ticks/sec

        # --- Turn on the spot ---
        turn_rate = 2 * wheel_vel * scale / baseline  # rad/s
        turn_time = abs(heading_error) / turn_rate
        turn_direction = 1 if heading_error > 0 else 0
        print(f"[INFO] Turning {heading_error:.3f} rad for {turn_time:.2f} s")
        print(heading_error)
        self.ppi.set_velocity([0, turn_direction], turning_tick=wheel_vel, time=turn_time)

        # --- Update heading after turn ---
        theta = self.wrap_angle(theta + heading_error)

        # --- Drive straight ---
        drive_speed = wheel_vel * scale  # m/s approx
        drive_time = distance / drive_speed
        print(f"[INFO] Driving straight {distance:.3f} m for {drive_time:.2f} s")
        self.ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)

        # --- Update position after drive ---
        x = waypoint[0]
        y = waypoint[1]

        # --- Save updated pose ---
        self.pose = np.array([x, y, theta])
        print(f"[INFO] Arrived at waypoint [{x:.3f}, {y:.3f}], heading {theta:.3f} rad")




    def follow_path(self, dt=0.1):
        """
        Follow the planned path using drive_to_point for each waypoint.
        Updates self.pose after each waypoint.
        """
        for waypoint in self.path:
            # Drive to the waypoint
            self.drive_to_point(waypoint)


    # ====== Helper to process detections during exploration ======
    def process_detections_update_map(self, detections):
        """
        For levels 3 and 4, update the semantic map online from YOLO detections.
        Obstacles are inflated to 0.05 m (via map.inflate) for planning.

        Level 3:
        - Targets are already seeded from GT; do NOT add occupied cells for targets.
        - Carve freespace toward target detections; add non-targets as obstacles.

        Level 4 (updated):
        - We are NOT given target positions; we must discover them.
        - If detection class is in the shopping list => add as a TARGET (occupied in target layer)
        - Else => add as an OBSTACLE (occupied in obstacle layer)
        """
        if self.level_number not in (3, 4):
            return

        rpose = Pose2D(*self.get_robot_pose())

        for det in detections:
            cls, xywh, conf = det
            pos = self.estimate_fruit_position(det)
            if pos is None:
                continue
            px, py = float(pos[0]), float(pos[1])

            if self.level_number == 3:
                if cls in self.search_list:
                    # Already seeded from GT — don't block targets; just clear the ray.
                    self.map.carve_freespace(rpose, px, py, layer="targets")
                else:
                    # Non-target → obstacle
                    self.map.carve_freespace(rpose, px, py, layer="obstacles")
                    self.map.add_object(px, py, 0.10, is_target=False)
                    self.obstacles.append(np.array([px, py]))

            else:  # level 4 (UPDATED LOGIC)
                if cls in self.search_list:
                    # Detected a target class → add as TARGET and carve freespace on target layer
                    self.map.carve_freespace(rpose, px, py, layer="targets")
                    self.map.add_object(px, py, 0.08, is_target=True)  # geometric size; planning inflation is on obstacles only
                    self.targets.append(np.array([px, py]))
                else:
                    # Unknown / distraction / marker → obstacle
                    self.map.carve_freespace(rpose, px, py, layer="obstacles")
                    self.map.add_object(px, py, 0.10, is_target=False)
                    self.obstacles.append(np.array([px, py]))

        # Keep inflated mask fresh; inflation radius is fixed at 0.05 m (obstacles only).
        self.map.inflate_obstacles()


        ###################### Path Visualization / Mapping ############################
    def map_path(self):
        """
        Save the semantic occupancy map and planned path to a PNG file.
        - ArUcos shown in blue squares
        - Targets shown in green dots
        - Distractions/obstacles shown in red dots
        - Planned path shown as a black line with points
        """

        if not self.path:
            print("[INFO] No path to save.")
            return

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title("Semantic Map + Planned Path")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")

        ax.set_xlim([self.map.ox, self.map.ox + self.map.W * self.map.res])
        ax.set_ylim([self.map.oy, self.map.oy + self.map.H * self.map.res])
        ax.set_aspect('equal')
        ax.grid(True)

        # --- plot OBSTACLES (distractions) in red ---
        if np.any(self.map.L_obstacles > 0):
            ys, xs = np.where(self.map.L_obstacles > 0)
            ox = self.map.ox + (xs + 0.5) * self.map.res
            oy = self.map.oy + (ys + 0.5) * self.map.res
            ax.scatter(ox, oy, c="red", s=15, label="Obstacles/Distractions")

        # --- plot TARGETS in green ---
        if np.any(self.map.L_targets > 0):
            ys, xs = np.where(self.map.L_targets > 0)
            tx = self.map.ox + (xs + 0.5) * self.map.res
            ty = self.map.oy + (ys + 0.5) * self.map.res
            ax.scatter(tx, ty, c="green", s=15, label="Targets")

        # --- plot ARUCO markers in blue ---
        if hasattr(self, "aruco_true_pos") and self.aruco_true_pos is not None:
            for (axu, ayu) in self.aruco_true_pos:
                ax.scatter(axu, ayu, c="blue", s=40, marker="s", label="ArUco")

        # --- plot PATH in black ---
        path_np = np.array(self.path)
        ax.plot(path_np[:, 0], path_np[:, 1], 'k-', lw=2, label="Planned Path")
        ax.scatter(path_np[:, 0], path_np[:, 1], c="k", s=10)

        # Clean up legend duplicates
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="best")

        plt.tight_layout()
        plt.savefig("planned_path.png")
        plt.close(fig)
        print("[INFO] Saved semantic map with path to planned_path.png")


    ###################### Occupancy Map Visualization ############################
    def map_occupancy(self, filename="occupancy_map.png"):
        """
        Plot the occupancy probability map with obstacles, targets, and ArUcos overlaid.
        Fixes alignment of scatter points to match grid cell centers.
        """
        if not hasattr(self.map, "L_obstacles") or not hasattr(self.map, "L_targets"):
            print("[ERROR] OccupancyGridMap missing L_obstacles/L_targets.")
            return

        def logodds_to_prob(L):
            return 1.0 - 1.0 / (1.0 + np.exp(L))

        prob_obstacles = logodds_to_prob(self.map.L_obstacles)
        prob_targets   = logodds_to_prob(self.map.L_targets)
        prob_map = np.maximum(prob_obstacles, prob_targets)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("Occupancy Probability Map")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")

        im = ax.imshow(
            np.flipud(prob_map),
            cmap="gray_r",
            vmin=0.0,
            vmax=1.0,
            extent=[
                self.map.ox,
                self.map.ox + self.map.W * self.map.res,
                self.map.oy,
                self.map.oy + self.map.H * self.map.res
            ],
            origin="upper"
        )

        # --- Scatter plots, centered on grid cells ---
        obs_y, obs_x = np.where(prob_obstacles > 0.65)
        ax.scatter(self.map.ox + (obs_x + 0.5) * self.map.res,
                   self.map.oy + (obs_y + 0.5) * self.map.res,
                   c='red', s=2, label='Obstacles')

        tgt_y, tgt_x = np.where(prob_targets > 0.65)
        ax.scatter(self.map.ox + (tgt_x + 0.5) * self.map.res,
                   self.map.oy + (tgt_y + 0.5) * self.map.res,
                   c='green', s=2, label='Targets')

        if hasattr(self.map, "aruco_coords"):
            ax.scatter([x for x, y in self.map.aruco_coords],
                       [y for x, y in self.map.aruco_coords],
                       c='blue', s=10, label='ArUcos')

        ax.set_aspect("equal")
        ax.legend(loc="upper right")
        plt.colorbar(im, ax=ax, label="Occupancy Probability")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)
        print(f"[INFO] Saved detailed occupancy map to {filename}")




# =========================
# main loop
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--level_number", type=int, default=3, help="Select which map to use for the corresponding level")
    parser.add_argument("--ip", metavar='', type=str, default='192.168.50.1')
    parser.add_argument("--port", metavar='', type=int, default=8080)
    args, _ = parser.parse_known_args()

    map_files = {2: "M3_prac_map_full.txt", 3: "M3_prac_map_part.txt", 4: "M3_prac_map_min.txt"}
    selected_map = map_files.get(args.level_number)
    if selected_map is None:
        raise ValueError(f"No map defined for number {args.level_number}")

    print(f"Using map: {selected_map}")
    ppi = PenguinPi(args.ip, args.port)
    robot = FruitSearch(ppi)

    fruit_list, fruit_true_pos, aruco_true_pos = robot.read_true_map(selected_map)
    search_list = robot.read_search_list("M3_prac_shopping_list.txt")
    robot.print_target_fruits_pos(search_list, fruit_list, fruit_true_pos)

    # Build occupancy according to level (L2 uses GT; L3/L4 start empty, filled during exploration)
    robot.create_occupancy_map(
        level_number=args.level_number,
        fruit_list=fruit_list,
        fruit_true_pos=fruit_true_pos,
        aruco_true_pos=aruco_true_pos,
        search_list_path="M3_prac_shopping_list.txt"
    )

    robot.map_occupancy()


    if args.level_number in (3, 4):
        print("[INFO] Starting exploration to build map...")
        exploration_path = robot.generate_exploration_path(arena_size=robot.arena_size)  # simple coverage pattern
        for waypoint in exploration_path:
            robot.drive_to_point(waypoint)

            # detect and immediately fold into map (inflated to 0.05 m for planning)
            bboxes, img_out = robot.detect_fruits()
            robot.process_detections_update_map(bboxes)

        # Return to (0,0)
        print("[INFO] Returning to start position...")
        robot.drive_to_point([0.0, 0.0])

    # Plan with RRT* over the inflated occupancy (0.05 m radius)
    robot.plan_path()
    robot.map_path()
    robot.follow_path()
