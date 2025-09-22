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

class DriveMeasurement:
    def __init__(self, left_speed, right_speed, dt, left_cov=0.1, right_cov=0.1):
        self.left_speed = left_speed
        self.right_speed = right_speed  
        self.dt = dt
        self.left_cov = left_cov
        self.right_cov = right_cov

class OccupancyGridMap:
    """
    Semantic log-odds occupancy grid with freespace carving and inflation.

    Layers:
      - obstacles (distractions + ArUco + any non-target in L3/L4)
      - targets
    We inflate OBSTACLES ONLY for planning.
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
                 inflation_radius_m: float = 0.02): 
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
        self.ekf.P[0:3, 0:3] = np.eye(3) * 0.1  # Small initial uncertainty


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
                                    inflation_radius_m=0.02)  # <<< fixed to 0.05 m
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

    def get_fruit_name_at_target(self, target_index):
        """
        Get the name of the fruit at the given target index based on search list order.
        
        Args:
            target_index: Index of the target in self.targets list
        
        Returns:
            String name of the fruit, or "Unknown Fruit" if not found
        """
        if target_index < len(self.search_list):
            return self.search_list[target_index]
        else:
            return "Unknown Fruit"

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
        
        # Store aruco positions for visualization
        self.aruco_true_pos = aruco_true_pos
        
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
    
    ################### Path Planning #########################
    def calculate_pickup_points(self, pickup_distance=0.3):
        """
        Calculate pickup points for each target at specified distance.
        Returns list of pickup points that are collision-free and accessible.
        
        Args:
            pickup_distance: Distance to maintain from target center (0.3m)
        
        Returns:
            List of [x, y] pickup points corresponding to each target
        """
        pickup_points = []
        robot_radius = 0.09
        
        # Get all obstacles (including other targets) for collision checking pickup points
        all_obstacles = self.get_all_obstacle_circles()
        
        for target in self.targets:
            tx, ty = float(target[0]), float(target[1])
            
            # Try multiple angles around the target to find a good pickup point
            best_point = None
            min_obstacle_dist = 0
            
            # Test 8 directions around the target
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                # Calculate potential pickup point
                px = tx + pickup_distance * np.cos(angle)
                py = ty + pickup_distance * np.sin(angle)
                
                # Check if pickup point is within arena bounds
                if not (self.map.ox <= px <= self.map.ox + self.map.W * self.map.res and
                        self.map.oy <= py <= self.map.oy + self.map.H * self.map.res):
                    continue
                
                # Check collision with all obstacles (except the current target)
                collision = False
                min_dist_to_obstacle = float('inf')
                
                for obs_x, obs_y, obs_r in all_obstacles:
                    # Skip collision check with the current target itself
                    if abs(obs_x - tx) < 0.05 and abs(obs_y - ty) < 0.05:
                        continue
                    
                    dist_to_obs = hypot(px - obs_x, py - obs_y)
                    min_dist_to_obstacle = min(min_dist_to_obstacle, dist_to_obs)
                    
                    if dist_to_obs <= obs_r + robot_radius + 0.02:  # Small safety margin
                        collision = True
                        break
                
                if not collision and min_dist_to_obstacle > min_obstacle_dist:
                    best_point = [px, py]
                    min_obstacle_dist = min_dist_to_obstacle
            
            if best_point is not None:
                pickup_points.append(best_point)
                print(f"[INFO] Pickup point for target at ({tx:.3f}, {ty:.3f}): ({best_point[0]:.3f}, {best_point[1]:.3f})")
            else:
                # Fallback: use closest feasible point even if not optimal
                fallback_angle = np.arctan2(ty, tx) + np.pi  # Point away from origin
                px = tx + pickup_distance * np.cos(fallback_angle)
                py = ty + pickup_distance * np.sin(fallback_angle)
                pickup_points.append([px, py])
                print(f"[WARN] Using fallback pickup point for target at ({tx:.3f}, {ty:.3f}): ({px:.3f}, {py:.3f})")
        
        return pickup_points

    def smooth_path_preserve_pickups(self, path, pickup_points, robot_radius=0.09):
        """
        Smooth the path while preserving pickup points.
        Only smooths segments between pickup points, never skips pickup points themselves.
        
        Args:
            path: List of [x, y] waypoints
            pickup_points: List of pickup point coordinates that must be preserved
            robot_radius: Robot collision radius for safety checking
        
        Returns:
            Smoothed path that still visits all pickup points
        """
        if len(path) <= 2:
            return path
        
        print(f"[INFO] Smoothing path while preserving {len(pickup_points)} pickup points...")
        print(f"[INFO] Original path: {len(path)} waypoints -> ", end="")
        
        obstacles = self.get_all_obstacle_circles()
        
        # Identify which waypoints are pickup points (with tolerance)
        pickup_indices = []
        for i, waypoint in enumerate(path):
            for pickup_point in pickup_points:
                distance = hypot(waypoint[0] - pickup_point[0], waypoint[1] - pickup_point[1])
                if distance <= 0.15:  # 15cm tolerance
                    pickup_indices.append(i)
                    break
        
        # Always include start and end points
        critical_indices = [0] + pickup_indices + [len(path) - 1]
        critical_indices = sorted(list(set(critical_indices)))  # Remove duplicates and sort
        
        print(f"[DEBUG] Pickup points found at waypoint indices: {pickup_indices}")
        print(f"[DEBUG] Critical waypoints to preserve: {critical_indices}")
        
        # Smooth segments between critical points
        smoothed_path = []
        
        for i in range(len(critical_indices) - 1):
            start_idx = critical_indices[i]
            end_idx = critical_indices[i + 1]
            
            # Extract segment between critical points
            segment = path[start_idx:end_idx + 1]
            
            if len(segment) <= 2:
                # Short segment, just add as-is
                if i == 0:
                    smoothed_path.extend(segment)
                else:
                    smoothed_path.extend(segment[1:])  # Skip first point to avoid duplication
            else:
                # Smooth this segment
                smoothed_segment = self.smooth_segment(segment, obstacles, robot_radius)
                
                if i == 0:
                    smoothed_path.extend(smoothed_segment)
                else:
                    smoothed_path.extend(smoothed_segment[1:])  # Skip first point to avoid duplication
        
        print(f"{len(smoothed_path)} waypoints")
        print(f"[INFO] Reduced waypoints by {len(path) - len(smoothed_path)} while preserving all pickup points")
        
        return smoothed_path

    def smooth_segment(self, segment, obstacles, robot_radius):
        """
        Smooth a single path segment between two critical points.
        """
        if len(segment) <= 2:
            return segment
        
        smoothed_segment = [segment[0]]  # Start with first point
        current_index = 0
        
        while current_index < len(segment) - 1:
            # Look ahead to find the furthest reachable waypoint
            furthest_reachable = current_index + 1
            
            for look_ahead_index in range(current_index + 2, len(segment)):
                # Check if we can go directly from current to look_ahead point
                start_point = Node(segment[current_index][0], segment[current_index][1])
                end_point = Node(segment[look_ahead_index][0], segment[look_ahead_index][1])
                
                # Test collision-free direct path
                if not self.line_collision(start_point, end_point, obstacles, robot_radius):
                    furthest_reachable = look_ahead_index
                else:
                    break  # Can't go further, use previous reachable point
            
            # Add the furthest reachable waypoint (skip intermediate ones)
            if furthest_reachable != current_index:
                smoothed_segment.append(segment[furthest_reachable])
                current_index = furthest_reachable
            else:
                # If we can't skip any points, just advance by one
                smoothed_segment.append(segment[current_index + 1])
                current_index += 1
        
        return smoothed_segment

    def plan_path(self):
        """
        Simplified plan_path - removes duplicate fallback calls since plan_single_segment
        already has built-in parameter adaptation and fallback strategies.
        """
        if not self.targets:
            print("[INFO] No targets to plan for.")
            self.path = []
            return
        
        # Calculate pickup points for all targets
        pickup_points = self.calculate_pickup_points(pickup_distance=0.3)
        
        if len(pickup_points) != len(self.targets):
            print("[ERROR] Could not calculate pickup points for all targets")
            return
        
        print(f"[DEBUG] Planning path through {len(pickup_points)} pickup points")
        print(f"[DEBUG] Robot starting position: {[self.pose[0], self.pose[1]]}")
        print(f"[DEBUG] Pickup points: {pickup_points}")
        
        # Debug: check obstacle density around pickup points
        obstacles = self.get_all_obstacle_circles()
        print(f"[DEBUG] Total obstacles in map: {len(obstacles)}")
        
        # Analyze problematic segments
        for i, pickup in enumerate(pickup_points):
            if i > 0:  # Check segments between pickup points
                prev_pickup = pickup_points[i-1]
                segment_distance = hypot(pickup[0] - prev_pickup[0], pickup[1] - prev_pickup[1])
                
                # Count obstacles along this segment
                obstacles_in_segment = 0
                for ox, oy, r in obstacles:
                    # Check if obstacle is near the line between pickup points
                    # Simple distance to line segment check
                    dist_to_segment = self.point_to_line_distance([ox, oy], prev_pickup, pickup)
                    if dist_to_segment <= r + 0.2:  # Include buffer
                        obstacles_in_segment += 1
                
                print(f"[DEBUG] Segment {i}: {prev_pickup} -> {pickup}")
                print(f"[DEBUG] Segment {i}: distance {segment_distance:.3f}m, {obstacles_in_segment} blocking obstacles")
                
                if obstacles_in_segment > 50:  # High obstacle density
                    print(f"[WARNING] Segment {i} has very high obstacle density!")
        
        # Build full path by connecting start -> pickup1 -> pickup2 -> ... -> pickupN
        full_path = [[self.pose[0], self.pose[1]]]
        current_pos = [self.pose[0], self.pose[1]]
        
        for i, pickup_point in enumerate(pickup_points):
            print(f"[INFO] Planning segment {i+1}/{len(pickup_points)}: {current_pos} -> {pickup_point}")
            
            # ONLY call plan_single_segment - it has built-in fallbacks
            segment_path = self.plan_single_segment(current_pos, pickup_point)
            
            if segment_path is None:
                print(f"[ERROR] All planning methods failed for segment {i+1}")
                print(f"[ERROR] This suggests the pickup point may be unreachable")
                
                # Try to find an alternative pickup point for this target
                print(f"[INFO] Attempting to find alternative pickup point for target {i+1}")
                alternative_pickup = self.find_alternative_pickup_point(self.targets[i], pickup_point)
                
                if alternative_pickup:
                    print(f"[INFO] Found alternative pickup: {alternative_pickup}")
                    segment_path = self.plan_single_segment(current_pos, alternative_pickup)
                    pickup_point = alternative_pickup  # Update pickup point
                
                if segment_path is None:
                    print(f"[FATAL] Cannot reach target {i+1} at all. Aborting path planning.")
                    return
            
            # Append segment (skip first point to avoid duplication)
            if len(segment_path) > 1:
                full_path.extend(segment_path[1:])
            
            # Update current position for next segment
            current_pos = pickup_point[:]
            print(f"[SUCCESS] Completed segment {i+1}/{len(pickup_points)}")
        
        print(f"[INFO] Raw path has {len(full_path)} waypoints")
        
        # Path optimization
        smoothed_path = self.smooth_path_preserve_pickups(full_path, pickup_points)
        final_path = self.optimize_path_preserve_pickups(smoothed_path, pickup_points)
        
        self.path = final_path
        print(f"[SUCCESS] Final path has {len(self.path)} waypoints")
        print(f"[SUCCESS] Waypoint reduction: {len(full_path)} -> {len(self.path)}")
        
        self.verify_pickup_points_in_path(pickup_points)
        self.map_path()

    def point_to_line_distance(self, point, line_start, line_end):
        """
        Calculate perpendicular distance from point to line segment.
        """
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Vector from line_start to line_end
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            # Line segment is a point
            return hypot(x0 - x1, y0 - y1)
        
        # Parameter t that represents position along line segment
        t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)
        
        # Clamp t to [0, 1] to stay on line segment
        t = max(0, min(1, t))
        
        # Find closest point on line segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        return hypot(x0 - closest_x, y0 - closest_y)

    def find_alternative_pickup_point(self, target, original_pickup, max_attempts=16):
        """
        Find an alternative pickup point if the original one is unreachable.
        """
        tx, ty = float(target[0]), float(target[1])
        pickup_distance = 0.35  # Slightly larger distance
        
        obstacles = self.get_all_obstacle_circles()
        robot_radius = 0.08
        
        print(f"[DEBUG] Searching for alternative pickup point for target at ({tx:.3f}, {ty:.3f})")
        
        # Try more angles around the target
        for angle in np.linspace(0, 2*np.pi, max_attempts, endpoint=False):
            px = tx + pickup_distance * np.cos(angle)
            py = ty + pickup_distance * np.sin(angle)
            
            # Check bounds
            if not (self.map.ox <= px <= self.map.ox + self.map.W * self.map.res and
                    self.map.oy <= py <= self.map.oy + self.map.H * self.map.res):
                continue
            
            # Check collision with obstacles (except current target)
            collision = False
            for obs_x, obs_y, obs_r in obstacles:
                # Skip the current target
                if abs(obs_x - tx) < 0.05 and abs(obs_y - ty) < 0.05:
                    continue
                    
                if hypot(px - obs_x, py - obs_y) <= obs_r + robot_radius + 0.05:
                    collision = True
                    break
            
            if not collision:
                print(f"[SUCCESS] Alternative pickup found at ({px:.3f}, {py:.3f})")
                return [px, py]
        
        print(f"[ERROR] No alternative pickup point found for target at ({tx:.3f}, {ty:.3f})")
        return None

        """
        Enhanced plan_single_segment with more aggressive parameters and better debugging.
        """
        # Map extents for sampling
        x_limits = (self.map.ox, self.map.ox + self.map.W * self.map.res)
        y_limits = (self.map.oy, self.map.oy + self.map.H * self.map.res)
        robot_radius = 0.07  # Further reduced robot radius
        
        # Get ALL obstacle circles
        obstacles = self.get_all_obstacle_circles()
        
        print(f"[DEBUG] Planning from {start_pos} to {goal_pos}")
        print(f"[DEBUG] Using {len(obstacles)} obstacles, robot radius: {robot_radius}")
        
        # Calculate direct distance for reference
        direct_distance = hypot(goal_pos[0] - start_pos[0], goal_pos[1] - start_pos[1])
        print(f"[DEBUG] Direct distance: {direct_distance:.3f}m")
        
        # Initialize RRT* tree
        start_node = Node(float(start_pos[0]), float(start_pos[1]))
        goal_node = Node(float(goal_pos[0]), float(goal_pos[1]))
        
        # More aggressive parameter sets
        parameter_sets = [
            # Very aggressive first attempt
            {"max_iter": 2000, "step_size": 0.20, "goal_radius": 0.25, "rewire_radius": 0.4, "goal_bias": 0.5},
            # Moderate parameters
            {"max_iter": 3000, "step_size": 0.15, "goal_radius": 0.20, "rewire_radius": 0.35, "goal_bias": 0.4},
            # Conservative parameters
            {"max_iter": 4000, "step_size": 0.12, "goal_radius": 0.18, "rewire_radius": 0.3, "goal_bias": 0.3},
            # Final attempt with maximum iterations
            {"max_iter": 6000, "step_size": 0.10, "goal_radius": 0.30, "rewire_radius": 0.5, "goal_bias": 0.2}
        ]
        
        for attempt, params in enumerate(parameter_sets):
            print(f"[DEBUG] RRT* Attempt {attempt + 1}/{len(parameter_sets)} - {params}")
            
            tree = [start_node]
            
            for iteration in range(params["max_iter"]):
                # Sample point with goal biasing
                if random.random() < params["goal_bias"]:
                    sx, sy = goal_node.x, goal_node.y
                else:
                    sx = random.uniform(*x_limits)
                    sy = random.uniform(*y_limits)
                
                # Find nearest node
                nearest = min(tree, key=lambda n: hypot(n.x - sx, n.y - sy))
                
                # Extend toward sample
                theta = atan2(sy - nearest.y, sx - nearest.x)
                new_x = nearest.x + params["step_size"] * np.cos(theta)
                new_y = nearest.y + params["step_size"] * np.sin(theta)
                
                new_node = Node(new_x, new_y)
                new_node.parent = nearest
                new_node.cost = nearest.cost + params["step_size"]
                
                # Collision check
                if self.line_collision(nearest, new_node, obstacles, robot_radius):
                    continue
                
                # RRT* rewiring
                for node in tree:
                    if node == nearest:
                        continue
                    dist_to_new = hypot(node.x - new_node.x, node.y - new_node.y)
                    if dist_to_new <= params["rewire_radius"]:
                        if not self.line_collision(node, new_node, obstacles, robot_radius):
                            potential_cost = node.cost + dist_to_new
                            if potential_cost < new_node.cost:
                                new_node.parent = node
                                new_node.cost = potential_cost
                
                tree.append(new_node)
                
                # Check if goal is reachable
                dist_to_goal = hypot(new_node.x - goal_node.x, new_node.y - goal_node.y)
                if dist_to_goal <= params["goal_radius"]:
                    if not self.line_collision(new_node, goal_node, obstacles, robot_radius):
                        goal_node.parent = new_node
                        goal_node.cost = new_node.cost + dist_to_goal
                        
                        # Reconstruct path
                        path = []
                        current = goal_node
                        while current is not None:
                            path.append([current.x, current.y])
                            current = current.parent
                        path.reverse()
                        
                        print(f"[SUCCESS] Path found: {len(path)} points in {iteration+1} iterations (attempt {attempt+1})")
                        return path
                
                # Progress logging
                if (iteration + 1) % 2000 == 0:
                    print(f"[DEBUG] Attempt {attempt+1}, iteration {iteration+1}, tree size: {len(tree)}")
        
        print(f"[ERROR] All RRT* attempts failed after {len(parameter_sets)} tries")
        return None
    
    def optimize_path_preserve_pickups(self, path, pickup_points, min_segment_length=0.05):
        """
        Remove very short segments while preserving pickup points.
        """
        if len(path) <= 2:
            return path
        
        # Identify pickup point indices
        pickup_indices = set()
        for i, waypoint in enumerate(path):
            for pickup_point in pickup_points:
                distance = hypot(waypoint[0] - pickup_point[0], waypoint[1] - pickup_point[1])
                if distance <= 0.15:  # 15cm tolerance
                    pickup_indices.add(i)
                    break
        
        optimized_path = [path[0]]  # Always keep start
        
        for i in range(1, len(path)):
            # Always keep pickup points and final point
            if i in pickup_indices or i == len(path) - 1:
                optimized_path.append(path[i])
                continue
            
            # For other points, check minimum distance
            last_point = optimized_path[-1]
            current_point = path[i]
            distance = hypot(current_point[0] - last_point[0], 
                            current_point[1] - last_point[1])
            
            if distance >= min_segment_length:
                optimized_path.append(current_point)
        
        return optimized_path

    def verify_pickup_points_in_path(self, pickup_points):
        """
        Verify that all pickup points are still reachable in the smoothed path.
        """
        print(f"[DEBUG] Verifying {len(pickup_points)} pickup points are in path...")
        
        for i, pickup_point in enumerate(pickup_points):
            closest_distance = float('inf')
            closest_waypoint_idx = -1
            
            for j, waypoint in enumerate(self.path):
                distance = hypot(waypoint[0] - pickup_point[0], waypoint[1] - pickup_point[1])
                if distance < closest_distance:
                    closest_distance = distance
                    closest_waypoint_idx = j
            
            if closest_distance <= 0.2:  # 20cm tolerance
                print(f"[DEBUG] Pickup point {i+1} OK - closest waypoint {closest_waypoint_idx} at {closest_distance:.3f}m")
            else:
                print(f"[ERROR] Pickup point {i+1} too far from path! Closest distance: {closest_distance:.3f}m")
                # Add the pickup point back to the path
                print(f"[FIX] Adding pickup point {i+1} back to path at position {closest_waypoint_idx+1}")
                self.path.insert(closest_waypoint_idx + 1, pickup_point)

    def plan_single_segment(self, start_pos, goal_pos):
        """
        Enhanced plan_single_segment with more aggressive parameters and better debugging.
        """
        # Map extents for sampling
        x_limits = (self.map.ox, self.map.ox + self.map.W * self.map.res)
        y_limits = (self.map.oy, self.map.oy + self.map.H * self.map.res)
        robot_radius = 0.07  # Further reduced robot radius
        
        # Get ALL obstacle circles
        obstacles = self.get_all_obstacle_circles()
        
        print(f"[DEBUG] Planning from {start_pos} to {goal_pos}")
        print(f"[DEBUG] Using {len(obstacles)} obstacles, robot radius: {robot_radius}")
        
        # Calculate direct distance for reference
        direct_distance = hypot(goal_pos[0] - start_pos[0], goal_pos[1] - start_pos[1])
        print(f"[DEBUG] Direct distance: {direct_distance:.3f}m")
        
        # Initialize RRT* tree
        start_node = Node(float(start_pos[0]), float(start_pos[1]))
        goal_node = Node(float(goal_pos[0]), float(goal_pos[1]))
        
        # More aggressive parameter sets
        parameter_sets = [
            # Very aggressive first attempt
            {"max_iter": 2000, "step_size": 0.20, "goal_radius": 0.25, "rewire_radius": 0.4, "goal_bias": 0.5},
            # Moderate parameters
            {"max_iter": 3000, "step_size": 0.15, "goal_radius": 0.20, "rewire_radius": 0.35, "goal_bias": 0.4},
            # Conservative parameters
            {"max_iter": 4000, "step_size": 0.12, "goal_radius": 0.18, "rewire_radius": 0.3, "goal_bias": 0.3},
            # Final attempt with maximum iterations
            {"max_iter": 6000, "step_size": 0.10, "goal_radius": 0.30, "rewire_radius": 0.5, "goal_bias": 0.2}
        ]
        
        for attempt, params in enumerate(parameter_sets):
            print(f"[DEBUG] RRT* Attempt {attempt + 1}/{len(parameter_sets)} - {params}")
            
            tree = [start_node]
            
            for iteration in range(params["max_iter"]):
                # Sample point with goal biasing
                if random.random() < params["goal_bias"]:
                    sx, sy = goal_node.x, goal_node.y
                else:
                    sx = random.uniform(*x_limits)
                    sy = random.uniform(*y_limits)
                
                # Find nearest node
                nearest = min(tree, key=lambda n: hypot(n.x - sx, n.y - sy))
                
                # Extend toward sample
                theta = atan2(sy - nearest.y, sx - nearest.x)
                new_x = nearest.x + params["step_size"] * np.cos(theta)
                new_y = nearest.y + params["step_size"] * np.sin(theta)
                
                new_node = Node(new_x, new_y)
                new_node.parent = nearest
                new_node.cost = nearest.cost + params["step_size"]
                
                # Collision check
                if self.line_collision(nearest, new_node, obstacles, robot_radius):
                    continue
                
                # RRT* rewiring
                for node in tree:
                    if node == nearest:
                        continue
                    dist_to_new = hypot(node.x - new_node.x, node.y - new_node.y)
                    if dist_to_new <= params["rewire_radius"]:
                        if not self.line_collision(node, new_node, obstacles, robot_radius):
                            potential_cost = node.cost + dist_to_new
                            if potential_cost < new_node.cost:
                                new_node.parent = node
                                new_node.cost = potential_cost
                
                tree.append(new_node)
                
                # Check if goal is reachable
                dist_to_goal = hypot(new_node.x - goal_node.x, new_node.y - goal_node.y)
                if dist_to_goal <= params["goal_radius"]:
                    if not self.line_collision(new_node, goal_node, obstacles, robot_radius):
                        goal_node.parent = new_node
                        goal_node.cost = new_node.cost + dist_to_goal
                        
                        # Reconstruct path
                        path = []
                        current = goal_node
                        while current is not None:
                            path.append([current.x, current.y])
                            current = current.parent
                        path.reverse()
                        
                        print(f"[SUCCESS] Path found: {len(path)} points in {iteration+1} iterations (attempt {attempt+1})")
                        return path
                
                # Progress logging
                if (iteration + 1) % 2000 == 0:
                    print(f"[DEBUG] Attempt {attempt+1}, iteration {iteration+1}, tree size: {len(tree)}")
        
        print(f"[ERROR] All RRT* attempts failed after {len(parameter_sets)} tries")
        return None

    def get_all_obstacle_circles(self):
        """
        Get ALL obstacle circles including both obstacles AND targets for collision detection.
        Fixed coordinate system issues.
        """
        if self.map.inflated_obstacles is None:
            self.map.inflate_obstacles()
        
        circles = []
        r_obstacle = self.map.inflation_radius_m
        
        # Add inflated obstacles (distractions, ArUco markers, etc.)
        ys, xs = np.where(self.map.inflated_obstacles > 0)
        for gy, gx in zip(ys, xs):
            # Fixed coordinate conversion
            x = self.map.ox + (gx + 0.5) * self.map.res
            y = self.map.oy + (gy + 0.5) * self.map.res
            circles.append((x, y, r_obstacle))
        
        # Add targets as obstacles too but with larger radius for safety
        target_ys, target_xs = np.where(self.map.L_targets > 0)
        r_target = 0.15  # Increased radius for targets to ensure better clearance
        for gy, gx in zip(target_ys, target_xs):
            x = self.map.ox + (gx + 0.5) * self.map.res
            y = self.map.oy + (gy + 0.5) * self.map.res
            circles.append((x, y, r_target))
        
        print(f"[DEBUG] Generated {len(circles)} obstacle circles")
        print(f"[DEBUG] Obstacle circles: {len(ys)} obstacles, {len(target_ys)} targets")
        
        return circles

    def line_collision(self, p1, p2, obstacles, robot_radius=0.08, step=0.02):
        """
        Enhanced collision checking with debugging info for failed paths.
        """
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        dist_total = hypot(dx, dy)
        steps = max(2, int(dist_total / step))
        
        collision_points = []  # Track where collisions occur for debugging
        
        for i in range(steps+1):
            x = p1.x + dx*i/steps
            y = p1.y + dy*i/steps
            for obs_idx, obs in enumerate(obstacles):
                ox, oy, r = obs
                collision_dist = hypot(x-ox, y-oy)
                if collision_dist <= r + robot_radius:
                    collision_points.append((x, y, ox, oy, collision_dist, r + robot_radius))
                    return True
        
        return False

    ############### Path following###########3

    def drive_to_point_with_ekf_updates(self, waypoint):
        """
        Enhanced navigation with proper EKF prediction and update steps.
        Uses both odometry (prediction) and landmarks (update) for pose tracking.
        """
        # Get current pose from EKF
        current_ekf_pose = self.get_robot_pose()
        x, y, theta = current_ekf_pose[0], current_ekf_pose[1], current_ekf_pose[2]
        self.pose = current_ekf_pose.copy()
        
        dx = waypoint[0] - x
        dy = waypoint[1] - y
        desired_theta = math.atan2(dy, dx)
        heading_error = self.wrap_angle(desired_theta - theta)
        distance = math.hypot(dx, dy)

        if distance < 0.02:
            print(f"[INFO] Already at waypoint [{waypoint[0]:.3f}, {waypoint[1]:.3f}]")
            return

        print(f"[INFO] EKF Nav: [{x:.3f}, {y:.3f}] -> [{waypoint[0]:.3f}, {waypoint[1]:.3f}]")

        # Calibration parameters
        scale = self.scale
        baseline = self.baseline
        wheel_vel = 25

        # TURN PHASE with EKF prediction + update
        if abs(heading_error) > 0.05:  # ~3 degrees
            print(f"[INFO] EKF Turning {math.degrees(heading_error):.1f}°")
            
            turn_rate = 2 * wheel_vel * scale / baseline
            turn_time = abs(heading_error) / turn_rate
            turn_direction = 1 if heading_error > 0 else 0
            
            # Create drive measurement for turning motion
            left_speed = -wheel_vel if heading_error > 0 else wheel_vel
            right_speed = wheel_vel if heading_error > 0 else -wheel_vel
            
            # Start turning
            start_time = time.time()
            self.ppi.set_velocity([0, turn_direction], turning_tick=wheel_vel, time=0)
            
            # Monitor turn with EKF updates
            update_interval = 0.1  # Update every 100ms
            last_update = start_time
            
            while time.time() - start_time < turn_time:
                current_time = time.time()
                
                # Create drive measurement for the time interval
                dt = min(update_interval, turn_time - (current_time - start_time))
                if dt > 0.01:  # Only predict if dt is meaningful
                    drive_meas = DriveMeasurement(
                        left_speed=left_speed,
                        right_speed=right_speed,
                        dt=dt
                    )
                    
                    try:
                        # EKF PREDICTION STEP (odometry)
                        self.ekf.predict(drive_meas)
                        
                        # EKF UPDATE STEP (landmarks)
                        img = self.ppi.get_image()
                        if img is not None:
                            lms, aruco_img = aruco.aruco_detector.detect_marker_positions(img)
                            if len(lms) > 0:
                                self.ekf.add_landmarks(lms)
                                self.ekf.update(lms)
                                print(f"[EKF] Updated with {len(lms)} landmarks")
                    
                    except Exception as e:
                        print(f"[WARNING] EKF update failed: {e}")
                        pass
                
                time.sleep(0.05)  # Small delay between updates
            
            # Stop turning
            self.ppi.set_velocity([0, 0], tick=0, time=0)
            time.sleep(0.2)  # Let robot settle

        # DRIVE PHASE with EKF prediction + update
        updated_pose = self.get_robot_pose()
        x, y, theta = updated_pose[0], updated_pose[1], updated_pose[2]
        dx = waypoint[0] - x
        dy = waypoint[1] - y
        distance = math.hypot(dx, dy)

        if distance > 0.02:
            print(f"[INFO] EKF Driving {distance:.3f}m")
            
            drive_speed = wheel_vel * scale
            drive_time = distance / drive_speed
            
            # Start driving straight
            start_time = time.time()
            self.ppi.set_velocity([1, 0], tick=wheel_vel, time=0)
            
            # Monitor drive with EKF updates
            update_interval = 0.1  # Update every 100ms
            
            while time.time() - start_time < drive_time:
                current_time = time.time()
                
                # Create drive measurement for straight driving
                dt = min(update_interval, drive_time - (current_time - start_time))
                if dt > 0.01:  # Only predict if dt is meaningful
                    drive_meas = DriveMeasurement(
                        left_speed=wheel_vel,
                        right_speed=wheel_vel,
                        dt=dt
                    )
                    
                    try:
                        # EKF PREDICTION STEP (odometry)
                        self.ekf.predict(drive_meas)
                        
                        # EKF UPDATE STEP (landmarks)
                        img = self.ppi.get_image()
                        if img is not None:
                            lms, aruco_img = aruco.aruco_detector.detect_marker_positions(img)
                            if len(lms) > 0:
                                self.ekf.add_landmarks(lms)
                                self.ekf.update(lms)
                                print(f"[EKF] Updated with {len(lms)} landmarks")
                    
                    except Exception as e:
                        print(f"[WARNING] EKF update failed: {e}")
                        pass
                
                time.sleep(0.1)
            
            # Stop driving
            self.ppi.set_velocity([0, 0], tick=0, time=0)
            time.sleep(0.1)

        # Final pose update and verification
        self.pose = self.get_robot_pose().copy()
        final_distance = math.hypot(waypoint[0] - self.pose[0], waypoint[1] - self.pose[1])
        
        print(f"[INFO] EKF Final pose: [{self.pose[0]:.3f}, {self.pose[1]:.3f}, {self.pose[2]:.3f}]")
        print(f"[INFO] Arrival error: {final_distance:.3f}m")
        
        if final_distance > 0.1:  # Warn if large error
            print(f"[WARNING] Large positioning error! Target: [{waypoint[0]:.3f}, {waypoint[1]:.3f}], Actual: [{self.pose[0]:.3f}, {self.pose[1]:.3f}]")
    
    def follow_path_ekf(self, dt=0.1):
        """
        Follow the planned path using EKF-enhanced navigation with SLAM updates.
        Detects pickup points and performs fruit collection behavior with target facing.
        Uses drive_to_point_with_ekf_updates for better pose accuracy.
        """
        if not self.path:
            print("[INFO] No path to follow.")
            return
        
        # Calculate pickup points to know where to stop for collection
        pickup_points = []
        if self.targets:
            pickup_points = self.calculate_pickup_points(pickup_distance=0.3)
        
        print(f"[INFO] Following path with EKF/SLAM - {len(self.path)} waypoints")
        print(f"[INFO] Will collect {len(pickup_points)} fruits during path execution")
        
        current_target_index = 0  # Track which target we're approaching
        
        for i, waypoint in enumerate(self.path):
            # Drive to the waypoint using EKF-enhanced navigation
            self.drive_to_point_with_ekf_updates(waypoint)
            
            # Wait 1 second at each waypoint for stability
            time.sleep(1.0)
            
            # Check if this waypoint is a pickup point
            is_pickup_point = False
            closest_pickup_idx = -1
            
            if current_target_index < len(pickup_points):
                pickup_point = pickup_points[current_target_index]
                
                # Check if current waypoint is close to the pickup point (within 0.1m tolerance)
                distance_to_pickup = hypot(waypoint[0] - pickup_point[0], 
                                        waypoint[1] - pickup_point[1])
                
                if distance_to_pickup <= 0.1:  # Close enough to pickup point
                    is_pickup_point = True
                    closest_pickup_idx = current_target_index
            
            # Also check if we're close to any pickup point (in case path smoothing affected order)
            if not is_pickup_point:
                for pickup_idx, pickup_point in enumerate(pickup_points):
                    if pickup_idx < current_target_index:  # Already collected
                        continue
                        
                    distance_to_pickup = hypot(waypoint[0] - pickup_point[0], 
                                            waypoint[1] - pickup_point[1])
                    
                    if distance_to_pickup <= 0.15:  # Slightly larger tolerance for fallback
                        is_pickup_point = True
                        closest_pickup_idx = pickup_idx
                        current_target_index = pickup_idx  # Update to correct index
                        break
            
            # If we're at a pickup point, perform fruit collection behavior
            if is_pickup_point and closest_pickup_idx >= 0:
                fruit_name = self.get_fruit_name_at_target(closest_pickup_idx)
                target_pos = self.targets[closest_pickup_idx]
                
                print(f"\n{'='*60}")
                print(f"[FRUIT COLLECTION - EKF] Arrived at pickup point {closest_pickup_idx + 1}/{len(pickup_points)}")
                print(f"[FRUIT COLLECTION - EKF] Collecting: {fruit_name}")
                print(f"[FRUIT COLLECTION - EKF] Target position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}]")
                print(f"[FRUIT COLLECTION - EKF] Pickup position: [{waypoint[0]:.3f}, {waypoint[1]:.3f}]")
                print(f"{'='*60}")
                
                # Face the target before collecting (uses EKF pose internally)
                self.face_target_for_collection_EKF(target_pos, fruit_name)
                
                # Look at the fruit for 5 seconds to "collect" it
                print(f"[FRUIT COLLECTION - EKF] Looking at {fruit_name} for 5 seconds to collect...")
                time.sleep(5.0)
                
                print(f"[FRUIT COLLECTION - EKF] Successfully collected {fruit_name}!")
                print(f"[FRUIT COLLECTION - EKF] Remaining fruits: {len(pickup_points) - closest_pickup_idx - 1}\n")
                
                # Move to next target
                current_target_index = closest_pickup_idx + 1
            
            # Progress update for regular waypoints
            elif i % 10 == 0 or i == len(self.path) - 1:  # Every 10 waypoints or last waypoint
                ekf_pose = self.get_robot_pose()
                print(f"[INFO] EKF Progress: waypoint {i+1}/{len(self.path)}, EKF pose: [{ekf_pose[0]:.3f}, {ekf_pose[1]:.3f}, {ekf_pose[2]:.3f}]")
        
        print(f"\n[SUCCESS] EKF path following complete!")
        print(f"[SUCCESS] Collected all {len(pickup_points)} fruits from the shopping list using EKF/SLAM")

    def drive_to_point_simple(self, waypoint):
        """
        Drive the robot to a specific waypoint WITHOUT EKF updates.
        Uses simple mathematical pose updates for testing purposes.
        
        Args:
            waypoint : [x, y] target coordinates
        """
        # Use internal pose tracking (no EKF)
        x, y, theta = self.pose[0], self.pose[1], self.pose[2]
        
        dx = waypoint[0] - x
        dy = waypoint[1] - y

        # --- Compute heading and distance ---
        desired_theta = math.atan2(dy, dx)
        heading_error = self.wrap_angle(desired_theta - theta)
        distance = math.hypot(dx, dy)

        if distance < 0.02:  # Increased tolerance for practical navigation
            print(f"[INFO] Already at waypoint [{waypoint[0]:.3f}, {waypoint[1]:.3f}] (within 2cm)")
            return

        print(f"[INFO] Simple nav: [{x:.3f}, {y:.3f}, {theta:.3f}] -> Target: [{waypoint[0]:.3f}, {waypoint[1]:.3f}]")
        print(f"[INFO] Distance: {distance:.3f}m, Required turn: {heading_error:.3f} rad ({math.degrees(heading_error):.1f}°)")

        # --- Calibration parameters ---
        scale = self.scale
        baseline = self.baseline
        wheel_vel = 25  # Reduced speed for better accuracy

        # --- Turn on the spot (only if significant heading error) ---
        min_turn_threshold = 0.05  # ~3 degrees minimum turn
        if abs(heading_error) > min_turn_threshold:
            turn_rate = 2 * wheel_vel * scale / baseline  # rad/s
            turn_time = abs(heading_error) / turn_rate
            turn_direction = 1 if heading_error > 0 else -1
            
            print(f"[INFO] Simple turning {heading_error:.3f} rad ({math.degrees(heading_error):.1f}°) for {turn_time:.2f} s")
            self.ppi.set_velocity([0, turn_direction], turning_tick=wheel_vel, time=turn_time)
            
            # Let the robot settle after turning
            time.sleep(0.2)
            
            # Update heading mathematically (no EKF)
            theta += heading_error  # Simple mathematical update
            theta = self.wrap_angle(theta)
            
            print(f"[INFO] After turn - mathematical heading: {theta:.3f} rad ({math.degrees(theta):.1f}°)")
        else:
            print(f"[INFO] Heading error {math.degrees(heading_error):.1f}° below threshold, skipping turn")

        # --- Drive straight ---
        # Recalculate distance after potential pose update
        dx = waypoint[0] - x
        dy = waypoint[1] - y
        distance = math.hypot(dx, dy)
        
        if distance > 0.02:  # Only drive if still far enough
            drive_speed = wheel_vel * scale  # m/s approx
            drive_time = distance / drive_speed
            
            print(f"[INFO] Simple driving straight {distance:.3f}m for {drive_time:.2f}s at {drive_speed:.3f}m/s")
            self.ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
            
            # Let the robot settle after driving
            time.sleep(0.1)
            
            # Update position mathematically (no EKF)
            x += distance * np.cos(theta)
            y += distance * np.sin(theta)

        # --- Update internal pose mathematically ---
        self.pose[0] = x
        self.pose[1] = y
        self.pose[2] = theta
        
        # Verify we reached the target
        final_distance = math.hypot(waypoint[0] - x, waypoint[1] - y)
        
        print(f"[INFO] Simple nav final pose: [{x:.3f}, {y:.3f}, {theta:.3f}]")
        print(f"[INFO] Distance to target: {final_distance:.3f}m")
        
        if final_distance > 0.1:  # Warn if we're far from target
            print(f"[WARN] Large positioning error! Expected: [{waypoint[0]:.3f}, {waypoint[1]:.3f}], Got: [{x:.3f}, {y:.3f}]")

    def follow_path_simple(self, dt=0.1):
        """
        Follow the planned path using simple mathematical pose updates (no EKF/SLAM).
        Detects pickup points and performs fruit collection behavior with target facing.
        Uses drive_to_point_simple for testing purposes without EKF complexity.
        """
        if not self.path:
            print("[INFO] No path to follow.")
            return
        
        # Calculate pickup points to know where to stop for collection
        pickup_points = []
        if self.targets:
            pickup_points = self.calculate_pickup_points(pickup_distance=0.3)
        
        print(f"[INFO] Following path with simple navigation - {len(self.path)} waypoints")
        print(f"[INFO] Will collect {len(pickup_points)} fruits during path execution")
        
        current_target_index = 0  # Track which target we're approaching
        
        for i, waypoint in enumerate(self.path):
            # Drive to the waypoint using simple navigation
            self.drive_to_point_simple(waypoint)
            
            # Wait 1 second at each waypoint for simplicity
            time.sleep(1.0)
            
            # Check if this waypoint is a pickup point
            is_pickup_point = False
            closest_pickup_idx = -1
            
            if current_target_index < len(pickup_points):
                pickup_point = pickup_points[current_target_index]
                
                # Check if current waypoint is close to the pickup point (within 0.1m tolerance)
                distance_to_pickup = hypot(waypoint[0] - pickup_point[0], 
                                        waypoint[1] - pickup_point[1])
                
                if distance_to_pickup <= 0.1:  # Close enough to pickup point
                    is_pickup_point = True
                    closest_pickup_idx = current_target_index
            
            # Also check if we're close to any pickup point (in case path smoothing affected order)
            if not is_pickup_point:
                for pickup_idx, pickup_point in enumerate(pickup_points):
                    if pickup_idx < current_target_index:  # Already collected
                        continue
                        
                    distance_to_pickup = hypot(waypoint[0] - pickup_point[0], 
                                            waypoint[1] - pickup_point[1])
                    
                    if distance_to_pickup <= 0.15:  # Slightly larger tolerance for fallback
                        is_pickup_point = True
                        closest_pickup_idx = pickup_idx
                        current_target_index = pickup_idx  # Update to correct index
                        break
            
            # If we're at a pickup point, perform fruit collection behavior
            if is_pickup_point and closest_pickup_idx >= 0:
                fruit_name = self.get_fruit_name_at_target(closest_pickup_idx)
                target_pos = self.targets[closest_pickup_idx]
                
                print(f"\n{'='*60}")
                print(f"[FRUIT COLLECTION - SIMPLE] Arrived at pickup point {closest_pickup_idx + 1}/{len(pickup_points)}")
                print(f"[FRUIT COLLECTION - SIMPLE] Collecting: {fruit_name}")
                print(f"[FRUIT COLLECTION - SIMPLE] Target position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}]")
                print(f"[FRUIT COLLECTION - SIMPLE] Pickup position: [{waypoint[0]:.3f}, {waypoint[1]:.3f}]")
                print(f"{'='*60}")
                
                # Face the target before collecting (modified to use simple pose)
                self.face_target_for_collection_simple(target_pos, fruit_name)
                
                # Look at the fruit for 5 seconds to "collect" it
                print(f"[FRUIT COLLECTION - SIMPLE] Looking at {fruit_name} for 5 seconds to collect...")
                time.sleep(5.0)
                
                print(f"[FRUIT COLLECTION - SIMPLE] Successfully collected {fruit_name}!")
                print(f"[FRUIT COLLECTION - SIMPLE] Remaining fruits: {len(pickup_points) - closest_pickup_idx - 1}\n")
                
                # Move to next target
                current_target_index = closest_pickup_idx + 1
            
            # Progress update for regular waypoints
            elif i % 10 == 0 or i == len(self.path) - 1:  # Every 10 waypoints or last waypoint
                print(f"[INFO] Simple Progress: waypoint {i+1}/{len(self.path)}, pose: [{self.pose[0]:.3f}, {self.pose[1]:.3f}, {self.pose[2]:.3f}]")
        
        print(f"\n[SUCCESS] Simple path following complete!")
        print(f"[SUCCESS] Collected all {len(pickup_points)} fruits from the shopping list using simple navigation")

    def face_target_for_collection_simple(self, target_pos, fruit_name):
        """
        Turn the robot to face the target fruit for collection using simple pose tracking.
        
        Args:
            target_pos: np.array([x, y]) position of the target
            fruit_name: string name of the fruit being collected
        """
        # Use internal pose tracking (no EKF)
        robot_x, robot_y, robot_theta = self.pose[0], self.pose[1], self.pose[2]
        
        # Calculate required heading to face the target
        target_x, target_y = float(target_pos[0]), float(target_pos[1])
        dx = target_x - robot_x
        dy = target_y - robot_y
        
        required_heading = math.atan2(dy, dx)
        heading_error = self.wrap_angle(required_heading - robot_theta)
        
        distance_to_target = math.hypot(dx, dy)
        
        print(f"[TARGET FACING - SIMPLE] Current robot pose: [{robot_x:.3f}, {robot_y:.3f}, {robot_theta:.3f}]")
        print(f"[TARGET FACING - SIMPLE] Target {fruit_name} at: [{target_x:.3f}, {target_y:.3f}]")
        print(f"[TARGET FACING - SIMPLE] Distance to target: {distance_to_target:.3f}m")
        print(f"[TARGET FACING - SIMPLE] Required heading: {required_heading:.3f} rad ({math.degrees(required_heading):.1f}°)")
        print(f"[TARGET FACING - SIMPLE] Current heading: {robot_theta:.3f} rad ({math.degrees(robot_theta):.1f}°)")
        print(f"[TARGET FACING - SIMPLE] Heading error: {heading_error:.3f} rad ({math.degrees(heading_error):.1f}°)")
        
        # Turn to face target if heading error is significant
        min_facing_threshold = 0.02  # ~1 degree - more precise for "looking"
        
        if abs(heading_error) > min_facing_threshold:
            print(f"[TARGET FACING - SIMPLE] Turning to face {fruit_name}...")
            
            # Calibration parameters
            scale = self.scale
            baseline = self.baseline
            wheel_vel = 20  # Slower speed for precise aiming
            
            turn_rate = 2 * wheel_vel * scale / baseline  # rad/s
            turn_time = abs(heading_error) / turn_rate
            turn_direction = 1 if heading_error > 0 else 0
            
            print(f"[TARGET FACING - SIMPLE] Executing precise turn: {heading_error:.3f} rad for {turn_time:.2f}s")
            self.ppi.set_velocity([0, turn_direction], turning_tick=wheel_vel, time=turn_time)
            
            # Let robot settle after precise turn
            time.sleep(0.3)
            
            # Update heading mathematically (no EKF)
            final_theta = robot_theta + heading_error
            final_theta = self.wrap_angle(final_theta)
            final_error = self.wrap_angle(required_heading - final_theta)
            
            print(f"[TARGET FACING - SIMPLE] Final heading: {final_theta:.3f} rad ({math.degrees(final_theta):.1f}°)")
            print(f"[TARGET FACING - SIMPLE] Final error: {final_error:.3f} rad ({math.degrees(final_error):.1f}°)")
            
            if abs(final_error) <= 0.05:  # ~3 degrees tolerance
                print(f"[TARGET FACING - SIMPLE] ✓ Successfully facing {fruit_name}")
            else:
                print(f"[TARGET FACING - SIMPLE] ⚠ Large aiming error: {math.degrees(final_error):.1f}° - may need recalibration")
            
            # Update internal pose tracking
            self.pose[2] = final_theta
        else:
            print(f"[TARGET FACING - SIMPLE] ✓ Already facing {fruit_name} (error: {math.degrees(heading_error):.1f}°)")
    
    def face_target_for_collection_EKF(self, target_pos, fruit_name):
        """
        Turn the robot to face the target fruit for collection using EKF pose tracking.
        
        Args:
            target_pos: np.array([x, y]) position of the target
            fruit_name: string name of the fruit being collected
        """
        # Get current robot pose from EKF
        current_pose = self.get_robot_pose()
        robot_x, robot_y, robot_theta = current_pose[0], current_pose[1], current_pose[2]
        
        # Calculate required heading to face the target
        target_x, target_y = float(target_pos[0]), float(target_pos[1])
        dx = target_x - robot_x
        dy = target_y - robot_y
        
        required_heading = math.atan2(dy, dx)
        heading_error = self.wrap_angle(required_heading - robot_theta)
        
        distance_to_target = math.hypot(dx, dy)
        
        print(f"[TARGET FACING - EKF] Current robot pose: [{robot_x:.3f}, {robot_y:.3f}, {robot_theta:.3f}]")
        print(f"[TARGET FACING - EKF] Target {fruit_name} at: [{target_x:.3f}, {target_y:.3f}]")
        print(f"[TARGET FACING - EKF] Distance to target: {distance_to_target:.3f}m")
        print(f"[TARGET FACING - EKF] Required heading: {required_heading:.3f} rad ({math.degrees(required_heading):.1f}°)")
        print(f"[TARGET FACING - EKF] Current heading: {robot_theta:.3f} rad ({math.degrees(robot_theta):.1f}°)")
        print(f"[TARGET FACING - EKF] Heading error: {heading_error:.3f} rad ({math.degrees(heading_error):.1f}°)")
        
        # Turn to face target if heading error is significant
        min_facing_threshold = 0.02  # ~1 degree - more precise for "looking"
        
        if abs(heading_error) > min_facing_threshold:
            print(f"[TARGET FACING - EKF] Turning to face {fruit_name}...")
            
            # Calibration parameters
            scale = self.scale
            baseline = self.baseline
            wheel_vel = 20  # Slower speed for precise aiming
            
            turn_rate = 2 * wheel_vel * scale / baseline  # rad/s
            turn_time = abs(heading_error) / turn_rate
            turn_direction = 1 if heading_error > 0 else 0
            
            print(f"[TARGET FACING - EKF] Executing precise turn: {heading_error:.3f} rad for {turn_time:.2f}s")
            self.ppi.set_velocity([0, turn_direction], turning_tick=wheel_vel, time=turn_time)
            
            # Let robot settle after precise turn
            time.sleep(0.3)
            
            # Verify final heading using EKF
            final_pose = self.get_robot_pose()
            final_theta = final_pose[2]
            final_error = self.wrap_angle(required_heading - final_theta)
            
            print(f"[TARGET FACING - EKF] Final heading: {final_theta:.3f} rad ({math.degrees(final_theta):.1f}°)")
            print(f"[TARGET FACING - EKF] Final error: {final_error:.3f} rad ({math.degrees(final_error):.1f}°)")
            
            if abs(final_error) <= 0.05:  # ~3 degrees tolerance
                print(f"[TARGET FACING - EKF] ✓ Successfully facing {fruit_name}")
            else:
                print(f"[TARGET FACING - EKF] ⚠ Large aiming error: {math.degrees(final_error):.1f}° - may need recalibration")
            
            # Update internal pose tracking
            self.pose = final_pose.copy()
        else:
            print(f"[TARGET FACING - EKF] ✓ Already facing {fruit_name} (error: {math.degrees(heading_error):.1f}°)")
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
        Enhanced visualization showing pickup points and target positions.
        """
        if not self.path:
            print("[INFO] No path to save.")
            return

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title("Semantic Map + Planned Path + Pickup Points")
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
            ax.scatter(ox, oy, c="red", s=15, label="Obstacles/Distractions", alpha=0.7)

        # --- plot TARGETS in green circles (showing they should be avoided) ---
        if np.any(self.map.L_targets > 0):
            ys, xs = np.where(self.map.L_targets > 0)
            tx = self.map.ox + (xs + 0.5) * self.map.res
            ty = self.map.oy + (ys + 0.5) * self.map.res
            ax.scatter(tx, ty, c="green", s=30, label="Target Fruits", alpha=0.8)

        # --- plot PICKUP POINTS in blue ---
        if hasattr(self, 'targets') and self.targets:
            pickup_points = self.calculate_pickup_points(pickup_distance=0.3)
            if pickup_points:
                pickup_x = [p[0] for p in pickup_points]
                pickup_y = [p[1] for p in pickup_points]
                ax.scatter(pickup_x, pickup_y, c="blue", s=25, marker="^", label="Pickup Points", alpha=0.9)
                
                # Draw lines from targets to pickup points
                for i, (target, pickup) in enumerate(zip(self.targets, pickup_points)):
                    ax.plot([target[0], pickup[0]], [target[1], pickup[1]], 'b--', alpha=0.5, linewidth=1)

        # --- plot ArUco markers in purple ---
        if hasattr(self, "aruco_true_pos") and self.aruco_true_pos is not None:
            ax.scatter(self.aruco_true_pos[:, 0], self.aruco_true_pos[:, 1], 
                    c="purple", s=40, marker="s", label="ArUco", alpha=0.8)

        # --- plot PATH in black ---
        path_np = np.array(self.path)
        ax.plot(path_np[:, 0], path_np[:, 1], 'k-', lw=2, label="Planned Path")
        ax.scatter(path_np[:, 0], path_np[:, 1], c="black", s=8, alpha=0.6)

        # Clean up legend duplicates
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="best")

        plt.tight_layout()
        plt.savefig("planned_path.png", dpi=150)
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
            robot.drive_to_point_simple(waypoint)

            # detect and immediately fold into map (inflated to 0.05 m for planning)
            bboxes, img_out = robot.detect_fruits()
            robot.process_detections_update_map(bboxes)

        # Return to (0,0)
        print("[INFO] Returning to start position...")
        robot.drive_to_point_simple([0.0, 0.0])

    robot.plan_path()
    robot.map_path()
    robot.follow_path_simple()
