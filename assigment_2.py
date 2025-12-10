""" 
Assignment 2: SLAM with Scan Matching

This implements a basic SLAM system using ICP scan matching, 
occupancy grid mapping, and loop closure detection.

The robot uses a Pioneer P3DX with lidar in CoppeliaSim.
"""

import robotica
import time
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line          # for ray tracing in occupancy grid
from scipy.ndimage import gaussian_filter  # smoothing the map
from scipy.spatial import KDTree       # fast nearest neighbor search for ICP
import copy


# =============================================================================
# Data Structures
# =============================================================================

class PoseNode:
    """
    Represents a single pose in the pose graph.
    Stores position (x, y), orientation (theta), and the lidar scan taken at this pose.
    We need to keep the scan around for loop closure detection later.
    """
    def __init__(self, idx, x, y, theta, scan):
        self.idx = idx
        self.x = x
        self.y = y
        self.theta = theta
        self.scan = scan  # raw lidar points in local frame


# =============================================================================
# Scan Matching (ICP)
# =============================================================================

class ScanMatcher:
    """
    Iterative Closest Point (ICP) scan matcher.
    
    Aligns two lidar scans by iteratively finding correspondences and 
    estimating the transformation. This is the core of scan-to-scan matching.
    
    """
    def __init__(self, max_iterations=20, tolerance=1e-4, 
                 max_correspondence_dist=0.3, min_points=10):
        self.max_iterations = max_iterations
        self.tolerance = tolerance                              # stop when change is smaller than this
        self.max_correspondence_dist = max_correspondence_dist  # reject matches farther than this
        self.min_points = min_points                            # need at least this many points for a valid match
        self.total_matches = 0
        self.successful_matches = 0
    
    def preprocess_scan(self, scan_points, max_range=4.0, min_range=0.1, downsample_factor=2):
        """
        Clean up raw lidar data before matching.
        - Filter out points that are too close (noise) or too far (unreliable)
        - Downsample to speed things up (we don't need every single point)
        """
        if len(scan_points) == 0: return np.array([]).reshape(0, 2)
        data = np.asarray(scan_points).reshape(-1, 3)
        x, y = data[:, 0], data[:, 1]
        distances = np.sqrt(x**2 + y**2)
        valid = (distances > min_range) & (distances < max_range)
        points = np.column_stack([x[valid], y[valid]])
        # Downsample by taking every nth point - makes ICP faster
        if len(points) > 0 and downsample_factor > 1:
            points = points[::downsample_factor]
        return points
    
    def find_correspondences(self, source_points, target_points):
        """
        Find matching points between source and target scans.
        Uses KDTree for fast nearest neighbor lookup.
        Rejects matches that are too far apart (probably wrong associations).
        """
        if len(source_points) == 0 or len(target_points) == 0:
            return np.array([]).reshape(0, 2), np.array([])
        # KDTree makes this O(n log n) instead of O(n^2) - huge speedup!
        tree = KDTree(target_points)
        distances, indices = tree.query(source_points)
        # Only keep correspondences that are close enough to be valid
        valid_mask = distances < self.max_correspondence_dist
        source_indices = np.where(valid_mask)[0]
        target_indices = indices[valid_mask]
        correspondences = np.column_stack([source_indices, target_indices])
        valid_distances = distances[valid_mask]
        return correspondences, valid_distances
    
    def estimate_transform_svd(self, source_matched, target_matched, fix_rotation=False):
        """
        Estimate 2D rigid transform using SVD (Singular Value Decomposition).
        This finds the rotation and translation that best aligns the matched points.
        
        If fix_rotation is True, we only estimate translation (dx, dy).
        This is useful when we trust the gyro for rotation (which we do!).
        
        """
        if len(source_matched) < 3: return 0.0, 0.0, 0.0
        
        # Center the point clouds (subtract mean)
        source_center = np.mean(source_matched, axis=0)
        target_center = np.mean(target_matched, axis=0)
        
        if fix_rotation:
            # If rotation is fixed (we trust the gyro), just align centroids
            t = target_center - source_center
            return t[0], t[1], 0.0
        
        # Center the points
        source_centered = source_matched - source_center
        target_centered = target_matched - target_center
        
        # Compute cross-covariance matrix and do SVD
        H = source_centered.T @ target_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Handle reflection case (det(R) should be +1 not -1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Extract rotation angle and translation
        dtheta = np.arctan2(R[1, 0], R[0, 0])
        t = target_center - (R @ source_center)
        return t[0], t[1], dtheta
    
    def apply_transform(self, points, dx, dy, dtheta):
        """Apply a 2D rigid transform (rotation then translation) to points."""
        if len(points) == 0: return points
        c, s = np.cos(dtheta), np.sin(dtheta)
        R = np.array([[c, -s], [s, c]])  # 2D rotation matrix
        transformed = (R @ points.T).T
        transformed[:, 0] += dx
        transformed[:, 1] += dy
        return transformed
    
    def icp(self, source_scan, target_scan, initial_guess=(0, 0, 0), fix_rotation=False):
        """
        Main ICP algorithm. Iteratively:
        1. Find correspondences (nearest neighbors)
        2. Estimate transform from correspondences
        3. Apply transform to source points
        4. Repeat until convergence
        
        Returns (dx, dy, dtheta, confidence)
        """
        source = self.preprocess_scan(source_scan)
        target = self.preprocess_scan(target_scan)
        if len(source) < self.min_points or len(target) < self.min_points:
            return initial_guess + (0.0,)  # not enough points, return guess with 0 confidence
        
        dx, dy, dtheta = initial_guess
        # Start with the odometry guess (this helps ICP converge faster)
        transformed_source = self.apply_transform(source, dx, dy, dtheta)
        total_dx, total_dy, total_dtheta = dx, dy, dtheta
        
        prev_error = float('inf')
        for iteration in range(self.max_iterations):
            # Step 1: Find correspondences
            correspondences, distances = self.find_correspondences(transformed_source, target)
            if len(correspondences) < self.min_points:
                return (total_dx, total_dy, total_dtheta, 0.0)
            
            # Check for convergence - if error stopped changing, we're done
            mean_error = np.mean(distances)
            if abs(prev_error - mean_error) < self.tolerance: break
            prev_error = mean_error
            
            # Get the matched point pairs
            source_matched = transformed_source[correspondences[:, 0]]
            target_matched = target[correspondences[:, 1]]
            
            # Step 2: Estimate incremental transform
            inc_dx, inc_dy, inc_dtheta = self.estimate_transform_svd(
                source_matched, target_matched, fix_rotation=fix_rotation
            )
            
            # Early exit if transform is tiny
            if (abs(inc_dx) < self.tolerance and abs(inc_dy) < self.tolerance and abs(inc_dtheta) < self.tolerance): break
            
            # Step 3: Apply transform and accumulate
            transformed_source = self.apply_transform(transformed_source, inc_dx, inc_dy, inc_dtheta)
            total_dx += inc_dx
            total_dy += inc_dy
            total_dtheta += inc_dtheta
        
        # Compute confidence score based on how well the scans aligned
        final_correspondences, final_distances = self.find_correspondences(transformed_source, target)
        if len(final_distances) > 0:
            mean_dist = np.mean(final_distances)
            correspondence_ratio = len(final_correspondences) / len(source)
            # Higher ratio + lower distance = higher confidence
            confidence = correspondence_ratio * (1.0 / (1.0 + mean_dist * 10))
            confidence = np.clip(confidence, 0.0, 1.0)
        else: confidence = 0.0
        return (total_dx, total_dy, total_dtheta, confidence)
    
    def match_scans(self, current_scan, previous_scan, odom_delta, fix_rotation=False):
        self.total_matches += 1
        dx, dy, dtheta, confidence = self.icp(
            current_scan, previous_scan, 
            initial_guess=odom_delta, 
            fix_rotation=fix_rotation
        )
        if confidence > 0.5: self.successful_matches += 1
        return dx, dy, dtheta, confidence


# =============================================================================
# Loop Closure Detection
# =============================================================================

class LoopClosureDetector:
    """
    Detects when the robot returns to a previously visited location.
    This helps correct accumulated drift in the trajectory.
    
    Uses scan signatures (basically a histogram of the lidar) to compare places.
    If the robot is close to an old pose AND the scans look similar, it's a loop!
    """
    def __init__(self, distance_threshold=0.5, scan_similarity_threshold=0.85, min_time_gap=100):
        self.distance_threshold = distance_threshold                # how close poses need to be (meters)
        self.scan_similarity_threshold = scan_similarity_threshold  # how similar scans need to be
        self.min_time_gap = min_time_gap                            # don't match with very recent poses (they're obviously similar)
    
    def compute_scan_signature(self, scan_points):
        """
        Create a compact representation of a scan for quick comparison.
        Uses a histogram of angles weighted by inverse distance.
        This is rotation-invariant which is nice for loop detection.
        """
        if len(scan_points) == 0: return None
        data = np.asarray(scan_points).reshape(-1, 3)
        x, y = data[:, 0], data[:, 1]
        angles = np.arctan2(y, x)
        distances = np.sqrt(x**2 + y**2)
        valid_mask = distances > 0.2  # ignore very close points
        angles = angles[valid_mask]
        distances = distances[valid_mask]
        if len(distances) == 0: return None
        # Create histogram weighted by 1/distance (closer walls contribute more)
        hist, _ = np.histogram(angles, bins=72, range=(-np.pi, np.pi), weights=1.0/distances)
        if np.sum(hist) > 0: hist = hist / np.sum(hist)  # normalize
        return hist
    
    def compare_scans(self, sig1, sig2):
        """
        Compare two scan signatures using cosine similarity.
        We try all rotations because the robot might be facing a different direction.
        Returns similarity score from 0 to 1.
        """
        if sig1 is None or sig2 is None: return 0.0
        max_similarity = 0.0
        norm1 = np.linalg.norm(sig1)
        norm2 = np.linalg.norm(sig2)
        if norm1 == 0 or norm2 == 0: return 0.0
        # Try all possible rotations of the histogram
        for shift in range(len(sig1)):
            rolled_sig2 = np.roll(sig2, shift)
            similarity = np.dot(sig1, rolled_sig2) / (norm1 * norm2)
            if similarity > max_similarity: max_similarity = similarity
        return max_similarity
    
    def detect_loop_closure(self, current_node, pose_graph):
        """
        Check if current pose matches any old pose.
        Returns (detected, match_index, similarity) tuple.
        """
        # Need enough poses to even check
        if len(pose_graph) < self.min_time_gap: return False, None, 0.0
        
        current_sig = self.compute_scan_signature(current_node.scan)
        if current_sig is None: return False, None, 0.0
        
        # Only check old poses (not recent ones - they'd obviously match)
        old_poses = pose_graph[:-self.min_time_gap]
        if len(old_poses) == 0: return False, None, 0.0
        
        # Find poses that are spatially close using KDTree
        positions = np.array([[n.x, n.y] for n in old_poses])
        tree = KDTree(positions)
        current_pos = np.array([current_node.x, current_node.y])
        indices = tree.query_ball_point(current_pos, self.distance_threshold)
        
        # Among close poses, find the one with most similar scan
        best_similarity = 0.0
        best_match_idx = None
        for idx in indices:
            candidate_node = old_poses[idx]
            similarity = self.compare_scans(current_sig, self.compute_scan_signature(candidate_node.scan))
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = candidate_node.idx
        
        if best_similarity > self.scan_similarity_threshold:
            return True, best_match_idx, best_similarity
        return False, None, 0.0


# =============================================================================
# Pose Graph Optimization
# =============================================================================

class PoseGraphOptimizer:
    """
    Simple pose graph optimizer.
    
    When a loop closure is detected, we need to correct the trajectory.
    This uses a simple linear interpolation approach.
    
    Basically distributes the error evenly across all poses in the loop.
    """
    def __init__(self):
        self.optimization_strength = 0.15  # how much to correct (0-1), lower = more conservative
    
    def optimize_trajectory(self, pose_graph, loop_closure_from, loop_closure_to):
        """Distribute loop closure error across poses between from and to."""
        if loop_closure_from >= loop_closure_to: return pose_graph
        
        node_from = pose_graph[loop_closure_from]
        node_to = pose_graph[loop_closure_to]
        
        # Calculate the error (how far off we are)
        error_x = node_to.x - node_from.x
        error_y = node_to.y - node_from.y
        error_theta = node_to.theta - node_from.theta
        # Wrap theta to [-pi, pi]
        while error_theta > np.pi: error_theta -= 2 * np.pi
        while error_theta < -np.pi: error_theta += 2 * np.pi
        
        # Distribute error linearly across the loop
        num_nodes = loop_closure_to - loop_closure_from
        for i in range(loop_closure_from, loop_closure_to + 1):
            # alpha goes from 0 to 1 as we go from start to end of loop
            alpha = (i - loop_closure_from) / num_nodes if num_nodes > 0 else 0
            alpha *= self.optimization_strength  # scale by how aggressive we want to be
            pose_graph[i].x -= alpha * error_x
            pose_graph[i].y -= alpha * error_y
            pose_graph[i].theta -= alpha * error_theta
        return pose_graph


# =============================================================================
# Occupancy Grid Mapping
# =============================================================================

class OccupancyMap:
    """
    Occupancy grid map using log-odds representation.
    
    Each cell stores the log-odds of being occupied. This is better than
    storing probabilities directly because we can just add observations
    instead of doing Bayesian updates (which involve multiplication).
    
    The map is centered at (0,0) in world coordinates.
    """
    def __init__(self, size_meters=20, resolution=0.1):
        self.resolution = resolution  # meters per cell
        self.size_meters = size_meters
        self.grid_size = int(size_meters / resolution)  # number of cells
        self.center = self.grid_size // 2  # center cell index
        
        # Log-odds grid (0 = unknown, positive = occupied, negative = free)
        self.log_odds = np.zeros((self.grid_size, self.grid_size))
        
        # Log-odds updates for observations
        # These values determine how confident we are in each observation
        self.log_occ = np.log(0.9 / 0.1)   # ~2.2, we're pretty sure it's occupied
        self.log_free = np.log(0.3 / 0.7)  # ~-0.85, we think it's free
        self.log_max = 10.0   # clamp to avoid overflow
        self.log_min = -10.0
        self.max_lidar_range = 5.0  # ignore readings beyond this
    
    def world_to_grid(self, x_world, y_world):
        """Convert world coordinates (meters) to grid cell indices."""
        x_world = np.array(x_world)
        y_world = np.array(y_world)
        grid_x = ((x_world / self.resolution) + self.center).astype(int)
        grid_y = ((y_world / self.resolution) + self.center).astype(int)
        return np.clip(grid_x, 0, self.grid_size - 1), np.clip(grid_y, 0, self.grid_size - 1)
    
    def get_probability_grid(self, threshold=True, smooth=True):
        """Convert log-odds to probability for visualization."""
        log_odds_display = self.log_odds.copy()
        if smooth: log_odds_display = gaussian_filter(log_odds_display, sigma=1.0)
        # Convert log-odds to probability: p = 1 / (1 + exp(-log_odds))
        prob = 1.0 - (1.0 / (1.0 + np.exp(log_odds_display)))
        if threshold:
            # Clean binary map: 0 = free, 0.5 = unknown, 1 = occupied
            clean_map = np.full_like(prob, 0.5)
            clean_map[prob > 0.8] = 1.0
            clean_map[prob < 0.2] = 0.0
            return clean_map
        return prob
    
    def update_map(self, lidar_local_points, robot_x, robot_y, robot_theta):
        """
        Update the map with a new lidar scan.
        Uses ray tracing to mark free space and obstacles.
        """
        if len(lidar_local_points) == 0: return
        
        # Parse lidar data
        data = np.asarray(lidar_local_points).reshape(-1, 3)
        local_x, local_y = data[:, 0], data[:, 1]
        distances = np.sqrt(local_x**2 + local_y**2)
        valid = distances < self.max_lidar_range
        local_x, local_y = local_x[valid], local_y[valid]
        if len(local_x) == 0: return
        
        # Transform points from robot frame to world frame
        global_x = (local_x * np.cos(robot_theta) - local_y * np.sin(robot_theta)) + robot_x
        global_y = (local_x * np.sin(robot_theta) + local_y * np.cos(robot_theta)) + robot_y
        
        # Convert to grid coordinates
        gx, gy = self.world_to_grid(global_x, global_y)
        rx, ry = self.world_to_grid(robot_x, robot_y)
        
        # Use sets to avoid updating the same cell multiple times per scan
        # This is more efficient than updating each ray independently
        free_cells = set()
        occupied_cells = set()
        
        # Ray trace from robot to each hit point
        for x, y in zip(gx, gy):
            rr, cc = line(int(rx), int(ry), x, y)  # Bresenham's line algorithm
            if len(cc) > 0:
                # The last point is the obstacle (where lidar hit)
                occupied_cells.add((cc[-1], rr[-1]))
                # All previous points along the ray are free space
                for i in range(len(cc) - 1):
                    free_cells.add((cc[i], rr[i]))
        
        # If a cell is marked both free and occupied, obstacle wins
        free_cells -= occupied_cells
        
        # Bulk update the log-odds grid
        if free_cells:
            fc, fr = zip(*free_cells)
            self.log_odds[fc, fr] += self.log_free
        
        if occupied_cells:
            oc, orow = zip(*occupied_cells)
            self.log_odds[oc, orow] += self.log_occ
        
        # Clamp values to prevent overflow
        self.log_odds = np.clip(self.log_odds, self.log_min, self.log_max)
    
    def rebuild_from_trajectory(self, pose_graph):
        """
        Rebuild the entire map from scratch using the corrected trajectory.
        Called after loop closure optimization.
        """
        self.log_odds = np.zeros((self.grid_size, self.grid_size))
        for node in pose_graph:
            self.update_map(node.scan, node.x, node.y, node.theta)


# =============================================================================
# Navigation (Wall Following)
# =============================================================================

class SimpleWallFollower:
    """
    A simple wall following controller using PID.
    
    Uses sonar sensors to maintain a constant distance from a wall.
    Has three modes: FIND (looking for wall), FOLLOW (tracking wall), CORNER (escaping corners).
    
    The PID gains were tuned by trial and error... kd helps a lot with oscillation.
    """
    def __init__(self, base_speed=0.6, follow_side='left', target_dist=0.15, kp=2.0, ki=0.05, kd=0.7, find_threshold=0.70, front_threshold=0.30):
        # Basic movement parameters
        self.base_speed = base_speed
        self.follow_side = follow_side      # which side to keep the wall on
        self.target_dist = target_dist      # desired distance from wall (meters)
        
        # PID gains - these took forever to tune
        self.kp = kp                        # proportional: react to current error
        self.ki = ki                        # integral: eliminate steady-state error
        self.kd = kd                        # derivative: damp oscillations
        self.integral_error = 0
        self.last_error = 0
        
        # Threshold for detecting walls
        self.find_threshold = find_threshold
        self.front_threshold = front_threshold  # obstacle ahead threshold
        
        # State machine variables
        self.mode = 'FIND'  # FIND, FOLLOW, or CORNER
        self.corner_step = 0
        self.corner_phase = None
        self.wall_lost_counter = 0
        self.turning_to_follow = False
        
        # Timing for corner escape maneuver
        self.wall_lost_steps = 40
        self.back_duration = 20
        self.turn_duration = 50
        self.forward_duration = 6
    def get_sensor(self, dist, idx):
        """Get a single sonar reading, returning None if invalid."""
        v = dist[idx]
        if v is None or v > 1.0: return None  # max sonar range is ~1m
        return v
    
    def get_side_distance(self, dist):
        """Get average distance to wall on the follow side."""
        # Different sensors for left vs right wall following
        if self.follow_side == 'left': front_idx, back_idx = 0, 15
        else: front_idx, back_idx = 7, 8
        front = self.get_sensor(dist, front_idx)
        back = self.get_sensor(dist, back_idx)
        readings = [r for r in [front, back] if r is not None]
        if not readings: return None, None, None
        return sum(readings)/len(readings), front, back
    
    def get_front_distance(self, dist):
        """Get distance to obstacle in front."""
        front_readings = []
        for idx in [3, 4]:  # front-facing sonars
            reading = self.get_sensor(dist, idx)
            if reading is not None: front_readings.append(reading)
        return min(front_readings) if front_readings else 1.0
    
    def start_corner_escape(self):
        """Begin the corner escape maneuver."""
        self.mode = 'CORNER'
        self.corner_phase = 'back'
        self.corner_step = 0
        self.integral_error = 0  # reset PID
    
    def handle_corner(self):
        """
        Execute corner escape: back up, turn, go forward.
        This is a simple open-loop maneuver - not elegant but it works.
        """
        self.corner_step += 1
        if self.corner_phase == 'back':
            # Back up while turning slightly away from wall
            if self.follow_side == 'left': left, right = -0.50, -0.75
            else: left, right = -0.75, -0.50
            if self.corner_step >= self.back_duration:
                self.corner_phase = 'turn'
                self.corner_step = 0
            return left, right
        elif self.corner_phase == 'turn':
            # Turn away from the wall
            if self.follow_side == 'left': left, right = 0.80, -0.80
            else: left, right = -0.80, 0.80
            if self.corner_step >= self.turn_duration:
                self.corner_phase = 'forward'
                self.corner_step = 0
            return left, right
        elif self.corner_phase == 'forward':
            # Go forward briefly
            left = right = self.base_speed
            if self.corner_step >= self.forward_duration:
                self.mode = 'FIND'  # back to finding wall
                self.corner_phase = None
            return left, right
    
    def step(self, dist):
        """
        Main control loop. Takes sonar readings, returns (left_speed, right_speed).
        """
        side_avg, side_front, side_back = self.get_side_distance(dist)
        front_dist = self.get_front_distance(dist)
        
        # Emergency: obstacle ahead!
        if front_dist < self.front_threshold:
            if self.mode != 'CORNER': self.start_corner_escape()
        if self.mode == 'CORNER': return self.handle_corner()
        
        # FIND mode: look for a wall
        if self.mode == 'FIND':
            if side_avg is not None and side_avg < self.find_threshold:
                self.mode = 'FOLLOW'
                self.integral_error = 0
                self.last_error = 0
            else: return self.base_speed, self.base_speed  # go straight until we find a wall
        # Wall lost - try to turn back towards it
        if side_avg is None or side_avg > 0.80:
            self.wall_lost_counter += 1
            if self.wall_lost_counter < self.wall_lost_steps:
                # Turn towards the wall side
                self.turning_to_follow = True
                if self.follow_side == 'left': left_speed, right_speed = 0.10, 0.50
                else: left_speed, right_speed = 0.50, 0.10
                return left_speed, right_speed
            else:
                # Give up, go back to FIND mode
                self.mode = 'FIND'
                self.integral_error = 0
                self.wall_lost_counter = 0
                self.turning_to_follow = False
                return self.base_speed, self.base_speed
        else:
            self.wall_lost_counter = 0
            self.turning_to_follow = False
        
        # === PID Controller ===
        # Distance error: positive means too far from wall
        distance_error = side_avg - self.target_dist
        
        # Angle error: are we parallel to the wall?
        if side_front is not None and side_back is not None: 
            angle_error = (side_front - side_back) / 0.18  # normalize by sensor spacing
        else: angle_error = 0
        
        # Integral term (with anti-windup)
        self.integral_error += distance_error
        self.integral_error = max(-0.5, min(0.5, self.integral_error))
        
        # Derivative term
        derivative_error = distance_error - self.last_error
        self.last_error = distance_error
        
        # Compute steering command
        steering = (self.kp * distance_error + self.ki * self.integral_error + self.kd * angle_error)
        steering = max(-0.8, min(0.8, steering))  # clamp steering
        
        # Convert steering to differential drive speeds
        forward = self.base_speed
        if self.follow_side == 'left':
            left_speed, right_speed = forward - steering, forward + steering
        else:
            left_speed, right_speed = forward + steering, forward - steering
        return max(0.1, min(1.2, left_speed)), max(0.1, min(1.2, right_speed))


# =============================================================================
# Main SLAM Class
# =============================================================================

class SLAMWithScanMatching:
    """
    Main SLAM system combining:
    - Odometry prediction (from wheel encoders + gyro)
    - ICP scan matching for correction
    - Pose graph for trajectory storage
    - Occupancy grid mapping
    - Loop closure detection and correction
    
    We trust the gyro for rotation and use ICP mainly to correct translation errors. 
    This is way more stable than letting ICP estimate both rotation AND translation.
    """
    def __init__(self, initial_pose=(0, 0, 0), use_scan_matching=True, use_loop_closure=True):
        # Components
        self.mapper = OccupancyMap(size_meters=20, resolution=0.1)
        self.loop_detector = LoopClosureDetector(distance_threshold=2.0, scan_similarity_threshold=0.96, min_time_gap=150)
        self.optimizer = PoseGraphOptimizer()
        self.pose_graph = []                    # list of PoseNodes
        self.current_pose = list(initial_pose)  # [x, y, theta]
        
        # Scan matcher configuration
        self.scan_matcher = ScanMatcher(max_iterations=20, tolerance=1e-4, max_correspondence_dist=0.3, min_points=10)
        self.last_scan = None
        self.last_odom_pose = None
        self.keyframe_odom_pose = list(initial_pose)  # odometry at last keyframe
        
        # Keyframe thresholds - only add a new node if we've moved enough
        self.keyframe_dist_thresh = 0.15   # meters
        self.keyframe_angle_thresh = 0.1   # radians (~6 degrees)
        
        # Validation gate - reject ICP results that disagree too much with odometry
        # This prevents bad matches from corrupting the trajectory
        self.gate_dist_thresh = 0.25      
        self.gate_angle_thresh = 0.15     
        self.confidence_threshold = 0.60  # minimum ICP confidence to accept
        
        # Feature flags
        self.use_scan_matching = use_scan_matching
        self.use_loop_closure = use_loop_closure
        
        # Statistics (useful for debugging)
        self.scan_match_used_count = 0
        self.odom_fallback_count = 0
        self.loop_closures_detected = 0

    def analyze_environment(self, points):
        """
        Detect if we're in a corridor .
        
        In corridors, ICP tends to slide along the walls because there's no
        features to lock onto in the direction of travel. In these cases,
        we're better off trusting odometry.
        
        Uses PCA to find the aspect ratio of the point cloud.
        """
        if len(points) < 10: return False
        data = np.asarray(points).reshape(-1, 3) 
        xy = data[:, 0:2]
        if xy.shape[0] < 5: return False
        
        # Compute covariance and eigenvalues
        cov = np.cov(xy.T)
        evals, evecs = np.linalg.eig(cov)
        idx = evals.argsort()[::-1]
        evals = evals[idx]
        
        # High aspect ratio = corridor
        if evals[1] > 0 and evals[0] / evals[1] > 10.0:
            return True
        return False

    def update(self, lidar_local_points, odom_x, odom_y, odom_theta):
        """
        Main SLAM update function. Called every time step with new sensor data.
        
        The overall flow is:
        1. If first call, initialize everything
        2. Predict pose from odometry (dead reckoning)
        3. Check if we should create a new keyframe
        4. If keyframe, run scan matching to refine pose
        5. Update pose graph and map
        6. Check for loop closures
        
        Returns: (x, y, theta) corrected pose
        """
        # === Step 1: First frame initialization ===
        if self.last_odom_pose is None:
            self.last_odom_pose = (odom_x, odom_y, odom_theta)
            self.keyframe_odom_pose = (odom_x, odom_y, odom_theta)
            self.last_scan = copy.deepcopy(lidar_local_points)
            node = PoseNode(0, self.current_pose[0], self.current_pose[1], self.current_pose[2], copy.deepcopy(lidar_local_points))
            self.pose_graph.append(node)
            return tuple(self.current_pose)

        # === Step 2: Prediction (Dead Reckoning) ===
        # Calculate how much we moved since last update
        step_global_dx = odom_x - self.last_odom_pose[0]
        step_global_dy = odom_y - self.last_odom_pose[1]
        step_dtheta = odom_theta - self.last_odom_pose[2]
        step_dtheta = (step_dtheta + np.pi) % (2 * np.pi) - np.pi  # normalize angle
        
        # Convert to local frame (robot's perspective)
        prev_theta = self.last_odom_pose[2]
        c, s = np.cos(prev_theta), np.sin(prev_theta)
        local_dx = step_global_dx * c + step_global_dy * s
        local_dy = -step_global_dx * s + step_global_dy * c
        
        # Apply motion to current pose estimate
        curr_theta = self.current_pose[2]
        c_curr, s_curr = np.cos(curr_theta), np.sin(curr_theta)
        
        self.current_pose[0] += local_dx * c_curr - local_dy * s_curr
        self.current_pose[1] += local_dx * s_curr + local_dy * c_curr
        self.current_pose[2] += step_dtheta
        self.current_pose[2] = (self.current_pose[2] + np.pi) % (2 * np.pi) - np.pi
        
        self.last_odom_pose = (odom_x, odom_y, odom_theta)

        # === Step 3: Keyframe Decision ===
        # Only create new nodes when we've moved far enough
        kf_dx = odom_x - self.keyframe_odom_pose[0]
        kf_dy = odom_y - self.keyframe_odom_pose[1]
        kf_dist = np.sqrt(kf_dx**2 + kf_dy**2)
        kf_dtheta = abs((odom_theta - self.keyframe_odom_pose[2] + np.pi) % (2*np.pi) - np.pi)
        
        # Exit if we haven't moved enough or don't have enough lidar points
        if (kf_dist < self.keyframe_dist_thresh and kf_dtheta < self.keyframe_angle_thresh) or len(lidar_local_points) < 10:
            return tuple(self.current_pose)

        # === Step 4: Prepare ICP Initial Guess ===
        # Transform odometry delta to local frame of last keyframe
        kf_theta_raw = self.keyframe_odom_pose[2]
        c_kf, s_kf = np.cos(kf_theta_raw), np.sin(kf_theta_raw)
        
        global_kf_dx = odom_x - self.keyframe_odom_pose[0]
        global_kf_dy = odom_y - self.keyframe_odom_pose[1]
        global_kf_dtheta = odom_theta - self.keyframe_odom_pose[2]
        global_kf_dtheta = (global_kf_dtheta + np.pi) % (2 * np.pi) - np.pi
        
        guess_dx = global_kf_dx * c_kf + global_kf_dy * s_kf
        guess_dy = -global_kf_dx * s_kf + global_kf_dy * c_kf
        guess_dtheta = global_kf_dtheta

        # === Step 5: Run Scan Matching ===
        # Start with odometry estimate
        scan_dx, scan_dy, scan_dtheta = guess_dx, guess_dy, guess_dtheta 
        
        if self.use_scan_matching:
            # Check if we're in a corridor
            is_corridor = self.analyze_environment(lidar_local_points)

            if is_corridor:
                # CORRIDOR LOCK: Don't trust ICP in hallways
                # It tends to "slide" along the walls because there's nothing
                # to anchor the position in the direction of travel
                scan_dx, scan_dy, scan_dtheta = guess_dx, guess_dy, guess_dtheta
                self.odom_fallback_count += 1
            else:
                # GYRO-ASSISTED ICP:
                # Key insight: The gyro is pretty accurate for rotation, but wheel
                # odometry drifts for translation. So we fix rotation and let ICP
                # only correct the translation (dx, dy).
                res_dx, res_dy, res_dtheta, confidence = self.scan_matcher.match_scans(
                    lidar_local_points, 
                    self.last_scan,
                    odom_delta=(guess_dx, guess_dy, guess_dtheta),
                    fix_rotation=True  # Trust the gyro for rotation!
                )
                
                # Validation Gate: Reject matches that disagree too much with odometry
                # This prevents occasional bad ICP matches from corrupting trajectory
                error_x = abs(res_dx - guess_dx)
                error_y = abs(res_dy - guess_dy)
                
                gate_passed = (error_x < self.gate_dist_thresh) and \
                              (error_y < self.gate_dist_thresh)
                
                if confidence > self.confidence_threshold and gate_passed:
                    # Accept ICP result
                    scan_dx, scan_dy, scan_dtheta = res_dx, res_dy, res_dtheta
                    self.scan_match_used_count += 1
                else:
                    # Fall back to odometry
                    self.odom_fallback_count += 1
        
        # === Step 6: Update Pose Graph ===
        # Transform local motion to global frame and add to last pose
        last_node = self.pose_graph[-1]
        c_node, s_node = np.cos(last_node.theta), np.sin(last_node.theta)
        
        new_x = last_node.x + (scan_dx * c_node - scan_dy * s_node)
        new_y = last_node.y + (scan_dx * s_node + scan_dy * c_node)
        new_theta = last_node.theta + scan_dtheta
        new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi
        
        # Update state
        self.current_pose = [new_x, new_y, new_theta]
        self.last_scan = copy.deepcopy(lidar_local_points)
        self.keyframe_odom_pose = (odom_x, odom_y, odom_theta)
        
        # Add new node to pose graph
        node = PoseNode(len(self.pose_graph), new_x, new_y, new_theta, copy.deepcopy(lidar_local_points))
        self.pose_graph.append(node)
        
        # Update occupancy grid map
        self.mapper.update_map(lidar_local_points, new_x, new_y, new_theta)
        
        # === Step 7: Loop Closure ===
        # Check periodically if we've returned to a previously visited location
        if self.use_loop_closure and len(self.pose_graph) % 10 == 0 and len(self.pose_graph) > 50:
             loop_detected, match_idx, _ = self.loop_detector.detect_loop_closure(node, self.pose_graph)
             if loop_detected:
                 # Found a loop! Optimize the trajectory to close it
                 self.pose_graph = self.optimizer.optimize_trajectory(self.pose_graph, match_idx, node.idx)
                 # Rebuild map with corrected trajectory
                 self.mapper.rebuild_from_trajectory(self.pose_graph)
                 latest = self.pose_graph[-1]
                 self.current_pose = [latest.x, latest.y, latest.theta]
                 self.loop_closures_detected += 1

        return tuple(self.current_pose)
    
    def get_trajectory(self):
        """Get the full trajectory for visualization."""
        if len(self.pose_graph) == 0: return [], []
        xs = [node.x for node in self.pose_graph]
        ys = [node.y for node in self.pose_graph]
        return xs, ys


# =============================================================================
# Main Function
# =============================================================================

def main(args=None):
    import sys
    
    # Check for mode
    USE_GROUND_TRUTH = False
    USE_SCAN_MATCHING = True
    USE_LOOP_CLOSURE = True
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--no-scan-matching':
            USE_SCAN_MATCHING = False
            print("\n*** RUNNING WITHOUT SCAN MATCHING (odometry only) ***\n")
        elif sys.argv[1] == '--no-loop-closure':
            USE_LOOP_CLOSURE = False
            print("\n*** RUNNING WITHOUT LOOP CLOSURE ***\n")
        elif sys.argv[1] == '--ground-truth':
            USE_GROUND_TRUTH = True
            USE_SCAN_MATCHING = False
            USE_LOOP_CLOSURE = False
            print("\n*** RUNNING WITH GROUND TRUTH POSE (from simulator) ***")
            print("*** This shows what the map SHOULD look like with perfect localization ***\n")
    
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX', use_lidar=True)
    coppelia.start_simulation(stepping=True)
    
    wf = SimpleWallFollower(
        base_speed=0.6,
        follow_side='left',
        target_dist=0.25,
        kp=2.0,
        ki=0.05,
        kd=0.7,
        find_threshold=0.50,
        front_threshold=0.30
    )
    
    # Get initial ground truth pose
    gt_x, gt_y, gt_theta = robot.get_ground_truth_pose()
    initial_gt_pose = (gt_x, gt_y, gt_theta)
    initial_pose = (0, 0, 0)
    print(f"\nInitial truth position: ({gt_x:.2f}, {gt_y:.2f}, {np.degrees(gt_theta):.1f}°)")
    
    slam = SLAMWithScanMatching(
        initial_pose=initial_pose,
        use_scan_matching=USE_SCAN_MATCHING,
        use_loop_closure=USE_LOOP_CLOSURE
    )
    
    # Track trajectories for comparison
    ground_truth_trajectory = []
    
    # Visualization
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    img = ax1.imshow(slam.mapper.get_probability_grid(threshold=True, smooth=True), 
                     cmap='gray', vmin=0, vmax=1, origin='lower')
    ax1.set_title("SLAM Map")
    ax1.set_xlabel("Grid X")
    ax1.set_ylabel("Grid Y")
    
    ax2.set_title("Robot Trajectory")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.grid(True)
    ax2.axis('equal')
    
    iteration = 0
    print("\nModes:")
    print("  python assignment_2_scan_match.py                    - Full SLAM with scan matching")
    print("  python assignment_2_scan_match.py --no-scan-matching - Odometry only")
    print("  python assignment_2_scan_match.py --no-loop-closure  - Scan match without loop closure")
    print("  python assignment_2_scan_match.py --ground-truth     - Use ground truth (perfect)")
    print("="*70 + "\n")
    
    while coppelia.is_running():
        # Call update_odometry every step
        robot.update_odometry()
        
        # Get odometry estimate
        odom_x, odom_y, odom_theta = robot.get_estimated_pose()
        
        # Get ground truth (just for comparison)
        gt_x, gt_y, gt_theta = robot.get_ground_truth_pose()
        ground_truth_trajectory.append((gt_x, gt_y))
        
        # Get lidar data
        raw_lidar = robot.read_lidar_data()
        
        # JUST FOR TESTING
        if USE_GROUND_TRUTH:
            # Directly use ground truth
            corrected_x, corrected_y, corrected_theta = gt_x, gt_y, gt_theta
            slam.mapper.update_map(raw_lidar, gt_x, gt_y, gt_theta)
            node = PoseNode(len(slam.pose_graph), gt_x, gt_y, gt_theta, copy.deepcopy(raw_lidar))
            slam.pose_graph.append(node)
        else:
            # SLAM
            corrected_x, corrected_y, corrected_theta = slam.update(
                raw_lidar, odom_x, odom_y, odom_theta
            )
        
        # Visualization
        if iteration % 10 == 0:
            prob_grid = slam.mapper.get_probability_grid(threshold=True, smooth=True)
            img.set_data(prob_grid)
            
            rx, ry = slam.mapper.world_to_grid(corrected_x, corrected_y)
            
            [p.remove() for p in ax1.lines]
            ax1.plot(rx, ry, 'go', markersize=8, label='Current')
            ax1.legend()
            
            # Update trajectory plot
            traj_x, traj_y = slam.get_trajectory()
            gt_traj_x = [p[0] for p in ground_truth_trajectory]
            gt_traj_y = [p[1] for p in ground_truth_trajectory]
            
            # If using SLAM, align GT to start at 0,0,0 for comparison
            if not USE_GROUND_TRUTH:
                x0, y0, theta0 = initial_gt_pose
                c = np.cos(-theta0)
                s = np.sin(-theta0)
                
                aligned_gt_x = []
                aligned_gt_y = []
                for gx, gy in zip(gt_traj_x, gt_traj_y):
                    dx = gx - x0
                    dy = gy - y0
                    aligned_gt_x.append(dx * c - dy * s)
                    aligned_gt_y.append(dx * s + dy * c)
                gt_traj_x = aligned_gt_x
                gt_traj_y = aligned_gt_y
                
                # Also align current GT marker
                curr_dx = gt_x - x0
                curr_dy = gt_y - y0
                curr_gt_x = curr_dx * c - curr_dy * s
                curr_gt_y = curr_dx * s + curr_dy * c
            else:
                curr_gt_x = gt_x
                curr_gt_y = gt_y
            
            ax2.clear()
            ax2.plot(gt_traj_x, gt_traj_y, 'g-', linewidth=2, alpha=0.8, label='Ground Truth')
            ax2.plot(traj_x, traj_y, 'b-', linewidth=1, alpha=0.7, label='SLAM Trajectory')
            ax2.plot(corrected_x, corrected_y, 'bo', markersize=8, label='Current SLAM')
            ax2.plot(curr_gt_x, curr_gt_y, 'g*', markersize=10, label='Current GT')
            
            # Add scan matching stats to title
            if USE_SCAN_MATCHING and slam.scan_matcher.total_matches > 0:
                success_rate = 100.0 * slam.scan_matcher.successful_matches / slam.scan_matcher.total_matches
                ax2.set_title(f"Trajectory (Scan Match Success: {success_rate:.1f}%)")
            else:
                ax2.set_title("Trajectory Comparison")
            
            ax2.set_xlabel("X (m)")
            ax2.set_ylabel("Y (m)")
            ax2.grid(True)
            ax2.axis('equal')
            ax2.legend(loc='upper left', fontsize=8)
            
            fig.canvas.draw_idle()
            plt.pause(0.001)
        
        # Wall following control
        dist = robot.get_sonar()
        left_speed, right_speed = wf.step(dist)
        robot.set_speed(left_speed, right_speed)
        
        # Step the simulation forward
        coppelia.step()
        
        if iteration % 50 == 0:
            print(f"Iter {iteration:04d} | GT: ({gt_x:.2f}, {gt_y:.2f}, {np.degrees(gt_theta):.0f}°) | "
                  f"SLAM: ({corrected_x:.2f}, {corrected_y:.2f}, {np.degrees(corrected_theta):.0f}°)")
        
        iteration += 1
        time.sleep(0.01)

    coppelia.stop_simulation()


if __name__ == "__main__":
    main()