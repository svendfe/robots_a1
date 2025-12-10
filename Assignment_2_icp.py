import robotica
import time
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
import copy


class PoseNode:
    """A node in the pose graph representing a robot pose"""
    def __init__(self, idx, x, y, theta, scan):
        self.idx = idx
        self.x = x
        self.y = y
        self.theta = theta
        self.scan = scan  # Store lidar scan for loop closure detection


class ScanMatcher:
    """2D ICP (Iterative Closest Point) scan matching for odometry correction"""
    
    def __init__(self, max_iterations=20, tolerance=1e-4, 
                 max_correspondence_dist=0.3, min_points=10):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.max_correspondence_dist = max_correspondence_dist
        self.min_points = min_points
        
        # Statistics
        self.total_matches = 0
        self.successful_matches = 0
    
    def preprocess_scan(self, scan_points, max_range=4.0, min_range=0.1, downsample_factor=2):
        """Convert raw scan to numpy array, filter and downsample"""
        if len(scan_points) == 0:
            return np.array([]).reshape(0, 2)
        
        # Convert to numpy array
        data = np.array(scan_points).reshape(-1, 3)
        x, y = data[:, 0], data[:, 1]
        
        # Filter by range
        distances = np.sqrt(x**2 + y**2)
        valid = (distances > min_range) & (distances < max_range)
        points = np.column_stack([x[valid], y[valid]])
        
        # Downsample for efficiency
        if len(points) > 0 and downsample_factor > 1:
            points = points[::downsample_factor]
        
        return points
    
    def find_correspondences(self, source_points, target_points):
        """Find nearest neighbors using KDTree"""
        if len(source_points) == 0 or len(target_points) == 0:
            return np.array([]).reshape(0, 2), np.array([])
        
        # Build KDTree of target points
        tree = KDTree(target_points)
        
        # Query for each source point
        distances, indices = tree.query(source_points)
        
        # Filter by maximum correspondence distance
        valid_mask = distances < self.max_correspondence_dist
        
        # Create array of correspondence pairs [source_idx, target_idx]
        source_indices = np.where(valid_mask)[0]
        target_indices = indices[valid_mask]
        
        correspondences = np.column_stack([source_indices, target_indices])
        valid_distances = distances[valid_mask]
        
        return correspondences, valid_distances
    
    def estimate_transform_svd(self, source_matched, target_matched):
        """Estimate 2D rigid transform (dx, dy, dtheta) using SVD"""
        if len(source_matched) < 3:
            return 0.0, 0.0, 0.0
        
        # Center the point clouds
        source_center = np.mean(source_matched, axis=0)
        target_center = np.mean(target_matched, axis=0)
        
        source_centered = source_matched - source_center
        target_centered = target_matched - target_center
        
        # Compute covariance matrix
        H = source_centered.T @ target_centered
        
        # SVD decomposition
        U, _, Vt = np.linalg.svd(H)
        
        # Compute rotation matrix
        R = Vt.T @ U.T
        
        # Handle reflection case (determinant should be 1, not -1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Extract rotation angle
        dtheta = np.arctan2(R[1, 0], R[0, 0])
        
        # Compute translation
        t = target_center - (R @ source_center)
        dx, dy = t[0], t[1]
        
        return dx, dy, dtheta
    
    def apply_transform(self, points, dx, dy, dtheta):
        """Apply 2D rigid transform to points"""
        if len(points) == 0:
            return points
        
        # Rotation matrix
        c = np.cos(dtheta)
        s = np.sin(dtheta)
        R = np.array([[c, -s], [s, c]])
        
        # Apply rotation then translation
        transformed = (R @ points.T).T
        transformed[:, 0] += dx
        transformed[:, 1] += dy
        
        return transformed
    
    def icp(self, source_scan, target_scan, initial_guess=(0, 0, 0)):
        """
        Run ICP algorithm to align source scan to target scan.
        
        Returns: (dx, dy, dtheta, confidence)
        """
        # Preprocess scans
        source = self.preprocess_scan(source_scan)
        target = self.preprocess_scan(target_scan)
        
        # Check minimum points
        if len(source) < self.min_points or len(target) < self.min_points:
            return initial_guess + (0.0,)  # Return with zero confidence
        
        # Apply initial guess to source
        dx, dy, dtheta = initial_guess
        transformed_source = self.apply_transform(source, dx, dy, dtheta)
        
        # Track cumulative transform
        total_dx, total_dy, total_dtheta = dx, dy, dtheta
        
        # ICP iterations
        prev_error = float('inf')
        for iteration in range(self.max_iterations):
            # Find correspondences
            correspondences, distances = self.find_correspondences(
                transformed_source, target
            )
            
            # Check if we have enough correspondences
            if len(correspondences) < self.min_points:
                confidence = 0.0
                return (total_dx, total_dy, total_dtheta, confidence)
            
            # Compute mean error
            mean_error = np.mean(distances)
            
            # Check convergence
            if abs(prev_error - mean_error) < self.tolerance:
                break
            prev_error = mean_error
            
            # Get matched point pairs
            source_matched = transformed_source[correspondences[:, 0]]
            target_matched = target[correspondences[:, 1]]
            
            # Estimate incremental transform
            inc_dx, inc_dy, inc_dtheta = self.estimate_transform_svd(
                source_matched, target_matched
            )
            
            # Check if incremental transform is too small
            if (abs(inc_dx) < self.tolerance and 
                abs(inc_dy) < self.tolerance and 
                abs(inc_dtheta) < self.tolerance):
                break
            
            # Apply incremental transform
            transformed_source = self.apply_transform(
                transformed_source, inc_dx, inc_dy, inc_dtheta
            )
            
            # Update total transform
            total_dx += inc_dx
            total_dy += inc_dy
            total_dtheta += inc_dtheta
        
        # Compute confidence based on final alignment quality
        # Good alignment = low mean distance and high number of correspondences
        final_correspondences, final_distances = self.find_correspondences(
            transformed_source, target
        )
        
        if len(final_distances) > 0:
            mean_dist = np.mean(final_distances)
            correspondence_ratio = len(final_correspondences) / len(source)
            
            # Confidence metric: higher when distance is low and many points match
            confidence = correspondence_ratio * (1.0 / (1.0 + mean_dist * 10))
            confidence = np.clip(confidence, 0.0, 1.0)
        else:
            confidence = 0.0
        
        return (total_dx, total_dy, total_dtheta, confidence)
    
    def match_scans(self, current_scan, previous_scan, odom_delta):
        """
        Match two consecutive scans using ICP with odometry as initial guess.
        
        Args:
            current_scan: Current lidar scan (raw points)
            previous_scan: Previous lidar scan (raw points)
            odom_delta: Odometry-based motion estimate (dx, dy, dtheta)
        
        Returns: (dx, dy, dtheta, confidence)
        """
        self.total_matches += 1
        
        # Run ICP with odometry as initial guess
        dx, dy, dtheta, confidence = self.icp(
            current_scan, previous_scan, initial_guess=odom_delta
        )
        
        if confidence > 0.5:
            self.successful_matches += 1
        
        return dx, dy, dtheta, confidence


class LoopClosureDetector:
    """Detects when robot returns to previously visited location"""
    
    def __init__(self, distance_threshold=0.5, scan_similarity_threshold=0.85, min_time_gap=100):
        self.distance_threshold = distance_threshold
        self.scan_similarity_threshold = scan_similarity_threshold
        self.min_time_gap = min_time_gap
        
    def compute_scan_signature(self, scan_points):
        """Create a rotation-invariant descriptor from lidar scan"""
        if len(scan_points) == 0:
            return None
        
        data = np.array(scan_points).reshape(-1, 3)
        x = data[:, 0]
        y = data[:, 1]
        
        angles = np.arctan2(y, x)
        distances = np.sqrt(x**2 + y**2)
        
        valid_mask = distances > 0.2
        angles = angles[valid_mask]
        distances = distances[valid_mask]
        
        if len(distances) == 0:
            return None
        
        bins = 72
        hist, _ = np.histogram(angles, bins=bins, range=(-np.pi, np.pi), weights=1.0/distances)
        
        if np.sum(hist) > 0:
            hist = hist / np.sum(hist)
        
        return hist
    
    def compare_scans(self, sig1, sig2):
        """Compare two scan signatures using correlation (rotation invariant)"""
        if sig1 is None or sig2 is None:
            return 0.0
        
        max_similarity = 0.0
        norm1 = np.linalg.norm(sig1)
        norm2 = np.linalg.norm(sig2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        for shift in range(len(sig1)):
            rolled_sig2 = np.roll(sig2, shift)
            correlation = np.dot(sig1, rolled_sig2)
            similarity = correlation / (norm1 * norm2)
            if similarity > max_similarity:
                max_similarity = similarity
        
        return max_similarity
    
    def detect_loop_closure(self, current_node, pose_graph):
        """Check if current pose is close to any previous pose"""
        if len(pose_graph) < self.min_time_gap:
            return False, None, 0.0
        
        current_sig = self.compute_scan_signature(current_node.scan)
        if current_sig is None:
            return False, None, 0.0
        
        old_poses = pose_graph[:-self.min_time_gap]
        
        if len(old_poses) == 0:
            return False, None, 0.0
        
        positions = np.array([[n.x, n.y] for n in old_poses])
        tree = KDTree(positions)
        
        current_pos = np.array([current_node.x, current_node.y])
        indices = tree.query_ball_point(current_pos, self.distance_threshold)
        
        if len(indices) == 0:
            return False, None, 0.0
        
        best_similarity = 0.0
        best_match_idx = None
        
        for idx in indices:
            candidate_node = old_poses[idx]
            candidate_sig = self.compute_scan_signature(candidate_node.scan)
            
            similarity = self.compare_scans(current_sig, candidate_sig)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = candidate_node.idx
        
        if best_similarity > self.scan_similarity_threshold:
            return True, best_match_idx, best_similarity
        
        return False, None, 0.0


class PoseGraphOptimizer:
    """Simple pose graph optimization for loop closure"""
    
    def __init__(self):
        self.optimization_strength = 0.15
    
    def optimize_trajectory(self, pose_graph, loop_closure_from, loop_closure_to):
        """Distribute error correction across trajectory between loop closure points"""
        if loop_closure_from >= loop_closure_to:
            return pose_graph
        
        node_from = pose_graph[loop_closure_from]
        node_to = pose_graph[loop_closure_to]
        
        error_x = node_to.x - node_from.x
        error_y = node_to.y - node_from.y
        error_theta = node_to.theta - node_from.theta
        
        while error_theta > np.pi:
            error_theta -= 2 * np.pi
        while error_theta < -np.pi:
            error_theta += 2 * np.pi
        
        num_nodes = loop_closure_to - loop_closure_from
        
        for i in range(loop_closure_from, loop_closure_to + 1):
            alpha = (i - loop_closure_from) / num_nodes if num_nodes > 0 else 0
            alpha *= self.optimization_strength
            
            pose_graph[i].x -= alpha * error_x
            pose_graph[i].y -= alpha * error_y
            pose_graph[i].theta -= alpha * error_theta
        
        return pose_graph


class OccupancyMap:
    """Occupancy grid map"""
    
    def __init__(self, size_meters=20, resolution=0.1):
        self.resolution = resolution
        self.size_meters = size_meters
        self.grid_size = int(size_meters / resolution)
        self.center = self.grid_size // 2
        
        self.log_odds = np.zeros((self.grid_size, self.grid_size))
        
        self.prob_occ = 0.9
        self.prob_free = 0.3
        
        self.log_occ = np.log(self.prob_occ / (1 - self.prob_occ))
        self.log_free = np.log(self.prob_free / (1 - self.prob_free))
        
        self.log_max = 10.0
        self.log_min = -10.0
        
        self.max_lidar_range = 5.0

    def world_to_grid(self, x_world, y_world):
        x_world = np.array(x_world)
        y_world = np.array(y_world)
        
        grid_x = ((x_world / self.resolution) + self.center).astype(int)
        grid_y = ((y_world / self.resolution) + self.center).astype(int)
        
        grid_x = np.clip(grid_x, 0, self.grid_size - 1)
        grid_y = np.clip(grid_y, 0, self.grid_size - 1)
        
        return grid_x, grid_y
    
    def get_probability_grid(self, threshold=True, smooth=True):
        """Convert log-odds to probability for visualization"""
        log_odds_display = self.log_odds.copy()
        if smooth:
            log_odds_display = gaussian_filter(log_odds_display, sigma=1.0)
        
        prob = 1.0 - (1.0 / (1.0 + np.exp(log_odds_display)))
        
        if threshold:
            clean_map = np.full_like(prob, 0.5)
            clean_map[prob > 0.8] = 1.0
            clean_map[prob < 0.2] = 0.0
            return clean_map
        else:
            return prob

    def update_map(self, lidar_local_points, robot_x, robot_y, robot_theta):
        """Update map with given pose"""
        if len(lidar_local_points) == 0:
            return

        data = np.array(lidar_local_points).reshape(-1, 3)
        local_x = data[:, 0]
        local_y = data[:, 1]
        
        distances = np.sqrt(local_x**2 + local_y**2)
        valid_mask = distances < self.max_lidar_range
        
        local_x = local_x[valid_mask]
        local_y = local_y[valid_mask]
        
        if len(local_x) == 0:
            return
        
        global_x = (local_x * np.cos(robot_theta) - local_y * np.sin(robot_theta)) + robot_x
        global_y = (local_x * np.sin(robot_theta) + local_y * np.cos(robot_theta)) + robot_y
        
        gx, gy = self.world_to_grid(global_x, global_y)
        rx, ry = self.world_to_grid(robot_x, robot_y)
        
        for x, y in zip(gx, gy):
            rr, cc = line(int(rx), int(ry), x, y)
            if len(cc) > 1:
                self.log_odds[cc[:-1], rr[:-1]] += self.log_free
            if len(cc) > 0:
                self.log_odds[cc[-1], rr[-1]] += self.log_occ
        
        self.log_odds = np.clip(self.log_odds, self.log_min, self.log_max)
    
    def rebuild_from_trajectory(self, pose_graph):
        """Rebuild entire map from optimized pose graph"""
        self.log_odds = np.zeros((self.grid_size, self.grid_size))
        
        for node in pose_graph:
            self.update_map(node.scan, node.x, node.y, node.theta)


class SimpleWallFollower:
    """Wall following controller"""
    
    def __init__(self,
                 base_speed=0.6,
                 follow_side='left',
                 target_dist=0.15,
                 kp=2.0,
                 ki=0.05,
                 kd=0.7,
                 find_threshold=0.70,
                 front_threshold=0.30):
        
        self.base_speed = base_speed
        self.follow_side = follow_side
        self.target_dist = target_dist
        
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_error = 0
        self.last_error = 0
        
        self.find_threshold = find_threshold
        self.front_threshold = front_threshold
        
        self.mode = 'FIND'
        self.corner_step = 0
        self.corner_phase = None
        
        self.wall_lost_counter = 0
        self.turning_to_follow = False
        self.wall_lost_steps = 25
        
        self.back_duration = 10
        self.turn_duration = 10
        self.forward_duration = 8
        
    def get_sensor(self, dist, idx):
        v = dist[idx]
        if v is None or v > 1.0:
            return None
        return v
    
    def get_side_distance(self, dist):
        if self.follow_side == 'left':
            front_idx, back_idx = 0, 15
        else:
            front_idx, back_idx = 7, 8
        
        front = self.get_sensor(dist, front_idx)
        back = self.get_sensor(dist, back_idx)
        
        readings = [r for r in [front, back] if r is not None]
        if not readings:
            return None, None, None
        
        return sum(readings)/len(readings), front, back
    
    def get_front_distance(self, dist):
        front_readings = []
        for idx in [3, 4]:
            reading = self.get_sensor(dist, idx)
            if reading is not None:
                front_readings.append(reading)
        return min(front_readings) if front_readings else 1.0
    
    def start_corner_escape(self):
        self.mode = 'CORNER'
        self.corner_phase = 'back'
        self.corner_step = 0
        self.integral_error = 0
    
    def handle_corner(self):
        self.corner_step += 1
        
        if self.corner_phase == 'back':
            if self.follow_side == 'left':
                left, right = -0.10, -0.15
            else:
                left, right = -0.15, -0.10
            
            if self.corner_step >= self.back_duration:
                self.corner_phase = 'turn'
                self.corner_step = 0
            
            return left, right
        
        elif self.corner_phase == 'turn':
            if self.follow_side == 'left':
                left, right = 0.25, -0.25
            else:
                left, right = -0.25, 0.25
            
            if self.corner_step >= self.turn_duration:
                self.corner_phase = 'forward'
                self.corner_step = 0
            
            return left, right
        
        elif self.corner_phase == 'forward':
            left = right = self.base_speed
            
            if self.corner_step >= self.forward_duration:
                self.mode = 'FIND'
                self.corner_phase = None
            
            return left, right
    
    def step(self, dist):
        side_avg, side_front, side_back = self.get_side_distance(dist)
        front_dist = self.get_front_distance(dist)
        
        if front_dist < self.front_threshold:
            if self.mode != 'CORNER':
                self.start_corner_escape()
        
        if self.mode == 'CORNER':
            return self.handle_corner()
        
        if self.mode == 'FIND':
            if side_avg is not None and side_avg < self.find_threshold:
                self.mode = 'FOLLOW'
                self.integral_error = 0
                self.last_error = 0
            else:
                return self.base_speed, self.base_speed
        
        if side_avg is None or side_avg > 0.80:
            self.wall_lost_counter += 1
            
            if self.wall_lost_counter < self.wall_lost_steps:
                self.turning_to_follow = True
                
                if self.follow_side == 'left':
                    left_speed = 0.22
                    right_speed = 0.45
                else:
                    left_speed = 0.45
                    right_speed = 0.22
                
                return left_speed, right_speed
            else:
                self.mode = 'FIND'
                self.integral_error = 0
                self.wall_lost_counter = 0
                self.turning_to_follow = False
                return self.base_speed, self.base_speed
        else:
            self.wall_lost_counter = 0
            self.turning_to_follow = False
        
        distance_error = side_avg - self.target_dist
        
        if side_front is not None and side_back is not None:
            angle_error = (side_front - side_back) / 0.18
        else:
            angle_error = 0
        
        self.integral_error += distance_error
        self.integral_error = max(-0.5, min(0.5, self.integral_error))
        
        derivative_error = distance_error - self.last_error
        self.last_error = distance_error
        
        steering = (self.kp * distance_error + 
                   self.ki * self.integral_error + 
                   self.kd * angle_error)
        
        steering = max(-0.8, min(0.8, steering))
        forward = self.base_speed
        
        if self.follow_side == 'left':
            left_speed = forward - steering
            right_speed = forward + steering
        else:
            left_speed = forward + steering
            right_speed = forward - steering
        
        left_speed = max(0.1, min(1.2, left_speed))
        right_speed = max(0.1, min(1.2, right_speed))
        
        return left_speed, right_speed


class SLAMWithScanMatching:
    """SLAM using scan matching to correct odometry drift"""
    
    def __init__(self, initial_pose=(0, 0, 0), use_scan_matching=True, use_loop_closure=True):
        self.mapper = OccupancyMap(size_meters=20, resolution=0.1)
        self.loop_detector = LoopClosureDetector(
            distance_threshold=2.0,
            scan_similarity_threshold=0.96,
            min_time_gap=150
        )
        self.optimizer = PoseGraphOptimizer()
        
        # Pose graph
        self.pose_graph = []
        self.current_pose = list(initial_pose)
        
        # Scan matching
        self.scan_matcher = ScanMatcher(
            max_iterations=20,
            tolerance=1e-4,
            max_correspondence_dist=0.3,
            min_points=10
        )
        self.last_scan = None
        self.last_odom_pose = None
        
        # Options
        self.use_scan_matching = use_scan_matching
        self.use_loop_closure = use_loop_closure
        
        # Fusion weights (how much to trust each source)
        self.scan_match_weight = 0.3  # Trust scan matching more
        self.odom_weight = 0.7
        self.confidence_threshold = 0.5  # Minimum confidence to use scan matching
        
        # Statistics
        self.loop_closures_detected = 0
        self.last_optimization_time = 0
        self.scan_match_used_count = 0
        self.odom_fallback_count = 0
    
    # Add this helper method to the SLAM class
    def check_front_wall(self, lidar_points):
        """Returns True if there is a wall directly in front (< 2m)"""
        if len(lidar_points) == 0: return False
        data = np.array(lidar_points).reshape(-1, 3)
        # Filter for points in front cone (+/- 15 degrees)
        angle = np.arctan2(data[:, 1], data[:, 0])
        dist = np.sqrt(data[:, 0]**2 + data[:, 1]**2)
        front_points = (np.abs(angle) < np.radians(15)) & (dist < 2.0)
        return np.sum(front_points) > 10

    def update(self, lidar_local_points, odom_x, odom_y, odom_theta):
        """Robust Update with Direction Consistency and Strict Corridor Lock"""
        
        # 1. Initialize
        if self.last_odom_pose is None:
            self.last_odom_pose = (odom_x, odom_y, odom_theta)
            self.last_scan = copy.deepcopy(lidar_local_points)
            node = PoseNode(0, self.current_pose[0], self.current_pose[1], self.current_pose[2], copy.deepcopy(lidar_local_points))
            self.pose_graph.append(node)
            return tuple(self.current_pose)

        # 2. Calculate Odometry Delta (Global -> Local)
        global_odom_dx = odom_x - self.last_odom_pose[0]
        global_odom_dy = odom_y - self.last_odom_pose[1]
        odom_dtheta = odom_theta - self.last_odom_pose[2]
        odom_dtheta = (odom_dtheta + np.pi) % (2 * np.pi) - np.pi

        prev_theta = self.last_odom_pose[2]
        cos_th = np.cos(prev_theta)
        sin_th = np.sin(prev_theta)
        
        # Local Odometry (Robot Frame)
        local_odom_dx = global_odom_dx * cos_th + global_odom_dy * sin_th
        local_odom_dy = -global_odom_dx * sin_th + global_odom_dy * cos_th
        
        # 3. Detect Context
        has_front_wall = self.check_front_wall(lidar_local_points)
        is_turning = abs(odom_dtheta) > np.radians(0.5) # Sensitive turn detection
        
        # 4. Scan Matching
        target_dx = local_odom_dx
        target_dy = local_odom_dy
        target_dtheta = odom_dtheta
        
        if self.use_scan_matching and self.last_scan is not None and len(lidar_local_points) > 20:
            scan_dx, scan_dy, scan_dtheta, confidence = self.scan_matcher.match_scans(
                lidar_local_points, 
                self.last_scan,
                odom_delta=(local_odom_dx, local_odom_dy, odom_dtheta)
            )
            
            if confidence > self.confidence_threshold:
                # Rule 1: Rotation and Lateral (Y) are usually safe from ICP
                target_dy = scan_dy
                target_dtheta = scan_dtheta
                
                # Rule 2: FORWARD (X) HANDLING
                if has_front_wall and not is_turning:
                    # We have a wall to lock onto -> Trust ICP
                    target_dx = scan_dx
                else:
                    # Corridor or Turning -> Trust ODOMETRY 100% for X
                    # (Previous code used 0.1 * scan_dx, which caused the backward jumps)
                    target_dx = local_odom_dx

                # Rule 3: DIRECTION CONSISTENCY (The "No Back in Time" Fix)
                # If wheels say Forward (>1cm), and ICP says Backward (<0), REJECT ICP X.
                if local_odom_dx > 0.01 and scan_dx < -0.01:
                    target_dx = local_odom_dx # Force Odometry
                
                self.scan_match_used_count += 1
            else:
                self.odom_fallback_count += 1

        # 5. Clamping (Shock Absorber)
        # Prevents teleportation if rules above fail
        diff_x = target_dx - local_odom_dx
        diff_y = target_dy - local_odom_dy
        diff_th = target_dtheta - odom_dtheta
        diff_th = (diff_th + np.pi) % (2 * np.pi) - np.pi
        
        # Strict limit: Max 5cm correction per step
        max_correction = 0.05
        if abs(diff_x) > max_correction: diff_x = np.sign(diff_x) * max_correction
        if abs(diff_y) > max_correction: diff_y = np.sign(diff_y) * max_correction
        
        final_dx = local_odom_dx + diff_x
        final_dy = local_odom_dy + diff_y
        final_dtheta = odom_dtheta + diff_th

        # 6. Update Pose (Local -> Global)
        avg_theta = self.current_pose[2] + final_dtheta / 2.0
        
        global_dx = final_dx * np.cos(avg_theta) - final_dy * np.sin(avg_theta)
        global_dy = final_dx * np.sin(avg_theta) + final_dy * np.cos(avg_theta)
        
        self.current_pose[0] += global_dx
        self.current_pose[1] += global_dy
        self.current_pose[2] += final_dtheta
        self.current_pose[2] = (self.current_pose[2] + np.pi) % (2 * np.pi) - np.pi
        
        # State & Graph
        self.last_scan = copy.deepcopy(lidar_local_points)
        self.last_odom_pose = (odom_x, odom_y, odom_theta)
        
        node = PoseNode(len(self.pose_graph), self.current_pose[0], self.current_pose[1], self.current_pose[2], copy.deepcopy(lidar_local_points))
        self.pose_graph.append(node)
        
        # (Loop Closure Logic)
        if self.use_loop_closure and len(self.pose_graph) % 20 == 0 and len(self.pose_graph) > 150:
             loop_detected, match_idx, _ = self.loop_detector.detect_loop_closure(node, self.pose_graph)
             if loop_detected:
                 self.pose_graph = self.optimizer.optimize_trajectory(self.pose_graph, match_idx, node.idx)
                 self.mapper.rebuild_from_trajectory(self.pose_graph)
                 self.last_optimization_time = len(self.pose_graph)

        latest = self.pose_graph[-1]
        if len(self.pose_graph) - self.last_optimization_time > 5:
            self.mapper.update_map(lidar_local_points, latest.x, latest.y, latest.theta)

        return tuple(self.current_pose)
    

    def get_trajectory(self):
        """Get full trajectory for visualization"""
        if len(self.pose_graph) == 0:
            return [], []
        
        xs = [node.x for node in self.pose_graph]
        ys = [node.y for node in self.pose_graph]
        return xs, ys

    def check_front_obstacle(self, lidar_points, angle_range=np.radians(20), dist_threshold=2.0):
        """
        Check if there are features in front of the robot that allow reliable X-correction.
        Returns: True if valid features exist in front (not just a corridor).
        """
        if len(lidar_points) == 0: return False
        
        # Extract X, Y
        data = np.array(lidar_points).reshape(-1, 3)
        x = data[:, 0]
        y = data[:, 1]
        
        # Calculate angles and distances
        angles = np.arctan2(y, x)
        dists = np.sqrt(x**2 + y**2)
        
        # Look for points in the front cone (+/- 20 degrees)
        # and within reliable range (< 2.0 meters)
        front_mask = (np.abs(angles) < angle_range) & (dists < dist_threshold) & (dists > 0.1)
        
        # If we have enough points in front, we can trust ICP for forward motion
        return np.sum(front_mask) > 15

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
        target_dist=0.15,
        kp=2.0,
        ki=0.05,
        kd=0.7,
        find_threshold=0.50,
        front_threshold=0.30
    )
    
    # Get initial ground truth pose
    gt_x, gt_y, gt_theta = robot.get_ground_truth_pose()
    initial_gt_pose = (gt_x, gt_y, gt_theta)
    initial_pose = (gt_x, gt_y, gt_theta)
    print(f"\nInitial ground truth pose: ({gt_x:.2f}, {gt_y:.2f}, {np.degrees(gt_theta):.1f}°)")
    
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
    print("\n" + "="*70)
    print("SLAM WITH SCAN MATCHING")
    print("="*70)
    print("This system will:")
    print("  ✓ Use robot.update_odometry() for initial pose estimation")
    if USE_SCAN_MATCHING:
        print("  ✓ Apply ICP scan matching to correct odometry drift")
    else:
        print("  ✗ Scan matching disabled (odometry only)")
    print("  ✓ Build occupancy grid map from lidar scans")
    print("  ✓ Track robot trajectory")
    if USE_LOOP_CLOSURE:
        print("  ✓ Detect loop closures for global correction")
    else:
        print("  ✗ Loop closure disabled")
    print("\nModes:")
    print("  python assignment_2_scan_match.py                    - Full SLAM with scan matching")
    print("  python assignment_2_scan_match.py --no-scan-matching - Odometry only")
    print("  python assignment_2_scan_match.py --no-loop-closure  - Scan match without loop closure")
    print("  python assignment_2_scan_match.py --ground-truth     - Use ground truth (perfect)")
    print("="*70 + "\n")
    
    while coppelia.is_running():
        # IMPORTANT: Call update_odometry every step
        robot.update_odometry()
        
        # Get odometry estimate
        odom_x, odom_y, odom_theta = robot.get_estimated_pose()
        
        # Get ground truth for comparison
        gt_x, gt_y, gt_theta = robot.get_ground_truth_pose()
        ground_truth_trajectory.append((gt_x, gt_y))
        
        # Get lidar data
        raw_lidar = robot.read_lidar_data()
        
        # Use ground truth or SLAM based on mode
        if USE_GROUND_TRUTH:
            # Directly use ground truth
            corrected_x, corrected_y, corrected_theta = gt_x, gt_y, gt_theta
            slam.mapper.update_map(raw_lidar, gt_x, gt_y, gt_theta)
            node = PoseNode(len(slam.pose_graph), gt_x, gt_y, gt_theta, copy.deepcopy(raw_lidar))
            slam.pose_graph.append(node)
        else:
            # SLAM update with scan matching
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
    
    coppelia.stop_simulation()
    
    print("\n" + "="*70)
    print("SLAM STATISTICS")
    print("="*70)
    print(f"Total iterations: {iteration}")
    print(f"Pose graph nodes: {len(slam.pose_graph)}")
    
    if USE_SCAN_MATCHING:
        print(f"\nScan Matching:")
        print(f"  Total matches attempted: {slam.scan_matcher.total_matches}")
        print(f"  Successful matches: {slam.scan_matcher.successful_matches}")
        if slam.scan_matcher.total_matches > 0:
            success_rate = 100.0 * slam.scan_matcher.successful_matches / slam.scan_matcher.total_matches
            print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Scan match used: {slam.scan_match_used_count} times")
        print(f"  Odometry fallback: {slam.odom_fallback_count} times")
    
    if USE_LOOP_CLOSURE:
        print(f"\nLoop Closure:")
        print(f"  Detections: {slam.loop_closures_detected}")
        print(f"  Status: {'✓ ACTIVE' if slam.loop_closures_detected > 0 else '✗ not triggered'}")
    
    print(f"\n{'✓' if USE_SCAN_MATCHING else '✗'} Scan matching {'ENABLED' if USE_SCAN_MATCHING else 'DISABLED'}")
    print(f"{'✓' if USE_LOOP_CLOSURE else '✗'} Loop closure {'ENABLED' if USE_LOOP_CLOSURE else 'DISABLED'}")
    print("="*70)


if __name__ == "__main__":
    main()
