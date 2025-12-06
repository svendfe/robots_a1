import robotica
import time
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from collections import deque
import copy


class ICPScanMatcher:
    """
    Improved ICP with keyframe-based matching and better angular estimation.
    Uses multiple reference frames to reduce drift accumulation.
    """
    
    def __init__(self, max_iterations=50, tolerance=1e-6, max_correspondence_dist=0.3):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.max_correspondence_dist = max_correspondence_dist
        
        # Keyframe system - stable reference points
        self.keyframes = []  # List of (points, global_x, global_y, global_theta)
        self.keyframe_interval = 15  # Add keyframe every N frames
        self.max_keyframes = 20  # Keep recent keyframes only
        self.frame_count = 0
        
        # Local map built from recent scans
        self.local_map_points = None
        self.local_map_size = 5  # Number of scans to aggregate
        self.recent_scans = deque(maxlen=self.local_map_size)
        
        # Last scan for frame-to-frame
        self.last_scan_points = None
        
    def extract_points_from_scan(self, lidar_local_points, min_range=0.15, max_range=4.0):
        """Extract valid 2D points from lidar scan data with stricter filtering."""
        if len(lidar_local_points) == 0:
            return np.array([]).reshape(0, 2)
        
        data = np.array(lidar_local_points).reshape(-1, 3)
        x = data[:, 0]
        y = data[:, 1]
        
        # Filter by range
        distances = np.sqrt(x**2 + y**2)
        valid_mask = (distances > min_range) & (distances < max_range)
        
        points = np.column_stack([x[valid_mask], y[valid_mask]])
        return points
    
    def transform_points(self, points, dx, dy, dtheta):
        """Apply a 2D rigid transformation to points."""
        if len(points) == 0:
            return points
        
        cos_t = np.cos(dtheta)
        sin_t = np.sin(dtheta)
        
        R = np.array([[cos_t, -sin_t],
                      [sin_t, cos_t]])
        
        rotated = points @ R.T
        transformed = rotated + np.array([dx, dy])
        
        return transformed
    
    def find_correspondences_robust(self, source_points, target_points, reject_ratio=0.2):
        """
        Find correspondences with outlier rejection.
        Rejects worst matches to improve robustness.
        """
        if len(source_points) == 0 or len(target_points) == 0:
            return np.array([]), np.array([]), np.array([])
        
        tree = KDTree(target_points)
        distances, indices = tree.query(source_points, k=1)
        
        # Filter by maximum correspondence distance
        valid_mask = distances < self.max_correspondence_dist
        
        valid_source = source_points[valid_mask]
        valid_target = target_points[indices[valid_mask]]
        valid_distances = distances[valid_mask]
        
        # Reject worst matches (outliers)
        if len(valid_distances) > 20:
            threshold = np.percentile(valid_distances, (1 - reject_ratio) * 100)
            inlier_mask = valid_distances < threshold
            valid_source = valid_source[inlier_mask]
            valid_target = valid_target[inlier_mask]
            valid_distances = valid_distances[inlier_mask]
        
        return valid_source, valid_target, valid_distances
    
    def compute_transform_weighted(self, source_points, target_points, distances):
        """
        Compute transform with distance-based weighting.
        Closer correspondences get more weight.
        """
        if len(source_points) < 5 or len(target_points) < 5:
            return 0.0, 0.0, 0.0
        
        # Weight inversely proportional to distance
        weights = 1.0 / (distances + 0.01)
        weights = weights / np.sum(weights)
        
        # Weighted centroids
        centroid_source = np.average(source_points, axis=0, weights=weights)
        centroid_target = np.average(target_points, axis=0, weights=weights)
        
        # Center the points
        source_centered = source_points - centroid_source
        target_centered = target_points - centroid_target
        
        # Weighted cross-covariance matrix
        H = np.zeros((2, 2))
        for i in range(len(source_points)):
            H += weights[i] * np.outer(source_centered[i], target_centered[i])
        
        # SVD
        U, S, Vt = np.linalg.svd(H)
        
        # Compute rotation
        R = Vt.T @ U.T
        
        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        t = centroid_target - R @ centroid_source
        
        dtheta = np.arctan2(R[1, 0], R[0, 0])
        dx = t[0]
        dy = t[1]
        
        return dx, dy, dtheta
    
    def icp_robust(self, source_points, target_points, initial_guess=(0, 0, 0)):
        """
        Robust ICP with outlier rejection and weighted optimization.
        """
        if len(source_points) < 15 or len(target_points) < 15:
            return initial_guess, False, float('inf')
        
        dx, dy, dtheta = initial_guess
        current_source = self.transform_points(source_points, dx, dy, dtheta)
        
        prev_error = float('inf')
        best_result = (dx, dy, dtheta)
        best_error = float('inf')
        
        for iteration in range(self.max_iterations):
            # Find correspondences with outlier rejection
            matched_source, matched_target, distances = self.find_correspondences_robust(
                current_source, target_points, reject_ratio=0.15
            )
            
            if len(matched_source) < 10:
                return best_result, False, best_error
            
            # Compute weighted error
            current_error = np.mean(distances**2)
            
            # Track best result
            if current_error < best_error:
                best_error = current_error
                best_result = (dx, dy, dtheta)
            
            # Check convergence
            if abs(prev_error - current_error) < self.tolerance:
                return (dx, dy, dtheta), True, current_error
            
            prev_error = current_error
            
            # Compute weighted incremental transform
            ddx, ddy, ddtheta = self.compute_transform_weighted(
                matched_source, matched_target, distances
            )
            
            # Limit angular correction per iteration to avoid oscillation
            max_angle_step = 0.05  # ~3 degrees
            ddtheta = np.clip(ddtheta, -max_angle_step, max_angle_step)
            
            # Compose transformations
            cos_t = np.cos(dtheta)
            sin_t = np.sin(dtheta)
            
            new_dx = dx + cos_t * ddx - sin_t * ddy
            new_dy = dy + sin_t * ddx + cos_t * ddy
            new_dtheta = dtheta + ddtheta
            
            # Normalize angle
            while new_dtheta > np.pi:
                new_dtheta -= 2 * np.pi
            while new_dtheta < -np.pi:
                new_dtheta += 2 * np.pi
            
            dx, dy, dtheta = new_dx, new_dy, new_dtheta
            current_source = self.transform_points(source_points, dx, dy, dtheta)
        
        return best_result, False, best_error
    
    def build_local_map(self, current_pose):
        """
        Build a local map from recent keyframes for more stable matching.
        """
        if len(self.keyframes) == 0:
            return None
        
        # Use keyframes near current position
        current_x, current_y, current_theta = current_pose
        
        all_points = []
        for kf_points, kf_x, kf_y, kf_theta in self.keyframes[-8:]:  # Last 8 keyframes
            # Distance check
            dist = np.sqrt((kf_x - current_x)**2 + (kf_y - current_y)**2)
            if dist > 3.0:  # Skip keyframes too far away
                continue
            
            # Transform keyframe points to current frame
            # First to global, then to current local
            cos_kf = np.cos(kf_theta)
            sin_kf = np.sin(kf_theta)
            
            # To global
            global_points = kf_points @ np.array([[cos_kf, sin_kf], [-sin_kf, cos_kf]])
            global_points = global_points + np.array([kf_x, kf_y])
            
            # To current local frame
            cos_cur = np.cos(-current_theta)
            sin_cur = np.sin(-current_theta)
            
            local_points = global_points - np.array([current_x, current_y])
            local_points = local_points @ np.array([[cos_cur, sin_cur], [-sin_cur, cos_cur]])
            
            all_points.append(local_points)
        
        if len(all_points) == 0:
            return None
        
        return np.vstack(all_points)
    
    def match_scan(self, current_scan, odom_delta, current_global_pose):
        """
        Match current scan using both frame-to-frame and keyframe matching.
        
        Args:
            current_scan: Current lidar points (local frame)
            odom_delta: (dx, dy, dtheta) from odometry since last scan
            current_global_pose: Current estimated (x, y, theta) for keyframe building
            
        Returns:
            corrected_delta: Refined (dx, dy, dtheta)
            correction: How much ICP corrected the odometry
        """
        self.frame_count += 1
        current_points = self.extract_points_from_scan(current_scan)
        
        if len(current_points) < 15:
            return odom_delta, (0, 0, 0)
        
        # Initialize on first scan
        if self.last_scan_points is None or len(self.last_scan_points) < 15:
            self.last_scan_points = current_points
            # Add first keyframe
            self.keyframes.append((
                current_points.copy(),
                current_global_pose[0],
                current_global_pose[1],
                current_global_pose[2]
            ))
            return odom_delta, (0, 0, 0)
        
        # === Method 1: Frame-to-frame ICP ===
        refined_f2f, converged_f2f, error_f2f = self.icp_robust(
            current_points,
            self.last_scan_points,
            initial_guess=odom_delta
        )
        
        # === Method 2: Frame-to-local-map ICP (more stable) ===
        local_map = self.build_local_map(current_global_pose)
        
        if local_map is not None and len(local_map) > 50:
            # For map matching, initial guess is near-zero (already in local frame)
            refined_f2m, converged_f2m, error_f2m = self.icp_robust(
                current_points,
                local_map,
                initial_guess=(0, 0, 0)  # Small correction expected
            )
            
            # Combine f2m correction with odometry delta
            refined_f2m = (
                odom_delta[0] + refined_f2m[0],
                odom_delta[1] + refined_f2m[1],
                odom_delta[2] + refined_f2m[2]
            )
        else:
            refined_f2m = refined_f2f
            error_f2m = float('inf')
            converged_f2m = False
        
        # === Fuse results ===
        # Weight by inverse error (lower error = more trust)
        if error_f2f < 0.001:
            error_f2f = 0.001
        if error_f2m < 0.001:
            error_f2m = 0.001
        
        w_f2f = 1.0 / error_f2f if converged_f2f else 0.3 / error_f2f
        w_f2m = 1.0 / error_f2m if converged_f2m else 0.3 / error_f2m
        
        # Normalize weights
        total_w = w_f2f + w_f2m
        w_f2f /= total_w
        w_f2m /= total_w
        
        # Weighted average
        refined_delta = (
            w_f2f * refined_f2f[0] + w_f2m * refined_f2m[0],
            w_f2f * refined_f2f[1] + w_f2m * refined_f2m[1],
            w_f2f * refined_f2f[2] + w_f2m * refined_f2m[2]
        )
        
        # Sanity check: don't allow huge corrections
        max_trans_correction = 0.1  # 10cm max
        max_rot_correction = 0.15   # ~8.5 degrees max
        
        correction = (
            np.clip(refined_delta[0] - odom_delta[0], -max_trans_correction, max_trans_correction),
            np.clip(refined_delta[1] - odom_delta[1], -max_trans_correction, max_trans_correction),
            np.clip(refined_delta[2] - odom_delta[2], -max_rot_correction, max_rot_correction)
        )
        
        final_delta = (
            odom_delta[0] + correction[0],
            odom_delta[1] + correction[1],
            odom_delta[2] + correction[2]
        )
        
        # Update last scan
        self.last_scan_points = current_points
        
        # Add keyframe periodically
        if self.frame_count % self.keyframe_interval == 0:
            self.keyframes.append((
                current_points.copy(),
                current_global_pose[0],
                current_global_pose[1], 
                current_global_pose[2]
            ))
            # Keep only recent keyframes
            if len(self.keyframes) > self.max_keyframes:
                self.keyframes.pop(0)
        
        return final_delta, correction


class SimpleWallFollower:
    """Wall following controller - unchanged"""
    
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


class PoseNode:
    """A node in the pose graph representing a robot pose"""
    def __init__(self, idx, x, y, theta, scan):
        self.idx = idx
        self.x = x
        self.y = y
        self.theta = theta
        self.scan = scan  # Store lidar scan for loop closure detection


class LoopClosureDetector:
    """Detects when robot returns to previously visited location"""
    
    def __init__(self, distance_threshold=0.5, scan_similarity_threshold=0.85, min_time_gap=100):
        self.distance_threshold = distance_threshold  # Closer proximity needed
        self.scan_similarity_threshold = scan_similarity_threshold  # Much higher bar
        self.min_time_gap = min_time_gap  # Must be at least this many frames apart
        
    def compute_scan_signature(self, scan_points):
        """Create a rotation-invariant descriptor from lidar scan"""
        if len(scan_points) == 0:
            return None
        
        data = np.array(scan_points).reshape(-1, 3)
        x = data[:, 0]
        y = data[:, 1]
        
        # Create angular histogram (simple descriptor)
        angles = np.arctan2(y, x)
        distances = np.sqrt(x**2 + y**2)
        
        # Filter very close points (noise)
        valid_mask = distances > 0.2
        angles = angles[valid_mask]
        distances = distances[valid_mask]
        
        if len(distances) == 0:
            return None
        
        # Bin into 72 angular sectors (5 degrees each) for better resolution
        bins = 72
        hist, _ = np.histogram(angles, bins=bins, range=(-np.pi, np.pi), weights=1.0/distances)
        
        # Normalize
        if np.sum(hist) > 0:
            hist = hist / np.sum(hist)
        
        return hist
    
    def compare_scans(self, sig1, sig2):
        """Compare two scan signatures using correlation"""
        if sig1 is None or sig2 is None:
            return 0.0
        
        # Compute normalized cross-correlation
        correlation = np.correlate(sig1, sig2, mode='valid')[0]
        norm1 = np.linalg.norm(sig1)
        norm2 = np.linalg.norm(sig2)
        
        if norm1 > 0 and norm2 > 0:
            similarity = correlation / (norm1 * norm2)
        else:
            similarity = 0.0
        
        return similarity
    
    def detect_loop_closure(self, current_node, pose_graph):
        """
        Check if current pose is close to any previous pose.
        Returns: (loop_detected, matched_node_idx, similarity_score)
        """
        # Need enough history AND time gap from last checked poses
        if len(pose_graph) < self.min_time_gap:
            return False, None, 0.0
        
        current_sig = self.compute_scan_signature(current_node.scan)
        if current_sig is None:
            return False, None, 0.0
        
        # Only check poses that are sufficiently old (exclude recent N frames)
        old_poses = pose_graph[:-self.min_time_gap]
        
        if len(old_poses) == 0:
            return False, None, 0.0
        
        # Build KD-tree from old pose positions
        positions = np.array([[n.x, n.y] for n in old_poses])
        tree = KDTree(positions)
        
        # Find nearby poses
        current_pos = np.array([current_node.x, current_node.y])
        indices = tree.query_ball_point(current_pos, self.distance_threshold)
        
        if len(indices) == 0:
            return False, None, 0.0
        
        # Check scan similarity for nearby poses
        best_similarity = 0.0
        best_match_idx = None
        
        for idx in indices:
            candidate_node = old_poses[idx]
            candidate_sig = self.compute_scan_signature(candidate_node.scan)
            
            similarity = self.compare_scans(current_sig, candidate_sig)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = candidate_node.idx  # Use actual node index from graph
        
        if best_similarity > self.scan_similarity_threshold:
            return True, best_match_idx, best_similarity
        
        return False, None, 0.0


class PoseGraphOptimizer:
    """Simple pose graph optimization for loop closure"""
    
    def __init__(self):
        self.optimization_strength = 0.5  # How much to trust loop closures
    
    def optimize_trajectory(self, pose_graph, loop_closure_from, loop_closure_to):
        """
        Distribute error correction across trajectory between loop closure points.
        Simple linear interpolation approach.
        """
        if loop_closure_from >= loop_closure_to:
            return pose_graph
        
        # Calculate error at loop closure
        node_from = pose_graph[loop_closure_from]
        node_to = pose_graph[loop_closure_to]
        
        error_x = node_to.x - node_from.x
        error_y = node_to.y - node_from.y
        error_theta = node_to.theta - node_from.theta
        
        # Normalize angle error
        while error_theta > np.pi:
            error_theta -= 2 * np.pi
        while error_theta < -np.pi:
            error_theta += 2 * np.pi
        
        # Distribute correction across affected nodes
        num_nodes = loop_closure_to - loop_closure_from
        
        for i in range(loop_closure_from, loop_closure_to + 1):
            alpha = (i - loop_closure_from) / num_nodes if num_nodes > 0 else 0
            alpha *= self.optimization_strength
            
            pose_graph[i].x -= alpha * error_x
            pose_graph[i].y -= alpha * error_y
            pose_graph[i].theta -= alpha * error_theta
        
        return pose_graph


class OccupancyMap:
    """Occupancy grid map with improved mapping"""
    
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


class SLAMWithLoopClosure:
    """Full SLAM system with loop closure and ICP scan matching"""
    
    def __init__(self, initial_pose=(0, 0, 0)):
        self.mapper = OccupancyMap(size_meters=20, resolution=0.1)
        self.loop_detector = LoopClosureDetector(
            distance_threshold=0.5,        # Must be within 50cm
            scan_similarity_threshold=0.85, # Must be 85% similar
            min_time_gap=100               # Must be 100+ frames apart
        )
        self.optimizer = PoseGraphOptimizer()
        
        # ICP scan matcher for drift correction
        self.scan_matcher = ICPScanMatcher(
            max_iterations=50,
            tolerance=1e-5,
            max_correspondence_dist=0.5
        )
        
        # Pose graph
        self.pose_graph = []
        self.current_pose = list(initial_pose)  # Corrected pose
        
        # Track previous odometry for computing deltas
        self.prev_odom = None
        
        # Statistics
        self.loop_closures_detected = 0
        self.last_optimization_time = 0
        self.total_icp_correction = [0.0, 0.0, 0.0]  # Cumulative correction stats
        
    def update(self, lidar_local_points, odom_x, odom_y, odom_theta):
        """Main SLAM update with ICP scan matching and loop closure"""
        
        # Compute odometry delta (motion since last update)
        if self.prev_odom is None:
            odom_delta = (0.0, 0.0, 0.0)
        else:
            # Compute relative motion in previous frame
            dx_global = odom_x - self.prev_odom[0]
            dy_global = odom_y - self.prev_odom[1]
            dtheta = odom_theta - self.prev_odom[2]
            
            # Transform to local frame of previous pose
            prev_theta = self.prev_odom[2]
            cos_t = np.cos(-prev_theta)
            sin_t = np.sin(-prev_theta)
            
            dx_local = cos_t * dx_global - sin_t * dy_global
            dy_local = sin_t * dx_global + cos_t * dy_global
            
            odom_delta = (dx_local, dy_local, dtheta)
        
        # Use ICP to refine the motion estimate
        # Pass current pose estimate for keyframe-based matching
        corrected_delta, icp_correction = self.scan_matcher.match_scan(
            lidar_local_points, odom_delta, tuple(self.current_pose)
        )
        
        # Track cumulative correction for statistics
        self.total_icp_correction[0] += abs(icp_correction[0])
        self.total_icp_correction[1] += abs(icp_correction[1])
        self.total_icp_correction[2] += abs(icp_correction[2])
        
        # Apply corrected delta to current pose
        if self.prev_odom is not None:
            # Transform corrected delta back to global frame
            current_theta = self.current_pose[2]
            cos_t = np.cos(current_theta)
            sin_t = np.sin(current_theta)
            
            dx_global = cos_t * corrected_delta[0] - sin_t * corrected_delta[1]
            dy_global = sin_t * corrected_delta[0] + cos_t * corrected_delta[1]
            
            self.current_pose[0] += dx_global
            self.current_pose[1] += dy_global
            self.current_pose[2] += corrected_delta[2]
            
            # Normalize angle
            while self.current_pose[2] > np.pi:
                self.current_pose[2] -= 2 * np.pi
            while self.current_pose[2] < -np.pi:
                self.current_pose[2] += 2 * np.pi
        
        # Store current odometry for next delta computation
        self.prev_odom = (odom_x, odom_y, odom_theta)
        
        # Create new pose node with CORRECTED pose (not raw odometry)
        node = PoseNode(
            idx=len(self.pose_graph),
            x=self.current_pose[0],
            y=self.current_pose[1],
            theta=self.current_pose[2],
            scan=copy.deepcopy(lidar_local_points)
        )
        
        self.pose_graph.append(node)
        
        # Check for loop closure every 20 frames (less frequent)
        if len(self.pose_graph) % 20 == 0 and len(self.pose_graph) > 150:
            loop_detected, match_idx, similarity = self.loop_detector.detect_loop_closure(
                node, self.pose_graph
            )
            
            if loop_detected:
                self.loop_closures_detected += 1
                print(f"\n🔄 LOOP CLOSURE DETECTED!")
                print(f"   Current node: {node.idx} matched with node: {match_idx}")
                print(f"   Similarity: {similarity:.3f}")
                print(f"   Optimizing pose graph...")
                
                # Optimize trajectory
                self.pose_graph = self.optimizer.optimize_trajectory(
                    self.pose_graph, match_idx, node.idx
                )
                
                # Rebuild map from corrected trajectory
                print(f"   Rebuilding map from optimized poses...")
                self.mapper.rebuild_from_trajectory(self.pose_graph)
                print(f"   ✓ Map updated!\n")
                
                self.last_optimization_time = len(self.pose_graph)
        
        # Update current pose to latest node (possibly corrected)
        latest_node = self.pose_graph[-1]
        self.current_pose = [latest_node.x, latest_node.y, latest_node.theta]
        
        # Update map incrementally if no recent optimization
        if len(self.pose_graph) - self.last_optimization_time > 5:
            self.mapper.update_map(lidar_local_points, latest_node.x, latest_node.y, latest_node.theta)
        
        return tuple(self.current_pose)
    
    def get_trajectory(self):
        """Get full trajectory for visualization"""
        if len(self.pose_graph) == 0:
            return [], []
        
        xs = [node.x for node in self.pose_graph]
        ys = [node.y for node in self.pose_graph]
        return xs, ys


def main(args=None):
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX', use_lidar=True)
    coppelia.start_simulation()
    
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
    
    slam = SLAMWithLoopClosure(initial_pose=(0, 0, 0))
    
    # Visualization
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    img = ax1.imshow(slam.mapper.get_probability_grid(threshold=True, smooth=True), 
                     cmap='gray', vmin=0, vmax=1, origin='lower')
    ax1.set_title("SLAM Map with Loop Closure")
    ax1.set_xlabel("Grid X")
    ax1.set_ylabel("Grid Y")
    
    ax2.set_title("Robot Trajectory")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.grid(True)
    ax2.axis('equal')
    
    iteration = 0
    print("\n" + "="*70)
    print("FULL SLAM WITH ICP SCAN MATCHING & LOOP CLOSURE")
    print("="*70)
    print("This system will:")
    print("  ✓ Use ICP scan matching to correct odometry drift in real-time")
    print("  ✓ Build a pose graph with corrected robot poses")
    print("  ✓ Detect when robot returns to known locations")
    print("  ✓ Use STRICT criteria to avoid false positives:")
    print("    - Distance < 0.5m")
    print("    - Similarity > 85%")
    print("    - Time gap > 100 frames")
    print("  ✓ Optimize trajectory and rebuild map on true loops")
    print("\nICP will reduce drift during turns!\n")
    
    while coppelia.is_running():
        robot.update_odometry()
        odom_x, odom_y, odom_theta = robot.get_estimated_pose()
        
        raw_lidar = robot.read_lidar_data()
        
        # SLAM update with loop closure
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
            ax2.clear()
            ax2.plot(traj_x, traj_y, 'b-', linewidth=1, alpha=0.7, label='Trajectory')
            ax2.plot(corrected_x, corrected_y, 'ro', markersize=8, label='Current')
            ax2.set_title(f"Trajectory (Loop Closures: {slam.loop_closures_detected})")
            ax2.set_xlabel("X (m)")
            ax2.set_ylabel("Y (m)")
            ax2.grid(True)
            ax2.axis('equal')
            ax2.legend()
            
            fig.canvas.draw_idle()
            plt.pause(0.001)
        
        # Wall following control
        dist = robot.get_sonar()
        left_speed, right_speed = wf.step(dist)
        robot.set_speed(left_speed, right_speed)
        
        if iteration % 50 == 0:
            icp_corr = slam.total_icp_correction
            print(f"Iter {iteration:04d} | Poses: {len(slam.pose_graph):04d} | "
                  f"Loops: {slam.loop_closures_detected} | "
                  f"ICP corr: ({icp_corr[0]:.3f}, {icp_corr[1]:.3f}, {icp_corr[2]:.2f}rad)")
        
        iteration += 1
        time.sleep(0.05)
    
    coppelia.stop_simulation()
    
    print("\n" + "="*70)
    print("SLAM STATISTICS")
    print("="*70)
    print(f"Total iterations: {iteration}")
    print(f"Pose graph nodes: {len(slam.pose_graph)}")
    print(f"Loop closures detected: {slam.loop_closures_detected}")
    print(f"\nICP Cumulative Corrections:")
    print(f"  X: {slam.total_icp_correction[0]:.3f}m")
    print(f"  Y: {slam.total_icp_correction[1]:.3f}m")
    print(f"  Theta: {slam.total_icp_correction[2]:.3f}rad ({np.degrees(slam.total_icp_correction[2]):.1f}°)")
    print(f"\n{'✓' if slam.loop_closures_detected > 0 else '✗'} Loop closure {'ACTIVE' if slam.loop_closures_detected > 0 else 'not triggered'}")
    print(f"{'✓'} ICP scan matching was active throughout")


if __name__ == "__main__":
    main()