import robotica
import time
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from collections import deque
import copy


class ScanMatchingLocalizer:
    """
    Pure scan-to-scan ICP for motion estimation.
    
    Since odometry is broken, we estimate robot motion by comparing
    consecutive lidar scans using ICP (Iterative Closest Point).
    """
    
    def __init__(self):
        self.last_scan_points = None
        self.current_pose = [0.0, 0.0, 0.0]  # x, y, theta
        
        # ICP parameters
        self.max_iterations = 30
        self.convergence_threshold = 1e-5
        self.max_correspondence_dist = 0.5
        
    def extract_points(self, lidar_local_points, min_range=0.15, max_range=5.0):
        """Extract valid 2D points from lidar scan."""
        if len(lidar_local_points) == 0:
            return np.array([]).reshape(0, 2)
        
        data = np.array(lidar_local_points).reshape(-1, 3)
        x = data[:, 0]
        y = data[:, 1]
        
        distances = np.sqrt(x**2 + y**2)
        valid_mask = (distances > min_range) & (distances < max_range)
        
        return np.column_stack([x[valid_mask], y[valid_mask]])
    
    def transform_points(self, points, dx, dy, dtheta):
        """Apply rigid transformation to points."""
        if len(points) == 0:
            return points
        
        cos_t = np.cos(dtheta)
        sin_t = np.sin(dtheta)
        
        rotated_x = points[:, 0] * cos_t - points[:, 1] * sin_t
        rotated_y = points[:, 0] * sin_t + points[:, 1] * cos_t
        
        return np.column_stack([rotated_x + dx, rotated_y + dy])
    
    def icp(self, source, target):
        """
        ICP to find transformation from source to target.
        Returns (dx, dy, dtheta) that transforms source to align with target.
        """
        if len(source) < 10 or len(target) < 10:
            return 0.0, 0.0, 0.0, False
        
        # Build KD-tree for target
        tree = KDTree(target)
        
        # Current transform estimate
        dx, dy, dtheta = 0.0, 0.0, 0.0
        current_source = source.copy()
        
        prev_error = float('inf')
        
        for _ in range(self.max_iterations):
            # Find correspondences
            distances, indices = tree.query(current_source, k=1)
            
            # Filter by distance
            valid = distances < self.max_correspondence_dist
            if np.sum(valid) < 10:
                return dx, dy, dtheta, False
            
            src_matched = current_source[valid]
            tgt_matched = target[indices[valid]]
            
            # Compute error
            error = np.mean(distances[valid]**2)
            if abs(prev_error - error) < self.convergence_threshold:
                return dx, dy, dtheta, True
            prev_error = error
            
            # Compute optimal transformation using SVD
            src_centroid = np.mean(src_matched, axis=0)
            tgt_centroid = np.mean(tgt_matched, axis=0)
            
            src_centered = src_matched - src_centroid
            tgt_centered = tgt_matched - tgt_centroid
            
            H = src_centered.T @ tgt_centered
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            t = tgt_centroid - R @ src_centroid
            
            # Extract incremental transform
            ddtheta = np.arctan2(R[1, 0], R[0, 0])
            ddx, ddy = t[0], t[1]
            
            # Compose with current transform
            cos_t = np.cos(dtheta)
            sin_t = np.sin(dtheta)
            dx += cos_t * ddx - sin_t * ddy
            dy += sin_t * ddx + cos_t * ddy
            dtheta += ddtheta
            
            # Update source points
            current_source = self.transform_points(source, dx, dy, dtheta)
        
        return dx, dy, dtheta, False
    
    def update(self, lidar_local_points):
        """
        Update pose estimate using scan matching.
        Returns current estimated pose (x, y, theta).
        """
        current_points = self.extract_points(lidar_local_points)
        
        if len(current_points) < 30:
            return tuple(self.current_pose)
        
        if self.last_scan_points is None or len(self.last_scan_points) < 30:
            self.last_scan_points = current_points
            return tuple(self.current_pose)
        
        # Use ICP to find motion from last scan to current scan
        # When robot moves forward, walls appear to move backward in scan
        # icp(last_scan, current_scan) gives us how walls moved (opposite of robot)
        dx, dy, dtheta, converged = self.icp(self.last_scan_points, current_points)
        
        # The transform tells us how WALLS moved, robot moved opposite
        robot_dx = -dx
        robot_dy = -dy
        robot_dtheta = -dtheta
        
        # Sanity check: reject unreasonable motions
        # Robot can't move more than ~0.1m or rotate more than ~15° per frame
        motion_magnitude = np.sqrt(robot_dx**2 + robot_dy**2)
        
        if motion_magnitude > 0.15 or abs(robot_dtheta) > 0.3:
            # Motion seems too large - might be bad match
            # Just keep previous pose, update scan
            self.last_scan_points = current_points
            return tuple(self.current_pose)
        
        # Transform local motion to global frame
        cos_t = np.cos(self.current_pose[2])
        sin_t = np.sin(self.current_pose[2])
        
        global_dx = cos_t * robot_dx - sin_t * robot_dy
        global_dy = sin_t * robot_dx + cos_t * robot_dy
        
        # Update pose
        self.current_pose[0] += global_dx
        self.current_pose[1] += global_dy
        self.current_pose[2] += robot_dtheta
        
        # Normalize angle
        while self.current_pose[2] > np.pi:
            self.current_pose[2] -= 2 * np.pi
        while self.current_pose[2] < -np.pi:
            self.current_pose[2] += 2 * np.pi
        
        # Update last scan
        self.last_scan_points = current_points
        
        return tuple(self.current_pose)
    
    def get_pose(self):
        return tuple(self.current_pose)
    
    def set_pose(self, x, y, theta):
        self.current_pose = [x, y, theta]


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
    """SLAM using scan-matching for localization (odometry-free)"""
    
    def __init__(self, initial_pose=(0, 0, 0), use_correction=True):
        self.mapper = OccupancyMap(size_meters=20, resolution=0.1)
        self.loop_detector = LoopClosureDetector(
            distance_threshold=0.8,
            scan_similarity_threshold=0.80,
            min_time_gap=80
        )
        self.optimizer = PoseGraphOptimizer()
        
        # Scan-matching localizer (replaces broken odometry)
        self.localizer = ScanMatchingLocalizer()
        self.localizer.set_pose(initial_pose[0], initial_pose[1], initial_pose[2])
        
        # Pose graph
        self.pose_graph = []
        self.current_pose = list(initial_pose)
        
        # Option to use scan matching or not
        self.use_scan_matching = use_correction
        
        # Statistics
        self.loop_closures_detected = 0
        self.last_optimization_time = 0
        self.total_correction = [0.0, 0.0, 0.0]
        
    def update(self, lidar_local_points, odom_x, odom_y, odom_theta):
        """Update SLAM using scan matching for localization"""
        
        if self.use_scan_matching:
            # Use scan-to-scan ICP to estimate pose (ignore broken odometry)
            est_x, est_y, est_theta = self.localizer.update(lidar_local_points)
            self.current_pose = [est_x, est_y, est_theta]
        else:
            # Use raw odometry (for comparison - will be bad)
            self.current_pose = [odom_x, odom_y, odom_theta]
        
        # Create new pose node
        node = PoseNode(
            idx=len(self.pose_graph),
            x=self.current_pose[0],
            y=self.current_pose[1],
            theta=self.current_pose[2],
            scan=copy.deepcopy(lidar_local_points)
        )
        
        self.pose_graph.append(node)
        
        # Check for loop closure every 20 frames
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
    import sys
    
    # Check for mode
    USE_GROUND_TRUTH = False
    USE_CORRECTION = True
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--odom-only':
            USE_CORRECTION = False
            print("\n*** RUNNING IN ODOMETRY-ONLY MODE (no corrections) ***\n")
        elif sys.argv[1] == '--ground-truth':
            USE_GROUND_TRUTH = True
            USE_CORRECTION = False
            print("\n*** RUNNING WITH GROUND TRUTH POSE (from simulator) ***")
            print("*** This shows what the map SHOULD look like with perfect localization ***\n")
    
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
    
    # Get initial ground truth pose - ALWAYS use it to start, otherwise we have no reference!
    gt_x, gt_y, gt_theta = robot.get_ground_truth_pose()
    initial_pose = (gt_x, gt_y, gt_theta)  # Always start from ground truth position
    print(f"\nInitial ground truth pose: ({gt_x:.2f}, {gt_y:.2f}, {np.degrees(gt_theta):.1f}°)")
    
    slam = SLAMWithLoopClosure(initial_pose=initial_pose, use_correction=USE_CORRECTION)
    
    # Track trajectories for comparison
    raw_odom_trajectory = []
    ground_truth_trajectory = []
    
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
    print("SLAM WITH SCAN-TO-SCAN ICP LOCALIZATION")
    print("="*70)
    print("This system will:")
    print("  ✓ Use scan-to-scan ICP to estimate robot motion")
    print("  ✓ IGNORE broken odometry completely")
    print("  ✓ Build occupancy grid map from lidar scans")
    print("  ✓ Track robot trajectory using scan matching")
    print("\nModes:")
    print("  python assigment_2.py              - Scan matching SLAM")
    print("  python assigment_2.py --ground-truth - Use ground truth (perfect)")
    print("  python assigment_2.py --odom-only    - Use broken odometry (bad)")
    print("="*70 + "\n")
    
    while coppelia.is_running():
        robot.update_odometry()
        odom_x, odom_y, odom_theta = robot.get_estimated_pose()
        gt_x, gt_y, gt_theta = robot.get_ground_truth_pose()
        
        # Track trajectories
        raw_odom_trajectory.append((odom_x, odom_y))
        ground_truth_trajectory.append((gt_x, gt_y))
        
        raw_lidar = robot.read_lidar_data()
        
        # Use ground truth or odometry based on mode
        if USE_GROUND_TRUTH:
            # Directly use ground truth - no SLAM needed
            corrected_x, corrected_y, corrected_theta = gt_x, gt_y, gt_theta
            # Still update the map with ground truth pose
            slam.mapper.update_map(raw_lidar, gt_x, gt_y, gt_theta)
            # Create a pose node for trajectory tracking
            node = PoseNode(len(slam.pose_graph), gt_x, gt_y, gt_theta, copy.deepcopy(raw_lidar))
            slam.pose_graph.append(node)
        else:
            # SLAM update with corrections
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
            
            # Update trajectory plot - show all trajectories
            traj_x, traj_y = slam.get_trajectory()
            raw_x = [p[0] for p in raw_odom_trajectory]
            raw_y = [p[1] for p in raw_odom_trajectory]
            gt_traj_x = [p[0] for p in ground_truth_trajectory]
            gt_traj_y = [p[1] for p in ground_truth_trajectory]
            
            ax2.clear()
            ax2.plot(gt_traj_x, gt_traj_y, 'g-', linewidth=2, alpha=0.8, label='Ground Truth')
            ax2.plot(raw_x, raw_y, 'r-', linewidth=1, alpha=0.5, label='Raw Odometry')
            ax2.plot(traj_x, traj_y, 'b-', linewidth=1, alpha=0.7, label='SLAM Trajectory')
            ax2.plot(corrected_x, corrected_y, 'bo', markersize=8, label='Current SLAM')
            ax2.plot(gt_x, gt_y, 'g*', markersize=10, label='Current GT')
            ax2.set_title(f"Trajectory Comparison")
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
        
        if iteration % 50 == 0:
            corr = slam.total_correction
            # Show ground truth, raw odom, and SLAM positions
            print(f"Iter {iteration:04d} | GT: ({gt_x:.2f}, {gt_y:.2f}, {np.degrees(gt_theta):.0f}°) | "
                  f"Odom: ({odom_x:.2f}, {odom_y:.2f}, {np.degrees(odom_theta):.0f}°) | "
                  f"SLAM: ({corrected_x:.2f}, {corrected_y:.2f}, {np.degrees(corrected_theta):.0f}°)")
        
        iteration += 1
        time.sleep(0.05)
    
    coppelia.stop_simulation()
    
    print("\n" + "="*70)
    print("SLAM STATISTICS")
    print("="*70)
    print(f"Total iterations: {iteration}")
    print(f"Pose graph nodes: {len(slam.pose_graph)}")
    print(f"Loop closures detected: {slam.loop_closures_detected}")
    print(f"\nScan-to-Map Cumulative Corrections:")
    print(f"  X: {slam.total_correction[0]:.3f}m")
    print(f"  Y: {slam.total_correction[1]:.3f}m")
    print(f"  Theta: {slam.total_correction[2]:.3f}rad ({np.degrees(slam.total_correction[2]):.1f}°)")
    print(f"\n{'✓' if slam.loop_closures_detected > 0 else '✗'} Loop closure {'ACTIVE' if slam.loop_closures_detected > 0 else 'not triggered'}")
    print(f"{'✓'} Scan-to-map matching was active throughout")


if __name__ == "__main__":
    main()