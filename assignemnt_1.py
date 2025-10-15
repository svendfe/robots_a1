import robotica
import time

class SimpleWallFollower:
    """
    Simple, reliable PID wall follower that sticks to one side.
    States: FIND -> FOLLOW -> CORNER (when needed) -> back to FOLLOW
    """
    
    def __init__(self,
                 base_speed=0.6,
                 follow_side='left',
                 target_dist=0.15,
                 kp=2.0,
                 ki=0.05,
                 kd=0.7,
                 find_threshold=0.70,
                 front_threshold=0.30):
        
        # Basic parameters
        self.base_speed = base_speed
        self.follow_side = follow_side
        self.target_dist = target_dist
        
        # PID gains
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_error = 0
        self.last_error = 0
        
        # Detection thresholds
        self.find_threshold = find_threshold
        self.front_threshold = front_threshold
        
        # State
        self.mode = 'FIND'  # FIND, FOLLOW, CORNER
        self.corner_step = 0
        self.corner_phase = None  # 'back', 'turn', 'forward'
        
        # Track when wall disappears to follow it around corners
        self.wall_lost_counter = 0
        self.turning_to_follow = False
        
        # Corner escape parameters - slower and more controlled
        self.back_duration = 10
        self.turn_duration = 10
        self.forward_duration = 8
        
    def get_sensor(self, dist, idx):
        """Get valid sensor reading or None."""
        v = dist[idx]
        if v is None or v >= 0.95:
            return None
        return v
    
    def get_side_distance(self, dist):
        """Get average distance from the followed side (left or right)."""
        if self.follow_side == 'left':
            front_idx, back_idx = 0, 15  # Left sensors
        else:
            front_idx, back_idx = 7, 8    # Right sensors
        
        front = self.get_sensor(dist, front_idx)
        back = self.get_sensor(dist, back_idx)
        
        readings = [r for r in [front, back] if r is not None]
        if not readings:
            return None, None, None
        
        return sum(readings) / len(readings), front, back
    
    def get_front_distance(self, dist):
        """Get minimum front distance."""
        front_readings = []
        for idx in [3, 4]:  # Front sensors
            reading = self.get_sensor(dist, idx)
            if reading is not None:
                front_readings.append(reading)
        return min(front_readings) if front_readings else 1.0
    
    def start_corner_escape(self):
        """Enter corner escape mode."""
        self.mode = 'CORNER'
        self.corner_phase = 'back'
        self.corner_step = 0
        self.integral_error = 0  # Reset integral
        print("CORNER DETECTED → Starting escape sequence")
    
    def handle_corner(self, dist):
        """Execute corner escape: back up, turn away from wall, move forward."""
        self.corner_step += 1
        
        if self.corner_phase == 'back':
            # Back up
            if self.follow_side == 'left':
                left, right = -0.10, -0.15
            else:
                left, right = -0.15, -0.10
            
            print(f"  ↩ Backing up [{self.corner_step}/{self.back_duration}]")
            
            if self.corner_step >= self.back_duration:
                self.corner_phase = 'turn'
                self.corner_step = 0
                print("   Starting turn...")
            
            return left, right
        
        elif self.corner_phase == 'turn':
            # Turn away from the wall - slower rotation for precision
            if self.follow_side == 'left':
                left, right = 0.25, -0.25  # Turn right (away from left wall)
            else:
                left, right = -0.25, 0.25  # Turn left (away from right wall)
            
            print(f"  🔄 Turning [{self.corner_step}/{self.turn_duration}]")
            
            if self.corner_step >= self.turn_duration:
                self.corner_phase = 'forward'
                self.corner_step = 0
                print("  ➡ Moving forward...")
            
            return left, right
        
        elif self.corner_phase == 'forward':
            # Move forward to clear the corner
            left = right = self.base_speed
            
            print(f"  ➡ Forward recovery [{self.corner_step}/{self.forward_duration}]")
            
            if self.corner_step >= self.forward_duration:
                self.mode = 'FIND'
                self.corner_phase = None
                print("✅ Corner escape complete → Searching for wall")
            
            return left, right
    
    def step(self, dist):
        """Main control loop."""
        side_avg, side_front, side_back = self.get_side_distance(dist)
        front_dist = self.get_front_distance(dist)
        
        # === PRIORITY 1: Check for obstacles/corners (dead-ends) ===
        if front_dist < self.front_threshold:
            if self.mode != 'CORNER':
                self.start_corner_escape()
        
        # === CORNER MODE: Execute escape sequence ===
        if self.mode == 'CORNER':
            return self.handle_corner(dist)
        
        # === FIND MODE: Look for wall ===
        if self.mode == 'FIND':
            if side_avg is not None and side_avg < self.find_threshold:
                self.mode = 'FOLLOW'
                self.integral_error = 0
                self.last_error = 0
                print(f"✅ Wall found at {side_avg:.2f}m → Starting FOLLOW mode")
            else:
                print(f"🔍 Searching for wall... (side: {side_avg}m if side_avg else 'none')")
                return self.base_speed, self.base_speed
        
        # === FOLLOW MODE: PID control ===
        # Check if wall disappeared (opening/corner where wall ends)
        if side_avg is None or side_avg > 0.80:
            self.wall_lost_counter += 1
            
            # Wall just disappeared - turn towards it to follow around the corner!
            if self.wall_lost_counter < 25:  # Keep turning for up to 25 steps
                self.turning_to_follow = True
                print(f"↪️  Wall disappeared! Turning LEFT to follow it [{self.wall_lost_counter}/25]")
                
                # Turn toward the wall side (left if following left wall)
                if self.follow_side == 'left':
                    # Turn left to follow the wall
                    left_speed = 0.22
                    right_speed = 0.45
                else:
                    # Turn right to follow the wall
                    left_speed = 0.45
                    right_speed = 0.22
                
                return left_speed, right_speed
            else:
                # Turned for too long without finding wall - give up and search
                print("⚠ Wall lost for too long → Back to FIND mode")
                self.mode = 'FIND'
                self.integral_error = 0
                self.wall_lost_counter = 0
                self.turning_to_follow = False
                return self.base_speed, self.base_speed
        else:
            # Wall is present - reset counters
            if self.wall_lost_counter > 0:
                print(f"✅ Wall reacquired at {side_avg:.2f}m after turning!")
            self.wall_lost_counter = 0
            self.turning_to_follow = False
        
        # Calculate errors
        distance_error = side_avg - self.target_dist
        
        # Angle error (how parallel are we to the wall?)
        if side_front is not None and side_back is not None:
            angle_error = (side_front - side_back) / 0.18  # sensor separation
        else:
            angle_error = 0
        
        # PID calculation
        self.integral_error += distance_error
        # Anti-windup: limit integral
        self.integral_error = max(-0.5, min(0.5, self.integral_error))
        
        derivative_error = distance_error - self.last_error
        self.last_error = distance_error
        
        # PID output (steering correction)
        steering = (self.kp * distance_error + 
                   self.ki * self.integral_error + 
                   self.kd * angle_error)
        
        # Limit steering
        steering = max(-0.8, min(0.8, steering))
        
        # Apply steering to wheel speeds
        forward = self.base_speed
        
        if self.follow_side == 'left':
            # If too close to left wall (distance_error < 0), steer right (left slower)
            left_speed = forward - steering
            right_speed = forward + steering
        else:
            # If too close to right wall, steer left (right slower)
            left_speed = forward + steering
            right_speed = forward - steering
        
        # Ensure minimum speed
        left_speed = max(0.1, min(1.2, left_speed))
        right_speed = max(0.1, min(1.2, right_speed))
        
        # Status output
        status = f"dist_err:{distance_error:+.3f} ang_err:{angle_error:+.3f} I:{self.integral_error:+.3f}"
        print(f"📏 Wall: {side_avg:.2f}m | {status} | L:{left_speed:.2f} R:{right_speed:.2f}")
        
        return left_speed, right_speed


def main(args=None):
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX')
    coppelia.start_simulation()
    
    # Create wall follower - adjusted for closer following and gentler turns
    wf = SimpleWallFollower(
        base_speed=0.6,
        follow_side='left',    # Follow left wall
        target_dist=0.15,      # Stay 25cm from wall (closer!)
        kp=2.0,                # Proportional gain
        ki=0.05,               # Integral gain
        kd=0.7,                # Derivative gain
        find_threshold=0.50,   # Detect wall within 70cm
        front_threshold=0.30   # Corner detection at 30cm
    )
    
    print("=" * 60)
    print("🤖 SIMPLE PID WALL FOLLOWER (LEFT SIDE)")
    print("=" * 60)
    
    iteration = 0
    while coppelia.is_running():
        dist = robot.get_sonar()
        
        # Debug sensor readings every 30 iterations
        if iteration % 30 == 0:
            left = [f'{dist[i]:.2f}' if dist[i] and dist[i] < 0.95 else '--' for i in [0, 15]]
            right = [f'{dist[i]:.2f}' if dist[i] and dist[i] < 0.95 else '--' for i in [7, 8]]
            front = [f'{dist[i]:.2f}' if dist[i] and dist[i] < 0.95 else '--' for i in [3, 4]]
            print(f"\n[Iter {iteration}] Sensors → Left:{left} Right:{right} Front:{front}")
        
        # Get motor speeds from controller
        left_speed, right_speed = wf.step(dist)
        robot.set_speed(left_speed, right_speed)
        
        iteration += 1
        time.sleep(0.05)
    
    coppelia.stop_simulation()


if __name__ == "__main__":
    main()