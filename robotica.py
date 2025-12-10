'''
robotica.py

Provides the communication between CoppeliaSim robotics simulator and
external Python applications via the ZeroMQ remote API.

Copyright (C) 2024 Javier de Lope

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

'''
robotica.py
'''
import numpy as np
import cv2
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class Coppelia():
    def __init__(self):
        print('*** connecting to coppeliasim')
        self.client = RemoteAPIClient()
        self.sim = self.client.getObject('sim')

    def start_simulation(self, stepping=False):
        self.default_idle_fps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)
        self.stepping = stepping
        if stepping:
            self.client.setStepping(True)
        self.sim.startSimulation()
    
    def step(self):
        if self.stepping:
            self.client.step()

    def stop_simulation(self):
        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.1)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, self.default_idle_fps)
        print('*** done')

    def is_running(self):
        return self.sim.getSimulationState() != self.sim.simulation_stopped

class P3DX():
    num_sonar = 16
    sonar_max = 1.0
    
    # Adjusted Wheel Base to match CoppeliaSim model perfectly
    WHEEL_RADIUS = 0.0975
    WHEEL_BASE = 0.3310    
    
    def __init__(self, sim, robot_id, use_camera=False, use_lidar=False):
        self.sim = sim
        print(f'*** getting handles for /{robot_id}')
        self.robot_base = self.sim.getObject(f'/{robot_id}') 
        self.left_motor = self.sim.getObject(f'/{robot_id}/leftMotor')
        self.right_motor = self.sim.getObject(f'/{robot_id}/rightMotor')
        self.sonar = []
        for i in range(self.num_sonar):
            self.sonar.append(self.sim.getObject(f'/{robot_id}/ultrasonicSensor[{i}]'))
        if use_camera:
            self.camera = self.sim.getObject(f'/{robot_id}/camera')
        if use_lidar:
            try:
                self.lidar_handle = self.sim.getObject(f'/{robot_id}/lidar')
                self.lidar_script = self.sim.getScript(self.sim.scripttype_childscript, self.lidar_handle)
            except:
                print("No lidar obtained")

        self.dt = self.sim.getSimulationTimeStep()
        
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def update_odometry(self):
        """
        Fused Odometry:
        - Distance (Forward/Backward) -> from Wheel Encoders
        - Rotation (Theta) -> from Simulated Gyroscope (Physics Engine)
        """
        dt = self.dt
        
        # 1. Get Distance from Wheels
        vl = self.sim.getJointVelocity(self.left_motor)
        vr = self.sim.getJointVelocity(self.right_motor)
        
        dist_left = vl * self.WHEEL_RADIUS * dt
        dist_right = vr * self.WHEEL_RADIUS * dt
        dist_center = (dist_left + dist_right) / 2.0
        
        # 2. Get Rotation from "Perfect" Gyro (Physics Engine)
        # This bypasses wheel slip completely.
        linear_vel, angular_vel = self.sim.getObjectVelocity(self.robot_base)
        gyro_z = angular_vel[2] # Pure angular velocity (no noise added)
        delta_theta = gyro_z * dt
        
        # 3. Update Pose
        mid_theta = self.theta + delta_theta / 2.0
        
        self.x += dist_center * np.cos(mid_theta)
        self.y += dist_center * np.sin(mid_theta)
        self.theta += delta_theta
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

    def get_estimated_pose(self):
        return self.x, self.y, self.theta
    
    def get_ground_truth_pose(self):
        position = self.sim.getObjectPosition(self.robot_base, self.sim.handle_world)
        orientation = self.sim.getObjectOrientation(self.robot_base, self.sim.handle_world)
        return position[0], position[1], orientation[2]
    
    def get_sonar(self):
        readings = []
        for i in range(self.num_sonar):
            res,dist,_,_,_ = self.sim.readProximitySensor(self.sonar[i])
            readings.append(dist if res == 1 else self.sonar_max)
        return readings

    def get_image(self):
        img, resX, resY = self.sim.getVisionSensorCharImage(self.camera)
        img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
        img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
        return img

    def read_lidar_data(self):
        try:
            data = self.sim.callScriptFunction('getLidarData', self.lidar_script)
            if not data: return []
            return data 
        except:
            return []

    def set_speed(self, left_speed, right_speed):
        self.sim.setJointTargetVelocity(self.left_motor, left_speed)
        self.sim.setJointTargetVelocity(self.right_motor, right_speed)

def main(args=None):
    coppelia = Coppelia()
    robot = P3DX(coppelia.sim, 'PioneerP3DX', use_camera=True)
    robot.set_speed(+1.2, -1.2)
    coppelia.start_simulation()
    while (t := coppelia.sim.getSimulationTime()) < 3:
        print(f'Simulation time: {t:.3f} [s]')
    coppelia.stop_simulation()


if __name__ == '__main__':
    main()
