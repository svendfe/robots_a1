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

import numpy as np
import cv2
import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


class Coppelia():

    def __init__(self):
        print('*** connecting to coppeliasim')
        client = RemoteAPIClient()
        self.sim = client.getObject('sim')

    def start_simulation(self):
        # print('*** saving environment')
        self.default_idle_fps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)
        self.sim.startSimulation()

    def stop_simulation(self):
        # print('*** stopping simulation')
        self.sim.stopSimulation()
        while self.sim.getSimulationState() != self.sim.simulation_stopped:
            time.sleep(0.1)
        # print('*** restoring environment')
        self.sim.setInt32Param(self.sim.intparam_idle_fps, self.default_idle_fps)
        print('*** done')

    def is_running(self):
        return self.sim.getSimulationState() != self.sim.simulation_stopped


class P3DX():

    num_sonar = 16
    sonar_max = 1.0

    WHEEL_RADIUS = 0.0975
    WHEEL_BASE = 0.331

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
            except:
                print("No lidar obtained")

            self.lidar_script = self.sim.getScript(self.sim.scripttype_childscript, self.lidar_handle)

        self.dt = self.sim.getSimulationTimeStep()
        # --- INTERNAL STATE (ODOMETRY) ---
        # We start at 0,0,0 relative to where we turned on the robot
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def update_odometry(self):
        """
        Calculates the new position based on wheel rotation.
        Call this EVERY simulation step.
        """
        # 1. Get current velocity of wheels (Rad/s)
        # We use getJointVelocity to see what the physics engine is actually doing
        vl = self.sim.getJointVelocity(self.left_motor)
        vr = self.sim.getJointVelocity(self.right_motor)
        
        # 2. Calculate Linear (v) and Angular (w) velocity of the robot
        v = (self.WHEEL_RADIUS / 2) * (vr + vl)
        w = (self.WHEEL_RADIUS / self.WHEEL_BASE) * (vr - vl)
        
        # 3. Update Position (Integration)
        # New_Pos = Old_Pos + (Velocity * Time)
        self.x += v * np.cos(self.theta) * self.dt
        self.y += v * np.sin(self.theta) * self.dt
        self.theta += w * self.dt
        
        # Normalize theta to -pi to +pi (optional but recommended)
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

    def get_estimated_pose(self):
        return self.x, self.y, self.theta
    
    def get_ground_truth_pose(self):
        """Get the actual position from CoppeliaSim (for debugging/comparison)"""
        position = self.sim.getObjectPosition(self.robot_base, self.sim.handle_world)
        orientation = self.sim.getObjectOrientation(self.robot_base, self.sim.handle_world)
        # orientation[2] is the rotation around Z axis (yaw)
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
        # We invoke the 'getLidarData' function inside the Lidar's specific script
        try:
            data = self.sim.callScriptFunction('getLidarData', self.lidar_script)
            
            if not data:
                return []
            return data 
        except Exception as e:
            # This handles cases where the script isn't initialized yet
            return []

    def set_speed(self, left_speed, right_speed):
        self.sim.setJointTargetVelocity(self.left_motor, left_speed)
        self.sim.setJointTargetVelocity(self.right_motor, right_speed)

    def move(self, velocity):
        self.set_speed(velocity, velocity)

    def turn(self, turnVelocity):
        self.set_speed(+turnVelocity, -turnVelocity)
    
    def stop(self):
        self.set_speed(0, 0)

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
