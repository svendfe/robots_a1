'''
avoid.py

Sample client for the Pioneer P3DX mobile robot that implements a
kind of heuristic, rule-based controller for collision avoidance.

Copyright (C) 2023 Javier de Lope

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

import robotica
import numpy as np


def avoid(readings):
    if (readings[3] < 0.1) or (readings[4] < 0.2):
        lspeed, rspeed = +0.1, -0.8
    elif readings[1] < 0.1:
        lspeed, rspeed = +1.3, +0.6
    elif readings[5] < 0.4:
        lspeed, rspeed = +0.1, +0.9
    else:
        lspeed, rspeed = +1.5, +1.5

    if (readings[3] < 0.3) and (readings[2] > readings[5]):
        lspeed, rspeed = -1.0, +1.0
    elif (readings[3] < 0.3) and (readings[2] < readings[5]):
        lspeed, rspeed = +1.0, -1.0
    else:
        lspeed, rspeed = +1.0, +1.0
    return lspeed, rspeed


def avoid_corrected(readings):

    if (np.array(readings[2:6]) < 0.15).any():
        if (readings[3] < 0.12) and (readings[4] < 0.12):
            lspeed, rspeed = -2.5, -2.5
            #print("Reversing...")
            return lspeed, rspeed
        elif (np.argmin(np.array(readings[2:6])) + 2) < 4:
            lspeed, rspeed = -1, -2.5
            #print("Reversing left...")
            return lspeed, rspeed
        else:
            lspeed, rspeed = -2.5, -1
            #print("Reversing right...")
            return lspeed, rspeed
    else:
        pass
        #print("No need to reverse...")        

    PI = np.pi
    sensor_loc=np.array(
        [-PI/2, -50/180.0*PI,-30/180.0*PI,-10/180.0*PI,
         10/180.0*PI,30/180.0*PI,50/180.0*PI,PI/2,
         PI/2,130/180.0*PI,150/180.0*PI,170/180.0*PI,
         -170/180.0*PI,-150/180.0*PI,-130/180.0*PI,-PI/2]) 

    
    sensor_sq = np.array(readings[:8])**2
    min_ind = np.argmin(sensor_sq)
    if sensor_sq[min_ind] < 0.2:
        steer = -1/sensor_loc[min_ind]
    else:
        steer = 0
    
    base_speed = 1.0
    phi = np.maximum((0.3-sensor_sq[min_ind]), 0)
    speed_adjust = phi * steer
    lspeed = base_speed + speed_adjust
    rspeed = base_speed - speed_adjust
    return lspeed, rspeed


def main(args=None):
    coppelia = robotica.Coppelia()
    robot = robotica.P3DX(coppelia.sim, 'PioneerP3DX')
    coppelia.start_simulation()
    while coppelia.is_running():
        readings = robot.get_sonar()
        #lspeed, rspeed = avoid(readings)
        lspeed, rspeed = avoid_corrected(readings)
        robot.set_speed(lspeed, rspeed)
    coppelia.stop_simulation()


if __name__ == '__main__':
    main()
