import numpy as np
from math import atan2, cos, sin

class PathPlanner():
    def __init__(self, trajectory):
        self.trajectory = np.array(trajectory)
        if self.trajectory.shape[-1] != 3:
            self.trajectory = self.trajectory.T
        self.len_trajectory = len(trajectory)
        self.past_vel = []

    def wrap_to_pi(self, number):
        if number > np.pi:
            return number - 2 * np.pi
        elif number < -np.pi:
            return number + 2 * np.pi
        else:
            return number

    def __call__(self, state, gate_pos):
        pos = np.array(state[:3])
        angles = state[3:]
        yaw = angles[-1]

        target_pos = self.trajectory[0]

        # print(np.linalg.norm(pos[:2]-target_pos[:2]))
        if np.linalg.norm(pos[:2]-target_pos[:2]) < 0.2 and abs(pos[2]-target_pos[2]) < 0.1:
            self.trajectory = self.trajectory[1:] 

        dpos_yaw = (np.array(gate_pos) - np.array(pos))[:2]
        yaw_desired = atan2(dpos_yaw[1], dpos_yaw[0]) - yaw 
        yaw_desired = self.wrap_to_pi(yaw_desired)


        if abs(yaw_desired) > np.pi:
            yaw_desired = 2*np.pi - abs(yaw_desired) 
        
        R = np.array(
            [[cos(yaw), -sin(yaw)],
             [sin(yaw), cos(yaw)]
        ])
        

        dpos = (np.array(target_pos) - np.array(pos))[:2]
        vel_desired = np.matmul(dpos, R)
        vel_desired = 1 * (vel_desired / np.linalg.norm(vel_desired)) #max(np.linalg.norm(vel_desired), 1)
        
        self.past_vel.append(vel_desired)
        self.past_vel = self.past_vel[-100:]

        NUM_HIST = 10
        ALPHA = 0.75
        output_vel_desired = np.ma.average(
            self.past_vel[-NUM_HIST:], 
            axis=0,
            weights=[ALPHA**i for i in range(min(NUM_HIST, len(self.past_vel[-NUM_HIST:])))][::-1]
            )
        self.past_vel[-1] = output_vel_desired

        des_height = target_pos[2]
        if abs(des_height - pos[2]) > 0.2:
            des_height = pos[2] + (0.2 * (-1 if des_height - pos[2] < 0 else 1) ) 

        return output_vel_desired, yaw_desired, des_height
