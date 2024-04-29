import numpy as np
from math import atan2, cos, sin

class PathPlanner():
    def __init__(self, trajectory):
        self.current_i = 0
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
        
    def getCurrentSP(self):
        return self.current_target

    def __call__(self, state, gate_pos):
        pos = np.array(state[:3])
        angles = state[3:]
        yaw = angles[-1]

        best_i = -1
        best_dist = float('inf')
        for i in range(100):
            if self.current_i + i >= len(self.trajectory):
                break

            proposed_target = self.trajectory[self.current_i + i]
            proposed_target_dist = abs(np.linalg.norm(pos-proposed_target) - 0.1)

            if proposed_target_dist < best_dist:
                best_i = i
                best_dist = proposed_target_dist
        
        self.current_i = self.current_i + best_i

        target_pos = self.trajectory[ self.current_i ]
        self.current_target = target_pos

        # print(np.linalg.norm(pos[:2]-target_pos[:2]))
        # if np.linalg.norm(pos[:2]-target_pos[:2]) < 0.2 and abs(pos[2]-target_pos[2]) < 0.1:
        if np.linalg.norm(pos-target_pos) < 0.2:
            self.trajectory = self.trajectory[1:] 

        dpos_yaw = (np.array(gate_pos) - np.array(pos))[:2]
        yaw_desired = atan2(dpos_yaw[1], dpos_yaw[0]) - yaw 
        yaw_desired = self.wrap_to_pi(yaw_desired)
        yaw_desired = np.clip(yaw_desired, -1, 1)


        if abs(yaw_desired) > np.pi:
            yaw_desired = 2*np.pi - abs(yaw_desired) 
        
        R = np.array(
            [[cos(yaw), -sin(yaw)],
             [sin(yaw), cos(yaw)]
        ])
        

        dpos = (np.array(target_pos) - np.array(pos))[:2]
        vel_desired = np.matmul(dpos, R)
        vel_desired = 1 * (vel_desired / np.linalg.norm(vel_desired)) #max(np.linalg.norm(vel_desired), 1)
        

        # First height, then dpos
        des_height = target_pos[2]
        # if abs(des_height - pos[2]) > 0.2:
        #     des_height = pos[2] + (0.2 * (-1 if des_height - pos[2] < 0 else 1) ) 
        #     if abs(des_height - pos[2]) > 0.1:
        #         vel_desired *= 0

        # Clip for stability
        MAX_VEL_NORM = .3
        if np.linalg.norm(vel_desired) > MAX_VEL_NORM:
            vel_desired = MAX_VEL_NORM * np.array(vel_desired) / np.linalg.norm(vel_desired) 

        # Average for stability
        self.past_vel.append(vel_desired)
        self.past_vel = self.past_vel[-100:]

        NUM_HIST = 20
        ALPHA = 0.9
        output_vel_desired = np.ma.average(
            self.past_vel[-NUM_HIST:], 
            axis=0,
            weights=[ALPHA**i for i in range(min(NUM_HIST, len(self.past_vel[-NUM_HIST:])))][::-1]
            )
        self.past_vel[-1] = output_vel_desired

        fin = len(self.trajectory) - self.current_i < 10

        return output_vel_desired, yaw_desired, des_height, fin
