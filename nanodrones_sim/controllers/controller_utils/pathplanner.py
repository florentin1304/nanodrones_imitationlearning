import numpy as np
from math import atan2, cos, sin

class PathPlanner():
    def __init__(self, trajectory):

        # trajectory: [x, y, z, yaw, t]
        if trajectory.shape[-1] != 5:
            trajectory = trajectory.T
        
        self.current_i = 0
        self.trajectory = np.array(trajectory)[:, :-2]
        self.yaw_traj = trajectory[:, -2]
        self.time_traj = trajectory[:, -1]

        self.len_trajectory = len(trajectory)
        self.resetHist()

    def resetHist(self):
        self.past_vel = [[0, 0, 0, 1] for i in range(100)]

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
            proposed_target_dist = abs(np.linalg.norm(pos-proposed_target) - 0.2)

            if proposed_target_dist < best_dist:
                best_i = i
                best_dist = proposed_target_dist
        
        self.current_i = self.current_i + best_i
        # self.current_i = best_i + 100

        target_pos = self.trajectory[ self.current_i ]
        self.current_target = target_pos

        # print(np.linalg.norm(pos[:2]-target_pos[:2]))
        # if np.linalg.norm(pos[:2]-target_pos[:2]) < 0.2 and abs(pos[2]-target_pos[2]) < 0.1:
        # if np.linalg.norm(pos-target_pos) < 0.2:
        #     self.trajectory = self.trajectory[1:] 

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

        # Clip for stability
        MAX_VEL_NORM = 0.35
        vel_norm = np.linalg.norm(vel_desired)
        vel_desired = MAX_VEL_NORM * np.array(vel_desired) / np.linalg.norm(vel_desired) 
        # vel_desired = vel_desired * (np.linalg.norm(gate_pos-pos)/3)

        # vel_norm = np.linalg.norm(vel_desired)
        # if vel_norm < 0.2:
        #     vel_desired = 0.2 * vel_desired / np.linalg.norm(vel_desired) 
        # elif vel_norm > MAX_VEL_NORM:
        #     vel_desired = MAX_VEL_NORM * vel_desired / np.linalg.norm(vel_desired) 
        # print("Dist",np.linalg.norm(gate_pos-pos)/3)
        # print(np.linalg.norm(vel_desired))

        # Average for stability
        self.past_vel.append([vel_desired[0], vel_desired[1], yaw_desired, des_height])
        self.past_vel = self.past_vel[-200:]

        NUM_HIST = 100
        ALPHA = 0.9
        output = np.ma.average(
            self.past_vel[-NUM_HIST:], 
            axis=0,
            weights=[ALPHA**i for i in range(min(NUM_HIST, len(self.past_vel[-NUM_HIST:])))][::-1]
            )
        
        output_vel_desired, yaw_desired, des_height = output[:2], output[2], output[3] 

        fin = len(self.trajectory) - self.current_i < 10

        return output_vel_desired, yaw_desired, des_height, fin
