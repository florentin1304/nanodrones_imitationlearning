import numpy as np
from math import atan2, cos, sin

class PathPlanner():
    def __init__(self, trajectory):

        # trajectory: [x, y, z]
        if trajectory.shape[-1] != 3:
            trajectory = trajectory.T
        
        self.current_i = 0
        self.trajectory = np.array(trajectory)

        # Velocity boundaries
        self.MAX_VEL_NORM = 1.3
        self.MIN_VEL_NORM = 0.5
        
        # Averaging smoothing parameters
        self.ALPHA = 0.95
        self.HIST_LEN = 100

        self.len_trajectory = len(trajectory)
        self.resetHist()

    def resetHist(self):
        self.history = [[0, 0, 0, 1] for i in range(self.HIST_LEN)]

    def wrap_to_pi(self, number):
        if number > np.pi:
            return number - 2 * np.pi
        elif number < -np.pi:
            return number + 2 * np.pi
        else:
            return number
        
    def getCurrentSP(self):
        return self.current_target

    def __call__(self, pos, yaw, gates, current_gate_index):
        ### Calculate target point
        best_i = -1
        best_dist = float('inf')
        for i in range(100):
            if self.current_i + i >= len(self.trajectory):
                break

            proposed_target = self.trajectory[self.current_i + i]
            proposed_target_dist = abs(np.linalg.norm(pos-proposed_target) - 0.3)

            if proposed_target_dist < best_dist:
                best_i = i
                best_dist = proposed_target_dist
        
        self.current_i = self.current_i + best_i
        # self.current_i = best_i + 100

        target_pos = self.trajectory[ self.current_i ]
        self.current_target = target_pos

        ######################################
        ### Translate to controller inputs ###
        ######################################
        if current_gate_index-1 == -1:
            past_gate_pos = np.array([0,0,0], dtype=float)
        else:
            past_gate_pos = gates[current_gate_index-1]['pos']
        current_gate_pos = gates[current_gate_index]['pos']
        next_gate_pos = gates[(current_gate_index+1) % len(gates)]['pos']

        ### Yawrate
        yawrate_desired = self.getDesiredYawRate(pos, yaw, current_gate_pos, past_gate_pos, next_gate_pos)

        ### Velocity
        vel_desired = self.getDesiredVelocity(pos, target_pos, yaw, current_gate_pos, past_gate_pos)

        # Height (easy)
        des_height = target_pos[2]

        # Average for stability
        output = [vel_desired[0], vel_desired[1], yawrate_desired, des_height]
        output_smoothed = self.getAverageSmoothing(output)

        output_vel_desired, yawrate_desired, des_height = output_smoothed[:2], output_smoothed[2], output_smoothed[3] 

        fin = len(self.trajectory) - self.current_i < 10

        return output_vel_desired, yawrate_desired, des_height, fin

    def getDesiredYawRate(self, pos, yaw, current_gate_pos, past_gate_pos, next_gate_pos):
        # Compute yaw
        dpos_current = (np.array(current_gate_pos) - np.array(pos))[:2]
        yaw_desired_current = atan2(dpos_current[1], dpos_current[0]) - yaw 
        yaw_desired_current = self.wrap_to_pi(yaw_desired_current)

        dpos_next = (np.array(next_gate_pos) - np.array(pos))[:2]
        yaw_desired_next = atan2(dpos_next[1], dpos_next[0]) - yaw
        yaw_desired_next = self.wrap_to_pi(yaw_desired_next)

        # dpos_past_current = np.linalg.norm(np.array(past_gate_pos) - np.array(current_gate_pos))
        near_current_gate_factor = min(np.linalg.norm(dpos_current)/2, 1)

        yaw_desired = near_current_gate_factor * yaw_desired_current + \
                      (1-near_current_gate_factor) * yaw_desired_next #self.wrap_to_pi(current_gate_yaw-yaw+(np.pi/2))
        yaw_desired = np.clip(yaw_desired, -1, 1)

        return yaw_desired
    
    def getDesiredVelocity(self, pos, target_pos, yaw, current_gate_pos, past_gate_pos):
        R = np.array(
            [[cos(yaw), -sin(yaw)],
             [sin(yaw), cos(yaw)]
        ])
        
        dpos = (np.array(target_pos) - np.array(pos))[:2]
        vel_desired = np.matmul(dpos, R)
        
        # Clip for stability
        dist = np.linalg.norm(current_gate_pos-pos)
        past_dist = np.linalg.norm(past_gate_pos-pos)
        dist_factor = min(dist, past_dist) / 2
        dist_factor = min(1, dist_factor)


        vel_norm = np.linalg.norm(vel_desired)
        vel_desired_normalized = np.array(vel_desired) / vel_norm

        scale = (self.MAX_VEL_NORM-self.MIN_VEL_NORM) * dist_factor + self.MIN_VEL_NORM
        vel_desired_fin = vel_desired_normalized * scale

        return vel_desired_fin



    def getAverageSmoothing(self, output):
        self.history.append(output)
        self.history = self.history[-self.HIST_LEN:]

        output_avg = np.ma.average(
            self.history, 
            axis=0,
            weights=list(reversed([self.ALPHA**i for i in range(len(self.history))]))
            )
        
        return output_avg
