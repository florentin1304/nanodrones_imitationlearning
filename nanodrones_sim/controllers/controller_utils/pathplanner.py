import numpy as np
from math import atan2, cos, sin
from sklearn.preprocessing import MinMaxScaler
from controller_utils.velocity_profiler import VelocityProfiler

class PathPlanner():
    def __init__(self, trajectory, smoothing_factor=0.95, velocity_profiler_config={}):

        # trajectory: [x, y, z]
        if trajectory.shape[-1] != 3:
            trajectory = trajectory.T
        
        self.trajectory = np.array(trajectory)

        vp = VelocityProfiler(**velocity_profiler_config)
        self.curvature = vp.run(self.trajectory)

        self.TARGET_DISTANCE = 0.5
        self.closest_i = 0
        self.target_i = 0

        # Velocity boundaries
        self.MAX_VEL_NORM = 2.5
        self.MIN_VEL_NORM = 0.8
        
        # Averaging smoothing parameters
        self.ALPHA = smoothing_factor
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
        return self.target_point
    
    def getCurrentProj(self):
        return self.closest_point

    def __call__(self, pos, yaw, vel, gates, current_gate_index):
        proposed_closest_dir = self.trajectory[self.closest_i + 1] - self.trajectory[self.closest_i] 
        proposed_closest_dist = np.linalg.norm(proposed_closest_dir)
        proposed_closest_dir = proposed_closest_dir / proposed_closest_dist

        offset = pos - self.trajectory[self.closest_i]
        x = offset.dot(proposed_closest_dir) / proposed_closest_dist

        while(x > 1.0):
            self.closest_i += 1
            self.closest_i = self.closest_i % len(self.trajectory)
            proposed_closest_dir = self.trajectory[self.closest_i + 1] - self.trajectory[self.closest_i] 
            proposed_closest_dist = np.linalg.norm(proposed_closest_dir)
            proposed_closest_dir = proposed_closest_dir / proposed_closest_dist

            offset = pos - self.trajectory[self.closest_i]
            x = offset.dot(proposed_closest_dir) / (proposed_closest_dist)
            # x = offset.dot(proposed_closest_dir) / (np.linalg.norm(offset))
        self.closest_point = self.trajectory[ self.closest_i ]

        self.target_i = self.closest_i
        while np.linalg.norm(self.closest_point - self.trajectory[ self.target_i ]) < self.TARGET_DISTANCE:
            self.target_i += 1
            self.target_i = self.target_i % len(self.trajectory)
            
        self.target_point = self.trajectory[ self.target_i ]
            

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
        vel_desired = self.getDesiredVelocity(pos, self.target_point, yaw, vel)

        # Height (easy)
        des_height = self.target_point[2]

        # Average for stability
        output = [vel_desired[0], vel_desired[1], yawrate_desired, des_height]
        output_smoothed = self.getAverageSmoothing(output)

        output_vel_desired, yawrate_desired, des_height = output_smoothed[:2], output_smoothed[2], output_smoothed[3] 

        fin = len(self.trajectory) - self.target_i < 10

        # print("Vel: ",np.linalg.norm(output_vel_desired))
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
    
    def getDesiredVelocity(self, pos, target_pos, yaw, actual_vel):
        R = np.array(
            [[cos(yaw), -sin(yaw)],
             [sin(yaw), cos(yaw)]
        ])
        
        dpos = (np.array(target_pos) - np.array(pos))[:2]
        vel_desired = np.matmul(dpos, R)
        # vel_deisred = self.trajectory[ self.target_i + 1] - self.trajectory[ self.target_i + 1]

        # Clip for stability
        # dist = np.linalg.norm(current_gate_pos-pos)
        # past_dist = np.linalg.norm(past_gate_pos-pos)
        # dist_factor = min(dist, past_dist) / 2
        # dist_factor = min(1, dist_factor)


        vel_des_norm = np.linalg.norm(vel_desired)
        vel_desired_normalized = np.array(vel_desired) / vel_des_norm

        # scale = (self.MAX_VEL_NORM-self.MIN_VEL_NORM) * dist_factor + self.MIN_VEL_NORM
        # print("yawdes factor", (1-(abs(yaw_des)/(np.pi/2))))
        
        # actual_vel_normalized = actual_vel / np.linalg.norm(actual_vel)
        # scale = vel_desired_normalized.dot(actual_vel_normalized) 
        # scale =  max(0, np.cos(5*np.arccos(scale)))
        # scale = self.curvature[ self.closest_i ]
        # scale = (self.MAX_VEL_NORM-self.MIN_VEL_NORM) * scale + self.MIN_VEL_NORM
        # scale = np.clip(scale, self.MIN_VEL_NORM, self.MAX_VEL_NORM)
        # scale = 1.0
        # print("Scale factor: ", scale)
        scale = self.curvature[ self.closest_i ]
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
        
        output_avg[2] = np.ma.average(
            [x[2] for x in self.history], 
            axis=0,
            weights=list(reversed([0.9**i for i in range(len(self.history))]))
            )
        
        return output_avg
    


    def old_find_target_point(self, pos):
        ## Calculate target point
        best_i = -1
        best_dist = float('inf')
        for i in range(100):
            if self.target_i + i >= len(self.trajectory):
                break

            proposed_target = self.trajectory[self.target_i + i]
            proposed_target_dist = abs(np.linalg.norm(pos-proposed_target) - 0.3)

            if proposed_target_dist < best_dist:
                best_i = i
                best_dist = proposed_target_dist
        
        self.target_i = self.target_i + best_i
        self.target_point = self.trajectory[ self.target_i ]

        ### /////////////////////
        dist_closest = float('inf')
        for i in range(min(0, self.closest_i-100), min(len(self.trajectory), self.closest_i+100)):
            proposed_closest = self.trajectory[i]
            
            if np.linalg.norm(pos-proposed_closest) < dist_closest:
                self.closest_i = i
                dist_closest = np.linalg.norm(pos-proposed_closest)
        self.closest_point = self.trajectory[self.closest_i]