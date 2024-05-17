import numpy as np
import math
from copy import deepcopy
import pandas as pd
import os
from pathlib import Path

class TrajectoryGenerator:
    def __init__(self, traj_type='ellipse', traj_conf={}, discretization=1/1000, fov_reduction_factor=0.5):

        self.__DISCRETIZATION = discretization # milimeter
        self.__FOV_REDUCTION_FACTOR = fov_reduction_factor
        self.__FOV = 1.52
        self.__type = traj_type
        self.__traj_conf = traj_conf
        self.__generate_functions = {
            'ellipse': self.__generate_ellipse,
            'line': self.__generate_line,
            'csv': self.__generate_from_csv
        }


    def generate_trajectory(self):
        gen_func = self.__generate_functions.get(self.__type)
        
        assert gen_func != None, f"Trajectory type {self.__type} unrecognized!"

        traj = gen_func(**self.__traj_conf)
        return traj
        

    def __generate_line(self, length=10):
        x = np.linspace(0,length,int(length*1000/2))
        y = np.zeros_like(x)
        z = np.ones_like(x)

        traj = np.vstack([x,y,z]).T
        return traj
    
    def __generate_ellipse(self, radius:float=6.0, other_radius_frac:float=0.6, shift_left=False):
        # Calculate circumference to ensure 1 cm points
        rx = radius
        ry = radius * other_radius_frac

        h = np.power(rx-ry, 2) / np.power(rx+ry, 2)
        circumference = np.pi * (rx + ry) * (1 + (3 * (h / (10 + np.sqrt(4-3*h)) ) ))

        # Create t parameter and adjust:
        ### -3pi/2 if going to the right
        t = np.linspace(0, (1.99)*np.pi, int(circumference/self.__DISCRETIZATION)) - ((int(shift_left) + 1/2)*np.pi)
        if shift_left: t = t[::-1] # has to be reversed to start in front of the 

        x = rx*np.cos(t)
        y = ry*np.sin(t) + ((-1) ** int(shift_left))*ry
        z = np.ones_like(x)
        traj = np.vstack([x,y,z]).T

        return traj


    def __generate_from_csv(self, csv_file='controllers/controller_utils/base_tracks/a.csv'):
        curr_dir = os.path.dirname( os.path.abspath(__file__) )
        path = Path(curr_dir).parent.parent.absolute()
        file_path = os.path.join(path, csv_file)
        
        df = pd.read_csv(file_path)
        df = df[['x','y','z']]
        traj = df.to_numpy()
        return traj

    

    def generate_gate_positions_recursive(self, trajectory):
        # trajectory should be discretized by 1 centimeter
        if trajectory.shape[0] != 3:
            trajectory = trajectory.T
        x, y, z = trajectory

        gate_states = []

        i = int(1/self.__DISCRETIZATION) # 1 meter
        dpos = np.array([x[i],y[i]]) - np.array([x[i-1], y[i-1]])
        yaw = math.atan2(dpos[1], dpos[0])
        first_g_state = {
                'pos': np.array([x[i], y[i], z[i]-1]),
                'rot': np.array([0.0, 0.0, 1.0, yaw-(np.pi/2)]),
                'index': i
             } 
        gate_states.append(first_g_state)

        gate_states, ok = self.generate_gate_positions_R(gate_states, 
                                                         None, 
                                                         i, 
                                                         trajectory, 
                                                         max_index=len(trajectory[0])-100)

        print("Gate state correctly generated: ", ok)
        return gate_states




    def generate_gate_positions_R(self, current, solution, index, trajectory, max_index):
        if 1 < self.__DISCRETIZATION*(len(trajectory[0]) - current[-1]['index']) < 2 and len(current) > 3:
            # if self.check_correct_gate_placement(trajectory, current):
            solution = deepcopy(current)
            return solution, True
            # return current, False
        
        ### Find possible next gate position indexes
        x,y,z = trajectory
        admissible_delta_indexes = []
        for delta_i in range(int(3.5//self.__DISCRETIZATION), 
                             int(0.5//self.__DISCRETIZATION), 
                             -int(max(1, 0.25//self.__DISCRETIZATION))): # Move 0.25 meters everytime
            if index+delta_i >= max_index:
                continue

            new_gate_pos = np.array([x[index+delta_i], y[index+delta_i], z[index+delta_i]])
            old_gate_pos = current[-1]['pos']
            old_gate_yaw = current[-1]['rot'][-1]
            gate_dpos = new_gate_pos[:2] - old_gate_pos[:2]
            gate_yaw = math.atan2(gate_dpos[1], gate_dpos[0])

            if abs(gate_yaw - (old_gate_yaw+(np.pi/2)))  < self.__FOV_REDUCTION_FACTOR*(self.__FOV/2):
                admissible_delta_indexes.append(delta_i)
        
        ### Recursively try from the furthest one
        for delta_i in admissible_delta_indexes:
            # Add new gate to current solution
            next_i = index + delta_i
            dpos = np.array([x[next_i],y[next_i]]) - np.array([x[next_i-1], y[next_i-1]])
            yaw = math.atan2(dpos[1], dpos[0])
            next_g_state = {
                'pos': np.array([x[next_i], y[next_i], z[next_i]-1]),
                'rot': np.array([0.0, 0.0, 1.0, yaw-(np.pi/2)]),
                'index': next_i
            } 
            current.append(next_g_state)

            # Recursive call
            sol, finish = self.generate_gate_positions_R(current, solution, next_i, trajectory, max_index)
            if finish:
                return sol, finish
            
            # Backtrack
            current.pop(-1)
        
        return current, False
    
    def check_correct_gate_placement(self, trajectory, solution):
        # The last gate doesnt have to be checked
        index_last_gate = solution[-1]['index']
        indexes_to_check = [ int((solution[i+1]['index'] + solution[i]['index'])/2) for i in range(len(solution[:-1]))]
        print((indexes_to_check))
        for index in indexes_to_check:
            point = trajectory[:, index]
            dpoint = trajectory[:, index+1] - trajectory[:, index+1]
            point_angle = math.atan2(dpoint[1], dpoint[0])

            # Select correct gate index
            correct_gate_i = -1
            for i, gate_other in enumerate(solution):
                if gate_other['index'] < index:
                    continue
                correct_gate_i = i
                break


            nearest_gate_in_fov = -1
            distance_nearest = float('inf')
            for i, gate_other in enumerate(solution):
                gate_dpos = gate_other['pos'] - point
                gate_angle = math.atan2(gate_dpos[1], gate_dpos[0])


                if abs(gate_angle - point_angle)  < (self.__FOV/2):
                    if np.linalg.norm(gate_dpos) < distance_nearest:
                        nearest_gate_in_fov = i
                        distance_nearest = np.linalg.norm(gate_dpos)


            # If the nearest gate in field of view of 
            if nearest_gate_in_fov != correct_gate_i:
                return False

        return True
        


    # def generate_gate_positions_ellipse(self, trajectory):
    #     # trajectory should be discretized by 1 centimeter
    #     if trajectory.shape[0] != 3:
    #         trajectory = trajectory.T
    #     x, y, z = trajectory

    #     gate_states = []

    #     i = 100
    #     dpos = np.array([x[i],y[i]]) - np.array([x[i-1], y[i-1]])
    #     yaw = math.atan2(dpos[1], dpos[0])
    #     first_g_state = {
    #             'pos': np.array([x[i], y[i], z[i]-1]),
    #             'rot': np.array([0.0, 0.0, 1.0, yaw-(np.pi/2)]),
    #             'index': i
    #         } 
    #     gate_states.append(first_g_state)

    #     gates_stop = False
    #     while i < len(x)-100 and not gates_stop:
    #         # Find all next admissible points (in the FOV)
    #         admissible_indexes = []
    #         for delta_i in range(100, 1000):
    #             if i+delta_i >= len(x) - 100:
    #                 if len(gate_states) > 2 and len(x)-next_i < 300:
    #                     gates_stop = True
    #                     break
    #                 continue

    #             new_gate_pos = np.array([x[i+delta_i], y[i+delta_i], z[i+delta_i]])
    #             old_gate_pos = gate_states[-1]['pos']
    #             old_gate_yaw = gate_states[-1]['rot'][-1]
    #             gate_dpos = new_gate_pos[:2] - old_gate_pos[:2]
    #             gate_yaw = math.atan2(gate_dpos[1], gate_dpos[0])

    #             if abs(gate_yaw - (old_gate_yaw+(np.pi/2)))  < 0.7*(self.__FOV/2):
    #                 admissible_indexes.append(i+delta_i)
            
    #         print(i)
    #         print(len(admissible_indexes))

    #         if admissible_indexes == []:
    #             if len(gate_states) == 1:
    #                 raise Exception("wtf")
    #             gate_states.pop(-1)
    #             i = gate_states[-1]['index']
    #             continue
                
            

    #         next_i = admissible_indexes[-1] #np.random.choice(admissible_indexes)
            
    #         dpos = np.array([x[next_i],y[next_i]]) - np.array([x[next_i-1], y[next_i-1]])
    #         yaw = math.atan2(dpos[1], dpos[0])
    #         next_g_state = {
    #             'pos': np.array([x[next_i], y[next_i], z[next_i]-1]),
    #             'rot': np.array([0.0, 0.0, 1.0, yaw-(np.pi/2)]),
    #             'index': i
    #         } 

    #         i = next_i
    #         gate_states.append(next_g_state)


            
    #     return gate_states
            
    
    def to_gate_squares(self, gate_poses):
        gate_square_poses = deepcopy(gate_poses)
        for sq in gate_square_poses:
            sq['pos'][2] += 1
        return gate_square_poses


import matplotlib.pyplot as plt
if __name__ == '__main__':

    tg = TrajectoryGenerator()
    rx = np.random.uniform(1,7)
    ry = np.random.uniform(1,7)
    left = np.random.choice([False, True])
    # trajectory = tg.generate_line()
    # print()
    traj= tg.generate_ellipse(rx=rx, ry=ry, shift_left=left)

    # x,y,z = traj

    tg.generate_gate_positions_recursive(traj)
