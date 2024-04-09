import numpy as np
from scipy.spatial.transform import Rotation as R
from copy import deepcopy

class TrackGenerator():
    def __init__(self, 
                 num_gate_poses: int = 4, 
                 min_distance: float = 2.5, 
                 delta_yaw: float = np.pi/9,
                 boundaries: dict = {"x":[-2, 2], "y":[-2, 2], "z":[-0.5, 1.0]}):
        self.n = num_gate_poses
        self.min_dist = min_distance
        self.delta_yaw = delta_yaw
        self.boundaries = boundaries

    def generate(self):
        gate_poses = []
        random_pos, random_rot, yaw = self.get_random_gate_pose()
        while np.linalg.norm(random_pos) < self.min_dist:
                random_pos, random_rot, yaw  = self.get_random_gate_pose()

        gate_poses.append({
            'pos': random_pos,
            'rot': random_rot
        })

        for _ in range(self.n-1):
            next_random_pos, next_random_rot, yaw = self.get_random_gate_pose(
                                                        yaw_bounds=[yaw-self.delta_yaw, yaw+self.delta_yaw]
                                                    )
            while np.linalg.norm(gate_poses[-1]['pos'] - next_random_pos) < self.min_dist:
                next_random_pos, next_random_rot, yaw = self.get_random_gate_pose(
                                                        yaw_bounds=[yaw-self.delta_yaw, yaw+self.delta_yaw]
                                                    )
            gate_poses.append({
                'pos': next_random_pos,
                'rot': next_random_rot
            })
            
        return gate_poses
    
    def generate_easy(self):
        gate_poses = []
        gate_poses.append({
            'pos': np.array([0,2,0]),
            'rot': np.array([0,0,1,0])
        })
        gate_poses.append({
            'pos': np.array([2,2,0]),
            'rot': np.array([0,0,1,np.pi/4])
        })
        gate_poses.append({
            'pos': np.array([2,-2,0]),
            'rot': np.array([0,0,1,np.pi/2])
        })
        return gate_poses

    def get_random_gate_pose(self, yaw_bounds: list = [0, 2*np.pi]):
        random_pos = [
                        np.random.uniform(self.boundaries['x'][0], self.boundaries['x'][1]),
                        np.random.uniform(self.boundaries['y'][0], self.boundaries['y'][1]),
                        np.random.uniform(self.boundaries['z'][0], self.boundaries['z'][1])
                    ]
        yaw = np.pi/2 + np.random.uniform(yaw_bounds[0], yaw_bounds[1])
        r = R.from_euler('zyx', [yaw, 0, 0], degrees=False)
        random_rot = r.as_quat().tolist()

        return np.array(random_pos), np.array(random_rot), yaw

    def to_gate_squares(self, gate_poses):
        gate_square_poses = deepcopy(gate_poses)
        for sq in gate_square_poses:
            sq['pos'][2] += 1
        return gate_square_poses