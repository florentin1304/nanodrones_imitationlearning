import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
from optimal_splines.DroneTrajectory import DroneTrajectory
from maveric_trajectory_planner import *
from copy import copy

def get_trajectory_waypoints(gates):
    waypoints = [np.array([0, 0, 0.5])]
    for gate in gates:
        waypoints += sorted(get_gate_waypoints(gate['pos'], gate['rot']),
                            key=lambda x: np.linalg.norm(x-waypoints[-1]))

    return waypoints

def get_gate_waypoints(gate_pos, gate_rot):
    r = R.from_quat(gate_rot)
    rot_mat = r.as_matrix()
    wp_before = np.array(gate_pos) + np.array([1.0,0,0]) @ rot_mat
    wp_after = np.array(gate_pos) - np.array([1.0,0,0]) @ rot_mat
    return [wp_before, np.array(gate_pos), wp_after]

def get_waypoints(drone_state, gate_state, next_gate_state):
    waypoints = []

    drone_pos = drone_state[:3]
    waypoints.append(drone_pos)

    # Add an extra point in front of the drone
    roll, pitch, yaw = drone_state[3:]
    r = R.from_euler("ZYX", (yaw, 0, 0))
    drone_padding_wp = np.array(drone_pos) - np.array([0.5,0,0]) @ r.as_matrix()
    drone_padding_wp[2] += 0.1
    waypoints.append(drone_padding_wp)

    # Add padding for before gate
    gate_pos = gate_state[:3]
    gate_angles = gate_state[3:]
    gate_yaw = gate_angles[3]
    print(f'{gate_yaw=}')
    r = R.from_euler("ZYX", (gate_yaw, 0, 0))
    gate_padding_wp = np.array(gate_pos) - np.array([0,1,0]) @ r.as_matrix()
    gate_padding_wp[2] += 1
    waypoints.append(gate_padding_wp)

    waypoints.append(gate_pos)
    waypoints[-1][2] = 1
    # Adding padding for after gate
    r = R.from_euler("ZYX", (gate_yaw, 0, 0))
    gate_padding_wp = np.array(gate_pos) + np.array([0,1,0]) @ r.as_matrix()
    gate_padding_wp[2] += 1
    waypoints.append(gate_padding_wp)

    # Add padding for
    next_gate_pos = next_gate_state[:3]
    next_gate_angles = next_gate_state[3:]
    next_gate_yaw = next_gate_angles[3]
    r = R.from_euler("ZYX", (next_gate_yaw, 0, 0),)
    next_gate_padding_wp = np.array(next_gate_pos) + np.array([0, 1,0]) @ r.as_matrix()
    gate_padding_wp[2] += 1
    waypoints.append(next_gate_padding_wp)


    return waypoints


def generate_trajecrory2(gates):
    aggr = 10_000 # CAPISCI POI PERCHE

    x0 = np.array([[0., 0., 0., 0., 0., 0., 0.]]).T
    drone_traj = DroneTrajectory()
    drone_traj.set_start(position=x0[:3], velocity=x0[3:6])
    drone_traj.set_end(position=x0[:3], velocity=x0[3:6])

    for gate in gates:
        trans, rot = gate['pos'], gate['rot']
        drone_traj.add_gate(trans, rot)
    drone_traj.solve(aggr)

    ref_traj = drone_traj.as_path(dt=0.001, frame='world', start_time=3)

    return ref_traj

def generate_trajectory(waypoints):
    waypoints = np.array(waypoints)
    x = waypoints[:, 0]
    y = waypoints[:, 1]
    z = waypoints[:, 2]

    tck,u = interpolate.splprep([x,y,z], s=0)

    wp_to_wp_distance = waypoints[1:] - waypoints[:-1]
    wp_to_wp_distance = np.linalg.norm(wp_to_wp_distance, axis=1) #forse axis 0
    total_distance = np.sum(wp_to_wp_distance)

    unew = np.arange(0, 1.01, 0.1/total_distance )
    out = interpolate.splev(unew, tck)
    # print(out)

    # ax = plt.figure().add_subplot(projection='3d')
    # ax.scatter(x,y,z)
    # ax.plot(out[0],out[1],out[2])
    # plt.title('Spline of parametrically-defined curve')
    # plt.show()
    out = np.array(out).T
    return out

