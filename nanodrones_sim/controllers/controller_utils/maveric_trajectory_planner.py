from numpy import *
from scipy.sparse import csc_matrix
import osqp
from scipy.spatial.transform import Rotation as R
import math

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from numpy import *
import matplotlib.pyplot as plot


def draw_traj(waypoints, trajectory):
    """
    Visualize the trajectories in every dimension by using matplotlib.

    The code is quite repetitive and might be optimized, but it works...
    """
    mpl.rcParams['legend.fontsize'] = 10
    mpl.rcParams['figure.figsize'] = (30, 30)

    # =============================
    # 3D Plot
    # =============================
    ax = plot.subplot2grid((23, 31), (0, 0), colspan=13, rowspan=13, projection='3d')  # create Axes3D object, which can plot in 3D
    for i in range(len(trajectory)):
        t = linspace(waypoints[i].time, waypoints[i+1].time, int((waypoints[i+1].time-waypoints[i].time)*20))
        x_path = trajectory[i][0] * t ** 4 + trajectory[i][1] * t ** 3 + trajectory[i][2] * t ** 2 + trajectory[i][3] * t + trajectory[i][4]
        y_path = trajectory[i][5] * t ** 4 + trajectory[i][6] * t ** 3 + trajectory[i][7] * t ** 2 + trajectory[i][8] * t + trajectory[i][9]
        z_path = trajectory[i][10] * t ** 4 + trajectory[i][11] * t ** 3 + trajectory[i][12] * t ** 2 + trajectory[i][13] * t + trajectory[i][14]

        ax.plot(x_path, y_path, z_path, label='[%d] to [%d]' %(i, i+1))  # plot trajectory
        ax.plot([waypoints[i+1].x], [waypoints[i+1].y], [waypoints[i+1].z],'ro')  # plot start
        if i == 0:
            ax.plot([waypoints[i].x], [waypoints[i].y], [waypoints[i].z], 'ro')  # plot end

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # =============================
    # Position Plots
    # =============================
    # add 2D plot of X over time
    ax = plot.subplot2grid((23, 31), (13, 0),  colspan = 6, rowspan=4)
    for i in range(len(trajectory)):
        t = linspace(waypoints[i].time, waypoints[i+1].time, int((waypoints[i+1].time-waypoints[i].time)*20))
        x_path = trajectory[i][0] * t ** 4 + trajectory[i][1] * t ** 3 + trajectory[i][2] * t ** 2 + trajectory[i][3] * t + trajectory[i][4]
        ax.plot(t, x_path)
    ax.set_ylabel('X')

    # add 2D plot of Y over time
    ax = plot.subplot2grid((23, 31), (19, 0),  colspan = 6, rowspan=4)
    for i in range(len(trajectory)):
        t = linspace(waypoints[i].time, waypoints[i+1].time, int((waypoints[i+1].time-waypoints[i].time)*20))
        y_path = trajectory[i][5] * t ** 4 + trajectory[i][6] * t ** 3 + trajectory[i][7] * t ** 2 + trajectory[i][8] * t + trajectory[i][9]
        ax.plot(t, y_path)
    ax.set_ylabel('Y')
    ax.set_xlabel('Time')

    # add 2D plot of Z over time
    ax = plot.subplot2grid((23, 31), (13, 7),  colspan = 6, rowspan=4)
    for i in range(len(trajectory)):
        t = linspace(waypoints[i].time, waypoints[i+1].time, int((waypoints[i+1].time-waypoints[i].time)*20))
        z_path = trajectory[i][10] * t ** 4 + trajectory[i][11] * t ** 3 + trajectory[i][12] * t ** 2 + trajectory[i][13] * t + trajectory[i][14]
        ax.plot(t, z_path)
    ax.set_ylabel('Z')

    # add 2D plot of Yaw over time
    ax = plot.subplot2grid((23, 31), (19, 7), colspan = 6, rowspan=4)
    for i in range(len(trajectory)):
        t = linspace(waypoints[i].time, waypoints[i+1].time, int((waypoints[i+1].time-waypoints[i].time)*20))
        yaw_path = trajectory[i][15] * t ** 2 + trajectory[i][16] * t + trajectory[i][17]
        ax.plot(t, yaw_path)
    ax.set_ylabel('Yaw')
    ax.set_xlabel('Time')

    # =============================
    # Velocity Plots
    # =============================
    # add 2D plot of X over time
    ax = plot.subplot2grid((23, 31), (0, 15), colspan=3, rowspan=4)
    for i in range(len(trajectory)):
        t = linspace(waypoints[i].time, waypoints[i + 1].time, int((waypoints[i + 1].time - waypoints[i].time) * 20))
        x_path = 4 * trajectory[i][0] * t ** 3 + 3 * trajectory[i][1] * t ** 2 + 2 * trajectory[i][2] * t + trajectory[i][3]
        ax.plot(t, x_path)
    ax.set_ylabel('X')

    # add 2D plot of Y over time
    ax = plot.subplot2grid((23, 31), (6, 15), colspan=3, rowspan=4)
    for i in range(len(trajectory)):
        t = linspace(waypoints[i].time, waypoints[i + 1].time, int((waypoints[i + 1].time - waypoints[i].time) * 20))
        y_path = 4 * trajectory[i][5] * t ** 3 + 3 * trajectory[i][6] * t ** 2 + 2* trajectory[i][7] * t + trajectory[i][8]
        ax.plot(t, y_path)
    ax.set_ylabel('Y')
    ax.set_xlabel('Time')

    # add 2D plot of Z over time
    ax = plot.subplot2grid((23, 31), (0, 19), colspan=3, rowspan=4)
    for i in range(len(trajectory)):
        t = linspace(waypoints[i].time, waypoints[i + 1].time, int((waypoints[i + 1].time - waypoints[i].time) * 20))
        z_path = 4 * trajectory[i][10] * t ** 3 + 3 * trajectory[i][11] * t ** 2 + 2 * trajectory[i][12] * t + trajectory[i][13]
        ax.plot(t, z_path)
    ax.set_ylabel('Z')

    # add 2D plot of Yaw over time
    ax = plot.subplot2grid((23, 31), (6, 19), colspan=3, rowspan=4)
    for i in range(len(trajectory)):
        t = linspace(waypoints[i].time, waypoints[i + 1].time, int((waypoints[i + 1].time - waypoints[i].time) * 20))
        yaw_path = 2 * trajectory[i][15] * t + trajectory[i][16]
        ax.plot(t, yaw_path)
    ax.set_ylabel('Yaw')
    ax.set_xlabel('Time')

    # =============================
    # Acceleration Plots
    # =============================
    # add 2D plot of X over time
    ax = plot.subplot2grid((23, 31), (13, 15), colspan=3, rowspan=4)
    for i in range(len(trajectory)):
        t = linspace(waypoints[i].time, waypoints[i + 1].time, int((waypoints[i + 1].time - waypoints[i].time) * 20))
        x_path = 12 * trajectory[i][0] * t ** 2 + 6 * trajectory[i][1] * t + 2 * trajectory[i][2]
        ax.plot(t, x_path)
    ax.set_ylabel('X')

    # add 2D plot of Y over time
    ax = plot.subplot2grid((23, 31), (19, 15), colspan=3, rowspan=4)
    for i in range(len(trajectory)):
        t = linspace(waypoints[i].time, waypoints[i + 1].time, int((waypoints[i + 1].time - waypoints[i].time) * 20))
        y_path = 12 * trajectory[i][5] * t ** 2 + 6 * trajectory[i][6] * t + 2 * trajectory[i][7]
        ax.plot(t, y_path)
    ax.set_ylabel('Y')
    ax.set_xlabel('Time')

    # add 2D plot of Z over time
    ax = plot.subplot2grid((23, 31), (13, 19), colspan=3, rowspan=4)
    for i in range(len(trajectory)):
        t = linspace(waypoints[i].time, waypoints[i + 1].time, int((waypoints[i + 1].time - waypoints[i].time) * 20))
        z_path = 12 * trajectory[i][10] * t ** 2 + 6 * trajectory[i][11] * t + 2 * trajectory[i][12]
        ax.plot(t, z_path)
    ax.set_ylabel('Z')

    # add 2D plot of Yaw over time
    ax = plot.subplot2grid((23, 31), (19, 19), colspan=3, rowspan=4)
    for i in range(len(trajectory)):
        t = linspace(waypoints[i].time, waypoints[i + 1].time, int((waypoints[i + 1].time - waypoints[i].time) * 20))
        oneVec = linspace(1,1, int((waypoints[i + 1].time - waypoints[i].time) * 20)) 
        yaw_path = 2 * trajectory[i][15] * oneVec
        ax.plot(t, yaw_path)
    ax.set_ylabel('Yaw')
    ax.set_xlabel('Time')

    # =============================
    # Jerk Plots
    # =============================
    # add 2D plot of X over time
    ax = plot.subplot2grid((23, 31), (0, 24), colspan=3, rowspan=4)
    for i in range(len(trajectory)):
        t = linspace(waypoints[i].time, waypoints[i + 1].time, int((waypoints[i + 1].time - waypoints[i].time) * 20))
        x_path = 24 * trajectory[i][0] * t + 6 * trajectory[i][1]
        ax.plot(t, x_path)
    ax.set_ylabel('X')

    # add 2D plot of Y over time
    ax = plot.subplot2grid((23, 31), (6, 24), colspan=3, rowspan=4)
    for i in range(len(trajectory)):
        t = linspace(waypoints[i].time, waypoints[i + 1].time, int((waypoints[i + 1].time - waypoints[i].time) * 20))
        y_path = 24 * trajectory[i][5] * t + 6 * trajectory[i][6]
        ax.plot(t, y_path)
    ax.set_ylabel('Y')
    ax.set_xlabel('Time')

    # add 2D plot of Z over time
    ax = plot.subplot2grid((23, 31), (0, 28), colspan=3, rowspan=4)
    for i in range(len(trajectory)):
        t = linspace(waypoints[i].time, waypoints[i + 1].time, int((waypoints[i + 1].time - waypoints[i].time) * 20))
        z_path = 24 * trajectory[i][10] * t + 6 * trajectory[i][11]
        ax.plot(t, z_path)
    ax.set_ylabel('Z')
    ax.set_xlabel('Time')

    # =============================
    # Snap Plots
    # =============================
    # add 2D plot of X over time
    ax = plot.subplot2grid((23, 31), (13, 24), colspan=3, rowspan=4)
    for i in range(len(trajectory)):
        t = linspace(waypoints[i].time, waypoints[i + 1].time, int((waypoints[i + 1].time - waypoints[i].time) * 20))
        oneVec = linspace(1,1, int((waypoints[i + 1].time - waypoints[i].time) * 20))
        x_path = 24 *trajectory[i][0] * oneVec
        ax.plot(t, x_path)
    ax.set_ylabel('X')

    # add 2D plot of Y over time
    ax = plot.subplot2grid((23, 31), (19, 24), colspan=3, rowspan=4)
    for i in range(len(trajectory)):
        t = linspace(waypoints[i].time, waypoints[i + 1].time, int((waypoints[i + 1].time - waypoints[i].time) * 20))
        oneVec = linspace(1,1, int((waypoints[i + 1].time - waypoints[i].time) * 20))
        y_path = 24 * trajectory[i][5] * oneVec
        ax.plot(t, y_path)
    ax.set_ylabel('Y')
    ax.set_xlabel('Time')

    # add 2D plot of Z over time
    ax = plot.subplot2grid((23, 31), (13, 28), colspan=3, rowspan=4)
    for i in range(len(trajectory)):
        t = linspace(waypoints[i].time, waypoints[i + 1].time, int((waypoints[i + 1].time - waypoints[i].time) * 20))
        oneVec = linspace(1,1, int((waypoints[i + 1].time - waypoints[i].time) * 20))
        z_path = 24 * trajectory[i][10] * oneVec
        ax.plot(t, z_path)
    ax.set_ylabel('Z')
    ax.set_xlabel('Time')

    # =============================
    # Labels
    # =============================
    ax = plot.subplot2grid((23, 31), (18, 6))
    ax.set_frame_on(False)
    ax.axis('off')
    ax.text(-0.3,0.7,"Position", fontweight='bold')

    ax = plot.subplot2grid((23, 31), (5, 18))
    ax.set_frame_on(False)
    ax.axis('off')
    ax.text(-0.3,0.7,"Velocity", fontweight='bold')

    ax = plot.subplot2grid((23, 31), (18, 18))
    ax.set_frame_on(False)
    ax.axis('off')
    ax.text(-0.7,0.7,"Acceleration", fontweight='bold')

    ax = plot.subplot2grid((23, 31), (5, 27))
    ax.set_frame_on(False)
    ax.axis('off')
    ax.text(0,0.7,"Jerk", fontweight='bold')

    ax = plot.subplot2grid((23, 31), (18, 27))
    ax.set_frame_on(False)
    ax.axis('off')
    ax.text(-0.1,0.7,"Snap", fontweight='bold')

    #plot.figtext(0, 0, 'Planned Trajectory:\n '
    #                   '(X,Y,Z,Yaw,X_dot,Y_dot,Z_dot)\n '
    #                   'Start: (%0.2f, %0.2f, %0.2f, %0.2f, %0.2f,%0.2f, %0.2f)\n '
    #                   'End: (%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f) \n'
    #                   'Time for segment: %0.2f'
    #             % (waypoint0.x, waypoint0.y, waypoint0.z, waypoint0.yaw, waypoint0.x_dot, waypoint0.y_dot,
    #                waypoint0.z_dot,
    #                waypoint1.x, waypoint1.y, waypoint1.z, waypoint1.yaw, waypoint1.x_dot, waypoint1.y_dot,
    #                waypoint1.z_dot,
    #                waypoint1.time - waypoint0.time))

    # print to screen
    plot.savefig('traj.png')

class Waypoint:
    """
    Store a waypoint.

    Attributes:
        x,y,z: position in world frame
        yaw: Euler angle of waypoint
        time: time at which the waypoint is to be reached
    """
    def __init__(self, x, y, z, yaw, time):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw


        r = R.from_euler('Z', yaw)
        vel = (r.as_matrix() @ array([1.0, 0, 0]))
        self.vel_x = vel[0]
        self.vel_y = vel[1]

        self.time = time


def calc_time(start, end):
    """
    Calculate a fitting time for the trajectory between two waypoints.
    """
    distance3D = sqrt((start[0]-end[0])**2 + (start[1]-end[1])**2 + (start[2]-end[2])**2)
    time = distance3D  # ... where an engineer cries and a programmer sees an easy solution
    return time

def joint(waypoints):
    """
    Calculate a trajectory by a joint operation.
    """
    # total number of segments
    numSegments = len(waypoints) - 1
    # every segment has its own polynomial of 4th degree for X,Y and Z and a polynomial of 2nd degree for Yaw
    numCoefficients = numSegments * (3*5+3)
    # list of calculated trajectory coefficients
    trajectory = []
    # start + end X,Y,Z,Yaw position for every segment: 8
    # rendezvous X,Y,Z,Yaw velocity: 4
    # absolute start + end X,Y,Z (+ start Yaw) velocity: 7
    numConstraints = numSegments * 8 + (numSegments - 1) * 4 + 7

    P_numpy = zeros((numCoefficients, numCoefficients))
    for i in range(numSegments):
        P_numpy[0 + i * 18, 0 + i * 18] = 1  # minimize snap for X
        # P_numpy[2 + i * 18, 2 + i * 18] = 100 # minimize acceleration for X
        P_numpy[5 + i * 18, 5 + i * 18] = 1  # minimize snap for Y
        P_numpy[7 + i * 18, 7 + i * 18] = 100  # minimize acceleration for Y
        P_numpy[10 + i * 18, 10 + i * 18] = 1  # minimize snap for Z
        P_numpy[12 + i * 18, 12 + i * 18] = 100  # minimize acceleration for Z
        # P_numpy[15 + i * 18, 15 + i * 18] = 100  # minimize acceleration for Yaw
    P = csc_matrix(P_numpy)  # convert to CSC for performance

    # =============================
    # Gradient vector (linear terms), we have none
    # =============================
    q = zeros((numCoefficients, 1))
    q = hstack(q)  # convert to hstack for performance

    # =============================
    # Inequality matrix (left side), we have none
    # =============================
    G = zeros((numConstraints, numCoefficients))

    # =============================
    # Inequality vector (right side), we have none
    # =============================
    h = zeros((numConstraints, 1))
    h = hstack(h)  # convert to hstack for performance

    # =============================
    # Equality matrix (left side)
    # =============================
    A = zeros((numConstraints, numCoefficients))

    # =============================
    # Equality vector (right side)
    # =============================
    b = zeros((numConstraints, 1))

    # =============================
    # Set up of Equality Constraints
    # =============================
    cc = -1  # Current Constraint
    for i in range(numSegments):
        # "start of segment" position constraints
        cc += 1  # X Position
        A[cc, 0 + i * 18] = waypoints[i].time ** 4
        A[cc, 1 + i * 18] = waypoints[i].time ** 3
        A[cc, 2 + i * 18] = waypoints[i].time ** 2
        A[cc, 3 + i * 18] = waypoints[i].time
        A[cc, 4 + i * 18] = 1
        b[cc, 0] = waypoints[i].x
        cc += 1  # Y Position
        A[cc, 5 + i * 18] = waypoints[i].time ** 4
        A[cc, 6 + i * 18] = waypoints[i].time ** 3
        A[cc, 7 + i * 18] = waypoints[i].time ** 2
        A[cc, 8 + i * 18] = waypoints[i].time
        A[cc, 9 + i * 18] = 1
        b[cc, 0] = waypoints[i].y
        cc += 1  # Z Position
        A[cc, 10 + i * 18] = waypoints[i].time ** 4
        A[cc, 11 + i * 18] = waypoints[i].time ** 3
        A[cc, 12 + i * 18] = waypoints[i].time ** 2
        A[cc, 13 + i * 18] = waypoints[i].time
        A[cc, 14 + i * 18] = 1
        b[cc, 0] = waypoints[i].z
        cc += 1  # Yaw Angle
        A[cc, 15 + i * 18] = waypoints[i].time ** 2
        A[cc, 16 + i * 18] = waypoints[i].time
        A[cc, 17 + i * 18] = 1
        b[cc, 0] = waypoints[i].yaw

        # "end of segment" position constraints
        cc += 1  # X Position
        A[cc, 0 + i * 18] = waypoints[i + 1].time ** 4
        A[cc, 1 + i * 18] = waypoints[i + 1].time ** 3
        A[cc, 2 + i * 18] = waypoints[i + 1].time ** 2
        A[cc, 3 + i * 18] = waypoints[i + 1].time
        A[cc, 4 + i * 18] = 1
        b[cc, 0] = waypoints[i + 1].x
        cc += 1  # Y Position
        A[cc, 5 + i * 18] = waypoints[i + 1].time ** 4
        A[cc, 6 + i * 18] = waypoints[i + 1].time ** 3
        A[cc, 7 + i * 18] = waypoints[i + 1].time ** 2
        A[cc, 8 + i * 18] = waypoints[i + 1].time
        A[cc, 9 + i * 18] = 1
        b[cc, 0] = waypoints[i + 1].y
        cc += 1  # Z Position
        A[cc, 10 + i * 18] = waypoints[i + 1].time ** 4
        A[cc, 11 + i * 18] = waypoints[i + 1].time ** 3
        A[cc, 12 + i * 18] = waypoints[i + 1].time ** 2
        A[cc, 13 + i * 18] = waypoints[i + 1].time
        A[cc, 14 + i * 18] = 1
        b[cc, 0] = waypoints[i + 1].z
        cc += 1  # Yaw Angle
        A[cc, 15 + i * 18] = waypoints[i + 1].time ** 2
        A[cc, 16 + i * 18] = waypoints[i + 1].time
        A[cc, 17 + i * 18] = 1
        b[cc, 0] = waypoints[i + 1].yaw

        # segment rendezvous constraints
        if i == 0:
            continue

        cc += 1  # X Velocity Rendezvous
        A[cc, 0 + i * 18] = 4 * waypoints[i].time ** 3
        A[cc, 1 + i * 18] = 3 * waypoints[i].time ** 2
        A[cc, 2 + i * 18] = 2 * waypoints[i].time
        A[cc, 3 + i * 18] = 1
        # b[cc, 0] = waypoints[i].vel_x #CONSTRAINT IN VELOCITA SU X

        A[cc, 0 + i * 18 - 18] = -1 * A[cc, 0 + i * 18]
        A[cc, 1 + i * 18 - 18] = -1 * A[cc, 1 + i * 18]
        A[cc, 2 + i * 18 - 18] = -1 * A[cc, 2 + i * 18]
        A[cc, 3 + i * 18 - 18] = -1 * A[cc, 3 + i * 18]
        cc += 1  # Y Velocity Rendezvous
        A[cc, 5 + i * 18] = 4 * waypoints[i].time ** 3
        A[cc, 6 + i * 18] = 3 * waypoints[i].time ** 2
        A[cc, 7 + i * 18] = 2 * waypoints[i].time
        A[cc, 8 + i * 18] = 1
        # b[cc, 0] =  waypoints[i].vel_y #CONSTRAINT IN VELOCITA SU Y


        A[cc, 5 + i * 18 - 18] = -1 * A[cc, 5 + i * 18]
        A[cc, 6 + i * 18 - 18] = -1 * A[cc, 6 + i * 18]
        A[cc, 7 + i * 18 - 18] = -1 * A[cc, 7 + i * 18]
        A[cc, 8 + i * 18 - 18] = -1 * A[cc, 8 + i * 18]
        cc += 1  # Z Velocity Rendezvous
        A[cc, 10 + i * 18] = 4 * waypoints[i].time ** 3
        A[cc, 11 + i * 18] = 3 * waypoints[i].time ** 2
        A[cc, 12 + i * 18] = 2 * waypoints[i].time
        A[cc, 13 + i * 18] = 1
        A[cc, 10 + i * 18 - 18] = -1 * A[cc, 10 + i * 18]
        A[cc, 11 + i * 18 - 18] = -1 * A[cc, 11 + i * 18]
        A[cc, 12 + i * 18 - 18] = -1 * A[cc, 12 + i * 18]
        A[cc, 13 + i * 18 - 18] = -1 * A[cc, 13 + i * 18]
        cc += 1  # Yaw Velocity Rendezvous
        A[cc, 15 + i * 18] = 2 * waypoints[i].time
        A[cc, 16 + i * 18] = 1
        A[cc, 15 + i * 18 - 18] = -1 * A[cc, 15 + i * 18]
        A[cc, 16 + i * 18 - 18] = -1 * A[cc, 16 + i * 18]

        
        # cc += 1  # X Acceleration Rendezvous
        # A[cc, 0 + i * 18] = 12 * waypoints[0].time ** 2
        # A[cc, 1 + i * 18] = 6 * waypoints[0].time
        # A[cc, 2 + i * 18] = 2
        # A[cc, 0 + i * 18 - 18] = -1 * A[cc, 0 + i * 18]
        # A[cc, 1 + i * 18 - 18] = -1 * A[cc, 1 + i * 18]
        # A[cc, 2 + i * 18 - 18] = -1 * A[cc, 2 + i * 18]
        # cc += 1  # Y Acceleration Rendezvous
        # A[cc, 5 + i * 18] = 12 * waypoints[0].time ** 2
        # A[cc, 6 + i * 18] = 6 * waypoints[0].time
        # A[cc, 7 + i * 18] = 2
        # A[cc, 5 + i * 18 - 18] = -1 * A[cc, 5 + i * 18]
        # A[cc, 6 + i * 18 - 18] = -1 * A[cc, 6 + i * 18]
        # A[cc, 7 + i * 18 - 18] = -1 * A[cc, 7 + i * 18]
        # cc += 1  # Z Acceleration Rendezvous
        # A[cc, 10 + i * 18] = 12 * waypoints[0].time ** 2
        # A[cc, 11 + i * 18] = 6 * waypoints[0].time
        # A[cc, 12 + i * 18] = 2
        # A[cc, 10 + i * 18 - 18] = -1 * A[cc, 10 + i * 18]
        # A[cc, 11 + i * 18 - 18] = -1 * A[cc, 11 + i * 18]
        # A[cc, 12 + i * 18 - 18] = -1 * A[cc, 12 + i * 18]
        # cc += 1  # Yaw Acceleration Rendezvous
        # A[cc, 15 + i * 18] = 2
        # A[cc, 15 + i * 18 - 18] = -1 * A[cc, 15 + i * 18]

        # cc += 1  # X Jerk Rendezvous
        # A[cc, 0] = 24 * waypoints[0].time
        # A[cc, 1] = 6
        # A[cc, 0 + i * 18 - 18] = -1 * A[cc, 0 + i * 18]
        # A[cc, 1 + i * 18 - 18] = -1 * A[cc, 1 + i * 18]
        # cc += 1  # Y Jerk Rendezvous
        # A[cc, 5] = 24 * waypoints[0].time
        # A[cc, 6] = 6
        # A[cc, 5 + i * 18 - 18] = -1 * A[cc, 5 + i * 18]
        # A[cc, 6 + i * 18 - 18] = -1 * A[cc, 6 + i * 18]
        # cc += 1  # Z Jerk Rendezvous
        # A[cc, 10] = 24 * waypoints[0].time
        # A[cc, 11] = 6
        # A[cc, 10 + i * 18 - 18] = -1 * A[cc, 10 + i * 18]
        # A[cc, 11 + i * 18 - 18] = -1 * A[cc, 11 + i * 18]
        #
        # cc += 1  # X Snap Rendezvous
        # A[cc, 0] = 24
        # A[cc, 0 + i * 18 - 18] = -1 * A[cc, 0 + i * 18]
        # cc += 1  # Y Snap Rendezvous
        # A[cc, 5] = 24
        # A[cc, 5 + i * 18 - 18] = -1 * A[cc, 5 + i * 18]
        # cc += 1  # Z Snap Rendezvous
        # A[cc, 10] = 24
        # A[cc, 10 + i * 18 - 18] = -1 * A[cc, 10 + i * 18]

    cc += 1 # absolute start X velocity
    A[cc, 0] = 4 * waypoints[0].time ** 3
    A[cc, 1] = 3 * waypoints[0].time ** 2
    A[cc, 2] = 2 * waypoints[0].time
    A[cc, 3] = 1
    cc += 1  # absolute start Y velocity
    A[cc, 5] = 4 * waypoints[0].time ** 3
    A[cc, 6] = 3 * waypoints[0].time ** 2
    A[cc, 7] = 2 * waypoints[0].time
    A[cc, 8] = 1
    cc += 1  # absolute start Z velocity
    A[cc, 10] = 4 * waypoints[0].time ** 3
    A[cc, 11] = 3 * waypoints[0].time ** 2
    A[cc, 12] = 2 * waypoints[0].time
    A[cc, 13] = 1
    cc += 1  # absolute start Yaw velocity
    A[cc, 15] = 2 * waypoints[0].time
    A[cc, 16] = 1

    cc += 1 # absolute end X velocity
    A[cc, numCoefficients - 18 + 0] = 4 * waypoints[-1].time ** 3
    A[cc, numCoefficients - 18 + 1] = 3 * waypoints[-1].time ** 2
    A[cc, numCoefficients - 18 + 2] = 2 * waypoints[-1].time
    A[cc, numCoefficients - 18 + 3] = 1
    cc += 1  # absolute end Y velocity
    A[cc, numCoefficients - 18 + 5] = 4 * waypoints[-1].time ** 3
    A[cc, numCoefficients - 18 + 6] = 3 * waypoints[-1].time ** 2
    A[cc, numCoefficients - 18 + 7] = 2 * waypoints[-1].time
    A[cc, numCoefficients - 18 + 8] = 1
    cc += 1  # absolute end Z velocity
    A[cc, numCoefficients - 18 + 10] = 4 * waypoints[-1].time ** 3
    A[cc, numCoefficients - 18 + 11] = 3 * waypoints[-1].time ** 2
    A[cc, numCoefficients - 18 + 12] = 2 * waypoints[-1].time
    A[cc, numCoefficients - 18 + 13] = 1
    #cc += 1  # absolute end Yaw velocity
    #A[cc, numCoefficients - 18 + 15] = 2 * waypoints[-1].time
    #A[cc, numCoefficients - 18 + 16] = 1

    #cc += 1 # absolute start X acceleration
    # A[c, 0] = 12 * waypoints[0].time ** 2
    # A[c, 1] = 6 * waypoints[0].time
    # A[c, 2] = 2
    #cc += 1  # absolute start Y acceleration
    # A[c, 5] = 12 * waypoints[0].time ** 2
    # A[c, 6] = 6 * waypoints[0].time
    # A[c, 7] = 2
    #cc += 1  # absolute start Z acceleration
    # A[cc, 10] = 12 * waypoints[0].time ** 2
    # A[cc, 11] = 6 * waypoints[0].time
    # A[cc, 12] = 2
    #cc += 1  # absolute start Yaw acceleration
    # A[cc, 15] = 2

    #cc += 1 # absolute end X acceleration
    # A[cc, numCoefficients - 18 + 0] = 12 * waypoints[-1].time ** 2
    # A[cc, numCoefficients - 18 + 1] = 6 * waypoints[-1].time
    # A[cc, numCoefficients - 18 + 2] = 2
    #cc += 1  # absolute end Y acceleration
    # A[cc, numCoefficients - 18 + 5] = 12 * waypoints[-1].time ** 2
    # A[cc, numCoefficients - 18 + 6] = 6 * waypoints[-1].time
    # A[cc, numCoefficients - 18 + 7] = 2
    #cc += 1  # absolute end Z acceleration
    # A[cc, numCoefficients - 18 + 10] = 12 * waypoints[-1].time ** 2
    # A[cc, numCoefficients - 18 + 11] = 6 * waypoints[-1].time
    # A[cc, numCoefficients - 18 + 12] = 2
    #cc += 1  # absolute end Yaw acceleration
    # A[cc, numCoefficients - 18 + 15] = 2

    #cc += 1 # absolute start X jerk
    # A[cc, 0] = 24 * waypoints[0].time
    # A[cc, 1] = 6
    #cc += 1  # absolute start Y jerk
    # A[cc, 5] = 24 * waypoints[0].time
    # A[cc, 6] = 6
    #cc += 1  # absolute start Z jerk
    # A[cc, 10] = 24 * waypoints[0].time
    # A[cc, 11] = 6

    #cc += 1 # absolute end X jerk
    # A[cc, numCoefficients - 18 + 0] = 24 * waypoints[-1].time
    # A[cc, numCoefficients - 18 + 1] = 6
    #cc += 1  # absolute end Y jerk
    # A[cc, numCoefficients - 18 + 5] = 24 * waypoints[-1].time
    # A[cc, numCoefficients - 18 + 6] = 6
    #cc += 1  # absolute end Z jerk
    # A[cc, numCoefficients - 18 + 10] = 24 * waypoints[-1].time
    # A[cc, numCoefficients - 18 + 11] = 6

    #cc += 1 # absolute start X snap
    # A[cc, 0] = 24
    #cc += 1  # absolute start Y snap
    # A[cc, 5] = 24
    #cc += 1  # absolute start Z snap
    # A[cc, 10] = 24

    #cc += 1 # absolute end X snap
    # A[cc, numCoefficients - 18 + 0] = 24
    #cc += 1  # absolute end Y snap
    # A[cc, numCoefficients - 18 + 5] = 24
    #cc += 1  # absolute end Z snap
    # A[cc, numCoefficients - 18 + 10] = 24

    # =============================
    # Solver Setup
    # =============================
    # OSQP needs:
    # P = quadratic terms
    # q = linear terms
    # A = constraint matrix of ALL constraints (inequality & equality)
    # l = lower constraints
    # u = upper constraints
    P = csc_matrix(P)
    q = hstack(q)
    h = hstack(h)
    b = hstack(b)

    A = vstack([G, A])
    A = csc_matrix(A)
    l = -inf * ones(len(h))
    l = hstack([l, b])
    u = hstack([h, b])

    # setup solver and solve
    m = osqp.OSQP()
    m.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)  # extra solver variables can be set here
    res = m.solve()

    # save to trajectory variable
    for i in range(0, size(res.x), 18):
        segment = res.x[i:i + 18]
        trajectory.append(segment)
    # print("QP solution Number following: ", res.x)

    return trajectory




def separate(waypoints):
    """
    Calculate a trajectory by separate operations.
    """
    # every segment has its own polynomial of 4th degree for X,Y and Z and a polynomial of 2nd degree for Yaw
    numCoefficients = 3*5+3
    # total number of segments
    numSegments = len(waypoints) - 1
    # list of calculated trajectory coefficients
    trajectory = []


    for i in range(numSegments):
        # X,Y,Z,Yaw position at start and end: 8
        # X,Y,Z,Yaw velocity at start: 4
        # X,Y,Z acceleration at start: 3
        numConstraints = 15
        # X,Y,Z velocity at absolute end: 3
        # they are initialized as zero, so no changes needed
        if i == numSegments-1:
            numConstraints += 3

        # =============================
        # Identity matrix for main part of QP (normally the Hesse matrix (quadratic terms), but this is a least squared problem)
        # =============================
        P = zeros((numCoefficients, numCoefficients))
        P[0, 0] = 1  # minimize snap for X
        P[5, 5] = 1  # minimize snap for Y
        P[7,7] = 100 # min acc for y
        P[10, 10] = 1  # minimize snap for Z
        P[15, 15] = 100  # minimize acceleration for Yaw

        # =============================
        # Gradient vector (linear terms), we have none
        # =============================
        q = zeros((numCoefficients, 1))

        # =============================
        # Inequality matrix (left side), we have none
        # =============================
        G = zeros((numConstraints, numCoefficients))

        # =============================
        # Inequality vector (right side), we have none
        # =============================
        h = zeros((numConstraints, 1))

        # =============================
        # Equality matrix (left side)
        # =============================
        A = zeros((numConstraints, numCoefficients))
        # X Position Start
        A[0, 0] = waypoints[i].time ** 4
        A[0, 1] = waypoints[i].time ** 3
        A[0, 2] = waypoints[i].time ** 2
        A[0, 3] = waypoints[i].time
        A[0, 4] = 1
        # Y Position Start
        A[1, 5] = waypoints[i].time ** 4
        A[1, 6] = waypoints[i].time ** 3
        A[1, 7] = waypoints[i].time ** 2
        A[1, 8] = waypoints[i].time
        A[1, 9] = 1
        # Z Position Start
        A[2, 10] = waypoints[i].time ** 4
        A[2, 11] = waypoints[i].time ** 3
        A[2, 12] = waypoints[i].time ** 2
        A[2, 13] = waypoints[i].time
        A[2, 14] = 1
        # Yaw Angle Start
        A[3, 15] = waypoints[i].time ** 2
        A[3, 16] = waypoints[i].time
        A[3, 17] = 1

        # X Position End
        A[4, 0] = waypoints[i + 1].time ** 4
        A[4, 1] = waypoints[i + 1].time ** 3
        A[4, 2] = waypoints[i + 1].time ** 2
        A[4, 3] = waypoints[i + 1].time
        A[4, 4] = 1
        # Y Position End
        A[5, 5] = waypoints[i + 1].time ** 4
        A[5, 6] = waypoints[i + 1].time ** 3
        A[5, 7] = waypoints[i + 1].time ** 2
        A[5, 8] = waypoints[i + 1].time
        A[5, 9] = 1
        # Z Position End
        A[6, 10] = waypoints[i + 1].time ** 4
        A[6, 11] = waypoints[i + 1].time ** 3
        A[6, 12] = waypoints[i + 1].time ** 2
        A[6, 13] = waypoints[i + 1].time
        A[6, 14] = 1
        # Yaw Angle End
        A[7, 15] = waypoints[i + 1].time ** 2
        A[7, 16] = waypoints[i + 1].time
        A[7, 17] = 1

        # X Velocity Start
        A[8, 0] = 4 * waypoints[i].time ** 3
        A[8, 1] = 3 * waypoints[i].time ** 2
        A[8, 2] = 2 * waypoints[i].time
        A[8, 3] = 1
        # Y Velocity Start
        A[9, 5] = 4 * waypoints[i].time ** 3
        A[9, 6] = 3 * waypoints[i].time ** 2
        A[9, 7] = 2 * waypoints[i].time
        A[9, 8] = 1
        # Z Velocity Start
        A[10, 10] = 4 * waypoints[i].time ** 3
        A[10, 11] = 3 * waypoints[i].time ** 2
        A[10, 12] = 2 * waypoints[i].time
        A[10, 13] = 1
        # Yaw Velocity Start
        A[11, 15] = 2 * waypoints[i].time
        A[11, 16] = 1

        # X Acceleration Start
        A[12, 0] = 12 * waypoints[i].time ** 2
        A[12, 1] = 6 * waypoints[i].time
        A[12, 2] = 2
        # Y Acceleration Start
        A[13, 5] = 12 * waypoints[i].time ** 2
        A[13, 6] = 6 * waypoints[i].time
        A[13, 7] = 2
        # Z Acceleration Start
        A[14, 10] = 12 * waypoints[i].time ** 2
        A[14, 11] = 6 * waypoints[i].time
        A[14, 12] = 2
        # Yaw Acceleration Start
        #A[15, 15] = 2

        # X Jerk Start
        #A[16, 0] = 24 * waypoints[i].time
        #A[16, 1] = 6
        # Y Jerk Start
        #A[17, 5] = 24 * waypoints[i].time
        #A[17, 6] = 6
        # Z Jerk Start
        #A[18, 10] = 24 * waypoints[i].time
        #A[18, 11] = 6

        # X Snap Start
        #A[19, 0] = 24
        # Y Snap Start
        #A[20, 5] = 24
        # Z Snap Start
        #A[21, 10] = 24

        # for full stop at absolute End
        if i == numSegments - 1:
            # X Velocity End
            A[15, 0] = 4 * waypoints[i + 1].time ** 3
            A[15, 1] = 3 * waypoints[i + 1].time ** 2
            A[15, 2] = 2 * waypoints[i + 1].time
            A[15, 3] = 1
            # Y Velocity End
            A[16, 5] = 4 * waypoints[i + 1].time ** 3
            A[16, 6] = 3 * waypoints[i + 1].time ** 2
            A[16, 7] = 2 * waypoints[i + 1].time
            A[16, 8] = 1
            # Z Velocity End
            A[17, 10] = 4 * waypoints[i + 1].time ** 3
            A[17, 11] = 3 * waypoints[i + 1].time ** 2
            A[17, 12] = 2 * waypoints[i + 1].time
            A[17, 13] = 1

        # =============================
        # Equality vector (right side)
        # =============================
        b = zeros((numConstraints, 1))

        b[0, 0] = waypoints[i].x
        b[1, 0] = waypoints[i].y
        b[2, 0] = waypoints[i].z
        b[3, 0] = waypoints[i].yaw
        b[4, 0] = waypoints[i+1].x
        b[5, 0] = waypoints[i+1].y
        b[6, 0] = waypoints[i+1].z
        b[7, 0] = waypoints[i+1].yaw

        # Derivatives = 0 for absolute Start, else Rendezvous of Segments
        if i != 0:
            b[8, 0] = 4 * trajectory[-1][0] * waypoints[i].time ** 3 + 3 * trajectory[-1][1] * waypoints[i].time ** 2 + 2 * trajectory[-1][2] * waypoints[i].time + trajectory[-1][3]
            b[9, 0] = 4 * trajectory[-1][5] * waypoints[i].time ** 3 + 3 * trajectory[-1][6] * waypoints[i].time ** 2 + 2* trajectory[-1][7] * waypoints[i].time + trajectory[-1][8]
            b[10, 0] = 4 * trajectory[-1][10] * waypoints[i].time ** 3 + 3 * trajectory[-1][11] * waypoints[i].time ** 2 + 2 * trajectory[-1][12] * waypoints[i].time + trajectory[-1][13]
            b[11, 0] = 2 * trajectory[-1][15] * waypoints[i].time + trajectory[-1][16]

            b[12, 0] = 12 * trajectory[-1][0] * waypoints[i].time ** 2 + 6 * trajectory[-1][1] * waypoints[i].time + 2 * trajectory[-1][2]
            b[13, 0] = 12 * trajectory[-1][5] * waypoints[i].time ** 2 + 6 * trajectory[-1][6] * waypoints[i].time + 2 * trajectory[-1][7]
            b[14, 0] = 12 * trajectory[-1][10] * waypoints[i].time ** 2 + 6 * trajectory[-1][11] * waypoints[i].time + 2 * trajectory[-1][12]


        # =============================
        # Solver Setup
        # =============================
        # OSQP needs:
        # P = quadratic terms
        # q = linear terms
        # A = constraint matrix of ALL constraints (inequality & equality)
        # l = lower constraints
        # u = upper constraints
        P = csc_matrix(P)
        q = hstack(q)
        h = hstack(h)
        b = hstack(b)

        A = vstack([G, A])
        A = csc_matrix(A)
        l = -inf * ones(len(h))
        l = hstack([l, b])
        u = hstack([h, b])

        # setup solver and solve
        m = osqp.OSQP()
        m.setup(P=P, q=q, A=A, l=l, u=u, verbose=False) # extra solver variables can be set here
        res = m.solve()

        # save to trajectory variable
        trajectory.append(res.x)
        # print("QP solution Number ", i, "following: ", res.x)

    return trajectory


def planner(waypoint_arr, isJoint=True):
    """
    Starting point of any generation.

    Waypoints are given as an array of arrays and transformed into an array of Waypoints.
    """
    # test waypoints
    # waypoint_arr = []
    # waypoint_arr.append([0, 0, 0, 2, 0])
    # waypoint_arr.append([1, 5, 0, 4, 3])
    # waypoint_arr.append([2, 5, 5, 3, 1])
    # waypoint_arr.append([3, 0, 5, 1, 5])
    # waypoint_arr.append([4, -5, 0, 2, 4])

    waypoints = []

    # the given "waypoints" are just the x,y,z,yaw values -> convert them to actual Waypoint objects
    for i in range(len(waypoint_arr)):
        # calculate time of waypoint
        if i == 0:
            time = 0
        else:
            time += calc_time(waypoint_arr[i-1], waypoint_arr[i])
        # create and append waypoint
        waypoint = Waypoint(waypoint_arr[i][0], waypoint_arr[i][1], waypoint_arr[i][2], waypoint_arr[i][3], time)
        waypoints.append(waypoint)

    # either calculate jointly or separately
    if isJoint:
        trajectory = joint(waypoints)
    else:
        trajectory = separate(waypoints)

    # after closing trajectory visualization
    return waypoints, trajectory

#### generate trajectory mavveric
def generate_waypoints_yaw(pos,gates):
    pos_higher = pos.copy()
    pos_higher[2] = pos_higher[2] + 1

    dpos_yaw = (array(gates[0]['pos']) - array(pos))[:2]
    yaw_desired = math.atan2(dpos_yaw[1], dpos_yaw[0])
    pos_higher.append(yaw_desired)


    waypoint_array = [pos_higher]
    for g in gates:
        gate_pos = g['pos']
        gate_yaw = g['rot'][3]

        r = R.from_euler('Z', gate_yaw)
        padding_point = (r.as_matrix() @ array([0, 0.2, 0]))
 
        # # Padding before
        gate_padding_wp = array(gate_pos) - padding_point
        waypoint_array.append(gate_padding_wp.tolist() + [gate_yaw])

        # Actual centerpoint
        waypoint_array.append(gate_pos.tolist() + [gate_yaw])

        # # Padding after
        gate_padding_wp = array(gate_pos) + padding_point
        waypoint_array.append(gate_padding_wp.tolist() + [gate_yaw])
    
    waypoint_array.append([0,0,1,0])
    return waypoint_array

def generate_trajectory_mavveric(pos,gates):
    wp_array = generate_waypoints_yaw(pos,gates)
    waypoints, trajectory = planner(wp_array)
    draw_traj(waypoints, trajectory)

    # for i, tj in enumerate(trajectory): print(i, tj)
    wp_list = [ [wp.x, wp.y, wp.z, wp.yaw, wp.time] for wp in waypoints]

    x_path_tot = []
    y_path_tot = []
    z_path_tot = []
    yaw_path_tot = []
    time_path_tot = []
    for i in range(len(trajectory)):
        t = linspace(waypoints[i].time, waypoints[i+1].time, int((waypoints[i+1].time-waypoints[i].time)*600))

        # print(i, waypoints[i].time, waypoints[i+1].time)
        time_path_tot += t.tolist()

        x_path = trajectory[i][0] * t ** 4 + trajectory[i][1] * t ** 3 + trajectory[i][2] * t ** 2 + trajectory[i][3] * t + trajectory[i][4]
        x_path_tot += x_path.tolist()
        
        y_path = trajectory[i][5] * t ** 4 + trajectory[i][6] * t ** 3 + trajectory[i][7] * t ** 2 + trajectory[i][8] * t + trajectory[i][9]
        y_path_tot += y_path.tolist()

        z_path = trajectory[i][10] * t ** 4 + trajectory[i][11] * t ** 3 + trajectory[i][12] * t ** 2 + trajectory[i][13] * t + trajectory[i][14]
        z_path_tot += z_path.tolist()

        yaw_path = trajectory[i][15] * t ** 2 + trajectory[i][16] * t + trajectory[i][17]
        yaw_path_tot += yaw_path.tolist()

        # print(len(t))
        # print(len(x_path))
        # print(len(y_path))
        # print(len(z_path))
        # print(len(yaw_path))

    trajectory = array([x_path_tot, y_path_tot, z_path_tot, yaw_path_tot, time_path_tot]).T
    return wp_list, trajectory
