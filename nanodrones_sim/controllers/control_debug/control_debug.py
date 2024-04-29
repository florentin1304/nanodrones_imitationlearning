from controller import Robot, Supervisor
from controller import TouchSensor
import numpy as np
from math import cos, sin, atan2
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
import sys
sys.path.append("..")
sys.path.append("../../..") #home

from controller_utils.pid_controller import pid_velocity_fixed_height_controller
from controller_utils.pathplanner import PathPlanner
from controller_utils.track_generator import TrackGenerator
from controller_utils.recorder import Recorder
from controller_utils.maveric_trajectory_planner import generate_trajectory_mavveric

from imitation_learning_simple.utils.pencil_filter import PencilFilter

class GateFollower(Supervisor):
    def __init__(self, record=False, display_path=True, randomisation=False):
        super().__init__()

        self.__record = record
        self.__display_path_flag = display_path
        self.__env_randomisation = randomisation
        self.__gate_node = self.getFromDef('imav2022-gate')

        self.__motors = []
        self.__sensors = {}
        self.__current_waypoint = 0
        self.__past_time = 0
        self.__num_steps = 0

        self.reset()

    def reset(self):
        self.simulationResetPhysics()
        self.simulationReset()
        self.__timestep = int(self.getBasicTimeStep())
        super().step(self.__timestep)
        robot = self
        
        # Initialize motors
        self.__motors = []
        for i in range(0,4):
            motor = robot.getDevice(f"m{i+1}_motor")
            motor.setPosition(float('inf'))
            motor.setVelocity(((-1)**(i+1)) )
            self.__motors.append(motor)

        # Initialize Sensors
        self.__sensors = {}
        self.__sensors['imu'] = robot.getDevice("inertial_unit")
        self.__sensors['imu'].enable(self.__timestep)
        self.__sensors['gps']= robot.getDevice("gps")
        self.__sensors['gps'].enable(self.__timestep)
        self.__sensors['gyro'] = robot.getDevice("gyro")
        self.__sensors['gyro'].enable(self.__timestep)
        self.__sensors['camera'] = robot.getDevice("camera")
        self.__sensors['camera'].enable(self.__timestep)
        self.__sensors['rangefinder'] = robot.getDevice("range-finder")
        self.__sensors['rangefinder'].enable(self.__timestep)
        self.__sensors['touch_sensor'] = robot.getDevice("touchsensor")
        self.__sensors['touch_sensor'].enable(self.__timestep)        
        self.__sensors['accelerometer'] = robot.getDevice("accelerometer")
        self.__sensors['accelerometer'].enable(self.__timestep)
        # self.__sensors['dist_sensors'] = []
        # for i in range(12):
        #     range_sensor = robot.getDevice(f"range_m_{i}")
        #     range_sensor.enable(timestep)
        #     self.__sensors['dist_sensors'].append(range_sensor)

        # Initialise controller
        self.pid_controller = pid_velocity_fixed_height_controller()

        # Randomise the environment
        if self.__env_randomisation:
            bg = np.random.choice(
                [
                "dawn_cloudy_empty", "dusk", "empty_office", "entrance_hall", "factory", "mars", "morning_cloudy_empty", 
                "mountains", "music_hall", "noon_building_overcast", "noon_cloudy_countryside", "noon_cloudy_empty", 
                "noon_cloudy_mountains", "noon_park_empty", "noon_stormy_empty", "noon_sunny_empty", "noon_sunny_garden", 
                "stadium", "stadium_dry", "twilight_cloudy_empty"
                ])
            self.getFromDef("TexBG").getField('texture').setSFString(bg)
            self.getFromDef("TexBG").getField('luminosity').setSFFloat(
                np.random.uniform(0.1, 2)
            )
            self.getFromDef("TexBGlight").getField('texture').setSFString(bg)
            self.getFromDef("TexBGlight").getField('luminosity').setSFFloat(
                np.random.uniform(0.1, 2)
            )



        # Setup recorder
        if self.__record:
            self.recorder = Recorder({'randomisation': False}, save_dir='data_debug')
            self.recorder.set_headers(["x", "y", "z", "roll", "pitch", "yaw", 
                                        "gate_x","gate_y", "gate_z", "gate_yaw", 
                                        "vx_global", "vy_global", "vz_global",
                                        "vx_local", "vy_local", 
                                        "roll_rate","pitch_rate","yaw_rate",
                                        "x_sp", "y_sp", "z_sp",
                                        "vx_local_sp", "vy_local_sp", 
                                        "alt_sp", 
                                        "yaw_rate_sp",
                                        "alt_command", "roll_command", "pitch_command", "yaw_command"])
            
        self.waypoints = [
            [0, 0, 1],
            [2, 0, 3],
            [0, 0, 3],
            [5, 0, 1],
        ]

        trajectory = []
        for i in range(1,len(self.waypoints)):
            delta = np.linalg.norm(
                np.array(self.waypoints[i]) - np.array(self.waypoints[i-1])
            )
            num = int(delta)*20
            x = np.linspace(self.waypoints[i-1][0], self.waypoints[i][0], num)
            y = np.linspace(self.waypoints[i-1][1], self.waypoints[i][1], num)
            z = np.linspace(self.waypoints[i-1][2], self.waypoints[i][2], num)
            for j in range(num):
                trajectory.append([x[j], y[j], z[j]])

        trajectory = np.array(trajectory)

        self.__pathplanner = PathPlanner(trajectory=trajectory) 

        # Initialise state variables
        self.__past_motor_power = np.array([-1,1,-1,1])
        self.__past_commands = np.array([0, 0, 0, 0])
        self.__past_time = 0
        self.__past_pos = self.__sensors['gps'].getValues()
        self.__num_steps = 0


    def step(self):
        robot = self
        self.__dt = robot.getTime() - self.__past_time

        # Get state information
        pos = np.array(self.__sensors['gps'].getValues())
        dpos = (pos - self.__past_pos) / self.__dt
        acceleration = self.__sensors['accelerometer'].getValues()

        rpy = self.__sensors['imu'].getRollPitchYaw()
        drpy = self.__sensors['gyro'].getValues()

        yaw = rpy[2]
        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)
        v_x = dpos[0] * cos_yaw + dpos[1] * sin_yaw
        v_y = - dpos[0] * sin_yaw + dpos[1] * cos_yaw
        dpos_ego = np.array([v_x, v_y])

        # Get pathplanner update
        drone_state = np.hstack([pos, rpy])
        gate_position = self.__gate_node.getPosition()
        if self.__num_steps < 300:
            vx_target, vy_target, yaw_desired, des_height = 0,0,0,1
            fin = False
        else:
            v_target, yaw_desired, des_height, fin = self.__pathplanner(drone_state, gate_position)
            vx_target, vy_target = v_target

        # PID velocity controller with fixed height
        roll, pitch, yaw = rpy
        roll_rate, pitch_rate, yaw_rate = drpy
        x_global, y_global, z_global = pos
        motor_power, commands = self.pid_controller.pid(self.__dt, vx_target, vy_target,
                                        yaw_desired, des_height,
                                        roll, pitch, yaw_rate,
                                        z_global, v_x, v_y)
        
        # Apply motor powers found by PID
        for i in range(4):
            self.__motors[i].setVelocity(motor_power[i] * ((-1) ** (i+1)) )

        # Record data
        if self.__record and self.__past_time != 0:
            current_pos_sp = [-1, -1, -1]
            gate_state = {'pos': [-1,-1,-1], 'rot': [-1,-1,-1,-1]}
            self.recorder.add_row([pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2], 
                                    *gate_state['pos'], gate_state['rot'][3], 
                                    dpos[0], dpos[1], dpos[2],
                                    dpos_ego[0], dpos_ego[1],
                                    drpy[0],drpy[1],drpy[2],
                                    current_pos_sp[0], current_pos_sp[1], current_pos_sp[2],
                                    vx_target, vy_target, 
                                    des_height,
                                    yaw_desired, 
                                    *commands])




        if self.__num_steps >= 2500 or fin:
            print("End of debug run")
            if self.__record:
                    self.recorder.save_data()
            self.reset()

        # if self.__num_steps > 500 and bool(self.__sensors['touch_sensor'].getValue()):
        #     print("Drone touched... resetting simulation")
        #     self.reset()

        # if abs(rpy[0]) > np.pi/2 or abs(rpy[1]) > np.pi/2:
        #     print("Drone ribaltated... resetting simulation")
        #     self.reset()

        # Update simulation and controller values
        self.__num_steps += 1
        self.__past_time = robot.getTime()
        self.__past_pos = pos
        self.__past_motor_power = motor_power
        self.__past_commands = commands
        return_value = super().step(self.__timestep)

        return return_value

    def __update_gate_pose(self):
        gate_state = self.__gate_poses[ self.__current_waypoint ]
        random_pos, random_rot = gate_state['pos'], gate_state['rot']
        self.__gate_node.getField('translation').setSFVec3f( random_pos.tolist() )
        self.__gate_node.getField('rotation').setSFRotation( random_rot.tolist() ) 

    def __is_passing_through_gate(self, pos):
        gate_position = self.__gate_node.getPosition()
        d_pos = np.array(gate_position[:2]) - np.array(pos[:2])

        if np.linalg.norm(d_pos) < 0.2 and abs(pos[2] - (gate_position[2] + 1)) < 0.1:
            return True
        return False

    def __get_current_gate_pos(self):
        return self.__gate_square_poses[ self.__current_waypoint ]

    def __display_path(self, path):
        root_node = self.getRoot()
        children_field = root_node.getField('children')
        
        #====================================================
        trail_str =  "DEF TRAIL Shape {\n" + \
                    "  appearance Appearance {\n" + \
                    "    material Material {\n" + \
                    "      diffuseColor 1 0 0\n" + \
                    "      emissiveColor 1 0 0\n" + \
                    "    }\n" + \
                    "  }\n" + \
                    "  geometry DEF TRAIL_LINE_SET IndexedLineSet {\n" + \
                    "    coord Coordinate {\n" + \
                    "      point [\n"
        for point in path:
            trail_str += f"      {point[0]} {point[1]} {point[2]}\n"
        trail_str += "      ]\n" +\
                    "    }\n" +\
                    "    coordIndex [\n"
        for i in range(len(path)-1):
            trail_str += f"      {i+1}\n"
        trail_str +=  "    ]\n" +\
                        "  }\n" +\
                        "}\n"
        #====================================================
        print("Displaying path...", end='')
        children_field.importMFNodeFromString(-1 ,trail_str)
        print("done")



if __name__ == "__main__":
    print("Starting simulation")
    gf = GateFollower(record=True)
    
    gf.reset()

    stop = False
    while not stop:
        sim_return = gf.step()
        stop = sim_return == -1


