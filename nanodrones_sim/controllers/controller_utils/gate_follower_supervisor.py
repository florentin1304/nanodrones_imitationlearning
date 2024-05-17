from controller import Robot, Supervisor
from controller import TouchSensor
import numpy as np
from math import cos, sin, atan2
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
import sys
import pandas as pd

sys.path.append("..")
sys.path.append("../../..") #home

from controller_utils.pid_controller import pid_velocity_fixed_height_controller
from controller_utils.pathplanner import PathPlanner
from controller_utils.track_generator import TrackGenerator
from controller_utils.recorder import Recorder
from controller_utils.maveric_trajectory_planner import generate_trajectory_mavveric
from controller_utils.trajectory_generator import TrajectoryGenerator

from imitation_learning_simple.utils.pencil_filter import PencilFilter

class GateFollowerSupervisor(Supervisor):
    def __init__(self, display_path=True):
        super().__init__()
        self.__display_path_flag = display_path
        self.__env_randomisation = False

        self.__setup_time = 7.5

        self.__pencil_filter = PencilFilter()

        self.__motors = []
        self.__sensors = {}
        self.__current_waypoint = 0
        self.__past_time = 0
        self.__num_steps = 0

    def reset(self, config):
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
            self.__randomize_environment()

        # Setup recorder
        self.recorder = Recorder(metadata=config, **config['recorder'])
        if self.recorder.is_on():
            self.recorder.set_headers(["sim_time","x", "y", "z", "roll", "pitch", "yaw", 
                                        "gate_x","gate_y", "gate_z", "gate_yaw", 
                                        "vx_global", "vy_global", "vz_global",
                                        "vx_local", "vy_local", 
                                        "roll_rate","pitch_rate","yaw_rate",
                                        "x_proj", "y_proj", "z_proj",
                                        "x_sp", "y_sp", "z_sp",
                                        "vx_local_sp", "vy_local_sp", 
                                        "alt_sp", 
                                        "yaw_rate_sp",
                                        "alt_command", "roll_command", "pitch_command", "yaw_command",
                                        "vx_ctrl_error","vy_ctrl_error", "alt_ctrl_error", #controller errors
                                        "roll_ctrl_error", "pitch_ctrl_error", "yaw_rate_ctrl_error", #controller errors
                                        "camera_img",
                                        "depth_img",
                                        "pencil_img"])
            
        # Generate new track
        tg = TrajectoryGenerator(**config['trajectory_generator'])
        trajectory = tg.generate_trajectory()

        self.__gate_poses = tg.generate_gate_positions_recursive(trajectory=trajectory)
        self.__gate_square_poses = tg.to_gate_squares(self.__gate_poses)

        self.__display_gates()

        self.__pathplanner = PathPlanner(trajectory=trajectory, **config['pathplanner'])
        if self.__display_path_flag: 
            self.__display_path(trajectory)
            self.__display_balls()
        
        ### Save track
        vel = self.__pathplanner.curvature 
        traj_report = np.hstack([trajectory, vel.reshape(-1, 1)])
        self.recorder.add_trajectory(traj_report)
        
        
        # Initialise state variables
        self.__current_waypoint = 0
        self.__past_motor_power = np.array([-1,1,-1,1])
        self.__past_commands = np.array([0, 0, 0, 0])
        self.__past_time = 0
        self.__past_pos = self.__sensors['gps'].getValues()
        self.__num_steps = 0
        self.__startRecording = False


    def step(self):
        robot = self
        self.__dt = robot.getTime() - self.__past_time

        # Get state information
        sim_time = robot.getTime() 
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

        # Get target update
        if self.__is_passing_through_gate(pos, rpy[2]):
            print(f"Passing thorugh gate {self.__current_waypoint} !!!")
            self.__startRecording = True
            self.__current_waypoint += 1
            if self.__current_waypoint == len(self.__gate_square_poses):
                print("Finished track...")
                return 'finished'

        # Get pathplanner update
        drone_pos = pos
        drone_yaw = rpy[-1]
        v_target, yaw_desired, des_height, fin = self.__pathplanner(drone_pos, drone_yaw, dpos_ego, self.__gate_square_poses, self.__current_waypoint)
        
        if robot.getTime() < self.__setup_time:
            v_target = [0,0]
            des_height = 1 
            self.__pathplanner.resetHist()

        # Show projection and target points
        if self.__display_path_flag:
            current_pos_sp = self.__pathplanner.getCurrentSP()
            current_projection = self.__pathplanner.getCurrentProj()
            self.__update_balls(projection_point=current_projection, target_point=current_pos_sp)

        # PID velocity controller with fixed height
        roll, pitch, yaw = rpy
        roll_rate, pitch_rate, yaw_rate = drpy
        x_global, y_global, z_global = pos
        motor_power, commands = self.pid_controller.pid(self.__dt, v_target[0], v_target[1],
                                        yaw_desired, des_height,
                                        roll, pitch, yaw_rate,
                                        z_global, v_x, v_y)
        
        # Apply motor powers found by PID
        for i in range(4):
            self.__motors[i].setVelocity(motor_power[i] * ((-1) ** (i+1)) )

        # Record data
        if self.recorder.is_on() and self.__startRecording:
            image_name = None
            pencil_name = None
            depth_image_name = None
            if self.recorder.is_recording_images():
                image_string = self.__sensors['camera'].getImage()
                image_data = np.frombuffer(image_string, dtype=np.uint8)
                image_data = image_data.reshape((self.__sensors['camera'].getHeight(), self.__sensors['camera'].getWidth(), 4))
                rgb_matrix = image_data[:, :, :3]
                image_name = self.recorder.add_image(rgb_matrix)

                pencil_matrix = self.__pencil_filter.apply(rgb_matrix)
                pencil_matrix = np.array(pencil_matrix).transpose(1,2,0) * 255
                pencil_matrix = pencil_matrix.astype(np.uint8)
                pencil_name = self.recorder.add_image(pencil_matrix, 'pencil')

                depth_array = self.__sensors['rangefinder'].getRangeImage(data_type="buffer")
                depth_array = np.ctypeslib.as_array(depth_array, 
                                                    (self.__sensors['rangefinder'].getWidth(), self.__sensors['rangefinder'].getHeight())
                                                    )
                depth_array = 255 * (depth_array / self.__sensors['rangefinder'].getMaxRange())
                depth_array[depth_array == float('inf')] = 255
                depth_array = depth_array.astype(np.uint8)
                depth_image_name = self.recorder.add_image(depth_array, 'depth')

            gate_state = self.__get_current_gate_state()
            current_pos_sp = self.__pathplanner.getCurrentSP()
            current_projection = self.__pathplanner.getCurrentProj()

            vx_error, vy_error, alt_error = self.pid_controller.past_vx_error, self.pid_controller.past_vy_error, self.pid_controller.past_alt_error
            roll_error, pitch_error, yaw_rate_error = self.pid_controller.past_roll_error, self.pid_controller.past_pitch_error, self.pid_controller.past_yaw_rate_error

            self.recorder.add_row([sim_time, pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2], 
                                    *gate_state['pos'], gate_state['rot'][3], 
                                    dpos[0], dpos[1], dpos[2],
                                    dpos_ego[0], dpos_ego[1],
                                    drpy[0],drpy[1],drpy[2],
                                    current_projection[0], current_projection[1], current_projection[2],
                                    current_pos_sp[0], current_pos_sp[1], current_pos_sp[2],
                                    v_target[0], v_target[1], 
                                    des_height,
                                    yaw_desired, 
                                    *commands,
                                    vx_error, vy_error, alt_error,
                                    roll_error, pitch_error, yaw_rate_error,
                                    image_name,
                                    depth_image_name,
                                    pencil_name])



        if robot.getTime() > self.__setup_time and bool(self.__sensors['touch_sensor'].getValue()):
            print("Drone touched... resetting simulation")
            return 'crashed'

        if abs(rpy[0]) > np.pi/2 or abs(rpy[1]) > np.pi/2:
            print("Drone ribaltated... resetting simulation")
            return 'crashed'

        # Update simulation and controller values
        self.__num_steps += 1
        self.__past_time = robot.getTime()
        self.__past_pos = deepcopy(pos)
        self.__past_motor_power = motor_power
        self.__past_commands = commands
        return_value = super().step(self.__timestep)

        return 'flying'
    
    def recorder_save_data(self):
        if self.recorder.mode == 'active':
            self.recorder.save_data()
        else:
            print("Warning: recorder_save_data was called, yet the recorder is disabled")

    def __is_passing_through_gate(self, pos, yaw):
        gate_position = self.__get_current_gate_state()['pos']
        gate_yaw = self.__get_current_gate_state()['rot'][-1] # dont add 90 degrees cause gate x axis is 'lateral'
        R_gate_yaw = np.array([
            [cos(gate_yaw), -sin(gate_yaw)],
            [sin(gate_yaw), cos(gate_yaw)]
        ])

        d_pos_past = np.array(gate_position[:2]) - np.array(self.__past_pos[:2]) 
        d_pos = np.array(gate_position[:2]) - np.array(pos[:2])

        if np.linalg.norm(d_pos) < 0.5:
            d_pos_gate_ref =  d_pos @ R_gate_yaw
            d_pos_past_gate_ref =  d_pos_past @ R_gate_yaw
            if d_pos_gate_ref[1]*d_pos_past_gate_ref[1] < 0:
                return True
        return False

    def __get_current_gate_state(self):
        return self.__gate_square_poses[ self.__current_waypoint ]

    def __randomize_environment(self):
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

    ### DISPLAY FUNCTIONS FOR DEBUG
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

    def __display_gates(self):
        root_node = self.getRoot()
        children_field = root_node.getField('children')
        for i in range(len(self.__gate_poses)-1):
            gate_node = self.getFromDef(f'imav2022-gate')
            gate_node_str = gate_node.exportString()
            gate_node_str_list = gate_node_str.split(" ")
            gate_node_str_list[1] += str(i+1)
            gate_node_str_new = ' '.join(gate_node_str_list)
            children_field.importMFNodeFromString(-1, gate_node_str_new)

        for i, gate_state in enumerate(self.__gate_poses):
            gate_node = self.getFromDef(f'imav2022-gate{'' if i==0 else i}')
            random_pos, random_rot = gate_state['pos'], gate_state['rot']
            gate_node.getField('translation').setSFVec3f( random_pos.tolist() )
            gate_node.getField('rotation').setSFRotation( random_rot.tolist() ) 

    def __display_balls(self):
        root_node = self.getRoot()
        children_field = root_node.getField('children')
        proj_ball_str = "DEF ProjectionBall Solid { \n" + \
                        " translation 0 0 -1 \n" + \
                        " children [ \n" + \
                        "  Shape { \n" + \
                        "   appearance PBRAppearance { \n" + \
                        "     baseColor 0 1 0 \n" + \
                        "     roughness 1 \n" + \
                        "     transparency 0 \n" + \
                        "     metalness 0 \n" + \
                        "   } \n" + \
                        "   geometry Sphere { \n" + \
                        "     radius 0.025   \n" + \
                        "   } \n" + \
                        "  } \n" + \
                        " ] \n" + \
                        "}"
        
        target_ball_str = "DEF TargetBall Solid { \n" + \
                        " translation 0 0 -1 \n" + \
                        " children [ \n" + \
                        "  Shape { \n" + \
                        "   appearance PBRAppearance { \n" + \
                        "     baseColor 1 0 0 \n" + \
                        "     roughness 1 \n" + \
                        "     transparency 0 \n" + \
                        "     metalness 0 \n" + \
                        "   } \n" + \
                        "   geometry Sphere { \n" + \
                        "     radius 0.025 \n" + \
                        "   } \n" + \
                        "  } \n" + \
                        " ] \n" + \
                        "}"

        
        children_field.importMFNodeFromString(-1, proj_ball_str)
        children_field.importMFNodeFromString(-1, target_ball_str)

    def __update_balls(self, projection_point, target_point):
        root_node = self.getRoot()

        proj_ball_node = self.getFromDef(f'ProjectionBall')
        proj_ball_node.getField('translation').setSFVec3f( projection_point.tolist() )

        target_ball_node = self.getFromDef(f'TargetBall')
        target_ball_node.getField('translation').setSFVec3f( target_point.tolist() )

