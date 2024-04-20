from controller import Robot, Supervisor
from controller import TouchSensor
import numpy as np
from math import cos, sin, atan2
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
import sys
import torch
sys.path.append("..")

from controller_utils.pid_controller import pid_velocity_fixed_height_controller
from controller_utils.pathplanner import PathPlanner
from controller_utils.track_generator import TrackGenerator
from controller_utils.recorder import Recorder
from controller_utils.maveric_trajectory_planner import generate_trajectory_mavveric

sys.path.append("../../..")
from imitation_learning_simple.utils.pencil_filter import PencilFilter
from imitation_learning_simple.models.resnet import ResNet18

class ResNetPolicy(Supervisor):
    def __init__(self, record=False, display_path=True):
        super().__init__()

        self.__record = record
        self.__display_path_flag = display_path
        self.__gate_node = self.getFromDef('imav2022-gate')

        self.__motors = []
        self.__sensors = {}
        self.__current_waypoint = 0
        self.__past_time = 0
        self.__num_steps = 0
        
        self.pencil_filter = PencilFilter()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.resnet = ResNet18().to(self.device)
        self.resnet.load_state_dict(
            torch.load('../../../imitation_learning_simple/weights/resnet_best.pth')
        )

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
        # self.__sensors['gyro'] = robot.getDevice("gyro")
        # self.__sensors['gyro'].enable(self.__timestep)
        self.__sensors['camera'] = robot.getDevice("camera")
        self.__sensors['camera'].enable(self.__timestep)
        self.__sensors['rangefinder'] = robot.getDevice("range-finder")
        self.__sensors['rangefinder'].enable(self.__timestep)
        self.__sensors['touch_sensor'] = robot.getDevice("touchsensor")
        self.__sensors['touch_sensor'].enable(self.__timestep)        
        # self.__sensors['accelerometer'] = robot.getDevice("accelerometer")
        # self.__sensors['accelerometer'].enable(self.__timestep)
        # self.__sensors['dist_sensors'] = []
        # for i in range(12):
        #     range_sensor = robot.getDevice(f"range_m_{i}")
        #     range_sensor.enable(timestep)
        #     self.__sensors['dist_sensors'].append(range_sensor)

        # TODO: Randomise the environment
        # bg = np.random.choice(
        #     [
        #        "dawn_cloudy_empty", "dusk", "empty_office", "entrance_hall", "factory", "mars", "morning_cloudy_empty", 
        #        "mountains", "music_hall", "noon_building_overcast", "noon_cloudy_countryside", "noon_cloudy_empty", 
        #        "noon_cloudy_mountains", "noon_park_empty", "noon_stormy_empty", "noon_sunny_empty", "noon_sunny_garden", 
        #        "stadium", "stadium_dry", "twilight_cloudy_empty"
        #     ])
        # self.getFromDef("TexBG").getField('texture').setSFString(bg)
        # self.getFromDef("TexBG").getField('luminosity').setSFFloat(
        #     np.random.uniform(0.1, 2)
        # )
        # self.getFromDef("TexBGlight").getField('texture').setSFString(bg)
        # self.getFromDef("TexBGlight").getField('luminosity').setSFFloat(
        #     np.random.uniform(0.1, 2)
        # )

        num_gates=3
        # Generate new track and trajectory
        tg = TrackGenerator(num_gate_poses=num_gates)
        # self.__gate_poses = tg.generate_easy()
        self.__gate_poses = tg.generate()
        self.__gate_square_poses = tg.to_gate_squares(self.__gate_poses)

        self.__current_waypoint = 0
        self.__update_gate_pose()

        # Initialise state variables
        self.__past_time = 0
        self.__past_motor_power = np.array([-1,1,-1,1])
        self.__past_commands = np.array([0, 0, 0, 0])
        self.__past_time = 0
        self.__past_pos = self.__sensors['gps'].getValues()
        self.__num_steps = 0


    def step(self):
        robot = self
        self.__dt = robot.getTime() - self.__past_time
        pos = self.__sensors['gps'].getValues()
        rpy = self.__sensors['imu'].getRollPitchYaw()

        # Get pencil image
        image_string = self.__sensors['camera'].getImage()
        image_data = np.frombuffer(image_string, dtype=np.uint8)
        image_data = image_data.reshape((self.__sensors['camera'].getHeight(), self.__sensors['camera'].getWidth(), 4))
        rgb_matrix = image_data[:, :, :3]
        pencil_image = self.pencil_filter.apply(rgb_matrix)
        pencil_image = torch.Tensor(pencil_image)

        # Get depth array
        depth_array = self.__sensors['rangefinder'].getRangeImage(data_type="buffer")
        depth_array = np.ctypeslib.as_array(depth_array, 
                                            (self.__sensors['rangefinder'].getWidth(), self.__sensors['rangefinder'].getHeight())
                                            )
        depth_array = (depth_array / self.__sensors['rangefinder'].getMaxRange())
        depth_array[depth_array == float('inf')] = 1
        depth_array = np.expand_dims(depth_array, axis=0)
        depth_array = torch.Tensor(depth_array)

        # Get output
        scalers = {
            "mean": torch.Tensor([ 5.5213e+01,  6.9066e-04, -5.0562e-03, -2.7119e-02]),
            "std": torch.Tensor([1.5544, 0.2182, 0.2224, 0.2800])
        }
        net_input = torch.cat([pencil_image, depth_array])
        net_input = torch.unsqueeze(net_input, dim=0)
        net_input= net_input.to(self.device)
        with torch.no_grad():
            output = self.resnet(net_input)
        
        output = output.to('cpu')
        commands = output * scalers['std'] + scalers['mean']
        alt_command, roll_command, pitch_command, yaw_command = commands[0].tolist()
        print(commands)
        motor_power = []
        motor_power.append(alt_command - roll_command + pitch_command + yaw_command)
        motor_power.append(alt_command - roll_command - pitch_command - yaw_command)
        motor_power.append(alt_command + roll_command - pitch_command + yaw_command)
        motor_power.append(alt_command + roll_command + pitch_command - yaw_command)

        # Apply motor powers
        for i in range(4):
            self.__motors[i].setVelocity(motor_power[i] * ((-1) ** (i+1)) )


        # Reset gate position
        if self.__is_passing_through_gate(pos):
            self.__current_waypoint += 1
            if self.__current_waypoint == len(self.__gate_poses):
                print("Finished track... restarting simulation")
                if self.__record:
                    self.recorder.save_data()

                ### TODO: Add way to distinguish between reset ending or close-sim ending
                self.reset()
            else:
                self.__update_gate_pose()

        if self.__num_steps > 300 and bool(self.__sensors['touch_sensor'].getValue()):
            print("Drone touched... resetting simulation")
            self.reset()

        if abs(rpy[0]) > np.pi/2 or abs(rpy[1]) > np.pi/2:
            print("Drone ribaltated... resetting simulation")
            self.reset()

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
    resnetpolicy = ResNetPolicy()
    resnetpolicy.reset()

    stop = False
    while not stop:
        sim_return = resnetpolicy.step()
        stop = sim_return == -1


