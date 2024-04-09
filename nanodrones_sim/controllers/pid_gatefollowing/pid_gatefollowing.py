"""gate_following_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Supervisor
from controller import TouchSensor
from pid_controller import pid_velocity_fixed_height_controller
import numpy as np
from math import cos, sin, atan2

from pathplanner import PathPlanner
from track_generator import TrackGenerator
from recorder import Recorder
from maveric_trajectory_planner import generate_trajectory_mavveric
from copy import deepcopy

from scipy.spatial.transform import Rotation as R

def display_path(supervisor, path):
    root_node = supervisor.getRoot()
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
    print("Displaying path")
    children_field.importMFNodeFromString(-1 ,trail_str)
    print("PATH DISPLAYED!")

DISPLAY_PATH = True
RECORD = True
NUM_GATES = 2

if __name__ == "__main__":
    print('Starting simulation...')
    # create the Robot instance.
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())

    # Initialize motors
    m1_motor = robot.getDevice("m1_motor")
    m1_motor.setPosition(float('inf'))
    m1_motor.setVelocity(-1)
    m2_motor = robot.getDevice("m2_motor")
    m2_motor.setPosition(float('inf'))
    m2_motor.setVelocity(1)
    m3_motor = robot.getDevice("m3_motor")
    m3_motor.setPosition(float('inf'))
    m3_motor.setVelocity(-1)
    m4_motor = robot.getDevice("m4_motor")
    m4_motor.setPosition(float('inf'))
    m4_motor.setVelocity(1)

    # Initialize Sensors
    imu = robot.getDevice("inertial_unit")
    imu.enable(timestep)
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    touch_sensor = robot.getDevice("touchsensor")
    touch_sensor.enable(timestep)
    range_finder = robot.getDevice("range-finder")
    range_finder.enable(timestep)
    dist_sensors = []
    for dir in ['l', 'm', 'u']:
        for i in range(12):
            range_sensor = robot.getDevice(f"range_m_{i}")
            range_sensor.enable(timestep)
            dist_sensors.append(range_sensor)


    PID_crazyflie = pid_velocity_fixed_height_controller()
    PID_update_last_time = robot.getTime()
    sensor_read_last_time = robot.getTime()
    
    ##### Supervisor link to nodes
    root_node = robot.getRoot()
    children_field = root_node.getField('children')
    gate_node = robot.getFromDef('imav2022-gate')

    ##### Initialize recorder
    if RECORD:
        recorder = Recorder({'randomisation': False})
        recorder.set_headers(["x", "y", "z", "yaw", 
                              "waypoint_num",
                              "vx_global", "vy_global", "vz_global",
                              "vx_local", "vy_local", 
                              "yaw_rate",
                              "vx_local_sp", "vy_local_sp", 
                              "alt_sp", 
                              "yaw_rate_sp", 
                              "camera_img",
                              "depth_img"])

    ##### Create track
    print('Generating track...', end='')
    tg = TrackGenerator(num_gate_poses=NUM_GATES)
    # gate_poses = tg.generate_easy()
    gate_poses = tg.generate()
    gate_square_poses = tg.to_gate_squares(gate_poses)


    ### My trajectory splines
    # waypoints = get_trajectory_waypoints(gate_square_poses)
    # trajectory = generate_trajectory(waypoints)

    ### Trajectory min snap MAVVERIC
    trajectory = generate_trajectory_mavveric(gate_square_poses)
    pathplanner = PathPlanner(trajectory=trajectory)
    if DISPLAY_PATH: display_path(robot, trajectory)

    ##### SET GATE POSE
    for i in range(len(gate_poses)): 
        yaw = gate_poses[i]['rot'][3]


    gate_state = gate_poses.pop(0)
    random_pos, random_rot = gate_state['pos'], gate_state['rot']
    gate_node.getField('translation').setSFVec3f( random_pos.tolist() )
    gate_node.getField('rotation').setSFRotation( random_rot.tolist() ) 

    # get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())
    
    # Initialize variables
    past_x_global = 0
    past_y_global = 0
    past_z_global = 0
    past_yaw = 0
    past_time = 0
    first_time = True
    
    # Main loop:
    # - perform simulation steps until Webots is stopping the controller
    i = 0
    waypoints_passed = 0

    hist = []
    while robot.step(timestep) != -1:
        dt = robot.getTime() - past_time
        # DEBUG
        # x = np.array([d.getValue() for d in dist_sensors])
        # if (x != 2000.0).any():
        #     print(x)
        # if touch_sensor.getValue() != 0.0:
        #     print("AAAAAAAAAA", i)
        # print(rangeData)
        
        ##### Get sensor data
        roll = imu.getRollPitchYaw()[0]
        pitch = imu.getRollPitchYaw()[1]
        yaw = imu.getRollPitchYaw()[2]
        yaw_rate = gyro.getValues()[2]
        
        x_global = gps.getValues()[0]
        v_x_global = (x_global - past_x_global)/dt
        
        y_global = gps.getValues()[1]
        v_y_global = (y_global - past_y_global)/dt
        
        z_global = gps.getValues()[2]
        v_z_global = (z_global - past_z_global)/dt
        #####
        
        # Get body fixed velocities 
        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)
        v_x = v_x_global * cos_yaw + v_y_global * sin_yaw
        v_y = - v_x_global * sin_yaw + v_y_global * cos_yaw


        #### MAGIC HAPPENS HERE
        drone_state = [x_global, y_global, z_global, roll, pitch, yaw]
        gate_position = gate_node.getPosition()
        v_target, yaw_desired, height = pathplanner(drone_state, gate_position)



        if RECORD:
            # camera_array = np.array(camera.getImageArray())[:, :, ::-1]
            # Convert string to numpy array of uint8
            image_string = camera.getImage()
            image_data = np.frombuffer(image_string, dtype=np.uint8)
            image_data = image_data.reshape((camera.getHeight(), camera.getWidth(), 4))
            rgb_matrix = image_data[:, :, :3]
            image_name = recorder.add_image(rgb_matrix)

            depth_array = range_finder.getRangeImage(data_type="buffer")
            depth_array = np.ctypeslib.as_array(depth_array, (range_finder.getWidth(), range_finder.getHeight()))
            depth_array = depth_array / range_finder.getMaxRange()
            depth_array = depth_array*255
            depth_array[depth_array == float('inf')] = 255
            depth_array = depth_array.astype(np.uint8)
            depth_image_name = recorder.add_image(depth_array, 'depth')

            recorder.add_row([x_global, y_global, z_global, yaw, 
                              waypoints_passed, 
                              v_x_global, v_y_global, v_z_global,
                              v_x, v_y,
                              yaw_rate,
                              v_target[0], v_target[1], 
                              height,
                              yaw_desired, 
                              image_name,
                              depth_image_name])
        
        
        ##### Reset target position
        gate_position = gate_node.getPosition()
        gate_rotation = gate_node.getField('rotation').getSFRotation()
        d_pos = np.array(gate_position[:2]) - np.array(drone_state[:2])
        if np.linalg.norm(d_pos) < 0.2 and abs(z_global - (gate_position[2] + 1)) < 0.1:
            waypoints_passed += 1
            if waypoints_passed < NUM_GATES:
                gate_state = gate_poses.pop(0)
            else:
                if RECORD:
                    recorder.save_data()
                # robot.simulationQuit(1)
            random_pos, random_rot = gate_state['pos'], gate_state['rot']

            ##### SET RANDOM GATE POSE
            gate_node.getField('translation').setSFVec3f( random_pos.tolist() )
            gate_node.getField('rotation').setSFRotation( random_rot.tolist() ) 


        ##### =====================

        # PID velocity controller with fixed height
        motor_power, a = PID_crazyflie.pid(dt, v_target[0], v_target[1],
                                        yaw_desired, height,
                                        roll, pitch, yaw_rate,
                                        z_global, v_x, v_y)
        
        # print("="*100)
        # print(a)
        # hist.append(a)
        # print(f"{np.mean(hist, axis=0)=}\n  {np.std(hist, axis=0)=}\n  {np.max(hist, axis=0)=}\n {np.min(hist, axis=0)=}\n")

        m1_motor.setVelocity(-motor_power[0])
        m2_motor.setVelocity(motor_power[1])
        m3_motor.setVelocity(-motor_power[2])
        m4_motor.setVelocity(motor_power[3])
        
        

        past_time = robot.getTime()
        past_x_global = x_global
        past_y_global = y_global
         
        i += 1
        
