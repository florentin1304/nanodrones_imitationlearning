import sys
sys.path.append("..")
from gatefollowing.gatefollowing import GateFollower
from controller import Supervisor

import torch
from math import cos, sin, atan2

from controller_utils.track_generator import TrackGenerator
from controller_utils.pid_controller import pid_velocity_fixed_height_controller
from controller_utils.maveric_trajectory_planner import generate_trajectory_mavveric
from controller_utils.pathplanner import PathPlanner

try:
    import gym
    import numpy as np
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

except ImportError:
    sys.exit(
        'Please make sure you have all dependencies installed. '
        'Run: "pip3 install numpy gym==0.21 stable_baselines3"'
    )


class OpenAIGymEnvironment(Supervisor, gym.Env):
    def __init__(self, get_images=False, images_size=(168,168), max_episode_steps=1_000):
        super().__init__()
        self.__get_images = get_images
        self.__images_size = images_size
        self.__max_episode_steps = max_episode_steps

        
        ### Action space = [alt_command, roll_command, pitch_command, yaw_command]
        ### To be used in motor-mixing algorithm + limit motor command
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        ### Observation space
        ### - BOX 1d: [x, y, z, dx, dy, dz, accx, accy, accz, 
        ###            dx_ego, dy_ego, roll, yaw, pitch, droll, dyaw, dpitch, 
        ###            gate_x, gate_y, gate_z, gate_yaw] 
        ###            + 12x distance_sensors + 4 actions
        ### - BOX 3d: [img(324,324), depth_img(324,324)]
        if self.__get_images:
            obs_spaces = {
                'state': gym.spaces.Box(low=float('inf'), high=float('inf'), shape=(21+12+4,)),
                'img_observation': gym.spaces.Box(low=0, high=1, shape=(2, *images_size))
            }
            self.observation_space = gym.spaces.Dict(obs_spaces)
        else:
            self.observation_space = gym.spaces.Box(low=-30_000, high=30_000, shape=(21+4,))

        self.state = None
        self.spec = gym.envs.registration.EnvSpec(id='BitcrazeEnv-v0', max_episode_steps=max_episode_steps)

        # Environment specific
        self.__timestep = int(self.getBasicTimeStep())
        self.__dt = self.__timestep / 1000
        self.__motors = []
        self.__sensors = {}
        self.__past_pos = np.zeros(shape=(3,))
        self.__current_waypoint = None
        self.__gate_square_poses = None


        # Gate node
        self.__root_node = self.getRoot()
        children_field = self.__root_node.getField('children')
        self.__gate_node = self.getFromDef('imav2022-gate')
        self.__drone_node = self.getFromDef('Crazyflie')

        # Tools
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.__timestep)

    def wait_keyboard(self):
        while self.keyboard.getKey() != ord('Y'):
            super().step(self.__timestep)
    
    def get_state_observation(self, actions):
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

        gate_pos = np.array(self.__get_current_gate_pose()['pos'])
        dgate_pos = np.array(self.__get_current_gate_pose()['pos']) - pos
        gate_yaw = np.array([ self.__get_current_gate_pose()['rot'][3] ])

        # distances = np.array([d.getValue() for d in self.__sensors['dist_sensors']])
        distances = []

        state = np.hstack([pos, dpos, acceleration, dpos_ego, rpy, drpy, dgate_pos, gate_yaw, distances, actions]).astype(np.float32)
        named_states = {
            "pos": pos,
            "dpos": dpos,
            "dpos_ego": dpos_ego,
            "rpy": rpy,
            "drpy": drpy,
            "gate_pos": gate_pos,
            "gate_yaw": gate_yaw,
            "distances": distances
        }
    
        if self.__get_images:
            raise Exception("Get images to be implemented!")
            images = np.zeros(shape=(2, *self.__images_size))
            full_state_observation = {
                'state': state,
                'img_observation': images
            }
            return full_state_observation

        return state, named_states

    def reset(self):
        # Reset the simulation
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.__timestep)

        ### Robot Init Sensors
        robot = self
        # self.__drone_node.getField('translation').setSFVec3f( [
        #     np.random.uniform(-0.5, 0.5),
        #     np.random.uniform(-0.5, 0.5),
        #     np.random.uniform(0,0) 
        #     ] )
        timestep = int(self.getBasicTimeStep())

        # Initialize motors
        self.__motors = []
        for i in range(0,4):
            motor = robot.getDevice(f"m{i+1}_motor")
            motor.setPosition(float('inf'))
            motor.setVelocity(((-1)**(i+1)) * 100 )
            self.__motors.append(motor)

        # Initialize Sensors
        self.__sensors = {}
        self.__sensors['imu'] = robot.getDevice("inertial_unit")
        self.__sensors['imu'].enable(timestep)
        self.__sensors['gps']= robot.getDevice("gps")
        self.__sensors['gps'].enable(timestep)
        self.__sensors['gyro'] = robot.getDevice("gyro")
        self.__sensors['gyro'].enable(timestep)
        self.__sensors['camera'] = robot.getDevice("camera")
        self.__sensors['camera'].enable(timestep)
        self.__sensors['touch_sensor'] = robot.getDevice("touchsensor")
        self.__sensors['touch_sensor'].enable(timestep)        
        self.__sensors['accelerometer'] = robot.getDevice("accelerometer")
        self.__sensors['accelerometer'].enable(timestep)
        # self.__sensors['dist_sensors'] = []
        # for i in range(12):
        #     range_sensor = robot.getDevice(f"range_m_{i}")
        #     range_sensor.enable(timestep)
        #     self.__sensors['dist_sensors'].append(range_sensor)


        # Initialise state variables
        self.__past_action = np.array([0, 0, 0, 0])
        self.__past_time = 0
        self.__past_pos = self.__sensors['gps'].getValues()

        # Internals
        ### TODO: CREATE ENV RANDOMIZER
        NUM_GATES = 3
        tg = TrackGenerator(num_gate_poses=NUM_GATES)
        self.__gate_poses = tg.generate_easy()
        self.__gate_square_poses = tg.to_gate_squares(self.__gate_poses)
        self.__current_waypoint = 0
        self.__update_gate_pose()

        trajectory = generate_trajectory_mavveric(self.__past_pos, self.__gate_square_poses)
        pathplanner = PathPlanner(trajectory=trajectory)
        pid_controller = pid_velocity_fixed_height_controller()

        for i in range(np.random.randint(200, 600)):
            self.__dt = robot.getTime() - self.__past_time

            # Get state information
            pos = np.array(self.__sensors['gps'].getValues())
            dpos = (pos - self.__past_pos) / self.__dt

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
            v_target, yaw_desired, des_height = pathplanner(drone_state, gate_position)

            # PID velocity controller with fixed height
            roll, pitch, yaw = rpy
            roll_rate, pitch_rate, yaw_rate = drpy
            x_global, y_global, z_global = pos
            motor_power, commands = pid_controller.pid(self.__dt, v_target[0], v_target[1],
                                            yaw_desired, des_height,
                                            roll, pitch, yaw_rate,
                                            z_global, v_x, v_y)
            
            # Apply motor powers found by PID
            for i in range(4):
                self.__motors[i].setVelocity(motor_power[i] * ((-1) ** (i+1)) )

            if self.__is_passing_through_gate(self.__sensors['gps'].getValues()):
                self.__current_waypoint += 1
                if self.__current_waypoint == len(self.__gate_poses):
                    self.reset()
                else:
                    self.__update_gate_pose()
                

            self.__past_pos = pos
            self.__past_action = commands
            self.__past_time = robot.getTime()
            super().step(self.__timestep)

        print("Start RL -> ", end='')

        # Open AI Gym generic
        # Internal
        super().step(self.__timestep)
        self.__dt = robot.getTime() - self.__past_time
        self.__past_time = robot.getTime()
        observation, _ = self.get_state_observation(self.__past_action)
        self.__iteration = 0 

        return observation
    
    def __get_reward(self, named_obs, action, istouching):
        d_me_gate = np.linalg.norm(
            np.array(named_obs['pos']) - np.array(named_obs['gate_pos'])
        )
        past_d_me_gate = np.linalg.norm(
            self.__past_pos - np.array(named_obs['gate_pos'])
        )
        r_adv = (1e3)*(past_d_me_gate-d_me_gate)
        
        dpos_yaw = np.array((named_obs['gate_pos']- named_obs['pos']))[:2]
        yaw = named_obs['rpy'][2]
    
        look_at_gate_reward = 0.2 * atan2(dpos_yaw[1], dpos_yaw[0]) - yaw
        daction_reward = -(2e-4) * np.linalg.norm(self.__past_action - action)
        dbodyrates_reward = -(5e-4) * np.linalg.norm(self.__past_action[1:] - action[1:])
        touching_reward = -4*istouching

        # return 10*np.exp(-abs(named_obs['gate_pos'][2] - named_obs['pos'][2])) + \
        #        np.exp(-abs(named_obs['rpy'][0] + named_obs['rpy'][1])) + \
        #        np.exp(-abs(named_obs['drpy'][0] + named_obs['drpy'][1]))
        return  r_adv + look_at_gate_reward + daction_reward + dbodyrates_reward + touching_reward



    def step(self, action):
        robot = self

        # motor_power = action * 300 + 300
        ##### Execute the action
        action = np.array([20, 4, 4, 4]) * action + np.array([60, 0, 0, 0])
        alt_command, roll_command, pitch_command, yaw_command = action
        
        # Motor mixing
        m1 = alt_command - roll_command + pitch_command + yaw_command
        m2 = alt_command - roll_command - pitch_command - yaw_command
        m3 = alt_command + roll_command - pitch_command + yaw_command
        m4 = alt_command + roll_command + pitch_command - yaw_command

        # Limit the motor command
        motor_power = np.array([m1,m2,m3,m4])
        motor_power = np.clip(motor_power, 0, 600)
        
        ### APPLY MOTORPOWER
        for i in range(4):
            self.__motors[i].setVelocity(motor_power[i] * ((-1) ** (i+1)) )

        ##### Step and observe
        super().step(self.__timestep)
        self.__dt = robot.getTime() - self.__past_time
        self.__past_time = robot.getTime()
        observation, named_states = self.get_state_observation(action)
        d_me_gate = np.array(named_states['gate_pos']) - np.array(named_states['pos'])

        finishedGates = False
        extraReward = 0
        if self.__is_passing_through_gate(self.__sensors['gps'].getValues()):
            extraReward += 100
            self.__current_waypoint += 1
            print("Entrato")
            if self.__current_waypoint == len(self.__gate_poses):
                print("Finished track... restarting simulation")
                finishedGates = True
            else:
                self.__update_gate_pose()
                


        # Reward & Done
        done = False
        if self.__iteration > self.__max_episode_steps or finishedGates:
            done = True
        rpy = self.__sensors['imu'].getRollPitchYaw()
        if abs(rpy[0]) > np.pi/2 + np.pi/12 or abs(rpy[1]) > np.pi/2 + np.pi/12:
            print("ribaltato")
            done = True
        reward = self.__get_reward(named_states, action, self.__sensors['touch_sensor'].getValue()) + extraReward
        # reward = extraReward

        self.__past_action = action
        self.__past_time = robot.getTime()
        self.__past_pos = np.array(observation[0:3])
        self.__iteration += 1

        return observation, reward, done, {}
    
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

    def __get_current_gate_pose(self):
        return self.__gate_square_poses[ self.__current_waypoint ]


def main():
    # Initialize the environment
    env = OpenAIGymEnvironment()
    check_env(env)

    # Train
    policy_kwargs = dict(activation_fn=torch.nn.Tanh)
                    #  net_arch=[256,512,1024,512,512,512,256,64,32])
    model = PPO('MlpPolicy', env, device='cuda' ,verbose=1, n_steps=2048, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=1e7, progress_bar=True)

    # Replay
    print('Training is finished, press `Y` for replay...')
    env.wait_keyboard()

    obs = env.reset()
    for _ in range(100_000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
        if done:
            obs = env.reset()


if __name__ == '__main__':
    main()
