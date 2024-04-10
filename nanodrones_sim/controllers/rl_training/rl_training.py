import sys
sys.path.append("..")
from gatefollowing.gatefollowing import GateFollower
from controller import Supervisor
from track_generator import TrackGenerator
from math import cos, sin, atan2
from pid_controller import pid_velocity_fixed_height_controller
import torch

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

        self.gatefollower = GateFollower(display_path=False, record=False)
        
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
        self.__gate_square_poses = None
        self.__current_gate_pose = None


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

        gate_pos = np.array(self.__current_gate_square_pose['pos'])
        dgate_pos = np.array(self.__current_gate_square_pose['pos']) - pos
        gate_yaw = np.array([ self.__current_gate_square_pose['rot'][3] ])

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
        self.__drone_node.getField('translation').setSFVec3f( [
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(0,0) 
            ] )
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
        self.__past_pos = np.zeros(shape=(3,))

        # Internals
        ### CREATE ENV RANDOMIZER
        self.__PID_crazyflie = pid_velocity_fixed_height_controller()
        NUM_GATES = 3
        tg = TrackGenerator(num_gate_poses=NUM_GATES)
        self.__gate_poses = tg.generate_easy()
        self.__gate_square_poses = tg.to_gate_squares(self.__gate_poses)
        self.__current_gate_pose = self.__gate_poses.pop(0)
        self.__current_gate_square_pose = self.__gate_square_poses.pop(0)
        self.__gate_node.getField('translation').setSFVec3f( self.__current_gate_pose['pos'].tolist() )
        self.__gate_node.getField('rotation').setSFRotation( self.__current_gate_pose['rot'].tolist() ) 


        # while self.__sensors['gps'].getValues()[2] < 0.1:
        #     motors = self.__PID_crazyflie.pid(self.__dt, 0, 0, 0, 1, 0, 0, 0,
        #                         self.__sensors['gps'].getValues()[2], 0, 0)
        #     for i in range(4):
        #         self.__motors[i].setVelocity(motors[i] * ((-1) ** (i+1)) )
        #     super().step(self.__timestep)
        #     self.__dt = robot.getTime() - self.__past_time

        # Open AI Gym generic
        # Internal
        super().step(self.__timestep)
        self.__dt = robot.getTime() - self.__past_time
        self.__past_time = robot.getTime()
        observation, _ = self.get_state_observation(np.zeros(shape=(4,)))
        self.__iteration = 0 

        return observation
    
    def __get_reward(self, named_obs, action, istouching):
        d_me_gate = np.linalg.norm(
            np.array(named_obs['pos']) - np.array(named_obs['gate_pos'])
        )
        past_d_me_gate = np.linalg.norm(
            self.__past_pos - np.array(named_obs['gate_pos'])
        )
        r_adv = 100*(past_d_me_gate-d_me_gate)
        
        dpos_yaw = np.array((named_obs['gate_pos']- named_obs['pos']))[:2]
        yaw = named_obs['rpy'][2]
    
        look_at_gate_reward = 0.2 * atan2(dpos_yaw[1], dpos_yaw[0]) - yaw
        daction_reward = -(2e-1) * np.linalg.norm(self.__past_action - action)
        dbodyrates_reward = -(5e-1) * np.linalg.norm(self.__past_action[1:] - action[1:])
        touching_reward = -4*istouching

        # return r_adv
        return  r_adv + look_at_gate_reward + daction_reward + dbodyrates_reward + touching_reward


    def step(self, action):
        robot = self

        # ##### Execute the action
        action = np.array([50, 4, 4, 4]) * action + np.array([100, 0, 0, 0])
        alt_command, roll_command, pitch_command, yaw_command = action
        
        # Motor mixing
        m1 = alt_command - roll_command + pitch_command + yaw_command
        m2 = alt_command - roll_command - pitch_command - yaw_command
        m3 = alt_command + roll_command - pitch_command + yaw_command
        m4 = alt_command + roll_command + pitch_command - yaw_command

        # Limit the motor command
        motor_power = np.array([m1,m2,m3,m4])
        motor_power = np.clip(motor_power, 0, 600)
        
        # motor_power = (0.5+action/2) * 20 + 40
        # action = action + np.array([0,0,0,1])
        # print(action)
        
        # v_x_target, v_y_target, yaw_desired, dheight = action

        # pos_global = self.__sensors['gps'].getValues()
        # v_global = (pos_global - self.__past_pos)/self.__dt
        # v_x_global, v_y_global, _ = v_global
        # height_desired = pos_global[2] + dheight

        # roll, pitch, yaw = self.__sensors['imu'].getRollPitchYaw()
        # cos_yaw = cos(yaw)
        # sin_yaw = sin(yaw)
        # v_x = v_x_global * cos_yaw + v_y_global * sin_yaw
        # v_y = - v_x_global * sin_yaw + v_y_global * cos_yaw
        # yaw_rate = self.__sensors['gyro'].getValues()[2]


        # motor_power = self.__PID_crazyflie.pid(self.__dt, v_x_target, v_y_target,
        #                         yaw_desired, height_desired,
        #                         roll, pitch, yaw_rate,
        #                         pos_global[2], v_x, v_y)
        
        ### APPLY MOTORPOWER
        # print(motor_power)
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
        if np.linalg.norm(d_me_gate) < 0.15:
            if len(self.__gate_square_poses) > 0:
                print('Entrato')
                self.__current_gate_pose = self.__gate_poses.pop(0)
                self.__current_gate_square_pose = self.__gate_square_poses.pop(0)
                self.__gate_node.getField('translation').setSFVec3f( self.__current_gate_pose['pos'].tolist() )
                self.__gate_node.getField('rotation').setSFRotation( self.__current_gate_pose['rot'].tolist() ) 

            else:
                finishedGates = True
            extraReward += 1000
                


        # Reward & Done
        done = False
        if self.__iteration > self.__max_episode_steps or finishedGates:
            done = True
        rpy = self.__sensors['imu'].getRollPitchYaw()
        if abs(rpy[0]) > np.pi/2 + np.pi/12 or abs(rpy[1]) > np.pi/2 + np.pi/12:
            print("ribaltato")
            done = True
            extraReward -= 100
        reward = self.__get_reward(named_states, action, self.__sensors['touch_sensor'].getValue()) + extraReward
        # reward = extraReward

        self.__past_action = action
        self.__past_time = robot.getTime()
        self.__past_pos = np.array(observation[0:3])
        self.__iteration += 1

        return observation, reward, done, {}


def main():
    # Initialize the environment
    env = OpenAIGymEnvironment()
    check_env(env)

    # Train
    policy_kwargs = dict(activation_fn=torch.nn.Tanh)
                    #  net_arch=[256,512,1024,512,512,512,256,64,32])
    model = PPO('MlpPolicy', env, device='cuda' ,verbose=1, n_steps=8192, policy_kwargs=policy_kwargs)
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
