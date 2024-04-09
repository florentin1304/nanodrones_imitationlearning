import sys
from controller import Supervisor
from track_generator import TrackGenerator
from math import cos, sin

try:
    import gym
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
except ImportError:
    sys.exit(
        'Please make sure you have all dependencies installed. '
        'Run: "pip3 install numpy gym==0.21 stable_baselines3"'
    )


class OpenAIGymEnvironment(Supervisor, gym.Env):
    def __init__(self, get_images=False, images_size=(168,168), max_episode_steps=100_000):
        super().__init__()
        self.__get_images = get_images
        self.__images_size = images_size

        ### Action space = [alt_command, roll_command, pitch_command, yaw_command]
        ### To be used in motor-mixing algorithm + limit motor command
        self.action_space = gym.spaces.Box(low=-30_000, high=30_000, shape=(4,), dtype=np.float32)

        ### Observation space
        ### - BOX 1d: [x, y, z, dx, dy, dz, dx_ego, dy_ego, roll, yaw, pitch, droll, dyaw, dpitch, gate_x, gate_y, gate_z, gate_yaw] + 12x distance_sensors
        ### - BOX 3d: [img(324,324), depth_img(324,324)]
        if self.__get_images:
            obs_spaces = {
                'state': gym.spaces.Box(low=float('inf'), high=float('inf'), shape=(18+12,)),
                'img_observation': gym.spaces.Box(low=0, high=1, shape=(2, *images_size))
            }
            self.observation_space = gym.spaces.Dict(obs_spaces)
        else:
            self.observation_space = gym.spaces.Box(low=-30_000, high=30_000, shape=(18+12,))

        self.state = None
        self.spec = gym.envs.registration.EnvSpec(id='BitcrazeEnv-v0', max_episode_steps=max_episode_steps)

        # Environment specific
        self.__timestep = int(self.getBasicTimeStep())
        self.__motors = []
        self.__sensors = {}
        self.__past_pos = np.zeros(shape=(3,))
        self.__gate_square_poses = None
        self.__current_gate_pose = None

        # Gate node
        root_node = self.getRoot()
        children_field = root_node.getField('children')
        self.__gate_node = self.getFromDef('imav2022-gate')

        # Tools
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.__timestep)

    def wait_keyboard(self):
        while self.keyboard.getKey() != ord('Y'):
            super().step(self.__timestep)
    
    def get_state_observation(self):
        robot = self.getSelf()
        pos = np.array(self.__sensors['gps'].getValues())
        dpos = (self.__past_pos - pos) / self.__dt
        rpy = self.__sensors['imu'].getRollPitchYaw()
        drpy = self.__sensors['gyro'].getValues()

        yaw = rpy[2]
        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)
        v_x = dpos[0] * cos_yaw + dpos[1] * sin_yaw
        v_y = - dpos[0] * sin_yaw + dpos[1] * cos_yaw
        dpos_ego = np.array([v_x, v_y])

        gate_pos = np.array(self.__current_gate_pose['pos'])
        gate_yaw = np.array([ self.__current_gate_pose['rot'][3] ])

        distances = np.array([d.getValue() for d in self.__sensors['dist_sensors']])

        state = np.hstack([pos, dpos, dpos_ego, rpy, drpy, gate_pos, gate_yaw, distances])
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
        robot = self.getSelf()
        timestep = int(self.getBasicTimeStep())

        # Initialize motors
        self.__motors = []
        for i in range(0,4):
            motor = robot.getDevice(f"m{i+1}_motor")
            motor.setPosition(float('inf'))
            motor.setVelocity(-1**(i))
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
        self.__sensors['dist_sensors'] = []
        for i in range(12):
            range_sensor = robot.getDevice(f"range_m_{i}")
            range_sensor.enable(timestep)
            self.__sensors['dist_sensors'].append(range_sensor)

        # Initialise state variables
        self.__past_time = 0
        self.__past_pos = np.zeros(shape=(3,))

        # Internals
        ### CREATE ENV RANDOMIZER
        NUM_GATES = 3
        tg = TrackGenerator(num_gate_poses=NUM_GATES)
        gate_poses = tg.generate()
        self.__gate_square_poses = tg.to_gate_squares(gate_poses)
        self.__current_gate_pose = self.__gate_square_poses.pop(0)

        # Open AI Gym generic
        # Internal
        super().step(self.__timestep)
        self.__dt = robot.getTime() - self.__past_time
        self.__past_time = robot.getTime()
        observation, _ = self.get_state_observation()

        return observation
    
    def __get_reward(self, named_obs):
        dpos = named_obs['dpos']
        d_me_gate = np.array(named_obs['gate_pos']) - np.array(named_obs['pos'])
        cos = np.dot(dpos, d_me_gate) / (np.linalg.norm(dpos) * np.linalg.norm(d_me_gate))

        return (cos/2) - 1


    def step(self, action):
        robot = self.getSelf()

        ##### Execute the action
        alt_command, roll_command, pitch_command, yaw_command = action

        # Motor mixing
        m1 = alt_command - roll_command + pitch_command + yaw_command
        m2 = alt_command - roll_command - pitch_command - yaw_command
        m3 = alt_command + roll_command - pitch_command + yaw_command
        m4 = alt_command + roll_command + pitch_command - yaw_command

        # Limit the motor command
        motor_power = np.array([m1,m2,m3,m4])
        motor_power = np.clip(motor_power, 0, 600)
        for i in range(4):
            self.__motors[i].setVelocity((-1**i) * motor_power[i])

        ##### Step and observe
        super().step(self.__timestep)
        self.__dt = robot.getTime() - self.__past_time
        self.__past_time = robot.getTime()
        observation, named_states = self.get_state_observation()

        d_me_gate = np.array(named_states['gate_pos']) - np.array(named_states['pos'])

        finishedGates = False
        extraReward = 0
        if np.linalg.norm(d_me_gate) < 0.15:
            if len(self.__gate_square_poses) > 0:
                gate_state = self.__gate_square_poses.pop(0)
                random_pos, random_rot = gate_state['pos'], gate_state['rot']

                ##### SET RANDOM GATE POSE
                self.__gate_node.getField('translation').setSFVec3f( random_pos.tolist() )
                self.__gate_node.getField('rotation').setSFRotation( random_rot.tolist() ) 
            else:
                finishedGates = True
            extraReward = 10
                


        # Reward & Done
        done = bool(self.__sensors['touch_sensor'].getValue()) or finishedGates
        reward = self.__get_reward(named_states) + extraReward

        self.__past_time = robot.getTime()
        self.__past_pos = np.array(observation[0:3])
        return observation, reward, done, {}


def main():
    # Initialize the environment
    env = OpenAIGymEnvironment()
    check_env(env)

    # Train
    model = PPO('MlpPolicy', env, n_steps=2048, verbose=1)
    model.learn(total_timesteps=1e5)

    # Replay
    print('Training is finished, press `Y` for replay...')
    env.wait_keyboard()

    obs = env.reset()
    for _ in range(100000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
        if done:
            obs = env.reset()


if __name__ == '__main__':
    main()
