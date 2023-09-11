import time
import stable_baselines3 as sb3

from stable_baselines3 import DQN, A2C
from gym import Wrapper, ObservationWrapper
from gym.spaces import MultiDiscrete, Box
import gym
import numpy as np

import smaclite  # noqa
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("error", category=RuntimeWarning)
RENDER = True
USE_CPP_RVO2 = False

class SingleAgentWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        act_dim = [space.n for space in env.action_space]
        l = [np.binary_repr(space.n) for space in env.action_space]
        l = int(''.join(l))
        action = np.base_repr(l, base=10)
        obs_low = np.concatenate([space.low for space in env.observation_space])
        obs_high = np.concatenate([space.high for space in env.observation_space])
        print(obs_low)
        self.action_space = MultiDiscrete(act_dim)
        self.observation_space = Box(low=obs_low, high=obs_high)

    def step(self, action):
        l = [np.binary_repr(a, width=self.env.action_space[i].n) for i, a in enumerate(action)]
        l = int(''.join(l))
        print('l', l)
        action = np.base_repr(l, base=10)
        action = np.array(action).astype(int)
        print("action", action)
        obs, _, terminated, truncated, info = self.env.step(action)
        #state = env.get_state()
        obs = np.concatenate(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, return_info=True,  **args):
        if return_info:
            obs, info = self.env.reset(return_info=True, **args)
            obs = np.concatenate(obs).reshape(1, -1)
            return obs, info
        else:
            obs =  self.env.reset(return_info=False, **args)
            obs = np.concatenate(obs).reshape(1, -1)
            print('obs', obs.shape)
            print('obs', obs)
            state = self.env.get_state()
            print('state', state.shape)
            return obs

class StateWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env.reset()
        state = self.env.get_state()
        high = np.ones(np.size(state))
        self.observation_space = Box(-high, high)

    def observation(self, obs):
        return self.env.get_state().reshape(1, -1)



def main():
    #env = "MMM2"
    env = "2s3z"
    env = gym.make(f"smaclite/{env}-v0",
                   use_cpp_rvo2=USE_CPP_RVO2)
    env = StateWrapper(env)
    #model = A2C("MlpPolicy", env, verbose=1)
    #model.learn(total_timesteps=10000, log_interval=4)
    episode_num = 20
    total_time = 0
    total_timesteps = 0
    for i in range(episode_num):
        obs, info = env.reset(return_info=True)
        if RENDER:
            env.render()
        done = False
        episode_reward = 0
        timer = time.time()
        episode_time = 0
        timestep_no = 0
        while not done and timestep_no < 200:
            actions = []
            avail_actions = info['avail_actions']
            for info in range(env.n_agents):
                avail_indices = [i for i, x
                                 in enumerate(avail_actions[info])
                                 if x]
                actions.append(int(np.random.choice(avail_indices)))
                # time.sleep(1/2)
            timer = time.time()
            print("obs", obs.shape)
            obs, reward, done, info = env.step(actions)
            episode_time += time.time() - timer
            episode_reward += reward
            timestep_no += 1
        print(f"Total reward in episode {episode_reward}")
        print(f"Episode {i} took {episode_time} seconds "
              f"and {timestep_no} timesteps.")
        print(f"Average per timestep: {episode_time/timestep_no}")
        total_time += episode_time
        total_timesteps += timestep_no
    print(f"Average time per timestep: {total_time/total_timesteps}")
    env.close()


if __name__ == "__main__":
    main()
