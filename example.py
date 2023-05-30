import time

import gym
import numpy as np

import smaclite  # noqa

RENDER = False
USE_CPP_RVO2 = False


def main():
    env = "MMM2"
    env = gym.make(f"smaclite/{env}-v0",
                   use_cpp_rvo2=USE_CPP_RVO2)
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
