import time

import gym
import numpy as np

import smaclite  # noqa

RENDER = True


def main():
    # np.random.seed(2)
    env = gym.make("smaclite/2s3z-v0")

    episode_num = 100
    for i in range(episode_num):
        obs, info = env.reset(return_info=True)
        done = False
        episode_reward = 0
        timer = time.time()
        timestep_no = 0
        while not done and timestep_no < 200:
            actions = []
            avail_actions = info['avail_actions']
            for info in range(env.n_agents):
                avail_indices = [i for i, x
                                 in enumerate(avail_actions[info])
                                 if x]
                actions.append(int(np.random.choice(avail_indices)))
            if RENDER:
                env.render()
                # time.sleep(1/2)
            obs, reward, done, info = env.step(actions)
            episode_reward += reward
            timestep_no += 1
        end_timer = time.time()
        secs = end_timer - timer
        print(f"Total reward in episode {episode_reward}")
        print(f"Episode {i} took {secs} seconds "
              f"and {timestep_no} timesteps.")
        print(f"Average per timestep: {secs/timestep_no}")
    env.close()


if __name__ == "__main__":
    main()
