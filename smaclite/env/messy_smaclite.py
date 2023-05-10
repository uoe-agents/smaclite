from typing import List, Tuple
from smaclite.env.smaclite import SMACliteEnv
import numpy as np

from smaclite.env.maps.map import MapInfo
from smaclite.env.units.unit import Unit


class MessySMACliteEnv(SMACliteEnv):
    """Messy Version of SMAClite from https://arxiv.org/abs/2301.01649"""

    def __init__(
        self,
        map_info: MapInfo = None,
        map_file: str = None,
        seed=None,
        use_cpp_rvo2=False,
        initial_random_steps=10,
        failure_obs_prob=0.15,
        failure_factor=-1.0,
        **kwargs,
    ):
        """Initializes the environment. Note that one of map_info or map_file
        is always required.

        Args:
            initial_random_steps (int, default=10): The amount of steps that the random walker takes at the start of the episode
            failure_obs_prob (float, default=0.15): The probability of a observation failing
            failure_factor (float, default=-1.0): The value that the observation is multiplied with, in case of failiure
        """
        # Unpack arguments from sacred
        self.initial_random_steps = initial_random_steps
        self.failure_obs_prob = failure_obs_prob
        self.failure_factor = failure_factor
        super().__init__(map_info, map_file, seed, use_cpp_rvo2, **kwargs)

    def __get_agent_obs(
        self, unit: Unit, visible_allies: List[Unit], visible_enemies: List[Unit]
    ):
        obs = super().__get_agent_obs(unit, visible_allies, visible_enemies)

        if np.random.rand() <= self.failure_obs_prob:
            return self.failure_factor * obs

        return obs

    def random_walk(self, max_steps: int):
        steps = 0

        while steps < max_steps:
            actions = []
            for agent_avail_actions in self.get_avail_actions():
                avail_actions_ind = np.nonzero(agent_avail_actions)[0]
                action = int(np.random.choice(avail_actions_ind))
                actions.append(action)

            _, _, done, _ = self.step(actions)
            steps += 1

            if done:
                return False

        return True

    def reset(
        self, seed=None, return_info=False, options=None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed, return_info, options)

        done = self.random_walk(self.initial_random_steps)

        # Fallback if episode has already terminated
        if done:
            return super().reset(seed, return_info, options)

        if return_info:
            return self.__get_obs, self.__get_info()

        return self.__get_obs()
