from typing import Tuple
import gym
import numpy as np

from smaclite.env.maps.map import Faction, MapInfo
from smaclite.env.units.unit_type import CombatType, Unit


class SMACLiteEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    """
    This is the SMACLite environment.
    """
    def __init__(self, map_info: MapInfo):
        self.map_info = map_info
        self.n_agents = map_info.num_allied_units
        self.n_enemies = map_info.num_enemy_units
        action_spaces = []
        for group in map_info.groups:
            if group.faction == Faction.ENEMY:
                continue
            base_actions = 6
            for unit_type, count in group.units:
                extra_actions = self.n_enemies \
                    if unit_type.combat_type == CombatType.DAMAGE \
                    else self.n_agents - 1
                total_actions = base_actions + extra_actions
                action_spaces.extend(gym.spaces.Discrete(total_actions)
                                     for _ in range(count))
        self.action_space = gym.spaces.Tuple(action_spaces)
        self.observation_space = gym.spaces.Box(float('-inf'), float('inf'))

        def _get_obs(self):
            raise NotImplementedError

        def _get_info(self):
            raise NotImplementedError

        def get_state(self):
            raise NotImplementedError

        def get_avail_actions(self):
            raise NotImplementedError

        def reset(self, seed=None, return_info=False, options=None) \
                -> Tuple[np.ndarray, dict]:
            self.units = []
            for group in self.map_info.groups:
                group_size = sum(count for _, count in group.units)
                square_side = np.ceil(np.sqrt(group_size)).astype(int)
                x = group.x
                y = group.y
                placed_in_row = 0
                max_height_in_row = 0
                width_in_row = 0
                row_heights = []
                # Plan out the layout of the units in the group
                for unit_type, count in group.units:
                    for _ in range(count):
                        max_height_in_row = max(max_height_in_row,
                                                unit_type.stats.size / 2)
                        width_in_row += unit_type.stats.size
                        placed_in_row += 1
                        if placed_in_row == square_side:
                            row_heights.append(max_height_in_row)
                            placed_in_row = 0
                            max_height_in_row = 0
                            width_in_row = 0
                # Actually place the units
                current_row = 0
                placed_in_row = 0
                for unit_type, count in group.units:
                    for _ in range(count):
                        self.units.append(Unit(unit_type, group.faction,
                                               x, y))
                        x += unit_type.stats.size
                        placed_in_row += 1
                        if placed_in_row == square_side \
                                and current_row + 1 < square_side:
                            x = group.x
                            y += (row_heights[current_row]
                                  + row_heights[current_row + 1])
            raise NotImplementedError

        def step(self, action):
            raise NotImplementedError

        def render(self, mode='human'):
            if mode == 'human':
                if self.renderer is None:
                    from smaclite.env.rendering.renderer import Renderer
                    self.renderer = Renderer()
                self.renderer.render(self.map_info)

        def close(self):
            if self.renderer is not None:
                self.renderer.close()
