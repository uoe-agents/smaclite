from enum import Enum
import math
from typing import List, Tuple

import gym
import numpy as np

from smaclite.env.maps.map import Faction, Group, MapInfo, TerrainType
from smaclite.env.units.unit import Unit
from smaclite.env.units.unit_type import CombatType, UnitType

GROUP_BUFFER = 0.05
AGENT_SIGHT_RANGE = 9
AGENT_TARGET_RANGE = 6
MOVE_AMOUNT = 2
STEP_MUL = 8
DIRECTION_TO_MOVEMENT = {
    0: (0, 1),
    1: (0, -1),
    2: (1, 0),
    3: (-1, 0),
}


class Direction(Enum):
    @property
    def dx_dy(self) -> Tuple[int, int]:
        return DIRECTION_TO_MOVEMENT[self.value]

    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class SMACliteEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}
    """
    This is the SMAClite environment.
    """
    def __init__(self, map_info: MapInfo):
        self.map_info = map_info
        self.n_agents = map_info.num_allied_units
        self.agents: List[Unit] = []
        self.n_enemies = map_info.num_enemy_units
        self.enemies: List[Unit] = []
        self.all_units: List[Unit] = []
        self.renderer = None
        self.last_actions: List[int] = [None for _ in range(self.n_agents)]
        self.enemy_feat_size = 5 + self.map_info.num_unit_types \
            + self.map_info.enemy_has_shields
        self.ally_feat_size = 5 + self.map_info.num_unit_types \
            + self.map_info.ally_has_shields
        self.obs_size = sum((
            # whether movement in the 4 directions is possible
            4,
            # enemy attackable, distance, x, y, health, shield, unit type
            (self.n_enemies) * self.enemy_feat_size,
            # aly visible, distance, x, y, health, shield, unit type
            (self.n_agents - 1) * self.ally_feat_size,
            # own health, shield, unit type
            1 + self.map_info.num_unit_types + self.map_info.ally_has_shields
        ))
        num_medivacs = sum(sum(count for t, count in group.units
                               if t == UnitType.MEDIVAC)
                           for group in map_info.groups
                           if group.faction == Faction.ALLY)
        # NOTE this has an assumption that medivacs can heal anything but
        # themselves, which is not exactly true in SC2
        self.n_actions = 6 + max((self.n_agents - num_medivacs),
                                 self.n_enemies)
        action_spaces = []
        for group in map_info.groups:
            if group.faction == Faction.ENEMY:
                continue
            base_actions = 6
            for unit_type, count in group.units:
                extra_actions = self.n_enemies \
                    if unit_type.stats.combat_type == CombatType.DAMAGE \
                    else self.n_agents - 1
                total_actions = base_actions + extra_actions
                action_spaces.extend(gym.spaces.Discrete(total_actions)
                                     for _ in range(count))
        self.action_space = gym.spaces.Tuple(action_spaces)
        self.observation_space = gym.spaces.Tuple(
            gym.spaces.Box(-1, 1, (self.obs_size,), dtype=np.float32)
            for _ in range(self.n_agents))

    def reset(self, seed=None, return_info=False, options=None) \
            -> Tuple[np.ndarray, dict]:
        self.agents = []
        self.enemies = []
        for group in self.map_info.groups:
            group_list = self.agents \
                if group.faction == Faction.ALLY \
                else self.enemies
            new_units = self.__place_group(group)
            group_list.extend(new_units)
            self.all_units.extend(new_units)

        assert len(self.agents) == self.n_agents and \
            len(self.enemies) == self.n_enemies
        obs = self.__get_obs()
        if return_info:
            info = self.__get_info() if return_info else None
            return obs, info
        return obs

    def step(self, actions):
        avail_actions = self.get_avail_actions()
        for i, (agent, action) in enumerate(zip(self.agents, actions)):
            if not avail_actions[i][action]:
                print(f"ERROR: invalid action for agent {i}: {action}")
                actions[i] = 1  # change to stop
        self.last_actions = actions
        done = False

        self.reward = 0
        for _ in range(STEP_MUL):
            self.__world_step()

        return self.__get_obs(), self.reward, done, self.__get_info()

    def render(self, mode='human'):
        if mode == 'human':
            if self.renderer is None:
                from smaclite.env.rendering.renderer import Renderer
                self.renderer = Renderer()
            self.renderer.render(self.map_info, self.all_units)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    def get_avail_actions(self):
        return [self.__get_agent_avail_actions(unit)
                for unit in self.agents.values()]

    def get_state(self):
        state = []
        for unit in self.agents:
            state.extend(self.__get_unit_state_features(unit, True))
        for enemy in self.enemies:
            state.extend(self.__get_unit_state_features(enemy, False))
        state.extend(self.last_actions)
        return np.array(state)

    def __world_step(self):
        raise NotImplementedError

    def __get_unit_state_features(self, unit: Unit, ally: bool):
        has_shields = self.map_info.ally_has_shields if ally else \
            self.map_info.enemy_has_shields
        num_features = 4 + has_shields + self.map_info.num_unit_types
        cx, cy = self.map_info.width / 2, self.map_info.height / 2
        if unit.hp == 0:
            return [0 for _ in range(num_features)]
        features = [
            unit.hp / unit.type.stats.hp,
        ]
        if ally:
            # TODO adjust for medivacs
            features.append(unit.cooldown / unit.type.stats.cooldown)
        features.extend([
            (unit.x - cx) / self.map_info.width,
            (unit.y - cy) / self.map_info.height,
        ])
        if has_shields:
            features.append(unit.shield / unit.type.stats.shield)
        if self.map_info.num_unit_types:
            features.extend(self.__get_unit_type_id(unit.type))
        return features

    def __get_agent_avail_actions(self, unit: Unit):
        if unit.hp == 0:
            return [1] + [0] * (self.n_actions - 1)
        actions = [0, 1]  # Can't noop, can stop
        actions.extend(int(self.__can_move(unit, direction))
                       for direction in Direction)
        targets = self.enemies \
            if unit.type.stats.combat_type == CombatType.DAMAGE \
            else [ally for ally in self.agents if ally != unit]
        actions.extend(int(self.__can_target(unit, target))
                       for target in targets)
        return actions

    def __get_obs(self):
        return tuple([self.__get_agent_obs(agent)
                      for agent in self.agents])

    def __get_info(self):
        raise NotImplementedError

    def __can_target(self, unit: Unit, target: Unit):
        if target.hp == 0 or unit.hp == 0:
            return 0
        distance = math.dist((unit.x, unit.y), (target.x, target.y))
        return distance <= AGENT_TARGET_RANGE

    def __get_unit_type_id(self, unit_type: UnitType):
        """"
        Args:
            unit_type (UnitType): The unit type to get the id for.

        Returns:
            _type_: A 1-hot list with all but the given unit type's index as 0,
        """
        type_list = [0 for _ in range(self.map_info.num_unit_types)]
        type_list[self.map_info.unit_type_ids[unit_type]] = 1
        return type_list

    def __get_agent_obs(self, unit: Unit):
        avail_actions = self.__get_agent_avail_actions(unit)
        obs = np.zeros(self.obs_size)
        if unit.hp == 0:
            return np.zeros(self.obs_size, dtype=np.float32)
        # Movement features
        obs = [avail_actions[2 + direction.value]
               for direction in Direction]
        # Enemy features
        for i, enemy in enumerate(self.enemies):
            distance = math.dist((unit.x, unit.y), (enemy.x, enemy.y))
            if distance > AGENT_SIGHT_RANGE or enemy.hp == 0:
                obs.extend(0 for _ in range(self.enemy_feat_size))
                continue
            obs.extend([
                avail_actions[6 + i],
                distance / AGENT_SIGHT_RANGE,
                (enemy.x - unit.x) / AGENT_SIGHT_RANGE,
                (enemy.y - unit.y) / AGENT_SIGHT_RANGE,
                enemy.hp / enemy.type.stats.hp,
            ])
            if self.map_info.enemy_has_shields:
                obs.append(enemy.shield / enemy.type.stats.shield)
            if self.map_info.num_unit_types:
                obs.extend(self.__get_unit_type_id(enemy.type))
        # Ally features
        for ally in self.agents:
            if ally == unit:
                continue
            distance = math.dist((unit.x, unit.y), (ally.x, ally.y))
            if distance > AGENT_SIGHT_RANGE or ally.hp == 0:
                obs.extend(0 for _ in range(self.ally_feat_size))
                continue
            obs.extend([
                1,  # visible
                distance / AGENT_SIGHT_RANGE,
                (ally.x - unit.x) / AGENT_SIGHT_RANGE,
                (ally.y - unit.y) / AGENT_SIGHT_RANGE,
                ally.hp / ally.type.stats.hp,
            ])
            if self.map_info.ally_has_shields:
                obs.append(ally.shield / ally.type.stats.shield)
            if self.map_info.num_unit_types:
                obs.extend(self.__get_unit_type_id(ally.type))
        # Own features
        obs.append(unit.hp / unit.type.stats.hp)
        if self.map_info.ally_has_shields:
            obs.append(unit.shield / unit.type.stats.shield)
        if self.map_info.num_unit_types:
            obs.extend(self.__get_unit_type_id(unit.type))

        return np.array(obs, dtype=np.float32)

    def __can_move(self, unit: Unit, direction: Direction):
        check_value = MOVE_AMOUNT / 2
        dx, dy = direction.dx_dy
        x, y = unit.x + dx * check_value, unit.y + dy * check_value
        return 0 <= x < self.map_info.width and 0 <= y < self.map_info.height \
            and self.map_info.terrain[int(x)][int(y)] == TerrainType.NORMAL

    def __place_group(self, group: Group) -> List[Unit]:
        faction = group.faction
        unit_list = []
        all_units_in_group = []
        for unit_type, count in group.units:
            all_units_in_group.extend([unit_type] * count)
        group_size = len(all_units_in_group)
        square_side = np.ceil(np.sqrt(group_size)).astype(int)
        unit_grid = [[None for _ in range(square_side)]
                     for _ in range(square_side)]
        a = b = 0
        # Plan out the layout of the units in the group
        for unit_type in all_units_in_group:
            unit_grid[b][a] = unit_type
            a += 1
            if a == square_side:
                a = 0
                b += 1
        row_radii = [max(u.stats.size / 2 if u else 0 for u in row)
                     for row in unit_grid]
        prev_row_height = 0
        group_height = 2 * sum(row_radii) + (square_side - 1) * GROUP_BUFFER
        row_widths = [sum(u.stats.size if u else 0 for u in row)
                      for row in unit_grid]
        group_width = max(row_widths)
        # this is so enemy units spawn facing allied units
        m = 1 if faction == Faction.ALLY else -1
        x0, y = group.x - m * group_width / 2, group.y - m * group_height / 2
        # Actually place the units
        for i, row in enumerate(unit_grid):
            x = x0
            y += m * (prev_row_height + row_radii[i])
            prev_row_height = row_radii[i]
            prev_unit_width = 0
            for u in row:
                if u is None:
                    continue
                x += m * (prev_unit_width + u.stats.size / 2)
                prev_unit_width = u.stats.size / 2
                unit = Unit(u, faction, x, y)
                unit.hp = np.random.choice([0, np.random.randint(unit.hp)])
                unit_list.append(unit)
                x += m * GROUP_BUFFER
            y += m * GROUP_BUFFER
        return unit_list
