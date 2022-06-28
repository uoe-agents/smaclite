from enum import Enum
from typing import List, Tuple

import gym
import numpy as np

from smaclite.env.maps.map import Faction, Group, MapInfo, TerrainType
from smaclite.env.units.unit import Unit
from smaclite.env.units.unit_command import AttackMoveCommand, \
    AttackUnitCommand, MoveCommand, NoopCommand, StopCommand
from smaclite.env.units.unit_type import CombatType, UnitType
from smaclite.env.rvo2.updater import VelocityUpdater

GROUP_BUFFER = 0.05
AGENT_SIGHT_RANGE = 9
AGENT_TARGET_RANGE = 6
MOVE_AMOUNT = 2
STEP_MUL = 8
DIRECTION_TO_MOVEMENT = {
    0: np.array([0, 1]),
    1: np.array([0, -1]),
    2: np.array([1, 0]),
    3: np.array([-1, 0]),
}


class Direction(Enum):
    @property
    def dx_dy(self) -> np.ndarray:
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
        self.velocity_updater = VelocityUpdater()
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
                    if unit_type.combat_type == CombatType.DAMAGE \
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
        self.all_units = []
        for group in self.map_info.groups:
            group_list = self.agents \
                if group.faction == Faction.ALLY \
                else self.enemies
            new_units = self.__place_group(group)
            group_list.extend(new_units)
            self.all_units.extend(new_units)
        assert len(self.agents) == self.n_agents and \
            len(self.enemies) == self.n_enemies
        self.velocity_updater.set_all_units(self.all_units)
        self.max_reward = self.n_enemies * 10 + 100 \
            + sum(enemy.hp + enemy.shield
                  for enemy in self.enemies)
        self.__enemy_attack()
        obs = self.__get_obs()
        if return_info:
            info = self.__get_info() if return_info else None
            return obs, info
        return obs

    def step(self, actions):
        assert len(actions) == self.n_agents
        avail_actions = self.get_avail_actions()
        for i, (agent, action) in enumerate(zip(self.agents, actions)):
            if not avail_actions[i][action]:
                print(f"ERROR: invalid action for agent {i}: {action}")
                actions[i] = 1  # change to stop
            agent.command = self.__get_command(agent, action)
        self.last_actions = actions
        reward = sum(self.__world_step() for _ in range(STEP_MUL))
        all_enemies_dead = all(enemy.hp <= 0 for enemy in self.enemies)
        if all_enemies_dead:
            reward += 100

        done = all_enemies_dead or all(a.hp == 0 for a in self.agents)

        reward /= self.max_reward / 20  # Scale reward between 0 and 20
        return self.__get_obs(), reward, done, self.__get_info()

    def render(self, mode='human'):
        if mode == 'human':
            if self.renderer is None:
                from smaclite.env.rendering.renderer import Renderer
                self.renderer = Renderer()
            self.renderer.render(self.map_info, self.all_units)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
        self.velocity_updater.close()

    def get_avail_actions(self):
        return [self.__get_agent_avail_actions(unit)
                for unit in self.agents]

    def get_state(self):
        state = []
        for unit in self.agents:
            state.extend(self.__get_unit_state_features(unit, True))
        for enemy in self.enemies:
            state.extend(self.__get_unit_state_features(enemy, False))
        state.extend(self.last_actions)
        return np.array(state)

    def __world_step(self):
        if self.renderer is not None:
            self.render()
        for unit in self.all_units:
            unit.prepare_velocity()
        self.velocity_updater.compute_new_velocities()
        return sum(unit.game_step() for unit in self.all_units)

    def __get_unit_state_features(self, unit: Unit, ally: bool):
        has_shields = self.map_info.ally_has_shields if ally else \
            self.map_info.enemy_has_shields
        num_features = 4 + has_shields + self.map_info.num_unit_types
        cx, cy = self.map_info.width / 2, self.map_info.height / 2
        if unit.hp == 0:
            return [0 for F_ in range(num_features)]
        features = [
            unit.hp / unit.max_hp,
        ]
        if ally:
            # TODO adjust for medivacs
            features.append(unit.cooldown / unit.max_cooldown)
        dx, dy = unit.pos - [cx, cy]
        features.extend([
            dx / self.map_info.width,
            dy / self.map_info.height,
        ])
        if has_shields:
            features.append(unit.shield / unit.max_shield)
        if self.map_info.num_unit_types:
            features.extend(self.__get_unit_type_id(unit.type))
        return features

    def __get_agent_avail_actions(self, unit: Unit):
        actions = np.zeros(self.n_actions)
        if unit.hp == 0:
            actions[0] = 1
            return actions
        actions[1] = 1
        for direction in Direction:
            actions[2 + direction.value] = self.__can_move(unit, direction)
        targets = self.enemies \
            if unit.combat_type == CombatType.DAMAGE \
            else [ally for ally in self.agents if ally != unit]
        for i, target in enumerate(targets):
            actions[6 + i] = self.__can_target(unit, target)
        return actions

    def __get_obs(self):
        return tuple(self.__get_agent_obs(agent) for agent in self.agents)

    def __get_info(self):
        return {'avail_actions': self.get_avail_actions(),
                'state': self.get_state()}

    def __can_target(self, unit: Unit, target: Unit):
        if target.hp == 0 or unit.hp == 0:
            return 0
        dpos = target.pos - unit.pos
        distance_sq = np.inner(dpos, dpos)
        return distance_sq <= AGENT_TARGET_RANGE ** 2

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
        if unit.hp == 0:
            return np.zeros(self.obs_size, dtype=np.float32)
        obs = np.zeros(self.obs_size, dtype=np.float32)
        # Movement features
        for direction in Direction:
            obs[direction.value] = avail_actions[2 + direction.value]
        # Enemy features
        base = 4
        for i, enemy in enumerate(self.enemies):
            dpos = enemy.pos - unit.pos
            distance = np.linalg.norm(dpos)
            if distance > AGENT_SIGHT_RANGE or enemy.hp == 0:
                continue
            dx, dy = enemy.pos - unit.pos
            obs[base] = avail_actions[6 + i]
            obs[base + 1] = distance / AGENT_SIGHT_RANGE
            obs[base + 2] = dx / self.map_info.width
            obs[base + 3] = dy / self.map_info.height
            obs[base + 4] = enemy.hp / enemy.max_hp
            if self.map_info.enemy_has_shields:
                obs[base + 5] = enemy.shield / enemy.max_shield
            if self.map_info.num_unit_types:
                obs[base + 6:base + 6 + self.map_info.num_unit_types] = self.__get_unit_type_id(enemy.type)
            base += self.enemy_feat_size
        # Ally features
        for ally in self.agents:
            if ally == unit:
                continue
            dpos = ally.pos - unit.pos
            distance = np.linalg.norm(dpos)
            if distance > AGENT_SIGHT_RANGE or ally.hp == 0:
                continue
            dx, dy = ally.pos - unit.pos
            obs[base] = 1
            obs[base + 1] = distance / AGENT_SIGHT_RANGE
            obs[base + 2] = dx / AGENT_SIGHT_RANGE
            obs[base + 3] = dy / AGENT_SIGHT_RANGE
            obs[base + 4] = ally.hp / ally.max_hp
            if self.map_info.ally_has_shields:
                obs[base + 5] = ally.shield / ally.max_shield
            if self.map_info.num_unit_types:
                obs[base + 6:base + 6 + self.map_info.num_unit_types] = self.__get_unit_type_id(ally.type)
            base += self.ally_feat_size
        # Own features
        obs[base] = unit.hp / unit.max_hp
        if self.map_info.ally_has_shields:
            obs[base + 1] = unit.shield / unit.max_shield
        if self.map_info.num_unit_types:
            obs[base + 2:base + 2 + self.map_info.num_unit_types] = self.__get_unit_type_id(unit.type)

        return obs

    def __can_move(self, unit: Unit, direction: Direction):
        check_value = MOVE_AMOUNT / 2
        dpos = direction.dx_dy
        npos = unit.pos + dpos * check_value
        return 0 <= npos[0] < self.map_info.width \
            and 0 <= npos[1] < self.map_info.height \
            and self.map_info.terrain[int(npos[0])][int(npos[1])] \
            == TerrainType.NORMAL

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
        row_radii = [max(u.radius if u else 0 for u in row)
                     for row in unit_grid]
        prev_row_height = 0
        group_height = 2 * sum(row_radii) + (square_side - 1) * GROUP_BUFFER
        row_widths = [sum(u.size if u else 0 for u in row)
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
                x += m * (prev_unit_width + u.radius)
                prev_unit_width = u.radius
                unit = Unit(u, faction, x, y,
                            len(self.all_units) + len(unit_list))
                # Uncomment to test killing units
                # unit.hp = np.random.choice([0, np.random.randint(unit.hp)])
                unit_list.append(unit)
                x += m * GROUP_BUFFER
            y += m * GROUP_BUFFER
        return unit_list

    def __enemy_attack(self):
        pos = np.array(self.map_info.attack_point)
        attack_move_command = AttackMoveCommand(pos,
                                                targets=self.agents)
        for enemy in self.enemies:
            if enemy.hp == 0:
                continue

            enemy.command = attack_move_command

    def __get_command(self, unit: Unit, action):
        if action == 0:
            return NoopCommand()
        if action == 1:
            return StopCommand()
        if 2 <= action <= 5:
            dpos = [2 * n for n in Direction(action - 2).dx_dy]
            return MoveCommand(unit.pos + dpos)
        return AttackUnitCommand(self.enemies[action - 6])
