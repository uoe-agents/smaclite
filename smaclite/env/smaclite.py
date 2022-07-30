import random
from typing import Dict, List, Tuple

import gym
import numpy as np

from smaclite.env.maps.map import Faction, Group, MapInfo
from smaclite.env.rvo2.neighbour_finder import NeighbourFinder
from smaclite.env.rvo2.velocity_updater import VelocityUpdater
from smaclite.env.terrain.terrain import TerrainType
from smaclite.env.units.unit import Unit
from smaclite.env.units.unit_command import (AttackMoveCommand,
                                             AttackUnitCommand, MoveCommand,
                                             NoopCommand, StopCommand)
from smaclite.env.units.unit_type import CombatType, StandardUnit, UnitType
from smaclite.env.util.direction import Direction

GROUP_BUFFER = 0.05
AGENT_SIGHT_RANGE = 9
AGENT_TARGET_RANGE = 6
AGENT_TARGET_RANGE_SQ = AGENT_TARGET_RANGE ** 2
MOVE_AMOUNT = 2
STEP_MUL = 8
REWARD_WIN = 200
REWARD_KILL = 10


class SMACliteEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}
    """
    This is the SMAClite environment.
    """
    def __init__(self,
                 map_info: MapInfo = None,
                 map_file: str = None,
                 seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        if map_info is None and map_file is None:
            raise ValueError("Either map_info or map_file must be provided.")
        if map_file is not None:
            map_info = MapInfo.from_file(map_file)
        self.map_info = map_info
        self.n_agents = map_info.num_allied_units
        self.agents: Dict[int, Unit] = []
        self.n_enemies = map_info.num_enemy_units
        self.enemies: Dict[int, Unit] = []
        self.all_units: Dict[int, Unit] = {}
        self.renderer = None
        self.neighbour_finder_ally: NeighbourFinder = NeighbourFinder()
        self.neighbour_finder_enemy: NeighbourFinder = NeighbourFinder()
        self.neighbour_finder_all: NeighbourFinder = NeighbourFinder()
        num_healers = sum(sum(count for t, count in group.units
                              if t.stats.combat_type == CombatType.HEALING)
                          for group in map_info.groups
                          if group.faction == Faction.ALLY)
        # NOTE this has an assumption that healers can heal anything but
        # themselves, which is not exactly true in SC2
        num_target_actions = max(self.n_agents - num_healers, self.n_enemies) \
            if num_healers else self.n_enemies

        self.n_actions = 6 + num_target_actions
        self.max_unit_radius = max(type.radius
                                   for group in self.map_info.groups
                                   for type, _ in group.units)
        self.velocity_updater = VelocityUpdater(self.neighbour_finder_all,
                                                self.max_unit_radius,
                                                map_info.terrain)
        self.last_actions: np.ndarray = np.zeros(self.n_actions
                                                 * self.n_agents)
        # enemy attackable, distance, x, y, health, shield, unit type
        self.enemy_feat_size = 5 + self.map_info.enemy_has_shields \
            + self.map_info.num_unit_types
        # aly visible, distance, x, y, health, shield, unit type
        self.ally_feat_size = 5 + self.map_info.ally_has_shields \
            + self.map_info.num_unit_types

        self.obs_size = sum((
            # whether movement in the 4 directions is possible
            4,
            (self.n_enemies) * self.enemy_feat_size,
            (self.n_agents - 1) * self.ally_feat_size,
            # own health, shield, unit type
            1 + self.map_info.ally_has_shields + self.map_info.num_unit_types
        ))
        self.ally_state_feat_size = 4 + self.map_info.ally_has_shields + \
            self.map_info.num_unit_types
        # hp, dx, dy, shields, unit type
        self.enemy_state_feat_size = 3 + self.map_info.enemy_has_shields + \
            self.map_info.num_unit_types
        self.state_size = sum((
            self.n_agents * self.ally_state_feat_size,
            self.n_enemies * self.enemy_state_feat_size,
            # previous actions
            self.n_agents * self.n_actions,
        ))
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
        self.cx_cy = np.array([map_info.width / 2, map_info.height / 2])

    def reset(self, seed=None, return_info=False, options=None) \
            -> Tuple[np.ndarray, dict]:
        self.agents = {}
        self.enemies = {}
        self.all_units = {}
        for group in self.map_info.groups:
            self.__place_group(group)
        assert len(self.agents) == self.n_agents and \
            len(self.enemies) == self.n_enemies
        self.neighbour_finder_ally.set_all_units(self.agents)
        self.neighbour_finder_enemy.set_all_units(self.enemies)
        self.neighbour_finder_all.set_all_units(self.all_units)
        self.max_reward = self.n_enemies * REWARD_KILL + REWARD_WIN \
            + sum(enemy.hp + enemy.shield
                  for enemy in self.enemies.values())
        self.__enemy_attack()
        obs = self.__get_obs()
        if return_info:
            return obs, self.__get_info()
        return obs

    def step(self, actions):
        assert len(actions) == self.n_agents
        assert all(type(action) == int for action in actions)
        self.last_actions = np.eye(self.n_actions)[np.array(actions)] \
            .flatten()
        avail_actions = self.get_avail_actions()
        for i, action in enumerate(actions):
            if i not in self.agents:
                assert actions[i] == 0
                continue
            agent = self.agents[i]
            if not avail_actions[i][action]:
                raise ValueError(f"Invalid action for agent {i}: {action}")
            agent.command = self.__get_command(agent, action)
        reward = sum(self.__world_step() for _ in range(STEP_MUL))
        all_enemies_dead = len(self.enemies) == 0
        if all_enemies_dead:
            reward += 200
        done = all_enemies_dead or len(self.agents) == 0
        reward /= self.max_reward / 20  # Scale reward between 0 and 20
        return self.__get_obs(), reward, done, self.__get_info()

    def render(self, mode='human'):
        if mode == 'human':
            if self.renderer is None:
                from smaclite.env.rendering.renderer import Renderer
                self.renderer = Renderer()
            self.renderer.render(self.map_info, self.all_units.values())

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    def get_avail_actions(self):
        avail_for_dead = np.zeros(self.n_actions)
        avail_for_dead[0] = 1
        return [self.__get_agent_avail_actions(self.agents[i])
                if i in self.agents else avail_for_dead
                for i in range(self.n_agents)]

    def get_state(self):
        state = np.zeros(self.state_size, dtype=np.float32)
        for unit in self.agents.values():
            base = unit.id_in_faction * self.ally_state_feat_size
            ally_feats = self.__get_unit_state_features(unit, True)
            state[base:base + self.ally_state_feat_size] = ally_feats
        base_offset = self.n_agents * self.ally_state_feat_size
        for enemy in self.enemies.values():
            base = base_offset + enemy.id_in_faction \
                * self.enemy_state_feat_size
            enemy_feats = self.__get_unit_state_features(enemy, False)
            state[base:base + self.enemy_state_feat_size] = enemy_feats
        base_offset += self.n_enemies * self.enemy_state_feat_size
        state[base_offset:base_offset + self.n_agents * self.n_actions] \
            = self.last_actions

        return np.array(state)

    def __get_targetter_neighbour_finder(self, unit: Unit):
        is_ally = unit.faction == Faction.ALLY
        is_healer = unit.combat_type == CombatType.HEALING
        if is_ally ^ is_healer:
            return self.neighbour_finder_enemy
        else:
            return self.neighbour_finder_ally

    def __world_step(self):
        if self.renderer is not None:
            self.render()
        for unit in self.all_units.values():
            unit.clean_up_target()
        # NOTE There is an assumption here that the set of attack-moving units
        # will never include any allied units. This is true right now,
        # but might change in the future.
        if attackmoving_units := [enemy for enemy
                                  in self.enemies.values()
                                  if enemy.combat_type == CombatType.DAMAGE
                                  and enemy.target is None]:
            attackmoving_radii = [unit.minimum_scan_range
                                  + self.max_unit_radius
                                  for unit in attackmoving_units]
            attackmoving_targets = \
                self.neighbour_finder_ally.query_radius(attackmoving_units,
                                                        attackmoving_radii,
                                                        return_distance=True,
                                                        targetting_mode=True)
            for unit, targets in zip(attackmoving_units, attackmoving_targets):
                unit.potential_targets = targets
            for unit in self.agents.values():
                if unit.target is not None \
                        and unit.plane in unit.target.valid_targets:
                    unit.target.potential_targets.append((unit, 2e9))
        if healmoving_units := [enemy for enemy
                                in self.enemies.values()
                                if enemy.combat_type == CombatType.HEALING
                                and enemy.target is None]:
            healmoving_radii = [unit.minimum_scan_range
                                + self.max_unit_radius
                                for unit in healmoving_units]
            attackhealing_targets = \
                self.neighbour_finder_enemy.query_radius(healmoving_units,
                                                         healmoving_radii,
                                                         return_distance=True,
                                                         targetting_mode=True)
            for unit, targets in zip(healmoving_units, attackhealing_targets):
                unit.potential_targets = targets
        if any(unit.combat_type == CombatType.HEALING
               for unit in self.agents.values()) \
                and (nonpriority_attackmoving := [enemy for enemy
                                                  in self.enemies.values()
                                                  if enemy.combat_type
                                                  == CombatType.DAMAGE
                                                  and enemy.target is not None
                                                  and enemy.target.combat_type
                                                  != CombatType.HEALING]):
            attackmoving_radii = [unit.minimum_scan_range
                                  + self.max_unit_radius
                                  for unit in nonpriority_attackmoving]
            attackmoving_targets = \
                self.neighbour_finder_ally.query_radius(
                    nonpriority_attackmoving,
                    attackmoving_radii,
                    return_distance=True,
                    targetting_mode=True)
            for unit, targets in zip(nonpriority_attackmoving,
                                     attackmoving_targets):
                unit.priority_targets = targets
        for unit in self.all_units.values():
            unit.prepare_velocity()
        self.velocity_updater.compute_new_velocities(self.all_units)

        shuffled_units = list(self.all_units.values())
        random.shuffle(shuffled_units)
        reward = sum(unit.game_step(
            neighbour_finder=self.__get_targetter_neighbour_finder(unit),
            max_radius=self.max_unit_radius
            )
                     for unit in shuffled_units)
        self.__update_deaths()
        self.neighbour_finder_all.update()
        self.neighbour_finder_ally.update()
        self.neighbour_finder_enemy.update()
        return reward

    def __get_unit_state_features(self, unit: Unit, ally: bool):
        lgt = self.ally_state_feat_size if ally else self.enemy_state_feat_size
        feats = np.zeros(lgt, dtype=np.float32)
        if unit.hp == 0:
            return feats
        has_shields = self.map_info.ally_has_shields if ally else \
            self.map_info.enemy_has_shields
        feats[0] = unit.hp / unit.max_hp
        if ally:
            if unit.combat_type != CombatType.HEALING:
                feats[1] = unit.cooldown / unit.max_cooldown
            else:
                feats[1] = unit.energy / unit.max_energy
        dx, dy = unit.pos - self.cx_cy
        feats[1 + ally] = dx / self.map_info.width
        feats[2 + ally] = dy / self.map_info.height
        base = 3 + ally
        if has_shields:
            feats[base] = unit.shield / unit.max_shield
            base += 1
        if self.map_info.num_unit_types:
            feats[base:
                  base + self.map_info.num_unit_types] \
                      = self.__get_unit_type_id(unit.type)
        assert base + self.map_info.num_unit_types == lgt
        return feats

    def __get_agent_avail_actions(self, unit: Unit, targets=None):
        assert unit.hp > 0
        actions = np.zeros(self.n_actions)
        actions[1] = 1
        for direction in Direction:
            actions[2 + direction.value] = self.__can_move(unit, direction)
        if targets is None:
            targets = self.enemies.values() \
                if unit.combat_type == CombatType.DAMAGE \
                else [ally for ally in self.agents.values() if ally != unit]
        distance = None
        for target in targets:
            print(type(target))
            if type(target) is tuple:
                target, distance = target
            if target is unit:
                continue
            actions[6 + target.id_in_faction] \
                = self.__can_target(unit, target, distance=distance)
        return actions

    def __get_obs(self):
        dead_obs = np.zeros(self.obs_size, dtype=np.float32)
        obs = [None for _ in range(self.n_agents)]
        agents = [None for _ in range(len(self.agents))]
        idx = 0
        for i in range(self.n_agents):
            if i not in self.agents:
                obs[i] = dead_obs
                continue
            agents[idx] = self.agents[i]
            idx += 1
        visible_allies_lists = \
            self.neighbour_finder_ally.query_radius(agents,
                                                    AGENT_SIGHT_RANGE,
                                                    True)
        visible_enemies_lists = \
            self.neighbour_finder_enemy.query_radius(agents,
                                                     AGENT_SIGHT_RANGE,
                                                     True)
        assert len(visible_allies_lists) == len(visible_enemies_lists) == \
            len(agents)
        for (agent,
             visible_allies,
             visible_enemies) in zip(agents,
                                     visible_allies_lists,
                                     visible_enemies_lists):
            obs[agent.id_in_faction] = self.__get_agent_obs(agent,
                                                            visible_allies,
                                                            visible_enemies)
        return tuple(obs)

    def __get_info(self):
        return {'avail_actions': self.get_avail_actions(),
                'state': self.get_state()}

    def __can_target(self, unit: Unit, target: Unit, distance=None):
        if target.hp == 0 or unit.hp == 0:
            return 0
        if distance is not None:
            return distance <= AGENT_TARGET_RANGE
        dpos = target.pos - unit.pos
        distance_sq = np.inner(dpos, dpos)
        return distance_sq <= AGENT_TARGET_RANGE_SQ

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

    def __get_agent_obs(self, unit: Unit,
                        visible_allies: List[Unit],
                        visible_enemies: List[Unit]):
        targets = visible_enemies \
            if unit.combat_type == CombatType.DAMAGE \
            else visible_allies
        avail_actions = self.__get_agent_avail_actions(unit, targets=targets)
        obs = np.zeros(self.obs_size, dtype=np.float32)
        # Movement features
        for direction in Direction:
            obs[direction.value] = avail_actions[2 + direction.value]
        # Enemy features
        base_offset = 4
        for enemy, distance in visible_enemies:
            assert distance < AGENT_SIGHT_RANGE
            if enemy.hp == 0:
                continue
            dpos = enemy.pos - unit.pos
            dx, dy = dpos
            base = base_offset + enemy.id_in_faction * self.enemy_feat_size
            obs[base] = avail_actions[6 + enemy.id_in_faction]
            obs[base + 1] = distance / AGENT_SIGHT_RANGE
            obs[base + 2] = dx / AGENT_SIGHT_RANGE
            obs[base + 3] = dy / AGENT_SIGHT_RANGE
            obs[base + 4] = enemy.hp / enemy.max_hp
            base += 5
            if self.map_info.enemy_has_shields:
                obs[base] = enemy.shield / enemy.max_shield
                base += 1
            if self.map_info.num_unit_types:
                obs[base:
                    base + self.map_info.num_unit_types] \
                        = self.__get_unit_type_id(enemy.type)
        base_offset += self.n_enemies * self.enemy_feat_size
        # Ally features
        for ally, distance in visible_allies:
            assert distance < AGENT_SIGHT_RANGE
            if ally is unit or ally.hp == 0:
                continue
            dpos = ally.pos - unit.pos
            dx, dy = dpos
            base = base_offset + (ally.id_in_faction
                                  - (ally.id_in_faction
                                     > unit.id_in_faction)) \
                * self.ally_feat_size
            obs[base] = 1
            obs[base + 1] = distance / AGENT_SIGHT_RANGE
            obs[base + 2] = dx / AGENT_SIGHT_RANGE
            obs[base + 3] = dy / AGENT_SIGHT_RANGE
            obs[base + 4] = ally.hp / ally.max_hp
            base += 5
            if self.map_info.ally_has_shields:
                obs[base] = ally.shield / ally.max_shield
                base += 1
            if self.map_info.num_unit_types:
                obs[base:
                    base + self.map_info.num_unit_types] \
                        = self.__get_unit_type_id(ally.type)
            base += self.ally_feat_size
        # Own features
        base = 4 + self.n_enemies * self.enemy_feat_size \
            + (self.n_agents - 1) * self.ally_feat_size
        obs[base] = unit.hp / unit.max_hp
        if self.map_info.ally_has_shields:
            obs[base + 1] = unit.shield / unit.max_shield
            base += 1
        if self.map_info.num_unit_types:
            obs[base + 1:
                base + 1 + self.map_info.num_unit_types] \
                    = self.__get_unit_type_id(unit.type)

        assert base + 1 + self.map_info.num_unit_types == self.obs_size
        return obs

    def __can_move(self, unit: Unit, direction: Direction):
        check_value = MOVE_AMOUNT / 2
        dpos = direction.dx_dy
        npos = unit.pos + dpos * check_value
        return 0 <= npos[1] < self.map_info.width \
            and 0 <= npos[0] < self.map_info.height \
            and self.map_info.terrain[int(npos[1])][int(npos[0])] \
            == TerrainType.NORMAL

    def __place_group(self, group: Group):
        faction = group.faction
        faction_dict = self.agents if faction == Faction.ALLY \
            else self.enemies
        all_types_in_group: List[UnitType] = []
        for unit_type, count in group.units:
            all_types_in_group.extend([unit_type] * count)
        group_size = len(all_types_in_group)
        square_side = np.ceil(np.sqrt(group_size)).astype(int)
        unit_grid = [[None for _ in range(square_side)]
                     for _ in range(square_side)]
        a = b = 0
        # Plan out the layout of the units in the group
        for unit_type in all_types_in_group:
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
        # This is so enemy units spawn opposite allied units
        # i.e. the layout is center-symmetric if the groups are equal
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
                id_overall = len(self.all_units)
                id_in_faction = len(faction_dict)
                unit = Unit(u, faction, x, y, id_overall, id_in_faction)
                # Uncomment to test killing units
                # unit.hp = np.random.choice([0, np.random.randint(unit.hp)])
                self.all_units[id_overall] = unit
                faction_dict[id_in_faction] = unit
                x += m * GROUP_BUFFER
            y += m * GROUP_BUFFER

    def __enemy_attack(self):
        pos = np.array(self.map_info.attack_point)
        attack_move_command = AttackMoveCommand(pos,
                                                targets=self.agents.values())
        for enemy in self.enemies.values():
            if enemy.hp == 0:
                continue

            enemy.command = attack_move_command

    def __get_command(self, unit: Unit, action: int):
        if action == 0:
            return NoopCommand()
        if action == 1:
            return StopCommand()
        if 2 <= action <= 5:
            dpos = Direction(action - 2).dx_dy * MOVE_AMOUNT
            return MoveCommand(unit.pos + dpos)
        if unit.combat_type == CombatType.HEALING:
            return AttackUnitCommand(self.agents[action - 6])
        return AttackUnitCommand(self.enemies[action - 6])

    def __update_deaths(self):
        to_remove_ally = [k for k, unit in self.agents.items()
                          if unit.hp == 0]
        for k in to_remove_ally:
            del self.agents[k]
        to_remove_enemy = [k for k, unit in self.enemies.items()
                           if unit.hp == 0]
        for k in to_remove_enemy:
            del self.enemies[k]
        to_remove_all = [k for k, unit in self.all_units.items()
                         if unit.hp == 0]
        for k in to_remove_all:
            del self.all_units[k]
