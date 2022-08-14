from typing import List

import numpy as np
from smaclite.env.units.combat_type import CombatType
from smaclite.env.units.unit import Unit


class Command:
    def clean_up_target(self, unit: Unit) -> None:
        """Remove the target from the unit if it is not valid anymore,
        e.g. it's too far away or it's dead or this command type
        doesn't support having a target.

        Args:
            unit (Unit): the unit to clean up for.
        """
        raise NotImplementedError

    def prepare_velocity(self, unit: Unit) -> np.ndarray:
        """Prepare the velocity for the unit in the current game step.
        This velocity will then be submitted to the collision avoidance
        algorithm and adjusted accordingly.

        Args:
            unit (Unit): the unit to prepare velocity for

        Returns:
            np.ndarray: [x, y] vector with the prepared velocity
        """
        raise NotImplementedError

    def execute(self, unit: Unit, **kwargs) -> float:
        """Execute the command using the unit

        Args:
            unit (Unit): the unit that should execute the command

        Returns:
            float: the reward obtained by the unit in the step
        """
        raise NotImplementedError


class AttackUnitCommand(Command):
    def __init__(self, target: Unit):
        self.target = target

    def clean_up_target(self, unit: Unit) -> None:
        assert self.target.hp >= 0
        unit.target = self.target if self.target.hp > 0 else None

    def prepare_velocity(self, unit: Unit) -> None:
        if not unit.has_within_attack_range(self.target) \
                or self.target.plane not in unit.valid_targets:
            # Target too far away or unit incapable of attacking target
            return MoveCommand(self.target.pos).prepare_velocity(unit)
        # Target is in range
        unit.attacking = True
        return np.zeros(2)

    def execute(self, unit: Unit, **kwargs) -> float:
        if not unit.attacking:
            return MoveCommand(self.target.pos).execute(unit, **kwargs)
        unit.attacking = False
        if unit.cooldown > 0:
            return 0
        unit.cooldown = unit.max_cooldown
        return unit.targeter.target(unit, self.target, **kwargs)


class MoveCommand(Command):
    def __init__(self, pos: np.ndarray):
        self.pos = pos

    def clean_up_target(self, unit: Unit) -> None:
        unit.target = None

    def prepare_velocity(self, unit: Unit) -> np.ndarray:
        if unit.max_velocity == 0:
            return np.zeros(2)
        if unit.combat_type == CombatType.HEALING:
            # healers shouldn't overtake allies
            max_velocity = min((u.max_velocity for u, _
                                in unit.potential_targets),
                               default=unit.max_velocity)
        else:
            max_velocity = unit.max_velocity
        dpos = self.pos - unit.pos
        distance = np.linalg.norm(dpos)
        return np.zeros(2) if distance == 0 \
            else dpos * max_velocity / distance

    def execute(self, unit: Unit, **kwargs) -> float:
        return 0


class AttackMoveCommand(Command):
    def __init__(self, pos: np.ndarray, targets: List[Unit]):
        self.pos = pos
        self.move_command = MoveCommand(pos)

    def clean_up_target(self, unit: Unit) -> None:
        if unit.target is None:
            # No target to lose.
            return
        if unit.target.hp == 0:
            # Always lose a dead target.
            unit.target = None
            return
        if unit.target.target == unit:
            # Don't lose an alive target who's attacking you.
            return
        if not unit.has_within_attack_range(unit.target):
            # Otherwise lose the target if it's not in range.
            unit.target = None

    def prepare_velocity(self, unit: Unit) -> np.ndarray:
        if unit.target is None:
            target_picking_func = self.__pick_target_damage \
            if unit.combat_type == CombatType.DAMAGE \
            else self.__pick_target_healing
            closest_target = target_picking_func(unit)

            if closest_target is None:
                return self.move_command.prepare_velocity(unit)
            unit.target = closest_target
        elif unit.combat_type == CombatType.DAMAGE \
                and unit.priority_targets:
            closest_priority_target = self.__pick_target_damage(unit,
                                                                True)
            if closest_priority_target is not None:
                unit.target = closest_priority_target

        return AttackUnitCommand(unit.target).prepare_velocity(unit)

    def execute(self, unit: Unit, **kwargs) -> float:
        if unit.target is None:
            return self.move_command.execute(unit, **kwargs)
        return AttackUnitCommand(unit.target).execute(unit, **kwargs)

    def __pick_target_damage(self, unit: Unit,
                             priority: bool = False) -> Unit:
        tgt = unit.priority_targets \
            if priority \
            else unit.potential_targets
        candidate = min(((u, d) for u, d in tgt
                         if u.hp > 0
                         and (d < unit.minimum_scan_range + u.radius
                              or u.target == unit)),
                        key=lambda p: (p[0].combat_type.priority,
                                       p[1]), default=None)
        return candidate[0] if candidate is not None else None

    def __pick_target_healing(self, unit: Unit) -> Unit:
        return min((u for u, d in unit.potential_targets
                    if u.hp > 0
                    and d < unit.minimum_scan_range + u.radius
                    and (u.hp < u.max_hp or u.target is not None)),
                   key=lambda u: u.hp, default=None)


class StopCommand(Command):
    def clean_up_target(self, unit: Unit) -> None:
        unit.target = None

    def prepare_velocity(self, unit: Unit) -> np.ndarray:
        return np.zeros(2)

    def execute(self, unit: Unit, **kwargs) -> float:
        unit.target = None
        return 0


class NoopCommand(Command):
    def clean_up_target(self, unit: Unit) -> None:
        unit.target = None

    def prepare_velocity(self, unit: Unit) -> np.ndarray:
        return np.zeros(2)

    def execute(self, unit: Unit, **kwargs) -> None:
        assert unit.hp == 0, f"Unit's hp is not 0: {unit.hp}"
        return 0
