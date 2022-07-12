from typing import List
import numpy as np

from smaclite.env.units.unit import Unit


class Command:
    def clean_up_target(self, unit: Unit) -> None:
        raise NotImplementedError

    def prepare_velocity(self, unit: Unit) -> np.ndarray:
        """Prepare the velocity for the unit in the current game step

        Args:
            unit (Unit): the unit to prepare velocity for

        Returns:
            np.ndarray: [x, y] vector with the prepared velocity
        """
        raise NotImplementedError

    def execute(self, unit: Unit) -> float:
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
        unit.target = None if self.target.hp == 0 else self.target

    def prepare_velocity(self, unit: Unit) -> None:
        if not unit.has_within_attack_range(self.target):
            return MoveCommand(self.target.pos).prepare_velocity(unit)
        unit.attacking = True
        return np.zeros(2)

    def execute(self, unit: Unit) -> float:
        if not unit.attacking:
            return MoveCommand(self.target.pos).execute(unit)
        unit.attacking = False
        if unit.cooldown > 0:
            return 0
        unit.cooldown = unit.max_cooldown
        return unit.deal_damage(self.target)


class MoveCommand(Command):
    def __init__(self, pos: np.ndarray):
        self.pos = pos

    def clean_up_target(self, unit: Unit) -> None:
        unit.target = None

    def prepare_velocity(self, unit: Unit) -> np.ndarray:
        dpos = self.pos - unit.pos
        distance = np.linalg.norm(dpos)
        return np.zeros(2) if distance == 0 \
            else dpos * unit.max_velocity / distance

    def execute(self, unit: Unit) -> float:
        return 0


class AttackMoveCommand(Command):
    def __init__(self, pos: np.ndarray, targets: List[Unit]):
        self.pos = pos
        self.move_command = MoveCommand(pos)
        self.targets = targets

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
            closest_target = min(((u, d) for u, d in unit.potential_targets
                                  if u.hp > 0
                                  and (d < unit.minimum_scan_range + u.radius
                                       or u.target == unit)),
                                 key=lambda p: p[1], default=None)

            if closest_target is None:
                return self.move_command.prepare_velocity(unit)
            unit.target = closest_target[0]
        return AttackUnitCommand(unit.target).prepare_velocity(unit)

    def execute(self, unit: Unit) -> float:
        if unit.target is None:
            return self.move_command.execute(unit)
        return AttackUnitCommand(unit.target).execute(unit)


class StopCommand(Command):
    def clean_up_target(self, unit: Unit) -> None:
        unit.target = None

    def prepare_velocity(self, unit: Unit) -> np.ndarray:
        return np.zeros(2)

    def execute(self, unit: Unit) -> float:
        unit.target = None
        return 0


class NoopCommand(Command):
    def clean_up_target(self, unit: Unit) -> None:
        unit.target = None

    def prepare_velocity(self, unit: Unit) -> np.ndarray:
        return np.zeros(2)

    def execute(self, unit: Unit) -> None:
        assert unit.hp == 0, f"Unit's hp is not 0: {unit.hp}"
        return 0
