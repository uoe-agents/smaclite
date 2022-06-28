from typing import List
import numpy as np

from smaclite.env.units.unit import GAME_TICK_TIME, Unit


class Command:
    def execute(self, unit: Unit) -> float:
        """Execute the command using the unit

        Args:
            unit (Unit): the unit that should execute the command

        Returns:
            float: the reward obtained by the unit in the step
        """
        raise NotImplementedError

    def prepare_velocity(self, unit: Unit) -> np.ndarray:
        """Prepare the velocity for the unit in the current game step

        Args:
            unit (Unit): the unit to prepare velocity for

        Returns:
            np.ndarray: [x, y] vector with the prepared velocity
        """
        raise NotImplementedError


class AttackUnitCommand(Command):
    def __init__(self, target: Unit):
        self.target = target

    def execute(self, unit: Unit) -> float:
        if not unit.attacking:
            return MoveCommand(self.target.pos).execute(unit)
        unit.attacking = False
        unit.target = self.target
        if unit.cooldown > 0:
            return 0
        unit.cooldown = unit.max_cooldown
        return unit.deal_damage(self.target)

    def prepare_velocity(self, unit: Unit) -> None:
        if not unit.has_within_attack_range(self.target):
            return MoveCommand(self.target.pos).prepare_velocity(unit)
        unit.attacking = True
        return np.zeros(2)


class MoveCommand(Command):
    def __init__(self, pos: np.ndarray):
        self.pos = pos

    def execute(self, unit: Unit) -> float:
        return 0

    def prepare_velocity(self, unit: Unit) -> np.ndarray:
        dpos = self.pos - unit.pos
        distance = np.linalg.norm(dpos)
        return np.zeros(2) if distance == 0 \
            else dpos * unit.max_velocity / distance


class AttackMoveCommand(Command):
    def __init__(self, pos: np.ndarray, targets: List[Unit]):
        self.pos = pos
        self.move_command = MoveCommand(pos)
        self.targets = targets

    def execute(self, unit: Unit) -> float:
        if unit.target is None:
            return self.move_command.execute(unit)
        return AttackUnitCommand(unit.target).execute(unit)

    def prepare_velocity(self, unit: Unit) -> np.ndarray:
        if (target := unit.target) is not None and \
                (target.hp == 0 or not unit.has_within_attack_range(target)):
            unit.target = None
        if unit.target is None:
            closest_target = min((t for t in self.targets if t.hp > 0),
                                 key=lambda t:
                                     np.inner(dist := t.pos - unit.pos, dist),
                                 default=None)

            if closest_target is None \
                    or not unit.has_within_scan_range(closest_target):
                return self.move_command.prepare_velocity(unit)
            unit.target = closest_target
        return AttackUnitCommand(unit.target).prepare_velocity(unit)


class StopCommand(Command):
    def execute(self, unit: Unit) -> float:
        unit.target = None
        return 0

    def prepare_velocity(self, unit: Unit) -> np.ndarray:
        return np.zeros(2)


class NoopCommand(Command):
    def execute(self, unit: Unit) -> None:
        assert unit.hp == 0, f"Unit's hp is not 0: {unit.hp}"
        return 0

    def prepare_velocity(self, unit: Unit) -> np.ndarray:
        return np.zeros(2)
