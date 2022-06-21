from smaclite.env.units.unit_type import UnitType
from smaclite.env.maps.map import Faction


class Unit(object):
    def __init__(self, unit_type: UnitType, faction: Faction,
                 x: float, y: float) -> None:
        self.type = unit_type
        self.hp = unit_type.stats.hp
        self.shield = unit_type.stats.shield
        self.faction = faction
        self.x = x
        self.y = y
        self.cooldown = 0

    def take_damage(self, attacker: 'Unit') -> float:
        raise NotImplementedError

    def decrease_cooldown(self, amount: float) -> None:
        raise NotImplementedError
