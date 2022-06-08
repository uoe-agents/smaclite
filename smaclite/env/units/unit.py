from dataclasses import dataclass

from enum import Enum

MELEE_ATTACK_RANGE = -1


@dataclass
class UnitStats(object):
    hp: int
    armor: int
    damage: int
    cooldown: float
    speed: float
    attack_range: int
    sight_range: int
    size: float
    shield: int = 0
    attacks: int = 1


class UnitType(Enum):
    """Various types of units adapted from Starcraft 2

    Statistics taken from https://liquipedia.net/starcraft2/Unit_Statistics_(Legacy_of_the_Void)  # noqa
    """
    @property
    def stats(self):
        return self.value

    # Zerg units
    ZERGLING = UnitStats(hp=35,
                         armor=0,
                         damage=5,
                         cooldown=0.497,
                         speed=4.13,
                         attack_range=MELEE_ATTACK_RANGE,
                         sight_range=8,
                         size=0.75)

    # Terran units
    MARINE = UnitStats(hp=45,
                       armor=0,
                       damage=6,
                       cooldown=0.61,
                       speed=3.15,
                       attack_range=5,
                       sight_range=9,
                       size=0.75)

    # Protoss units
    ZEALOT = UnitStats(hp=100,
                       armor=1,
                       shield=50,
                       damage=8,
                       cooldown=0.86,
                       attacks=2,
                       speed=3.15,
                       attack_range=MELEE_ATTACK_RANGE,
                       sight_range=9,
                       size=1)


class Unit:
    def __init__(self, type: UnitType) -> None:
        self.type = type
