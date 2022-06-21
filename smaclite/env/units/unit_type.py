from dataclasses import dataclass

from enum import Enum

MELEE_ATTACK_RANGE = -1


class CombatType(Enum):
    DAMAGE = 1
    HEALING = 2


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
    combat_type: CombatType = CombatType.DAMAGE


class UnitType(Enum):
    """Various types of units adapted from Starcraft 2

    Statistics taken from https://liquipedia.net/starcraft2/Unit_Statistics_(Legacy_of_the_Void)  # noqa
    """
    @property
    def stats(self) -> UnitStats:
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
    MEDIVAC = UnitStats(hp=150,
                        armor=1,
                        damage=0,
                        cooldown=0,
                        speed=3.5,
                        size=1.5,
                        attack_range=4,
                        sight_range=11,
                        combat_type=CombatType.HEALING)

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
    STALKER = UnitStats(hp=80,
                        armor=1,
                        shield=80,
                        damage=13,
                        cooldown=1.34,
                        speed=4.13,
                        attack_range=6,
                        sight_range=10,
                        size=1.25)
