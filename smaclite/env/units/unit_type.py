from dataclasses import dataclass

from enum import Enum
from typing import Dict, Set

# NOTE 0.25 for banelings
MELEE_ATTACK_RANGE = 0.1


class CombatType(Enum):
    DAMAGE = 1
    HEALING = 2


class Attribute(Enum):
    LIGHT = 0
    ARMORED = 1
    MASSIVE = 2
    BIOLOGICAL = 3
    MECHANICAL = 4
    PSIONIC = 5
    STRUCTURE = 6
    HEROIC = 7


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
    attributes: Set[Attribute]
    shield: int = 0
    attacks: int = 1
    combat_type: CombatType = CombatType.DAMAGE
    minimum_scan_range: int = 5
    bonuses: Dict[Attribute, float] = None


class UnitType(Enum):
    """Various types of units adapted from Starcraft 2

    Statistics taken from https://liquipedia.net/starcraft2/Unit_Statistics_(Legacy_of_the_Void)  # noqa
    """
    @property
    def stats(self) -> UnitStats:
        return self.value

    @property
    def radius(self):
        return self.stats.size / 2

    @property
    def size(self):
        return self.stats.size

    @property
    def combat_type(self):
        return self.stats.combat_type

    # Zerg units
    ZERGLING = UnitStats(hp=35,
                         armor=0,
                         damage=5,
                         cooldown=0.497,
                         speed=4.13,
                         attack_range=MELEE_ATTACK_RANGE,
                         sight_range=8,
                         size=0.75,
                         attributes={Attribute.LIGHT, Attribute.BIOLOGICAL})

    # Terran units
    MARINE = UnitStats(hp=45,
                       armor=0,
                       damage=6,
                       cooldown=0.61,
                       speed=3.15,
                       attack_range=5,
                       sight_range=9,
                       size=0.75,
                       attributes={Attribute.LIGHT, Attribute.BIOLOGICAL})
    MEDIVAC = UnitStats(hp=150,
                        armor=1,
                        damage=0,
                        cooldown=0,
                        speed=3.5,
                        size=1.5,
                        attack_range=4,
                        sight_range=11,
                        combat_type=CombatType.HEALING,
                        attributes={Attribute.ARMORED, Attribute.MECHANICAL}),

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
                       size=1,
                       attributes={Attribute.LIGHT, Attribute.BIOLOGICAL})
    STALKER = UnitStats(hp=80,
                        armor=1,
                        shield=80,
                        damage=13,
                        cooldown=1.34,
                        speed=4.13,
                        attack_range=6,
                        sight_range=10,
                        size=1.25,
                        attributes={Attribute.ARMORED, Attribute.MECHANICAL},
                        bonuses={Attribute.ARMORED: 5})
