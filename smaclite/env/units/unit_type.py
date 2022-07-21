from dataclasses import dataclass

from enum import Enum
import json
import os
from typing import Dict, Set

# NOTE 0.25 for banelings
MELEE_ATTACK_RANGE = 0.1


class CombatType(Enum):
    DAMAGE = 'DAMAGE'
    HEALING = 'HEALING'


class Attribute(Enum):
    LIGHT = 'LIGHT'
    ARMORED = 'ARMORED'
    MASSIVE = 'MASSIVE'
    BIOLOGICAL = 'BIOLOGICAL'
    MECHANICAL = 'MECHANICAL'
    PSIONIC = 'PSIONIC'
    STRUCTURE = 'STRUCTURE'
    HEROIC = 'HEROIC'


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

    @classmethod
    def from_file(cls, filename, custom_unit_path):
        if not os.path.isabs(filename):
            filename = os.path.join(os.path.abspath(custom_unit_path),
                                    filename)
        if not filename.endswith(".json"):
            filename += ".json"
        with open(filename) as f:
            stats_dict = json.load(f)
        if stats_dict['attack_range'] == "MELEE":
            stats_dict['attack_range'] = MELEE_ATTACK_RANGE
        if 'attributes' in stats_dict:
            stats_dict['attributes'] = set(map(Attribute,
                                               stats_dict['attributes']))
        if 'bonuses' in stats_dict:
            stats_dict['bonuses'] = {Attribute(k): v for k, v
                                     in stats_dict['bonuses'].items()}
        if 'combat_type' in stats_dict:
            stats_dict['combat_type'] = CombatType(stats_dict['combat_type'])

        return cls(**stats_dict)


class UnitType(object):
    @property
    def stats(self) -> UnitStats:
        raise NotImplementedError

    @property
    def radius(self):
        return self.stats.size / 2

    @property
    def size(self):
        return self.stats.size

    @property
    def combat_type(self):
        return self.stats.combat_type

    @classmethod
    def from_str(cls, s, custom_unit_path):
        return STANDARD_UNIT_TYPES[s] \
            if s in STANDARD_UNIT_TYPES \
            else CustomUnitType.from_file(s, custom_unit_path)


STANDARD_UNIT_PATH = os.path.join(os.path.dirname(__file__), "smaclite_units")


class StandardUnit(UnitType, Enum):
    """Various types of units adapted from Starcraft 2

    Statistics taken from https://liquipedia.net/starcraft2/Unit_Statistics_(Legacy_of_the_Void)  # noqa
    """
    @property
    def stats(self) -> UnitStats:
        return self.value

    # Zerg units
    ZERGLING = UnitStats.from_file("zergling", STANDARD_UNIT_PATH)

    # Terran units
    MARINE = UnitStats.from_file("marine", STANDARD_UNIT_PATH)
    MEDIVAC = UnitStats.from_file("medivac", STANDARD_UNIT_PATH)

    # Protoss units
    ZEALOT = UnitStats.from_file("zealot", STANDARD_UNIT_PATH)
    STALKER = UnitStats.from_file("stalker", STANDARD_UNIT_PATH)


STANDARD_UNIT_TYPES = {unit_type.name: unit_type
                       for unit_type in StandardUnit}


class CustomUnitType(UnitType):
    CUSTOM_UNIT_TYPES: Dict[str, 'CustomUnitType'] = {}

    def __init__(self, stats: UnitStats, filename: str):
        self._stats = stats
        self.filename = filename

    @classmethod
    def from_file(cls, filename, custom_unit_path):
        if filename in cls.CUSTOM_UNIT_TYPES:
            return cls.CUSTOM_UNIT_TYPES[filename]
        stats = UnitStats.from_file(filename, custom_unit_path)

        unit_type = cls(stats, filename)
        cls.CUSTOM_UNIT_TYPES[filename] = unit_type
        return unit_type

    @property
    def stats(self):
        return self._stats

    def __eq__(self, other: object) -> bool:
        return self.filename == other.filename \
            if isinstance(other, CustomUnitType) \
            else False

    def __hash__(self) -> int:
        return hash(self.filename)
