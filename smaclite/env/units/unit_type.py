import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Set

import smaclite.env.units.targeters.targeter as t
from smaclite.env.units.combat_type import CombatType
from smaclite.env.util.plane import Plane

# NOTE 0.25 for banelings
MELEE_ATTACK_RANGE = 0.1


class Attribute(Enum):
    LIGHT = 'LIGHT'
    ARMORED = 'ARMORED'
    MASSIVE = 'MASSIVE'
    BIOLOGICAL = 'BIOLOGICAL'
    MECHANICAL = 'MECHANICAL'
    PSIONIC = 'PSIONIC'
    STRUCTURE = 'STRUCTURE'
    HEROIC = 'HEROIC'


TARGETER_CACHE: Dict[str, t.Targeter] = {}


@dataclass
class UnitStats(object):
    name: str
    hp: int
    armor: int
    damage: int
    cooldown: float
    speed: float
    attack_range: int
    size: float
    attributes: Set[Attribute]
    valid_targets: Set[Plane]
    shield: int = 0
    energy: int = 0
    starting_energy: int = 0
    attacks: int = 1
    combat_type: CombatType = CombatType.DAMAGE
    minimum_scan_range: int = 5
    bonuses: Dict[Attribute, float] = None
    plane: Plane = Plane.GROUND
    hp_regen: float = 0

    @classmethod
    def from_file(cls, filename, custom_unit_path):
        if not os.path.isabs(filename):
            filename = os.path.join(os.path.abspath(custom_unit_path),
                                    filename)
        if not filename.endswith(".json"):
            filename += ".json"
        with open(filename) as f:
            stats_dict = json.load(f)
        stats_dict['name'] = os.path.splitext(os.path.basename(filename))[0]
        if stats_dict['attack_range'] == "MELEE":
            stats_dict['attack_range'] = MELEE_ATTACK_RANGE
        stats_dict['attributes'] = set(map(Attribute,
                                           stats_dict['attributes']))
        stats_dict['valid_targets'] = set(map(Plane,
                                              stats_dict['valid_targets']))
        if 'bonuses' in stats_dict:
            stats_dict['bonuses'] = {Attribute(k): v for k, v
                                     in stats_dict['bonuses'].items()}
        if 'combat_type' in stats_dict:
            stats_dict['combat_type'] = CombatType(stats_dict['combat_type'])
        if 'plane' in stats_dict:
            stats_dict['plane'] = Plane(stats_dict['plane'])
        targeter_kwargs = stats_dict.pop('targeter_kwargs', {})
        TARGETER_CACHE[stats_dict['name']] = \
            t.TargeterType[stats_dict.pop(
                'targeter', 'STANDARD')].value(**targeter_kwargs)
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
    BANELING = UnitStats.from_file("baneling", STANDARD_UNIT_PATH)
    SPINE_CRAWLER = UnitStats.from_file("spine_crawler", STANDARD_UNIT_PATH)

    # Terran units
    MARINE = UnitStats.from_file("marine", STANDARD_UNIT_PATH)
    MEDIVAC = UnitStats.from_file("medivac", STANDARD_UNIT_PATH)
    MARAUDER = UnitStats.from_file("marauder", STANDARD_UNIT_PATH)

    # Protoss units
    ZEALOT = UnitStats.from_file("zealot", STANDARD_UNIT_PATH)
    STALKER = UnitStats.from_file("stalker", STANDARD_UNIT_PATH)
    COLOSSUS = UnitStats.from_file("colossus", STANDARD_UNIT_PATH)


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
