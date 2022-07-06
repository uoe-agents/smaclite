from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

from smaclite.env.units.unit_type import UnitType


class Faction(Enum):
    ALLY = 1
    ENEMY = 2


class TerrainType(Enum):
    NORMAL = 1
    CLIFF = 2
    NONE = 3


@dataclass
class Group(object):
    x: int
    y: int
    faction: Faction
    units: List[Tuple[UnitType, int]]


@dataclass
class MapInfo(object):
    name: str
    num_allied_units: int
    num_enemy_units: int
    groups: List[Group]
    attack_point: Tuple[int, int]
    terrain: List[List[int]]
    ally_has_shields: bool
    enemy_has_shields: bool
    width: int = 32
    height: int = 32
    num_unit_types: int = 0  # note: 0 for single-type maps
    unit_type_ids: Dict[UnitType, int] = None


class TerrainPreset(Enum):
    SIMPLE = [*[[TerrainType.NONE for _ in range(32)] for _ in range(8)],
              *[[TerrainType.NORMAL for _ in range(32)] for _ in range(16)],
              *[[TerrainType.NONE for _ in range(32)] for _ in range(8)]
              ]


class MapPreset(Enum):
    """Maps adapted from SMAC.
    Details about the maps were found using the Starcraft II map editor.

    Args:
        Enum (_type_): _description_
    """

    @property
    def map_info(self) -> MapInfo:
        return self.value

    MAP_10M_VS_11M = MapInfo(
        name="10m_vs_11m",
        num_allied_units=10,
        num_enemy_units=11,
        groups=[
            Group(9, 16, Faction.ALLY, [(UnitType.MARINE, 10)]),
            Group(23, 16, Faction.ENEMY, [(UnitType.MARINE, 11)])
        ],
        attack_point=(9, 16),
        terrain=TerrainPreset.SIMPLE.value,
        num_unit_types=0,
        ally_has_shields=False,
        enemy_has_shields=False,
    )
    MAP_27M_VS_30M = MapInfo(
        name="27m_vs_30m",
        num_allied_units=27,
        num_enemy_units=30,
        groups=[
            Group(9, 16, Faction.ALLY, [(UnitType.MARINE, 27)]),
            Group(23, 16, Faction.ENEMY, [(UnitType.MARINE, 30)])
        ],
        attack_point=(9, 16),
        terrain=TerrainPreset.SIMPLE.value,
        num_unit_types=0,
        ally_has_shields=False,
        enemy_has_shields=False,
    )
    MAP_3S5Z_VS_3S6Z = MapInfo(
        name="3s5z_vs_3s6z",
        num_allied_units=8,
        num_enemy_units=9,
        groups=[
            Group(9, 16, Faction.ALLY, [(UnitType.STALKER, 3),
                                        (UnitType.ZEALOT, 5)]),
            Group(23, 16, Faction.ENEMY, [(UnitType.STALKER, 3),
                                          (UnitType.ZEALOT, 6)])
        ],
        attack_point=(9, 16),
        terrain=TerrainPreset.SIMPLE.value,
        num_unit_types=2,
        ally_has_shields=True,
        enemy_has_shields=True,
        unit_type_ids={
            UnitType.STALKER: 0,
            UnitType.ZEALOT: 1,
        }
    )
    MAP_2S3Z = MapInfo(
        name="2s3z",
        num_allied_units=5,
        num_enemy_units=5,
        groups=[
            Group(9, 16, Faction.ALLY, [(UnitType.STALKER, 2),
                                        (UnitType.ZEALOT, 3)]),
            Group(23, 16, Faction.ENEMY, [(UnitType.STALKER, 2),
                                            (UnitType.ZEALOT, 3)])
        ],
        attack_point=(9, 16),
        terrain=TerrainPreset.SIMPLE.value,
        num_unit_types=2,
        ally_has_shields=True,
        enemy_has_shields=True,
        unit_type_ids={
            UnitType.STALKER: 0,
            UnitType.ZEALOT: 1,
        }
    )

