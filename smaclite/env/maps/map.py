from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

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
    terrain: List[List[int]]
    width: int = 32
    height: int = 32


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
    MAP_10M_VS_11M = MapInfo(
        name="10m_vs_11m",
        num_allied_units=10,
        num_enemy_units=11,
        groups=[
            Group(9, 16, Faction.ALLY, [(UnitType.MARINE, 10)]),
            Group(23, 16, Faction.ENEMY, [(UnitType.MARINE, 11)])
        ],
        terrain=TerrainPreset.SIMPLE.value
    )
