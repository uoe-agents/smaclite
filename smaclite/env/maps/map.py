import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

from smaclite.env.units.unit_type import StandardUnit, UnitType
from smaclite.env.util.terrain import TERRAIN_PRESETS, TerrainPreset, TerrainType


class Faction(Enum):
    ALLY = 'ALLY'
    ENEMY = 'ENEMY'


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
    terrain: List[List[TerrainType]]
    ally_has_shields: bool
    enemy_has_shields: bool
    width: int = 32
    height: int = 32
    num_unit_types: int = 0  # note: 0 for single-type maps
    unit_type_ids: Dict[UnitType, int] = None

    @classmethod
    def from_file(cls, filename):
        with open(filename) as f:
            map_info_dict = json.load(f)
        custom_unit_path = map_info_dict.get('custom_unit_path', '.')
        if 'custom_unit_path' in map_info_dict:
            del map_info_dict['custom_unit_path']
        groups = []
        for group in map_info_dict['groups']:
            group['faction'] = Faction(group['faction'])
            group['units'] = [(UnitType.from_str(t, custom_unit_path), c)
                              for (t, c) in group['units']]
            groups.append(Group(**group))
        map_info_dict['groups'] = groups
        map_info_dict['attack_point'] = tuple(map_info_dict['attack_point'])
        if "terrain_preset" in map_info_dict:
            map_info_dict["terrain"] = \
                TERRAIN_PRESETS[map_info_dict["terrain_preset"]].value
            del map_info_dict["terrain_preset"]
        else:
            for i, row in enumerate(map_info_dict['terrain']):
                new_row = [None] * len(row)
                for j, terrain_type in enumerate(row):
                    new_row[j] = TerrainType(terrain_type)
                map_info_dict['terrain'][i] = new_row
        if 'unit_type_ids' in map_info_dict:
            map_info_dict['unit_type_ids'] = \
                {UnitType.from_str(k, custom_unit_path): v
                 for k, v in group['units'].items()}

        if map_info_dict['num_unit_types'] == 1:
            map_info_dict['num_unit_types'] = 0
        return cls(**map_info_dict)


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
            Group(9, 16, Faction.ALLY, [(StandardUnit.MARINE, 10)]),
            Group(23, 16, Faction.ENEMY, [(StandardUnit.MARINE, 11)])
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
            Group(9, 16, Faction.ALLY, [(StandardUnit.MARINE, 27)]),
            Group(23, 16, Faction.ENEMY, [(StandardUnit.MARINE, 30)])
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
            Group(9, 16, Faction.ALLY, [(StandardUnit.STALKER, 3),
                                        (StandardUnit.ZEALOT, 5)]),
            Group(23, 16, Faction.ENEMY, [(StandardUnit.STALKER, 3),
                                          (StandardUnit.ZEALOT, 6)])
        ],
        attack_point=(9, 16),
        terrain=TerrainPreset.SIMPLE.value,
        num_unit_types=2,
        ally_has_shields=True,
        enemy_has_shields=True,
        unit_type_ids={
            StandardUnit.STALKER: 0,
            StandardUnit.ZEALOT: 1,
        }
    )
    MAP_2S3Z = MapInfo(
        name="2s3z",
        num_allied_units=5,
        num_enemy_units=5,
        groups=[
            Group(9, 16, Faction.ALLY, [(StandardUnit.STALKER, 2),
                                        (StandardUnit.ZEALOT, 3)]),
            Group(23, 16, Faction.ENEMY, [(StandardUnit.STALKER, 2),
                                          (StandardUnit.ZEALOT, 3)])
        ],
        attack_point=(9, 16),
        terrain=TerrainPreset.SIMPLE.value,
        num_unit_types=2,
        ally_has_shields=True,
        enemy_has_shields=True,
        unit_type_ids={
            StandardUnit.STALKER: 0,
            StandardUnit.ZEALOT: 1,
        }
    )
