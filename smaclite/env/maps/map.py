import json
from dataclasses import dataclass
from enum import Enum
import os
from typing import Dict, List, Tuple

from smaclite.env.units.unit_type import StandardUnit, UnitType
from smaclite.env.terrain.terrain import TERRAIN_PRESETS, TerrainPreset, TerrainType
from smaclite.env.util.faction import Faction


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
                              for t, c in group['units'].items()]
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
                 for k, v in map_info_dict['unit_type_ids'].items()}

        if map_info_dict['num_unit_types'] == 1:
            map_info_dict['num_unit_types'] = 0
        return cls(**map_info_dict)


def get_standard_map(map_name):
    return MapInfo.from_file(os.path.join(os.path.dirname(__file__),
                                          'smaclite_maps', f"{map_name}.json"))


MAP_PRESET_DIR = os.path.join(os.path.dirname(__file__), 'smaclite_maps')


class MapPreset(Enum):
    """Maps adapted from SMAC.
    Details about the maps were found using the Starcraft II map editor.

    Args:
        Enum (_type_): _description_
    """

    @property
    def map_info(self) -> MapInfo:
        return self.value

    MAP_10M_VS_11M = get_standard_map('10m_vs_11m')
    MAP_27M_VS_30M = get_standard_map('27m_vs_30m')
    MAP_3S5Z_VS_3S6Z = get_standard_map('3s5z_vs_3s6z')
    MAP_2S3Z = get_standard_map("2s3z")
    MAP_3S5Z = get_standard_map("3s5z")
    MAP_MMM = get_standard_map("mmm")
    MAP_MMM2 = get_standard_map("mmm2")
    MAP_2C_VS_64ZG = get_standard_map("2c_vs_64zg")
    MAP_BANE_VS_BANE = get_standard_map("bane_vs_bane")
    MAP_CORRIDOR = get_standard_map("corridor")
    MAP_2S_VS_1SC = get_standard_map("2s_vs_1sc")
    MAP_3S_VS_5Z = get_standard_map("3s_vs_5z")
