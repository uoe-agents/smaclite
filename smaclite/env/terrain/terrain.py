import os
from enum import Enum


class TerrainType(Enum):
    NORMAL = '_'
    CLIFF = 'C'
    NONE = 'X'


def from_file(name):
    filename = os.path.join(os.path.dirname(__file__),
                            "smaclite_terrain",
                            f"{name}.slt")
    with open(filename) as f:
        strs = list(map(str.strip, f.readlines()))
    return [
        list(map(TerrainType, row))
        for row in reversed(strs)
    ]


class TerrainPreset(Enum):
    SIMPLE = from_file('simple')
    CHECKERBOARD = from_file('checkerboard')
    NARROW = from_file('narrow')
    RAVINE = from_file('ravine')
    OCTAGON = from_file('octagon')
    CORRIDOR = from_file('corridor')
    PENTAGON = from_file('pentagon')
    ALL_GREEN = from_file('all_green')


TERRAIN_PRESETS = {t.name: t for t in TerrainPreset}
