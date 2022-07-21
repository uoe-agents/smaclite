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
        strs = map(str.strip, f.readlines())
    return [
        list(map(TerrainType, row))
        for row in strs
    ]


class TerrainPreset(Enum):
    SIMPLE = from_file('simple')
    CHECKERBOARD = from_file('checkerboard')
    NARROW = from_file('narrow')


TERRAIN_PRESETS = {t.name: t for t in TerrainPreset}
