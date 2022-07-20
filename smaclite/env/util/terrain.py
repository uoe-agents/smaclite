from enum import Enum


class TerrainType(Enum):
    NORMAL = '_'
    CLIFF = 'C'
    NONE = 'X'


class TerrainPreset(Enum):
    SIMPLE = [*[[TerrainType.NONE for _ in range(32)] for _ in range(8)],
              *[[TerrainType.NORMAL for _ in range(32)] for _ in range(16)],
              *[[TerrainType.NONE for _ in range(32)] for _ in range(8)]
              ]
    CHECKERBOARD = [[TerrainType.NONE, TerrainType.NONE, TerrainType.NONE],
                    [TerrainType.NORMAL, TerrainType.NONE, TerrainType.NORMAL],
                    [TerrainType.NONE, TerrainType.NORMAL, TerrainType.NONE]]
    NARROW = [*[[TerrainType.NONE for _ in range(32)] for _ in range(8)],
              *[[*[TerrainType.NORMAL for _ in range(15)],
                 *[TerrainType.NONE for _ in range(2)],
                 *[TerrainType.NORMAL for _ in range(15)]] for _ in range(7)],
              *[[TerrainType.NORMAL for _ in range(32)] for _ in range(2)],
              *[[*[TerrainType.NORMAL for _ in range(15)],
                 *[TerrainType.NONE for _ in range(2)],
                 *[TerrainType.NORMAL for _ in range(15)]] for _ in range(7)],
              *[[TerrainType.NONE for _ in range(32)] for _ in range(8)]
              ]


TERRAIN_PRESETS = {t.name: t for t in TerrainPreset}
