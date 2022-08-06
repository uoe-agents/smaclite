from dataclasses import dataclass
from typing import List
import numpy as np
from smaclite.env.units.unit import Unit
from smaclite.env.util.direction import Direction
from smaclite.env.terrain.terrain import TerrainType


@dataclass
class ObstacleLine(object):
    point: np.ndarray = None
    unit_direction: np.ndarray = None

    def __eq__(self, other):
        return np.allclose(self.point, other.point) \
            and np.allclose(self.unit_direction, other.unit_direction)


class StaticObstacle(object):
    def __init__(self, idd, x, y, width, height):
        self.id = idd
        self.x = x
        self.y = y
        self.cx = x + width / 2
        self.cy = y + height / 2
        self.width = width
        self.height = height
        self.half_width = width / 2
        self.half_height = height / 2
        self.lines = [
            ObstacleLine(np.array([x, y]), Direction.EAST.dx_dy),
            ObstacleLine(np.array([x + width, y]), Direction.NORTH.dx_dy),
            ObstacleLine(np.array([x + width, y + height]),
                         Direction.WEST.dx_dy),
            ObstacleLine(np.array([x, y + height]), Direction.SOUTH.dx_dy),
        ]

    def __repr__(self) -> str:
        return f"StaticObstacle(x={self.x}, y={self.y}, " \
               f"width={self.width}, height={self.height})"

    def is_within_range(self, unit: Unit, radius: float, radius_sq: float):
        dx = max(0, abs(unit.pos[0] - self.cx) - self.half_width)
        if dx > radius:
            return False
        dy = max(0, abs(unit.pos[1] - self.cy) - self.half_height)
        if dy > radius:
            return False
        return True if dx + dy <= radius else dx**2 + dy**2 <= radius_sq

    def rtree_coords(self):
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @classmethod
    def from_terrain(cls, terrain: List[List[TerrainType]]) \
            -> List['StaticObstacle']:
        idd = 0
        h = len(terrain)
        w = len(terrain[0])
        visited = set()
        obstacles = []
        for y in range(h):
            for x in range(w):
                if (x, y) in visited:
                    continue
                visited.add((x, y))
                if terrain[y][x] == TerrainType.NORMAL:
                    continue
                x_start = x
                x_end = x + 1
                for x2 in range(x_start + 1, w):
                    if terrain[y][x2] == TerrainType.NORMAL:
                        break
                    x_end = x2 + 1
                    visited.add((x2, y))
                y_start = y
                y_end = y + 1
                for y2 in range(y_start + 1, h):
                    if any(terrain[y2][x2] == TerrainType.NORMAL
                           for x2 in range(x_start, x_end)):
                        break
                    y_end = y2 + 1
                    for x2 in range(x_start, x_end):
                        visited.add((x2, y2))
                obstacles.append(cls(idd,
                                     x_start,
                                     y_start,
                                     x_end - x_start,
                                     y_end - y_start))
                idd += 1
        assert len(visited) == w * h
        return obstacles
