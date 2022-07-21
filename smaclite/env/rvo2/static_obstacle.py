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
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
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

    def distance_sq_to(self, unit: Unit):
        cx = self.x + self.width / 2
        cy = self.y + self.height / 2
        dx = max(0, abs(unit.pos[0] - cx) - self.width / 2)
        dy = max(0, abs(unit.pos[1] - cy) - self.height / 2)
        return dx ** 2 + dy ** 2

    @classmethod
    def from_terrain(cls, terrain: List[List[TerrainType]]) \
            -> List['StaticObstacle']:
        h = len(terrain)
        w = len(terrain[0])
        visited = set()
        obstacles = []
        for x in range(w):
            for y in range(h):
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
                obstacles.append(cls(x_start,
                                     y_start,
                                     x_end - x_start,
                                     y_end - y_start))
        assert len(visited) == w * h
        return obstacles
