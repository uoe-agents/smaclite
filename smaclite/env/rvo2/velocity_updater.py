from typing import Dict, List

from smaclite.env.rvo2.neighbour_finder import NeighbourFinder
import smaclite.env.rvo2.rvo2 as rvo2
from smaclite.env.rvo2.static_obstacle import StaticObstacle
from smaclite.env.units.unit import Unit
from smaclite.env.terrain.terrain import TerrainType


class VelocityUpdater:
    def __init__(self, kd_tree: NeighbourFinder, max_radius: float,
                 terrain: List[List[TerrainType]]):
        self.kd_tree = kd_tree
        self.max_radius = max_radius
        self.obstacles = list(StaticObstacle.from_terrain(terrain))

    def compute_new_velocities(self, all_units: Dict[int, Unit]):
        all_units_list = list(all_units.values())
        radii = [self.max_radius + unit.radius for unit in all_units_list]
        neighbour_lists = self.kd_tree.query_radius(all_units_list, radii,
                                                    return_distance=True,
                                                    same_plane_only=True)
        for unit, neighbour_list in zip(all_units_list, neighbour_lists):
            unit.next_velocity = rvo2.compute_new_velocity(unit,
                                                           neighbour_list,
                                                           self.obstacles)
