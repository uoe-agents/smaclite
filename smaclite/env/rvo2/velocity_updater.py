from typing import Dict, List, Set

import smaclite.env.rvo2.rvo2 as rvo2
from smaclite.env.rvo2.neighbour_finder import NeighbourFinder
from smaclite.env.rvo2.obstacle_finder import ObstacleFinder
from smaclite.env.rvo2.static_obstacle import StaticObstacle
from smaclite.env.terrain.terrain import TerrainType
from smaclite.env.units.unit import Unit
from smaclite.env.util.plane import Plane


class VelocityUpdater(object):
    def __init__(self, kd_tree: NeighbourFinder,
                 max_radius: float,
                 terrain: List[List[TerrainType]],
                 planes: Set[Plane]) -> None:
        raise NotImplementedError

    def reset_all_units(self, all_units: Dict[int, Unit]):
        raise NotImplementedError

    def compute_new_velocities(self, all_units: Dict[int, Unit]):
        raise NotImplementedError


class NumpyVelocityUpdater(VelocityUpdater):
    def __init__(self, kd_tree: NeighbourFinder, max_radius: float,
                 terrain: List[List[TerrainType]],
                 planes: Set[Plane]) -> None:
        print("Using the numpy RVO2 port")
        self.kd_tree = kd_tree
        self.max_radius = max_radius
        self.obstacles = list(StaticObstacle.from_terrain(terrain))
        self.obstacle_finder = ObstacleFinder(self.obstacles)

    def reset_all_units(self, all_units: Dict[int, Unit]):
        pass

    def compute_new_velocities(self, all_units: Dict[int, Unit]):
        all_units_list = list(all_units.values())
        radii = [self.max_radius + unit.radius for unit in all_units_list]
        neighbour_lists = self.kd_tree.query_radius(all_units_list, radii,
                                                    return_distance=True,
                                                    same_plane_only=True)
        for unit, neighbour_list in zip(all_units_list, neighbour_lists):
            obstacle_neighbours = self.obstacle_finder.query(unit) \
                if unit.plane == Plane.GROUND \
                else []
            unit.next_velocity = rvo2.compute_new_velocity(unit,
                                                           neighbour_list,
                                                           obstacle_neighbours)


class CPPVelocityUpdater(VelocityUpdater):
    def __init__(self, kd_tree: NeighbourFinder,
                 max_radius: float,
                 terrain: List[List[TerrainType]],
                 planes: Set[Plane]) -> None:
        # Only import this dependency if we actually get here,
        # since it is optional
        from smaclite.env.rvo2.rvo2_cpp import CPPRVO2Simulator
        print("Using the C++ RVO2 library")
        self.rvo2_cpp = CPPRVO2Simulator(max_radius, planes)
        obstacles = list(StaticObstacle.from_terrain(terrain))
        self.rvo2_cpp.add_obstacles(obstacles)

    def reset_all_units(self, all_units: Dict[int, Unit]):
        self.rvo2_cpp.remove_all_units()
        for unit in all_units.values():
            self.rvo2_cpp.add_agent(unit)

    def compute_new_velocities(self, all_units: Dict[int, Unit]):
        self.rvo2_cpp.step()
