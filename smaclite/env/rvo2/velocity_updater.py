from typing import Dict

from smaclite.env.rvo2.kdtree_facade import KDTreeFacade
import smaclite.env.rvo2.rvo2 as rvo2
from smaclite.env.units.unit import Unit


class VelocityUpdater:
    def __init__(self, kd_tree: KDTreeFacade, max_radius: float):
        self.kd_tree = kd_tree
        self.max_radius = max_radius

    def compute_new_velocities(self, all_units: Dict[int, Unit]):
        all_units_list = list(all_units.values())
        radii = [self.max_radius + unit.radius for unit in all_units_list]
        neighbour_lists = self.kd_tree.query_radius(all_units_list, radii, True)
        for unit, neighbour_list in zip(all_units_list, neighbour_lists):
            unit.next_velocity = rvo2.compute_new_velocity(unit,
                                                           neighbour_list)
