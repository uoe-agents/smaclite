

from typing import Dict, List, Union
import numpy as np

from sklearn.neighbors import KDTree
from smaclite.env.units.unit import Unit


class KDTreeFacade:
    def __init__(self):
        self.all_units = []
        self.pos_array = None
        self.kd_tree = None

    def set_all_units(self, all_units: Dict[int, Unit]):
        self.all_units = all_units
        self.update()

    def update(self):
        if not self.all_units:
            return
        self.all_units_list = list(self.all_units.values())
        self.kd_tree = KDTree(np.vstack([u.pos for u in self.all_units_list]))

    def query_radius(self, units: List[Unit],
                     radius: Union[List[float], float],
                     return_distance: bool = False):
        if type(radius) is list:
            assert len(radius) == len(units)
        if not units or not self.all_units:
            return []
        self.update()
        poses = [unit.pos for unit in units]
        neighbour_idx_lists = \
            self.kd_tree.query_radius(poses, radius,
                                      return_distance=return_distance)
        if return_distance:
            return [[(self.all_units_list[idx], dist)
                     for idx, dist in zip(idx_list, dist_list)]
                    for idx_list, dist_list in zip(*neighbour_idx_lists)]
        else:
            return [[self.all_units_list[idx] for idx in idx_list]
                    for idx_list in neighbour_idx_lists]
