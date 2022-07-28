

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Union

from sklearn.neighbors import KDTree
from smaclite.env.util.plane import Plane

ALL_PLANES = set(Plane)


class NeighbourFinder:
    def __init__(self):
        self.all_units = []
        self.pos_array = None
        self.kd_tree = None

    def set_all_units(self, all_units: Dict[int, Any]):
        self.all_units = all_units
        self.update()

    def update(self):
        if not self.all_units:
            return
        self.lists = defaultdict(list)
        self.kd_trees = {}
        for unit in self.all_units.values():
            self.lists[unit.plane].append(unit)
        for plane, units in self.lists.items():
            self.kd_trees[plane] = KDTree([unit.pos for unit in units])

    def query_radius(self, units: List[Any],
                     radius: Union[List[float], float],
                     return_distance: bool = False,
                     same_plane_only: bool = False,
                     targetting_mode: bool = False,
                     planes: Iterable[Plane] = ALL_PLANES) -> List[Any]:
        return [sum(a, start=[]) for a in
                zip(*(self.query_radius_plane(units, radius,
                                              return_distance,
                                              same_plane_only,
                                              targetting_mode,
                                              plane)
                      for plane in planes))]

    def query_radius_plane(self, units: List[Any],
                           radius: Union[List[float], float],
                           return_distance: bool,
                           same_plane_only: bool,
                           targetting_mode: bool,
                           plane: Plane):
        if not units:
            return []
        if plane not in self.kd_trees or not self.lists[plane]:
            return [[] for _ in range(len(units))]
        if type(radius) is list:
            assert len(radius) == len(units)
            if same_plane_only:
                radius = [r for r, u in zip(radius, units)
                          if u.plane == plane]
            if targetting_mode:
                radius = [r for r, u in zip(radius, units)
                          if plane in u.valid_targets]
        valids = [(i, unit.pos) for i, unit in enumerate(units)
                  if (not same_plane_only and not targetting_mode)
                  or (same_plane_only and unit.plane == plane)
                  or (targetting_mode and (plane in unit.valid_targets
                                           or plane == Plane.COLOSSUS))]
        if not valids:
            return [[] for _ in range(len(units))]
        valid_idxes, poses = zip(*valids)
        reverse_idx_mapping = {idx: i for i, idx in enumerate(valid_idxes)}
        neighbour_idx_lists = \
            self.kd_trees[plane].query_radius(poses, radius,
                                              return_distance=return_distance)
        if return_distance:
            return [[(self.lists[plane][idx], dist)
                     for idx, dist in zip(neighbour_idx_lists[0][reverse_idx_mapping[i]],
                                          neighbour_idx_lists[1][reverse_idx_mapping[i]])]
                    if i in reverse_idx_mapping else []
                    for i in range(len(units))]
        else:
            return [[self.lists[plane][idx]
                     for idx in neighbour_idx_lists[reverse_idx_mapping[i]]]
                    if i in reverse_idx_mapping else []
                    for i in range(len(units))]
