from typing import Dict
from rtree import index
from smaclite.env.rvo2.static_obstacle import StaticObstacle


TAU = 1


class ObstacleFinder(object):
    def __init__(self, obstacles: StaticObstacle):
        self.index = index.Index()
        self.index_mapping: Dict[int, StaticObstacle] = {o.id:
                                                         o for o in obstacles}
        for o in obstacles:
            self.index.insert(o.id, o.rtree_coords())

    def query(self, unit):
        radius = unit.radius + TAU * unit.max_velocity
        rtree_coords = (unit.pos[0] - radius, unit.pos[1] - radius,
                        unit.pos[0] + radius, unit.pos[1] + radius)
        rtree_coords = (-100, -100, 100, 100)

        return [self.index_mapping[i] for i in
                self.index.intersection(rtree_coords)
                if self.index_mapping[i].is_within_range(unit,
                                                         radius,
                                                         radius ** 2)]
