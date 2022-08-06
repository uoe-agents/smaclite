from typing import Dict, List, Set

import numpy as np
from smaclite.env.rvo2.static_obstacle import StaticObstacle
from smaclite.env.units.unit import Unit
from smaclite.env.util.plane import Plane

import rvo2

TAU = 1
TAU_OBST = 1


class CPPRVO2Simulator(object):
    def __init__(self, max_radius: float, planes: Set[Plane]):
        self.sims: Dict[Plane, rvo2.PyRVOSimulator] = {}
        self.units_per_plane: Dict[Plane, List[Unit]] = {}
        self.planes = planes
        for plane in planes:
            # We will always use the agent constructor with all the parameters,
            # so we can set absurd values here to fail quickly in case of
            # trouble.
            self.sims[plane] = rvo2.PyRVOSimulator(timeStep=1/16,
                                                   neighborDist=-100,
                                                   maxNeighbors=0,
                                                   timeHorizon=-100,
                                                   timeHorizonObst=-100,
                                                   radius=-100,
                                                   maxSpeed=-100)
            self.units_per_plane[plane] = []
        self.max_radius = max_radius

    def add_agent(self, unit: Unit):
        neighbour_dist = (self.max_radius + unit.radius) * TAU
        plane = unit.plane
        rvo_id = self.sims[plane].addAgent(tuple(unit.pos),
                                           neighborDist=neighbour_dist,
                                           maxNeighbors=999,
                                           timeHorizon=TAU,
                                           timeHorizonObst=TAU_OBST,
                                           radius=unit.radius,
                                           maxSpeed=unit.max_velocity,
                                           velocity=unit.velocity)
        assert rvo_id == len(self.units_per_plane[plane])

        self.units_per_plane[plane].append(unit)

    def add_obstacles(self, obstacles: List[StaticObstacle]):
        if Plane.GROUND not in self.planes:
            return
        for obstacle in obstacles:
            self.sims[Plane.GROUND].addObstacle([tuple(ll.point)
                                                 for ll in obstacle.lines])
        self.sims[Plane.GROUND].processObstacles()

    def remove_all_units(self):
        for plane in self.planes:
            self.sims[plane].resetAgents()
            self.units_per_plane[plane] = []

    def step(self):
        for plane in self.planes:
            sim = self.sims[plane]
            units = self.units_per_plane[plane]
            to_delete = [k for k, v in enumerate(units) if v.hp == 0]
            for i in reversed(to_delete):
                id_to_delete = sim.removeAgent(i)
                units[i] = units[id_to_delete]
                del units[id_to_delete]
            for k, unit in enumerate(units):
                sim.setAgentPrefVelocity(k, tuple(unit.pref_velocity))
                sim.setAgentPosition(k, tuple(unit.pos))
            sim.doStep()
            for k, unit in enumerate(units):
                x, y = sim.getAgentVelocity(k)
                unit.next_velocity = np.array([x, y])
