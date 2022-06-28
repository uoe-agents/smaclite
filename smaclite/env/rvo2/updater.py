import multiprocessing as mp
from typing import List

import numpy as np
import smaclite.env.rvo2.rvo2 as rvo2
from sklearn.neighbors import KDTree
from smaclite.env.units.unit import Unit

USE_RVO = True
USE_MULTIPROCESSING = False
USE_KD_TREE = True


class VelocityUpdater:
    def __init__(self):
        if USE_RVO and USE_MULTIPROCESSING:
            self.in_queue = mp.Queue()
            self.out_queue = mp.Queue()
            n_workers = mp.cpu_count()
            self.all_units = None
            self.workers = [mp.Process(target=rvo2.velocity_computing_job,
                                       args=(self.in_queue,
                                             self.out_queue))
                            for _ in range(n_workers)]
            for worker in self.workers:
                worker.start()

    def set_all_units(self, all_units: List[Unit]):
        self.all_units = all_units

    def compute_new_velocities(self):
        all_units = self.all_units
        if not USE_RVO:
            for unit in all_units:
                unit.next_velocity = unit.pref_velocity
        else:
            if USE_KD_TREE:
                kd_tree = KDTree(np.vstack([unit.pos for unit in all_units]))
                poses = np.vstack([unit.pos for unit in all_units])
                unit_idss = kd_tree.query_radius(poses,
                                                 4 * max(unit.radius
                                                         for unit
                                                         in all_units))
            else:
                unit_idss = [[unit.id for unit in all_units]
                             for _ in all_units]
            if USE_MULTIPROCESSING:
                for unit, unit_ids in zip(all_units, unit_idss):
                    self.in_queue.put((unit.id, unit_ids, all_units))
                for _ in all_units:
                    idx, next_velocity = self.out_queue.get()
                    all_units[idx].next_velocity = next_velocity

            else:
                for unit, unit_ids in zip(all_units, unit_idss):
                    unit.next_velocity = rvo2.compute_new_velocity(unit.id,
                                                                   unit_ids,
                                                                   all_units)

    def close(self):
        if not (USE_RVO and USE_MULTIPROCESSING):
            return
        for _ in self.workers:
            self.in_queue.put((None, None, None))
        for w in self.workers:
            w.join()
            w.close()
