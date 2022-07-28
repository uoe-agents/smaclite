from enum import Enum


class Plane(Enum):
    GROUND = 'GROUND'
    AIR = 'AIR'
    COLOSSUS = 'COLOSSUS'

    @property
    def z(self):
        return 1 if self == Plane.AIR else 0
