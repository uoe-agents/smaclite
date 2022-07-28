from enum import Enum


class CombatType(Enum):
    DAMAGE = 'DAMAGE'
    HEALING = 'HEALING'

    @property
    def priority(self):
        return 1 if self == CombatType.DAMAGE else 0
