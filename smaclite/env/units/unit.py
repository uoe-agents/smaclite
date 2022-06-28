import numpy as np
from smaclite.env.units.unit_type import UnitType
from smaclite.env.maps.map import Faction

TICKS_PER_SECOND = 16
GAME_TICK_TIME = 1 / TICKS_PER_SECOND  # 30 ticks per second
SHIELD_REGEN = 2


class Unit(object):
    def __init__(self, unit_type: UnitType, faction: Faction,
                 x: float, y: float, idd: int) -> None:
        self.id = idd
        self.type = unit_type
        self.max_hp = unit_type.stats.hp
        self.max_shield = unit_type.stats.shield
        self.hp = self.max_hp
        self.shield = self.max_shield
        self.faction = faction
        self.pos = np.array([x, y], dtype=np.float32)
        self.cooldown = 0
        self.command = None
        self.target: 'Unit' = None
        self.shield_cooldown = 0
        self.velocity = np.array([0, 0], dtype=np.float32)
        self.next_velocity: np.ndarray = None
        self.pref_velocity: np.ndarray = None
        self.max_cooldown = unit_type.stats.cooldown
        self.attack_range = unit_type.stats.attack_range
        self.bonuses = unit_type.stats.bonuses
        self.minimum_scan_range = unit_type.stats.minimum_scan_range
        self.damage = unit_type.stats.damage
        self.attributes = unit_type.stats.attributes
        self.size = unit_type.stats.size
        self.radius = unit_type.radius
        self.armor = unit_type.stats.armor
        self.max_velocity = unit_type.stats.speed
        self.combat_type = unit_type.stats.combat_type
        self.attacking = False

    def prepare_velocity(self):
        self.pref_velocity = self.command.prepare_velocity(self)

    def game_step(self):
        if self.hp == 0:
            return 0
        if np.linalg.norm(self.next_velocity) - self.max_velocity > 1e-6:
            print(f"Unit {self.type.name} is too fast! {self.next_velocity} "
                  f"{np.linalg.norm(self.next_velocity)} > {self.max_velocity}")
        self.velocity = self.next_velocity
        self.pos += self.velocity * GAME_TICK_TIME
        self.next_velocity = None
        self.pref_velocity = None
        if self.cooldown > 0:
            self.__decrease_cooldown()
        if self.shield_cooldown > 0:
            self.shield_cooldown = max(self.shield_cooldown - GAME_TICK_TIME,
                                       0)
        if self.shield_cooldown == 0 and self.shield < self.max_shield:
            self.shield += 2 / TICKS_PER_SECOND
        return self.command.execute(self)

    def has_within_attack_range(self, target: 'Unit'):
        dpos = target.pos - self.pos
        radius = self.attack_range + target.radius + self.radius
        return np.inner(dpos, dpos) <= radius ** 2

    def has_within_scan_range(self, target: 'Unit'):
        dpos = target.pos - self.pos
        radius = self.minimum_scan_range + target.radius
        return np.inner(dpos, dpos) <= radius ** 2

    def deal_damage(self, target: 'Unit') -> float:
        damage = self.damage
        if not self.bonuses:
            return target.take_damage(damage)
        for attribute, amount in self.bonuses.items():
            if attribute in target.attributes:
                damage += amount
        return target.take_damage(damage)

    def take_damage(self, amount) -> float:
        if self.hp == 0:
            return 0
        if self.max_shield > 0:
            self.shield_cooldown = 10
        reward = 0
        if self.shield > 0:
            amount_shielded = min(amount, self.shield)
            self.shield -= amount_shielded
            amount -= amount_shielded
            reward += amount_shielded
        amount_dealt = min(amount - self.armor, self.hp)
        self.hp -= amount_dealt
        reward += amount_dealt
        if self.faction == Faction.ALLY:
            # No rewards for allies taking damage
            return 0
        if self.hp == 0:
            reward += 10
        return reward

    def __decrease_cooldown(self) -> None:
        if self.cooldown > 0:
            self.cooldown = max(0, self.cooldown - GAME_TICK_TIME)
