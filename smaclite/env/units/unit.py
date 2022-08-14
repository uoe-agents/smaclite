import numpy as np
import smaclite.env.units.unit_type as ut
from smaclite.env.units.combat_type import CombatType
from smaclite.env.util import point_inside_circle
from smaclite.env.util.faction import Faction

TICKS_PER_SECOND = 16
GAME_TICK_TIME = 1 / TICKS_PER_SECOND
SHIELD_REGEN = 2
SHIELD_PER_SECOND = 2
HEAL_PER_SECOND = 9
HEAL_PER_ENERGY = 3
ENERGY_PER_SECOND = 0.5625


class Unit(object):
    def __init__(self, unit_type: ut.UnitType, faction: Faction,
                 x: float, y: float, idd: int, idd_in_faction: int) -> None:
        self.id = idd
        self.id_in_faction = idd_in_faction
        self.type = unit_type
        self.max_hp = unit_type.stats.hp
        self.hp_regen = unit_type.stats.hp_regen
        self.max_shield = unit_type.stats.shield
        self.max_energy = unit_type.stats.energy
        self.hp = self.max_hp
        self.shield = self.max_shield
        self.energy = unit_type.stats.starting_energy
        self.healing_available = 0
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
        self.radius_sq = self.radius ** 2
        self.armor = unit_type.stats.armor
        self.max_velocity = unit_type.stats.speed
        self.combat_type = unit_type.stats.combat_type
        self.attacking = False
        self.attacks = self.type.stats.attacks
        self.targeter = ut.TARGETER_CACHE[self.type.stats.name]
        # Used for the purpose of attack-moving
        self.potential_targets = []
        self.priority_targets = []
        self.prev_target: 'Unit' = None
        self.plane = unit_type.stats.plane
        self.hit = False
        self.valid_targets = unit_type.stats.valid_targets

    def clean_up_target(self):
        self.hit = False
        self.potential_targets = []
        self.priority_targets = []
        self.prev_target = self.target
        self.command.clean_up_target(self)

    def prepare_velocity(self):
        self.pref_velocity = self.command.prepare_velocity(self)

    def game_step(self, **kwargs):
        if self.hp == 0:
            return 0
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
            self.shield = min(self.shield +
                              SHIELD_PER_SECOND / TICKS_PER_SECOND,
                              self.max_shield)
        if self.energy < self.max_energy:
            self.energy = min(self.energy +
                              ENERGY_PER_SECOND / TICKS_PER_SECOND,
                              self.max_energy)
        if self.hp_regen and self.hp < self.max_hp:
            self.hp = min(self.hp + self.hp_regen / TICKS_PER_SECOND,
                          self.max_hp)
        return self.command.execute(self, **kwargs)

    def has_within_attack_range(self, target: 'Unit'):
        radius = self.attack_range + target.radius + self.radius
        return point_inside_circle(self.pos, target.pos, radius)

    def has_within_scan_range(self, target: 'Unit'):
        radius = self.minimum_scan_range + target.radius
        return point_inside_circle(self.pos, target.pos, radius)

    def heal(self, target: 'Unit'):
        if self.combat_type != CombatType.HEALING:
            raise ValueError("Can't heal with this unit.")
        if self.prev_target != self.target and self.energy < 5 \
                or target.hp == target.max_hp:
            self.target = None
            return
        target_hp_missing = target.max_hp - target.hp
        desired_heal_amount = min(target_hp_missing,
                                  HEAL_PER_SECOND / TICKS_PER_SECOND)
        if self.healing_available < desired_heal_amount:
            energy_spent = min(1, self.energy)
            self.energy -= energy_spent
            self.healing_available += energy_spent * HEAL_PER_ENERGY
        heal_amount = min(desired_heal_amount,
                          self.healing_available)
        self.healing_available -= heal_amount
        target.hp += heal_amount

    def deal_damage(self, target: 'Unit') -> float:
        damage = self.damage
        if self.bonuses:
            for attribute, amount in self.bonuses.items():
                if attribute in target.attributes:
                    damage += amount
        return sum(target.take_damage(damage) for _ in range(self.attacks))

    def take_damage(self, amount) -> float:
        self.hit = True
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
        amount_dealt = max(0, min(amount - self.armor, self.hp))
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

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Unit) and self.id == other.id
