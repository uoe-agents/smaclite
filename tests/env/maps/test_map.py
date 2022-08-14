from smaclite.env.maps.map import MapPreset
from smaclite.env.util.faction import Faction


def test_number_of_units():
    for preset in MapPreset:
        groups = preset.map_info.groups
        total_allies = sum(count for group in groups for _, count
                           in group.units if group.faction == Faction.ALLY)
        total_enemies = sum(count for group in groups for _, count
                            in group.units if group.faction == Faction.ENEMY)

        assert total_allies == preset.map_info.num_allied_units
        assert total_enemies == preset.map_info.num_enemy_units
