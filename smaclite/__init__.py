import gym
from smaclite.env.maps.map import MapPreset

for preset in MapPreset:
    map_info = preset.value
    gym.register(
        f"smaclite/{map_info.name}-v0",
        entry_point="smaclite.env:SMACliteEnv",
        kwargs={"map_info": map_info},
    )
    gym.register(
        f"messy-smaclite/{map_info.name}-v0",
        entry_point="smaclite.env:MessySMACliteEnv",
        kwargs={"map_info": map_info},
    )
gym.register("smaclite/custom-v0", entry_point="smaclite.env:SMACliteEnv")
gym.register("messy-smaclite/custom-v0", entry_point="smaclite.env:MessySMACliteEnv")
