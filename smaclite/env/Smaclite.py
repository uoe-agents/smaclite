import gym

from smaclite.smaclite.env.maps.map import MapInfo


class SMACLiteEnv(gym.Env):
    """
    This is the SMACLite environment.
    """
    def __init__(self, map_info: MapInfo):
        self.n_agents = map_info.num_allied_units
        self.n_opponents = map_info.num_enemy_units
        self.action_space = None  # TODO
        self.observation_space = None  # TODO
