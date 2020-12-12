from pysc2.agents import base_agent
from pysc2.lib import actions, features
import numpy as np

FUNCTIONS = actions.FUNCTIONS
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL

def _xy_locs(mask):
  """Mask should be a set of bools from comparison with a feature layer."""
  y, x = mask.nonzero()
  return list(zip(x, y))

def do_stuff_with_reward(reward):
    """reward will be 1 if last frame the marine entered beacon else 0"""
    # print(reward)
    pass

def decide_coords(player_relative):
    beacon = _xy_locs(player_relative == _PLAYER_NEUTRAL)
    beacon_center = np.mean(beacon, axis=0).round()
    return beacon_center

class MoveToBeacon(base_agent.BaseAgent):
    def step(self, obs):
        super(MoveToBeacon, self).step(obs)
        do_stuff_with_reward(obs.reward)
        if FUNCTIONS.Move_screen.id not in obs.observation.available_actions:
            return FUNCTIONS.select_army(False)
        coords_to_go_to = decide_coords(obs.observation.feature_screen.player_relative)
        return FUNCTIONS.Move_screen(False, coords_to_go_to)


if __name__ == "__main__":
    agent1 = SmartAgent()
    agent2 = RandomAgent()
    try:
        with sc2_env.SC2Env(
                map_name="Simple64",
                players=[sc2_env.Agent(sc2_env.Race.terran),
                         sc2_env.Agent(sc2_env.Race.terran)],
                agent_interface_format=features.AgentInterfaceFormat(
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution=64,
                ),
                step_mul=48,
                disable_fog=True,
        ) as env:
            run_loop.run_loop([agent1, agent2], env, max_episodes=1000)
    except KeyboardInterrupt:
        pass