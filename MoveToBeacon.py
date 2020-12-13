from pysc2.env import sc2_env, run_loop
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from pysc2.agents import base_agent
from pysc2.lib import actions, features
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = 0.0001
FUNCTIONS = actions.FUNCTIONS
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
screen_size = 32

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.state_size = screen_size*screen_size
        self.action_size = 2
        self.linear1 = nn.Linear(self.state_size, self.state_size*2)
        self.linear2 = nn.Linear(self.state_size*2, self.state_size*4)
        self.linear3 = nn.Linear(self.state_size*4, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = F.sigmoid(self.linear3(output))
        return output


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.state_size = screen_size*screen_size
        self.action_size = 2
        self.linear1 = nn.Linear(self.state_size, self.state_size*2)
        self.linear2 = nn.Linear(self.state_size*2, self.state_size*4)
        self.linear3 = nn.Linear(self.state_size*4, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value



class MoveToBeacon(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        self.actor = Actor()
        self.critic = Critic()

    def _xy_locs(self, mask):
        """Mask should be a set of bools from comparison with a feature layer."""
        y, x = mask.nonzero()
        return list(zip(x, y))

    def do_stuff_with_reward(self, reward):
        """reward will be 1 if last frame the marine entered beacon else 0"""
        # print(reward)
        pass

    def decide_coords(self, player_relative):
        beacon = self._xy_locs(player_relative == _PLAYER_NEUTRAL)
        beacon_center = np.mean(beacon, axis=0).round()
        return beacon_center

    def save(self):
        torch.save(self.actor, 'model/actor.pkl')
        torch.save(self.critic, 'model/critic.pkl')

    def learn(self, state, reward):
        return

    def predict(self, state):
        processed_state = torch.FloatTensor(np.array(state.player_relative).flatten()).to(device)
        dist = self.actor(processed_state).cpu().detach().numpy().tolist()
        print(dist)
        dist = np.floor(np.multiply(dist, 32))
        return dist


    def step(self, obs):
        super(MoveToBeacon, self).step(obs)

        state = obs.observation.feature_screen
        reward = obs.reward
        self.learn(state, reward)
        coords_to_go_to = self.predict(state)

        if FUNCTIONS.Move_screen.id not in obs.observation.available_actions:
            return FUNCTIONS.select_army(False)

        # coords_to_go_to = self.decide_coords(obs.observation.feature_screen.player_relative)

        return FUNCTIONS.Move_screen(False, coords_to_go_to)


if __name__ == "__main__":
    agent = MoveToBeacon()
    try:
        with sc2_env.SC2Env(
              map_name="MoveToBeacon",
              players=[sc2_env.Agent(sc2_env.Race.terran)],
              agent_interface_format=sc2_env.parse_agent_interface_format(
                  feature_screen=screen_size,
                  feature_minimap=screen_size,
                  action_space=None,
                  use_feature_units=False,
                  use_raw_units=False),
              step_mul=8,
              game_steps_per_episode=None,
              disable_fog=False,
              visualize=True) as env:
            run_loop.run_loop([agent], env, max_episodes=1000)
    except KeyboardInterrupt:
        pass