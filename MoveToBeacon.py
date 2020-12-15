from pysc2.env import sc2_env, run_loop
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from pysc2.agents import base_agent
from pysc2.lib import actions, features
import numpy as np
import sys
from absl import flags
import os

actor_path = 'model/actor.pkl'
critic_path = 'model/critic.pkl'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 0.01
FUNCTIONS = actions.FUNCTIONS
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
screen_size = 32


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.state_size = screen_size * screen_size
        self.action_size = 2
        self.linear1 = nn.Linear(self.state_size, self.state_size * 2)
        self.linear2 = nn.Linear(self.state_size * 2, self.state_size * 4)
        self.linear3 = nn.Linear(self.state_size * 4, self.action_size)
        self.cuda(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = F.sigmoid(self.linear3(output))
        return output


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.state_size = screen_size * screen_size
        self.action_size = 2
        self.linear1 = nn.Linear(self.state_size, self.state_size * 2)
        self.linear2 = nn.Linear(self.state_size * 2, self.state_size * 4)
        self.linear3 = nn.Linear(self.state_size * 4, self.action_size)
        self.cuda(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value


class MoveToBeacon(base_agent.BaseAgent):
    def __init__(self, probability_of_random_action=1):
        super().__init__()
        self.probability_of_random_action = probability_of_random_action
        self.actor = None
        self.critic = None
        if os.path.isfile(actor_path):
            self.actor = torch.load(actor_path)
            self.actor.eval()
        else:
            self.actor = Actor()
        if os.path.isfile(critic_path):
            self.critic = torch.load(critic_path)
            self.critic.eval()
        else:
            self.critic = Critic()
        self.current_memory = []
        self.global_memory = []
        self.last = {}
        self.gamma = 0.99

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
        if self.last != {}:
            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()

            reward = torch.tensor(reward, dtype=torch.float).to(device)
            delta = reward + self.gamma * self.critic(state) - self.critic(self.last['state'])
            actor_loss = - self.last['prediction'] * delta
            critic_loss = delta ** 2

            actor_loss.backward()
            critic_loss.backward()

            self.actor.optimizer.step()
            self.critic.optimizer.step()

    def predict(self, state):
        processed_state = torch.FloatTensor(np.array(state).flatten()).to(device)
        prediction = self.actor(processed_state)
        dist = prediction.detach().numpy().tolist()
        dist = np.floor(np.multiply(dist, screen_size))
        self.last['prediction'] = prediction
        return dist

    def add_to_global_memory(self):
        last_q = 0
        for i in reversed(self.current_memory):
            last_q = i[1] + self.gamma * last_q
            self.global_memory.append([i[0], last_q])

    def save_global_memory(self):
        pass

    def reset(self):
        super(MoveToBeacon, self).reset()
        self.add_to_global_memory()
        self.save_global_memory()

    def step(self, obs):
        super(MoveToBeacon, self).step(obs)

        if FUNCTIONS.Move_screen.id not in obs.observation.available_actions:
            return FUNCTIONS.select_army(False)

        state = obs.observation.feature_screen.player_relative
        reward = obs.reward

        self.learn(state, reward)

        should_act_randomly = np.random.rand() < self.probability_of_random_action
        coords_to_go_to = self.predict(state) if not should_act_randomly else np.random.randint(0, screen_size, [2])

        # print(coords_to_go_to)
        self.current_memory.append([state, obs.reward])

        self.last['state'] = state
        return FUNCTIONS.Move_screen(False, coords_to_go_to)


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)

    agent = MoveToBeacon(1)

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
            run_loop.run_loop([agent], env, max_episodes=1)
    except KeyboardInterrupt:
        pass
