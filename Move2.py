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
actor_x_path = 'model/actor_x.pkl'
actor_y_path = 'model/actor_y.pkl'
critic_path = 'model/critic.pkl'
dir_path = 'model'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 0.01
FUNCTIONS = actions.FUNCTIONS
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
screen_size = 32


class GenericNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(GenericNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, observation):
        state = torch.flatten(torch.tensor(observation, dtype=torch.float)).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Agent(base_agent.BaseAgent):
    def __init__(self, alpha=learning_rate, input_dims=screen_size * screen_size, gamma=0.99, n_actions=2,
                 layer1_size=256, layer2_size=256, n_outputs=1):
        super().__init__()
        if input_dims is None:
            input_dims = (int(screen_size), int(screen_size))
        self.gamma = gamma
        self.log_probs_x = None
        self.log_probs_y = None
        self.n_outputs = n_outputs
        self.last = {}
        self.actor_x = None
        if os.path.isfile(actor_x_path):
            self.actor_x = torch.load(actor_x_path)
            self.actor_x.eval()
        else:
            self.actor_x = GenericNetwork(alpha, input_dims, layer1_size,
                                          layer2_size, n_actions=n_actions)
        self.actor_y = None
        if os.path.isfile(actor_y_path):
            self.actor_y = torch.load(actor_y_path)
            self.actor_y.eval()
        else:
            self.actor_y = GenericNetwork(alpha, input_dims, layer1_size,
                                          layer2_size, n_actions=n_actions)
        self.critic = None
        if os.path.isfile(critic_path):
            self.critic = torch.load(critic_path)
            self.critic.eval()
        else:
            self.critic = GenericNetwork(alpha, input_dims, layer1_size,
                                         layer2_size, n_actions=n_actions)

    def choose_action(self, observation):
        mu_x, sigma_x = self.actor_x.forward(observation).to(device)
        sigma_x = torch.exp(sigma_x)
        action_probs_x = torch.distributions.Normal(mu_x, sigma_x)
        probs_x = action_probs_x.sample(sample_shape=torch.Size([self.n_outputs]))
        self.log_probs_x = action_probs_x.log_prob(probs_x).to(device)
        action_x = torch.tanh(probs_x)

        mu_y, sigma_y = self.actor_y.forward(observation).to(device)
        sigma_y = torch.exp(sigma_y)
        action_probs_y = torch.distributions.Normal(mu_y, sigma_y)
        probs_y = action_probs_y.sample(sample_shape=torch.Size([self.n_outputs]))
        self.log_probs_y = action_probs_y.log_prob(probs_y).to(device)
        action_y = torch.tanh(probs_y)

        return action_x.item(), action_y.item()

    def learn(self, state, reward, new_state, done):
        self.actor_x.optimizer.zero_grad()
        self.actor_y.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        critic_value_ = self.critic.forward(new_state) if not done else 0
        critic_value = self.critic.forward(state)
        reward = torch.tensor(reward, dtype=torch.float).to(device)
        delta = (reward + self.gamma * critic_value_) - critic_value

        actor_x_loss = -self.log_probs_x * delta
        actor_y_loss = -self.log_probs_y * delta
        critic_loss = delta ** 2

        (actor_x_loss + actor_y_loss + critic_loss).backward()

        self.actor_x.optimizer.step()
        self.actor_y.optimizer.step()
        self.critic.optimizer.step()

    def save(self):
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        torch.save(self.actor_x, actor_x_path)
        torch.save(self.actor_y, actor_y_path)
        torch.save(self.critic, critic_path)

    def reset(self):
        super(Agent, self).reset()
        self.save()

    def step(self, obs):
        super(Agent, self).step(obs)
        if FUNCTIONS.Move_screen.id not in obs.observation.available_actions:
            return FUNCTIONS.select_army(False)

        state = obs.observation.feature_screen.player_relative
        reward = obs.reward

        if self.last != {}:
            self.learn(self.last['state'], reward, state, False)

        # should_act_randomly = np.random.rand() < self.probability_of_random_action
        # coords_to_go_to = self.predict(state) if not should_act_randomly else np.random.randint(0, screen_size, [2])
        coords_to_go_to = [np.floor(a) for a in self.choose_action(state)]
        coords_to_go_to = [(x + 1) * screen_size / 2 for x in coords_to_go_to]
        # print(coords_to_go_to)
        # self.current_memory.append([state, obs.reward])

        self.last['state'] = state
        print(coords_to_go_to)
        return FUNCTIONS.Move_screen(False, coords_to_go_to)

    def test(self):
        obs = torch.randn([32, 32])
        x, y = self.choose_action(obs)
        x = (x + 1) * screen_size / 2
        y = (y + 1) * screen_size / 2
        return [np.floor(a) for a in [x, y]]


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)

    agent = Agent()

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
# if __name__ == "__main__":
#     FLAGS = flags.FLAGS
#     FLAGS(sys.argv)
#
#     agent = Agent(1)
#     for i in range(10):
#         print(agent.test())
