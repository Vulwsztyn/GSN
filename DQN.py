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
from collections import namedtuple
from tensorboardX import SummaryWriter
import math
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

target_path = 'model/dqn_target.pkl'
act_path = 'model/dqn_act.pkl'

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

learning_rate = 0.01
FUNCTIONS = actions.FUNCTIONS
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
screen_size = 32

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(32 * 32, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 64)

    def forward(self, x):
        x = torch.reshape(x, (np.prod(x.shape) // 1024, 1024))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_prob = self.fc3(x)
        return action_prob


class DQN(base_agent.BaseAgent):
    capacity = 3600
    learning_rate = 1e-3
    memory_count = 0
    batch_size = min(360, capacity)
    gamma = 0.995
    update_count = 0

    def __init__(self):
        super(DQN, self).__init__()
        self.target_net, self.act_net = Net(), Net()
        self.target_net = None
        self.act_net = None
        if os.path.isfile(target_path):
            self.target_net = torch.load(target_path)
            self.target_net.eval()
        else:
            self.target_net = Net()
        if os.path.isfile(act_path):
            self.act_net = torch.load(act_path)
            self.act_net.eval()
        else:
            self.act_net = Net()
        self.memory = [None] * self.capacity
        self.optimizer = torch.optim.Adam(self.act_net.parameters(), self.learning_rate)
        self.loss_func = nn.MSELoss()
        self.writer = SummaryWriter('./DQN/logs')
        self.last = {}
        self.probability_of_random_action = 0
        self.learnings_count = 0

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        value = self.act_net.forward(state)
        x_ratings_tensor = value[0][:32]
        y_ratings_tensor = value[0][32:]

        x = torch.argmax(x_ratings_tensor)
        y = torch.argmax(y_ratings_tensor)
        if np.random.rand(1) <= self.probability_of_random_action:  # epslion greedy
            rand_coords = torch.randint(0, 32, (2,))
            x = rand_coords[0]
            y = rand_coords[1]
        return x.item(), y.item()

    def store_transition(self, transition):
        index = self.memory_count % self.capacity
        self.memory[index] = transition
        self.memory_count += 1
        return self.memory_count >= self.capacity

    def update(self):
        if self.memory_count >= self.capacity: # pominiemy pierwsze ileś 10 uczeń, ale nie chce mi sie
            states = torch.tensor([t.state for t in self.memory]).float()
            actions = torch.tensor([t.action for t in self.memory]).float()
            rewards = torch.tensor([t.reward for t in self.memory]).float()
            next_states = torch.tensor([t.next_state for t in self.memory]).float()

            #  idk if konieczne
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
            target_v = None
            with torch.no_grad():
                target_v = rewards + self.gamma * self.target_net.forward(next_states).max(1)[0]

            # Update...
            for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))), batch_size=self.batch_size,
                                      drop_last=False):
                # v = (self.act_net.forward(states).gather(1, actions))[index]
                t = self.act_net.forward(states)  # np. 50 na 64 
                t = torch.reshape(t, (self.capacity * 2, 32))  # np. 100 na 32
                actions_reshaped = torch.reshape(actions, (
                    self.capacity * 2, 1)).long()  # akcje jako wektor np 100 na 1 x1,y2,x2,y2,x3,y3,x4,y4,...
                loss = self.loss_func(target_v[index].unsqueeze(1),
                                      t.gather(1, actions_reshaped)[index])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar('loss/value_loss', loss, self.update_count)
                self.update_count += 1
                if self.update_count % 100 == 0:
                    self.target_net.load_state_dict(self.act_net.state_dict())
            self.learnings_count += 1
            print('learned {} times'.format(self.learnings_count))

    def reset(self):
        super(DQN, self).reset()
        print('memory_count', self.memory_count)
        self.update()
        torch.save(self.act_net, act_path)
        torch.save(self.target_net, target_path)

    def step(self, obs):
        super(DQN, self).step(obs)
        if FUNCTIONS.Move_screen.id not in obs.observation.available_actions:
            return FUNCTIONS.select_army(False)

        state = obs.observation.feature_screen.player_relative

        if self.last != {}:
            self.store_transition(Transition(self.last['state'], self.last['action'], obs.reward, state))
            # self.update()
        # should_act_randomly = np.random.rand() < self.probability_of_random_action
        # coords_to_go_to = self.predict(state) if not should_act_randomly else np.random.randint(0, screen_size, [2])        coords_to_go_to = [np.floor(a) for a in self.choose_action(state)]
        # coords_to_go_to = [(x + 1) * screen_size / 2 for x in coords_to_go_to]
        coords_to_go_to = self.choose_action(state)
        # print(coords_to_go_to)
        # self.current_memory.append([state, obs.reward])

        self.last['state'] = state
        self.last['action'] = coords_to_go_to
        return FUNCTIONS.Move_screen(False, coords_to_go_to)

    def set_probability_of_random_action(self, probability_of_random_action):
        self.probability_of_random_action = probability_of_random_action


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)

    agent = DQN()
    print(device)
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
            for i in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
                print('probability_of_random_action:', i)
                agent.set_probability_of_random_action(i)
                run_loop.run_loop([agent], env, max_episodes=600)
    except KeyboardInterrupt:
        pass
