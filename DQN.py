import pickle

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

target_path = 'model/dqn_target_3.pkl'
act_path = 'model/dqn_act_3.pkl'
memory_path = 'model/memory.pkl'
memory_positive_path = 'model/memory_positive.pkl'

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

learning_rate = 1e-3
FUNCTIONS = actions.FUNCTIONS
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
screen_size = 32

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])

def get_beacon_location(ai_relative_view):
    '''returns the location indices of the beacon on the map'''
    return (ai_relative_view == 3).nonzero()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(32 * 32, 32 * 32)
        self.fc2 = nn.Linear(32 * 32, 32 * 32)
        self.fc3 = nn.Linear(32 * 32, 32 * 32)
        self.fc4 = nn.Linear(32 * 32, 32 * 32)
        self.fc_last = nn.Linear(32 * 32, 64)

    def forward(self, x):
        x = torch.reshape(x, (np.prod(x.shape) // 1024, 1024))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        action_prob = self.fc_last(x)
        return action_prob


class DQN(base_agent.BaseAgent):
    capacity = 100
    learning_rate = 1e-3

    memory = None
    memory_positive = None
    memory_count = 0
    memory_positive_count = 0

    batch_size = min(20, capacity)
    gamma = 0.995
    update_count = 0

    current_move_reward = 0
    moving_gamma = 0.95

    should_learn = True
    moves_since_reward = []

    def load_nets(self):
        if os.path.isfile(target_path):
            self.target_net = torch.load(target_path)
            self.target_net.eval()
            print('target net loaded')
        else:
            self.target_net = Net()
            print('target net NOT loaded')
        if os.path.isfile(act_path):
            self.act_net = torch.load(act_path)
            self.act_net.eval()
            print('act net loaded')
        else:
            self.act_net = Net()
            print('act net NOT loaded')

    def load_memory(self):
        if os.path.isfile(memory_path):
            with open(memory_path, 'rb') as handle:
                self.memory = pickle.load(handle)
            self.memory_count = self.capacity - len([x for x in self.memory if x == 0])
            print('memory loaded', self.memory_count)
        else:
            self.memory = [0] * self.capacity
            print('memory NOT loaded')

        if os.path.isfile(memory_positive_path):
            with open(memory_positive_path, 'rb') as handle:
                self.memory_positive = pickle.load(handle)
            self.memory_positive_count = self.capacity - len([x for x in self.memory_positive if x == 0])
            print('memory positive loaded', self.memory_positive_count)
        else:
            self.memory_positive = [0] * self.capacity
            print('memory positive NOT loaded')

    def __init__(self):
        super(DQN, self).__init__()
        self.target_net, self.act_net = Net(), Net()
        self.target_net = None
        self.act_net = None
        self.load_nets()
        self.load_memory()
        self.optimizer = torch.optim.Adam(self.act_net.parameters(), self.learning_rate)
        self.loss_func = nn.MSELoss()
        self.writer = SummaryWriter('./DQN/logs')
        self.last = {}
        self.probability_of_random_action = 0
        self.learnings_count = 0
        self.moves = 0
        self.probability_of_perfect_action = 0.1

    def choose_action(self, state):
        should_move_be_random = np.random.rand(1) <= self.probability_of_random_action
        x,y = None, None
        if not should_move_be_random:
            self.last['action_type'] = 'network'
            state = torch.tensor((state==3)*3, dtype=torch.float).unsqueeze(0)
            value = self.act_net.forward(state)
            x_ratings_tensor = value[0][:32]
            y_ratings_tensor = value[0][32:]

            x = torch.argmax(x_ratings_tensor)
            y = torch.argmax(y_ratings_tensor)
        if should_move_be_random:
        # if should_move_be_random or (len(self.moves_since_reward) > 0 and self.moves_since_reward[-1] == (x.item(), y.item())):
            should_move_be_perfect = np.random.rand(1) <= self.probability_of_perfect_action
            if should_move_be_perfect:
                self.last['action_type'] = 'perfect'
                beacon_xs, beacon_ys = get_beacon_location(state)
                # if not beacon_ys.any():
                #     return actions.FunctionCall(_NO_OP, [])
                # get the middle of the beacon and move there
                target = [beacon_ys.mean(), beacon_xs.mean()]
                rand_coords = torch.tensor(target)
                x = rand_coords[0]
                y = rand_coords[1]
            else:
                self.last['action_type'] = 'random'
                rand_coords = torch.randint(0, 32, (2,))
                x = rand_coords[0]
                y = rand_coords[1]
        return x.item(), y.item()

    def store_transition(self, transition):
        if transition.reward > 0:
            index = self.memory_positive_count % self.capacity
            self.memory_positive[index] = transition
            self.memory_positive_count += 1
        else:
            index = self.memory_count % self.capacity
            self.memory[index] = transition
            self.memory_count += 1

        return self.memory_count >= self.capacity

    def update(self):
        if not self.should_learn: return
        is_memory_filled = self.memory_count >= self.capacity and self.memory_positive_count >= self.capacity
        if is_memory_filled:
            sample_index = np.random.choice(self.capacity, self.batch_size)
            sample_index_positive = np.random.choice(self.capacity, self.batch_size)
            batch_memory = np.concatenate((np.array(self.memory, dtype=Transition)[sample_index, :],
                                           np.array(self.memory_positive,
                                                    dtype=Transition)[
                                           sample_index_positive, :]))

            states = torch.tensor([t[0] for t in batch_memory]).float()
            actions = torch.tensor([t[1] for t in batch_memory]).float()
            rewards = torch.tensor([t[2] for t in batch_memory]).float()
            next_states = torch.tensor([t[3] for t in batch_memory]).float()

            t = self.act_net.forward(states)  # np. 50 na 64 
            t = torch.reshape(t, (len(batch_memory) * 2, 32))
            actions_reshaped = torch.reshape(actions, (
                len(batch_memory) * 2, 1)).long()
            q_eval = t.gather(1, actions_reshaped)
            q_next = self.target_net.forward(next_states).detach()
            q_target = rewards + self.gamma * q_next.max(1)[0]  # .view(self.batch_size, 1)
            q_target = torch.reshape(torch.transpose(torch.stack((q_target, q_target), 0), 0, 1),
                                     (len(batch_memory) * 2, 1))  # robi z 1,2,3 1,1,2,2,3,3
            loss = self.loss_func(q_eval, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return
            # #  idk if konieczne
            # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
            # target_v = None
            # with torch.no_grad():
            #     target_v = rewards + self.gamma * self.target_net.forward(next_states).max(1)[0]
            #
            # # Update...
            # for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))), batch_size=self.batch_size,
            #                           drop_last=False):
            #     # v = (self.act_net.forward(states).gather(1, actions))[index]
            #     t = self.act_net.forward(states)  # np. 50 na 64
            #     t = torch.reshape(t, (self.capacity * 2, 32))  # np. 100 na 32
            #     actions_reshaped = torch.reshape(actions, (
            #         self.capacity * 2, 1)).long()  # akcje jako wektor np 100 na 1 x1,y2,x2,y2,x3,y3,x4,y4,...
            #     loss = self.loss_func(target_v[index].unsqueeze(1),
            #                           t.gather(1, actions_reshaped)[index])
            #     self.optimizer.zero_grad()
            #     loss.backward()
            #     self.optimizer.step()
            #     self.writer.add_scalar('loss/value_loss', loss, self.update_count)
            #     self.update_count += 1
            #     if self.update_count % 100 == 0:
            #         self.target_net.load_state_dict(self.act_net.state_dict())
            # self.learnings_count += 1
            # print('learned {} times'.format(self.learnings_count))

    def save(self, prefix):
        torch.save(self.act_net, prefix + act_path)
        torch.save(self.target_net, prefix + target_path)
        with open(prefix + memory_path, 'wb') as handle:
            pickle.dump(self.memory, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(prefix + memory_positive_path, 'wb') as handle:
            pickle.dump(self.memory_positive, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def reset(self):
        super(DQN, self).reset()
        # print('memory_count', self.memory_count)
        # print('reset')
        print(self.reward, self.episodes)
        # print(self.moves)
        self.update()
        self.last = {}
        if self.episodes % 10 == 0:
            self.save("")

    def step(self, obs):
        super(DQN, self).step(obs)
        available_actions = obs.observation.available_actions
        # print([x.order_length for x in obs.observation.raw_units])
        # print([x.unit_type for x in obs.observation.raw_units])
        self.current_move_reward = self.current_move_reward * self.moving_gamma + obs.reward
        if FUNCTIONS.Move_screen.id not in available_actions:
            return FUNCTIONS.select_army(False)
        if sum([x.order_length for x in obs.observation.raw_units]) > 0:
            return FUNCTIONS.no_op()
        state = obs.observation.feature_screen.player_relative


        if self.last != {}:
            # print(self.last['action_type'],self.current_move_reward)
            self.store_transition(Transition(self.last['state'], self.last['action'], self.current_move_reward, state))
            self.update()
        # should_act_randomly = np.random.rand() < self.probability_of_random_action
        # coords_to_go_to = self.predict(state) if not should_act_randomly else np.random.randint(0, screen_size, [2])        coords_to_go_to = [np.floor(a) for a in self.choose_action(state)]
        # coords_to_go_to = [(x + 1) * screen_size / 2 for x in coords_to_go_to]
        # print(coords_to_go_to)
        # self.current_memory.append([state, obs.reward])
        coords_to_go_to = self.choose_action(state)
        self.last['state'] = state
        self.last['action'] = coords_to_go_to
        self.current_move_reward = 0
        # if np.random.rand(1) <= 0.5:  # epslion greedy
        #     print('stop')
        #     return FUNCTIONS.Stop_quick(False)
        self.moves += 1

        if obs.reward > 0:
            self.moves_since_reward = []
        else:
            self.moves_since_reward.append(coords_to_go_to)
        return FUNCTIONS.Move_screen(False, coords_to_go_to)

    def set_probability_of_random_action(self, probability_of_random_action):
        self.probability_of_random_action = probability_of_random_action

    def set_should_learn(self, should_learn):
        self.should_learn = should_learn


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
                    use_raw_units=True),
                step_mul=8,
                game_steps_per_episode=None,
                disable_fog=False,
                visualize=True) as env:
            # for i in [0]:
            #     print('probability_of_random_action:', i)
            #     agent.set_probability_of_random_action(i)
            #     run_loop.run_loop([agent], env, max_episodes=10)
            # ,
            # 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1
            # for i in [0.5]:
            #     print('probability_of_random_action:', i)
            #     agent.set_probability_of_random_action(i)
            #     run_loop.run_loop([agent], env, max_episodes=800)
            for i in [0]:
                print('probability_of_random_action:', i)
                agent.set_should_learn(False)
                agent.set_probability_of_random_action(i)
                run_loop.run_loop([agent], env, max_episodes=100)
    except KeyboardInterrupt:
        pass
