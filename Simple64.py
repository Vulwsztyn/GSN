from collections import deque

from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
import numpy as np
import sys
from absl import flags
from pysc2.lib import actions, features, units
import random
import os
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout, Activation, Convolution2D, Flatten
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import LeakyReLU

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb

import pickle

reward_list = []

BUILD_UNIT = 0.2
SENT_ATTACK = 0.4
IDLE_FREED = 0.05
NO_QUEUE = 0
QUEUED = 0.1


class QNetwork():

    def __init__(self,
                 legal_actions,
                 state_size,
                 gamma=0.9,
                 alpha=0.0005,
                 **args):

        self.state_size = state_size
        self.legal_actions = legal_actions
        self.gamma = gamma
        self.alpha = alpha
        self.lastAction = None

        sgd = SGD(lr=self.alpha)

        if not os.path.isfile('network.json'):
            input_size = self.state_size
            print('Initializing network ...')
            self.network = Sequential()
            self.network.add(
                Dense(input_size, input_dim=input_size, kernel_initializer="uniform", activation='relu'))
            self.network.add(
                Dense(input_size * 2, kernel_initializer="uniform", activation='relu'))
            self.network.add(
                Dense(input_size * 1, kernel_initializer="uniform", activation='relu'))
            self.network.add(Dense(len(self.legal_actions), kernel_initializer="uniform", activation='tanh'))

            self.network.compile(loss='mean_squared_error', optimizer=sgd)

            json_string = self.network.to_json()
            open('network.json', 'w').write(json_string)
            self.network.save_weights('weights.h5')

        else:
            print('Loading network ...')
            self.network = model_from_json(open('network.json').read())
            self.network.load_weights('weights.h5')
            self.network.compile(loss='mean_squared_error', optimizer=sgd)
        self.target_network = self.network
        print('Network is ready to use.')

    def learn(self, replay_memory):

        learning_rate = 0.7
        discount_factor = self.gamma
        MIN_REPLAY_SIZE = 200

        if len(replay_memory) < MIN_REPLAY_SIZE:
            return

        print("Getting smarter...")
        batch_size = 64 * 2
        mini_batch = random.sample(replay_memory, batch_size)
        current_states = np.array([transition[0] for transition in mini_batch])
        current_qs_list = self.network.predict(current_states)

        new_current_states = np.array([transition[3] for transition in mini_batch])
        future_qs_list = self.target_network.predict(new_current_states)

        X = []
        Y = []

        for index, (state, action, reward, new_state) in enumerate(mini_batch):
            if new_state != 'terminal':
                max_future_q = reward + discount_factor * np.max(future_qs_list[index])
            else:
                max_future_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

            X.append(state)
            Y.append(current_qs)
        self.network.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)

    def update_target(self):
        self.target_network.set_weights(self.network.get_weights())

    def predict(self, state):
        return self.network.predict(state)

    def save(self):
        self.network.save_weights('weights.h5')

class Agent(base_agent.BaseAgent):
    actions = ["do_nothing",
               "harvest_minerals",
               "build_supply_depot",
               "build_barracks",
               "train_marine",
               "attack"]

    def getActions(self):
        return self.actions

    def get_legal_actions(self, state):
        legal_actions = [1, 0, 0, 0, 0, 0]

        legal_action_numbers = [0]
        for i in self.actions:
            if i == "harvest_minerals":
                if state[0] > 0:
                    legal_actions[1] = 1
                    legal_action_numbers.append(1)
            if i == "build_supply_depot":
                if state[2] < 5:
                    legal_actions[2] = 1
                    legal_action_numbers.append(2)
            if i == "build_barracks":
                if state[3] > 0 and state[4] < 9:
                    legal_actions[3] = 1
                    legal_action_numbers.append(3)
            if i == "train_marine":
                if state[5] > 0:
                    legal_actions[4] = 1
                    legal_action_numbers.append(4)
            if i == "attack":
                if state[1]:
                    legal_actions[5] = 1
                    legal_action_numbers.append(5)
        return legal_actions, legal_action_numbers

    def get_state(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        idle_marines = [marine for marine in marines if marine.order_length == 0]

        queued_marines = np.sum([barracks.order_length for barracks in completed_barrackses])

        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        can_afford_supply_depot = obs.observation.player.minerals >= 100
        can_afford_barracks = obs.observation.player.minerals >= 150
        can_afford_marine = obs.observation.player.minerals >= 50

        print(
            'Idle SCVs: ', len(idle_scvs),
            ' Idle Marines: ', len(idle_marines),
            ' Depots: ', len(supply_depots),
            ' Completed Depots: ', len(completed_supply_depots),
            ' Barracks: ', len(barrackses),
            ' Completed Barracks: ', len(completed_barrackses),
            ' Marines: ', len(marines),
            ' Queued Marines: ', queued_marines,
            ' Free supply: ', free_supply,
            ' Affort Supply: ', can_afford_supply_depot,
            ' Afford Barracks: ', can_afford_barracks,
            ' Affort Marine: ', can_afford_marine,
        )

        return (len(idle_scvs),
                len(idle_marines),
                len(supply_depots),
                len(completed_supply_depots),
                len(barrackses),
                len(completed_barrackses),
                len(marines),
                queued_marines,
                free_supply,
                can_afford_supply_depot,
                can_afford_barracks,
                can_afford_marine)

    def get_my_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.SELF]

    def get_enemy_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.ENEMY]

    def get_my_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.SELF]

    def get_enemy_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.ENEMY]

    def get_distances(self, obs, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

    def step(self, obs):
        super(Agent, self).step(obs)
        if obs.first():
            command_center = self.get_my_units_by_type(
                obs, units.Terran.CommandCenter)[0]
            self.base_top_left = (command_center.x < 32)

    def do_nothing(self, obs):
        return actions.RAW_FUNCTIONS.no_op()

    def harvest_minerals(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        if len(idle_scvs) > 0:
            mineral_patches = [unit for unit in obs.observation.raw_units
                               if unit.unit_type in [
                                   units.Neutral.BattleStationMineralField,
                                   units.Neutral.BattleStationMineralField750,
                                   units.Neutral.LabMineralField,
                                   units.Neutral.LabMineralField750,
                                   units.Neutral.MineralField,
                                   units.Neutral.MineralField750,
                                   units.Neutral.PurifierMineralField,
                                   units.Neutral.PurifierMineralField750,
                                   units.Neutral.PurifierRichMineralField,
                                   units.Neutral.PurifierRichMineralField750,
                                   units.Neutral.RichMineralField,
                                   units.Neutral.RichMineralField750
                               ]]
            scv = random.choice(idle_scvs)
            distances = self.get_distances(obs, mineral_patches, (scv.x, scv.y))
            mineral_patch = mineral_patches[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
                "now", scv.tag, mineral_patch.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def build_supply_depot(self, obs):
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (obs.observation.player.minerals >= 100 and
                len(scvs) > 0):
            supply_depot_xy = (20 - 2 * len(supply_depots), 28) if self.base_top_left else (
                37 + 2 * len(supply_depots), 40)
            scv = random.choice(scvs)
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
                "now", scv.tag, supply_depot_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_barracks(self, obs):
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(completed_supply_depots) > 0 and
                obs.observation.player.minerals >= 150 and len(scvs) > 0):
            x_offset = 0
            count = len(barrackses)
            if count > 3:
                x_offset = 2
                count = count - 4
            barracks_xy = (22 + x_offset, 21 + count * 2 - x_offset) if self.base_top_left else (
                35 - x_offset, 47 - count * 2 + x_offset)
            scv = random.choice(scvs)
            return actions.RAW_FUNCTIONS.Build_Barracks_pt(
                "now", scv.tag, barracks_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def train_marine(self, obs):
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if (len(completed_barrackses) > 0 and obs.observation.player.minerals >= 100
                and free_supply > 0):
            barracks = random.choice(self.get_my_units_by_type(obs, units.Terran.Barracks))
            if barracks.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def attack(self, obs):
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        idle_marines = [marine for marine in marines if marine.order_length == 0]
        if len(idle_marines) > 0:
            attack_xy = (40, 44) if self.base_top_left else (17, 23)
            distances = self.get_distances(obs, idle_marines, attack_xy)
            marine = idle_marines[np.argmax(distances)]
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", marine.tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()


class TerranAgent(Agent):
    def __init__(self):
        super(TerranAgent, self).__init__()
        self.train = 1

        self.replay_memory = deque(maxlen=50_000)
        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0
        self.marines = 0
        self.idle_marines = 0
        self.idle_scvs = 0
        self.queue = 0
        self.steps_to_update_target_model = 0

        self.reward_list = []

        self.previous_state = None
        self.previous_action = None
        self.base_top_left = None

        self.epsilon = 1
        self.qnetwork = QNetwork(self.actions, 12)
        self.new_game()

    def reset(self):
        super(TerranAgent, self).reset()
        global reward_list
        reward_list.append(np.mean(self.reward_list))
        pickle.dump(reward_list, open("rewards.p", "wb"))
        self.reward_list = []

        self.epsilon = self.epsilon * 0.98
        self.new_game()

    def new_game(self):
        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0
        self.marines = 0
        self.idle_marines = 0
        self.idle_scvs = 0
        self.base_top_left = None
        self.previous_state = None
        self.previous_action = None
        self.qnetwork.save()


    def customReward(self, obs, state, previous_action, previous_state):
        killed_unit_score = obs.observation['score_cumulative'][5]
        killed_building_score = obs.observation['score_cumulative'][6]
        marines = state[6]
        queue = state[7]
        barrackses = state[5]
        idle_marines = previous_state[1]
        idle_scvs = previous_state[0]

        reward = obs.reward
        if reward == 0:
            # if killed_unit_score > self.previous_killed_unit_score:
            #     reward += KILL_UNIT_REWARD
            #
            # if killed_building_score > self.previous_killed_building_score:
            #     reward += KILL_BUILDING_REWARD

            if marines > self.marines:
                reward += BUILD_UNIT

            if previous_action == 'attack' and idle_marines > 0:
                reward += SENT_ATTACK

            if previous_action == 'harvest_minerals' and idle_scvs > 0:
                reward += IDLE_FREED

            if queue == 0 and barrackses > 0:
                reward += NO_QUEUE

            if barrackses > queue > self.queue:
                reward += QUEUED

            self.previous_killed_unit_score = killed_unit_score
            self.previous_killed_building_score = killed_building_score
            self.marines = marines
            self.idle_marines = idle_marines
            self.idle_scvs = idle_scvs
            self.queue = queue

        return reward

    def step(self, obs):
        super(TerranAgent, self).step(obs)

        state = self.get_state(obs)
        predicted = self.qnetwork.predict([state]).flatten()
        predicted = predicted+1

        legal_action_predictions, legal_action_numbers = self.get_legal_actions(state)

        predicted = legal_action_predictions * predicted
        predicted = predicted-1

        action = np.argmax(predicted)
        legal_actions = self.getActions()

        for i in range(len(legal_actions)):
            print('Chances: ', legal_actions[i], predicted[i])

        self.steps_to_update_target_model += 1

        if self.train == 0:
            if 0.2 > random.random():
                action = random.choice(legal_action_numbers)

        if self.train == 1:

            if self.epsilon > random.random():
                action = random.choice(legal_action_numbers)

            if self.previous_action is not None:
                reward = self.customReward(obs, state, legal_actions[self.previous_action], self.previous_state)
                self.reward_list.append(reward)
                self.replay_memory.append([self.previous_state, self.previous_action, reward, state])

                if self.steps_to_update_target_model % 4 == 0:
                    self.qnetwork.learn(self.replay_memory)
                print('Reward: ', reward, ' Action: ', legal_actions[action], )
            self.previous_state = state
            self.previous_action = action
            if self.steps_to_update_target_model >= 100:
                print('Copying main network weights to the target network weights')
                self.qnetwork.update_target()
                self.steps_to_update_target_model = 0
        return getattr(self, legal_actions[action])(obs)


class RandomAgent(Agent):
    def step(self, obs):
        super(RandomAgent, self).step(obs)
        state = self.get_state(obs)
        legal_actions = self.getActions()
        legal_action_predictions, legal_action_numbers = self.get_legal_actions(state)
        action = random.choice(legal_action_numbers)
        return getattr(self, legal_actions[action])(obs)


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)

    agent1 = TerranAgent()
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
                disable_fog=True) as env:
            run_loop.run_loop([agent1,agent2], env, max_episodes=1000)
    except KeyboardInterrupt:
        pass

    # try:
    #     with sc2_env.SC2Env(
    #             map_name="Simple64",
    #             players=[sc2_env.Agent(sc2_env.Race.terran),
    #                      sc2_env.Bot(sc_common.Terran, sc_pb.Easy, sc_pb.RandomBuild)],
    #             agent_interface_format=features.AgentInterfaceFormat(
    #                 action_space=actions.ActionSpace.RAW,
    #                 use_raw_units=True,
    #                 raw_resolution=64,
    #             ),
    #             step_mul=48,
    #             disable_fog=True) as env:
    #         run_loop.run_loop([agent1], env, max_episodes=1000)
    # except KeyboardInterrupt:
    #     pass
