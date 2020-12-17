from pysc2.env import sc2_env, run_loop
import torch
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

actor_path = 'model/actor.pkl'
critic_path = 'model/critic.pkl'

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

BUILD_UNIT = 0.1
KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5


class QNetwork():

    def __init__(self,
                 legal_actions,
                 state_size,
                 gamma=0.9,
                 alpha=0.001,
                 train=1,
                 **args):

        self.state_size = state_size
        self.legal_actions = legal_actions
        self.gamma = gamma
        self.alpha = alpha
        self.experience = {}
        self.experience['positivy'] = []
        self.experience['negativy'] = []
        self.train = train
        self.lastAction = random.choice(['East', 'West', 'North', 'South'])

        if not os.path.isfile('network.json'):
            input_size = len(self.legal_actions) + self.state_size
            print('Initializing network ...')
            self.network = Sequential()
            self.network.add(
                Dense(input_size, input_dim=input_size, kernel_initializer="uniform", activation='relu'))
            self.network.add(
                Dense(input_size * 2, kernel_initializer="uniform", activation='relu'))
            self.network.add(
                Dense(input_size * 1, kernel_initializer="uniform", activation='relu'))
            self.network.add(Dense(1, kernel_initializer="uniform", activation='sigmoid'))

            sgd = SGD(lr=self.alpha)
            self.network.compile(loss='mean_squared_error', optimizer=sgd)

            json_string = self.network.to_json()
            open('network.json', 'w').write(json_string)
            self.network.save_weights('weights.h5')

        else:
            print('Loading network ...')
            self.network = model_from_json(open('network.json').read())
            sgd = SGD(lr=self.alpha, momentum=0.9, nesterov=True)
            self.network.load_weights('weights.h5')
            self.network.compile(loss='mean_squared_error', optimizer=sgd)
        print('Network is ready to use.')

    def getQValue(self, state, action):
        encoded_action = np.array([1 if x == action else 0 for x in self.legal_actions])
        input = np.array([np.concatenate((np.array(state), encoded_action))])
        return self.network.predict(input)[0][0]

    def computeValueFromQValues(self, state):
        return max([self.getQValue(state, x) for x in self.legal_actions])

    def computeActionFromQValues(self, state):
        values = [self.getQValue(state, x) for x in self.legal_actions]
        max_values = [x for x in values if x == max(values)]
        if len(max_values) > 1:
            maxValue = random.choice(max_values)
            for x in max_values:
                if x == self.lastAction:
                    maxValue = x
                    break
        else:
            maxValue = max(values)
        return random.choice([a for a, v in zip(self.legal_actions, values) if v == maxValue])

    def getAction(self, state):
        action = self.computeActionFromQValues(state)
        self.lastAction = action
        return action

    def learn(self, state, action, new_state, reward):
        if new_state != 'terminal':
            q_target = reward + self.gamma * self.computeValueFromQValues(new_state)
        else:
            q_target = reward
        encoded_action = np.array([1 if x == action else 0 for x in self.legal_actions])
        input = np.array([np.concatenate((np.array(state), encoded_action))])
        self.network.fit(input, np.array([q_target]), batch_size=1, verbose=0)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class Agent(base_agent.BaseAgent):
    actions = ["do_nothing",
               "harvest_minerals",
               "build_supply_depot",
               "build_barracks",
               "train_marine",
               "attack"]

    def get_legal_actions:

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
        if (len(supply_depots) == 0 and obs.observation.player.minerals >= 100 and
                len(scvs) > 0):
            supply_depot_xy = (22, 26) if self.base_top_left else (35, 42)
            distances = self.get_distances(obs, scvs, supply_depot_xy)
            scv = scvs[np.argmin(distances)]
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt(
                "now", scv.tag, supply_depot_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def build_barracks(self, obs):
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        if (len(completed_supply_depots) > 0 and len(barrackses) == 0 and
                obs.observation.player.minerals >= 150 and len(scvs) > 0):
            barracks_xy = (22, 21) if self.base_top_left else (35, 45)
            distances = self.get_distances(obs, scvs, barracks_xy)
            scv = scvs[np.argmin(distances)]
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
            barracks = self.get_my_units_by_type(obs, units.Terran.Barracks)[0]
            if barracks.order_length < 5:
                return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag)
        return actions.RAW_FUNCTIONS.no_op()

    def attack(self, obs):
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        if len(marines) > 0:
            attack_xy = (38, 44) if self.base_top_left else (19, 23)
            distances = self.get_distances(obs, marines, attack_xy)
            marine = marines[np.argmax(distances)]
            x_offset = random.randint(-4, 4)
            y_offset = random.randint(-4, 4)
            return actions.RAW_FUNCTIONS.Attack_pt(
                "now", marine.tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
        return actions.RAW_FUNCTIONS.no_op()


class TerranAgent(Agent):
    def __init__(self):
        super(TerranAgent, self).__init__()

        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0
        self.marines = 0

        self.epsilon = 0.9
        self.qnetwork = QNetwork(self.actions, 13)
        self.new_game()

    def reset(self):
        super(TerranAgent, self).reset()
        self.epsilon = self.epsilon * 0.9
        self.new_game()

    def new_game(self):
        self.base_top_left = None
        self.previous_state = None
        self.previous_action = None

    def get_state(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)

        queued_marines = (completed_barrackses[0].order_length
                          if len(completed_barrackses) > 0 else 0)

        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        can_afford_supply_depot = obs.observation.player.minerals >= 100
        can_afford_barracks = obs.observation.player.minerals >= 150
        can_afford_marine = obs.observation.player.minerals >= 100

        return (len(command_centers),
                len(scvs),
                len(idle_scvs),
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

    def customReward(self, obs, state):
        killed_unit_score = obs.observation['score_cumulative'][5]
        killed_building_score = obs.observation['score_cumulative'][6]
        marines = state[7]

        reward = obs.reward
        if reward == 0:
            if killed_unit_score > self.previous_killed_unit_score:
                reward += KILL_UNIT_REWARD

            if killed_building_score > self.previous_killed_building_score:
                reward += KILL_BUILDING_REWARD

            if marines > self.marines:
                reward += BUILD_UNIT

            self.previous_killed_unit_score = killed_unit_score
            self.previous_killed_building_score = killed_building_score
            self.marines = marines

        return reward

    def step(self, obs):
        super(TerranAgent, self).step(obs)
        state = self.get_state(obs)

        reward = self.customReward(obs, state)
        print(reward)

        action = self.qnetwork.getAction(state)
        if self.previous_action is not None:
            self.qnetwork.learn(self.previous_state,
                                self.previous_action,
                                'terminal' if obs.last() else state,
                                reward
                                )
        if self.epsilon > random.random():
            action = random.choice(self.actions)

        self.previous_state = state
        self.previous_action = action
        print(action)
        return getattr(self, action)(obs)


class RandomAgent(Agent):
    def step(self, obs):
        super(RandomAgent, self).step(obs)
        action = random.choice(self.actions)
        return getattr(self, action)(obs)


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
            run_loop.run_loop([agent1, agent2], env, max_episodes=1000)
    except KeyboardInterrupt:
        pass
