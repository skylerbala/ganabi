from utils import parse_args
import gin
from subprocess import call
import pickle
import random
import numpy as np


@gin.configurable
class Dataset(object):
    def __init__(self,
                 game_type='Hanabi-Full',
                 num_players=2,
                 num_unique_agents=6,
                 num_games=150):
        self.game_type = game_type
        self.num_players = num_players
        self.num_unique_agents = num_unique_agents
        self.num_games = num_games
        self.current_index = 0

        self.train_data = {}  # gameplay data given to model
        self.validation_data = {}  # gameplay data not given to model, from same agents as train
        self.test_data = {}  # gameplay data from agents totally unseen to model

    def read(self, raw_data):
        # split up raw_data into train, validation, and test
        test_agent = random.choice(list(raw_data.keys()))

        for agent in raw_data:
            if agent == test_agent and len(raw_data.keys()) != 1:
                continue
            split_idx = int(0.5 * len(raw_data[agent]))
            self.train_data[agent] = raw_data[agent][:10]
            self.validation_data[agent] = raw_data[agent][10: 15]
            self.test_data[agent] = raw_data[agent][15: 20]
        # self.test_data[test_agent] = raw_data[test_agent]

    def naive_generator(self, batch_size, batch_type='train'):
        '''
        10*moves = samples
        batch_size = 10
        steps = 100
        1 epoch = 100 steps
        :param batch_size:
        :param batch_type:
        :return:
        '''

        while True:
            observations, actions = [], []

            if batch_type == 'train':
                data_bank = self.train_data
            elif batch_type == 'validation':
                data_bank = self.validation_data
            elif batch_type == 'test':
                data_bank = self.test_data

            agent = random.choice(list(data_bank))


            '''
            Basic Neural Network Sample
            Return: Sample of (batch_size) pairs of randomly selected
            observation-action pairs
            '''

            '''
            for _ in range(batch_size):
                game = random.choice(data_bank[agent])
                move = random.randint(0, len(game[0]) - 1)
                observations.append(game[0][move])
                actions.append(game[1][move])

            x = np.array(observations)
            y = np.array(actions)
            '''

            '''
            Recurrent Neural Network Sample
            Return: Sample of (batch_size) pairs of 3 consecutive moves
            from player X (i.e., 6 total moves passed when including player Y)
            '''
            steps = 6

            x = [[] for _ in range(batch_size)]
            y = []

            for i in range(batch_size):
                game = random.choice(data_bank[agent])
                self.current_index = random.randint(0, len(game[0]) - 1)
                buffer = random.randint(0, 1)

                if self.current_index + steps + 1 >= len(game[0]):
                    self.current_index = 0
                start = self.current_index + buffer
                end = self.current_index + buffer + steps

                for step in range(start, end, 2):
                    x[i].append(game[0][step])
                if start == 1:
                    y.append(game[1][self.current_index + steps - 1])
                else:
                    y.append(game[1][self.current_index + steps - 2])
                # x[i, :] = game[0][self.current_index: self.current_index + steps]
                # y[i] = game[1][self.current_index + steps - 1]

                self.current_index += steps

            x = np.array(x)
            y = np.array(y)

            yield x, y

        def generator(self, batch_type='train'):
            pass
            '''
            if batch_type == 'train':
                data_bank = self.train_data
            elif batch_type == 'validation':
                data_bank = self.validation_data
            elif batch_type == 'test':
                data_bank = self.test_data
    
            # data_bank: [AgentName][num_games][0 = 
            #         obs_vec, 1 = act_vec][game_step][index into vec]
            agent = random.choice(data_bank.keys())
            adhoc_games = [random.choice(list(data_bank[agent]))
                           for _ in range(NUM_ADHOC_GAMES)]
            game_lengths = [len(game[0]) for game in adhoc_games]
    
            # adhoc_games: [-->[[obs_act_vec],[obs_act_vec],...]<--game1, 
            #               -->[[obs_act_vec],[obs_act_vec],...]<--game2...]
            adhoc_games = [[adhoc_games[i][0][l] + adhoc_games[i][1][l]
                            for l in range(game_lengths[i])]
                           for i in range(NUM_ADHOC_GAMES)]
    
            # assemble generated agent observations and target actions
            agent_obs, agent_act = [], []
            for i in range(NUM_AGENT_OBS):
                game = random.choice(list(data_bank[agent]))
                step_num = random.randint(0, len(game[0]) - 1)
                agent_obs.append(game[0][step_num])
                agent_act.append(game[1][step_num])
    
            # convert nested uneven list of adhoc games into padded numpy array
            np_adhoc_games = np.zeros(shape=(NUM_ADHOC_GAMES, MAX_GAME_LEN, OBS_ACT_VEC_LEN))
            game_lengths = np.array(game_lengths)
            for game_num, game_len in enumerate(game_lengths):
                np_adhoc_games[game_num, :game_lengths[game_num], :] = \
                    np.asarray(adhoc_games[game_num])
    
            agent_obs = np.array(agent_obs)
            agent_act = np.array(agent_act)
    
            # FIXME: needs to return same_act
            return ([np_adhoc_games, game_lengths, agent_obs], agent_act)
            '''


def main(args):
    data = Dataset()
    args = parse_args.resolve_datapath(args,
                                       data.game_type,
                                       data.num_players,
                                       data.num_unique_agents,
                                       data.num_games)

    try:
        raw_data = pickle.load(open(args.datapath, "rb"), encoding='latin1')

    except IOError:
        call("python create_data.py --datapath " + args.datapath, shell=True)
        raw_data = pickle.load(open(args.datapath, "rb"), encoding='latin1')

    data.read(raw_data)

    return data


if __name__ == '__main__':
    args = parse_args.parse()
    args = parse_args.resolve_configpath(args)

    gin.parse_config_file(args.configpath)

    main(args)
