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

        self.train_data = {}  # gameplay data given to model
        self.validation_data = {}  # gameplay data not given to model, from same agents as train
        self.test_data = {}  # gameplay data from agents totally unseen to model

    def read(self, raw_data):
        # split up raw_data into train, validation, and test
        test_agent = random.choice(list(raw_data.keys()))

        for agent in raw_data:
            if agent == test_agent:
                continue
            split_idx = int(0.75 * len(raw_data[agent]))
            self.train_data[agent] = raw_data[agent][:split_idx]
            self.validation_data[agent] = raw_data[agent][split_idx:]

        self.test_data[test_agent] = raw_data[test_agent]

    def naive_generator(self, batch_size, batch_type='train'):
        observations, actions = [], []

        if batch_type == 'train':
            data_bank = self.train_data
        elif batch_type == 'validation':
            data_bank = self.validation_data
        elif batch_type == 'test':
            data_bank = self.test_data

        agent = list(data_bank)[0]
        games = random.sample(data_bank[agent], batch_size)
        game_lengths = [len(game[0]) for game in games]

        final_games = []

        for i in range(len(games)):
            for i_len in range(game_lengths[i]):
                final_games.append(games[i][0][i_len] + games[i][1][i_len])

        for i in range(len(games)):
            rand_game = random.choice(list(data_bank[agent]))
            step_num = random.randint(0, len(rand_game[0]) - 1)
            observations.append(rand_game[0][step_num])
            actions.append(rand_game[1][step_num])

        observations = np.array(observations)
        actions = np.array(actions)
        game_lengths = np.array(game_lengths)

        obs_act_len = np.shape(observations)[1] + np.shape(actions)[1] # Add lens of observations and action vectors
        max_game_len = len(max(data_bank[agent], key=len))

        # convert nested uneven list of adhoc games into padded numpy array
        games = np.zeros(shape=(len(games), max_game_len, obs_act_len))
        for game_index in range(len(game_lengths)):
            games[game_index, :game_lengths[game_index], :] = \
                np.asarray(games[game_index])

        return [observations, actions], actions

    '''
    def generator(self, batch_type='train'):
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
