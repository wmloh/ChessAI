import chess
import chess.pgn
import numpy as np
import os
import warnings
import pandas as pd
from joblib import dump
from datetime import datetime
# from scipy.stats import expon
from parsing.math_encode import tensor_encode, get_action_tensor
from parsing.parsing_constant import WHITE_WIN, STATE_DEPTH, ACTION_DEPTH, BOTH_DRAW

PATH_TO_DATA_DIR = os.path.join('..', 'data')
DEFAULT_TENSOR_NAME = "tensor_"
STATE_SUFFIX = "_state"
ACTION_SUFFIX = "_action"
DEFAULT_LABELS_NAME = "labels_"

# obtained from "A Chess Composer of Two-Move Mate Problems"
# AVERAGE_BRANCHING_FACTOR = 33
SAMPLING_THRESHOLD = 6
INITIAL_STATE_PROB = 0.001325  # 0.0009696
INITIAL_SAMPLING_RATE = np.array([INITIAL_STATE_PROB * 2 ** i
                                  for i in range(SAMPLING_THRESHOLD)])


# INITIAL_SAMPLING_RATE = expon.pdf(range(1, SAMPLING_THRESHOLD + 1),
#                                   scale=1 / np.log(AVERAGE_BRANCHING_FACTOR))
# INITIAL_SAMPLING_RATE = INITIAL_SAMPLING_RATE / INITIAL_SAMPLING_RATE.sum()


def sample_intermediate_states(game, states=None, labels=None, sources=None, targets=None,
                               sampling_rate=0.1, initial_sampling_rate=INITIAL_SAMPLING_RATE):
    '''
    For a particular game, converts each board state into Tensors where the first
        n=len(initial_sampling_rate) is randomly sampled depending on the probability and
        the rest are sampled with probability sampling_rate.

    Returns the states, labels (0 or 1) and actions.

    Note: Draws will not be distinguished from losses!

    :param game: pgn.Game - game to be sampled
    :param states: None/list - list of states to be appended to (if None, it will return a new list)
    :param labels: None/list - list of labels to be appended to (if None, it will return a new list)
    :param sources: None/list - list of source policy to be appended to (if None, it will return a new list)
    :param targets: None/list - list of targets policy to be appended to (if None, it will return a new list)
    :param sampling_rate: float - probability of each state being sampled
    :param initial_sampling_rate: np.array(float) - probabilities of initial states being sampled
    :return: tuple(list(Tensor), list(int), list(Tensor))
    '''
    current_board = chess.Board()

    # sampled probabilities (initial and regular)
    initial_sample_array = np.random.binomial(1, p=initial_sampling_rate,
                                              size=(len(initial_sampling_rate, )))
    sample_array = np.random.binomial(1, p=sampling_rate, size=(100,))

    if states is None or labels is None or sources is None or targets is None:
        states = list()
        labels = list()
        sources = list()
        targets = list()

    outcome = game.headers['Result']
    outcome = 1 if outcome == WHITE_WIN else 0  # no draws

    prob_count = -1
    mirror = False  # since White always starts first
    current_sampling_arr = initial_sample_array

    for move in game.mainline_moves():
        prob_count += 1

        # restarts cycle if exceeded and use the usual sampling array
        if prob_count >= len(current_sampling_arr):
            prob_count = 0
            current_sampling_arr = sample_array

        if current_sampling_arr[prob_count]:  # saves to dataset only if 1
            states.append(tensor_encode(current_board, mirror=mirror))
            src, tgt = get_action_tensor(move.uci(), mirror=mirror)
            sources.append(src)
            targets.append(tgt)
            labels.append(outcome)

        current_board.push(move)
        mirror = not mirror  # inverts mirror
        outcome = 1 - outcome  # alternating pattern of win/lose

    return states, labels, sources, targets


def generate_dataset(data_path, LIMIT=-1, sampling_rate=0.1):
    '''
    Samples and generates the dataset as a Numpy ndarray which includes
        states, labels and actions (source and target policy).

    Note: Skips games that results in draws

    :param data_path: str - path to the PGN file
    :param LIMIT: -1/int - number of games to store (-1 means all)
    :param sampling_rate: float - probability of each state being sampled
    :return: tuple(np.ndarray, np.array, np.ndarray, np.ndarray)
    '''

    all_states = list()
    all_labels = list()
    all_sources = list()
    all_targets = list()

    with open(data_path, 'r') as f:
        game = chess.pgn.read_game(f)

        count = 1
        while (count <= LIMIT or LIMIT == -1) and game is not None:
            if game.headers['Result'] != BOTH_DRAW:  # skips draws
                sample_intermediate_states(game,
                                           states=all_states,
                                           labels=all_labels,
                                           sources=all_sources,
                                           targets=all_targets,
                                           sampling_rate=sampling_rate)
                count += 1
            game = chess.pgn.read_game(f)

    return np.array(all_states, dtype=np.int8), \
           np.array(all_labels, dtype=np.int8), \
           np.array(all_sources, dtype=np.int8), \
           np.array(all_targets, dtype=np.int8)


def parse_pgn(data_path, LIMIT=-1):
    '''
    Collects all games in the specified file into a list

    Primarily used for debugging because this is not memory efficient

    :param data_path: str - path to the PGN file
    :param LIMIT: -1/int - number of games to store (-1 means all)
    :return: list(chess.Game)
    '''
    games = list()

    with open(data_path, 'r') as f:
        game = chess.pgn.read_game(f)
        count = 1
        while (count <= LIMIT or LIMIT == -1) and game is not None:
            games.append(game)
            game = chess.pgn.read_game(f)
            count += 1

    return games


def save_tensor_data(data, suffix=None, use_default_name=True, custom_name=None):
    '''
    Saves tensors as a Joblib file (without compression)

    :param data: np.ndarray - Tensor to be saved
    :param suffix: None/str - string to be appended to default file name (if use_default_name=False)
    :param use_default_name: bool - uses autogenerated unique name
    :param custom_name: None/str - name of file (only if use_default_name = False)
    :return: None
    '''
    if use_default_name:
        current_date = datetime.now().date().strftime('%d%m')
        file_path = os.path.join(PATH_TO_DATA_DIR, DEFAULT_TENSOR_NAME + current_date)
        if suffix is None:
            if data.shape[-1] == STATE_DEPTH:  # board state
                file_path += STATE_SUFFIX
            elif data.shape[-1] == ACTION_DEPTH:  # action
                file_path += ACTION_SUFFIX
            else:
                warnings.warn(f'Tensor with shape {data.shape} is neither state nor action')
        elif isinstance(suffix, str):
            file_path += suffix
        else:
            raise TypeError(f'Expected suffix to be of type str but given {type(suffix)} instead')

    else:  # use custom_name
        if not isinstance(custom_name, str):
            raise TypeError('The custom_name parameter must be a string')
        file_path = custom_name

    if not os.path.exists(file_path):  # simple case; just dump there
        dump(data, file_path + '.joblib')
    else:  # not-so-simple case, append a unique integer until it can be saved without overwriting
        if not use_default_name:
            raise IOError('A file with the given file name already exists')

        count = 2
        while os.path.exists(file_path + f'_{count}.joblib'):
            count += 1
        dump(data, file_path + f'_{count}.joblib')


def save_labels_data(data, use_default_name=True, custom_name=None):
    '''
    Saves labels as a CSV file

    :param data: np.array - Labels to be saved
    :param use_default_name: bool - uses autogenerated unique name
    :param custom_name: None/str - name of file (only if use_default_name = False)
    :return: None
    '''
    if use_default_name:
        current_date = datetime.now().date().strftime('%d%m')
        file_path = os.path.join(PATH_TO_DATA_DIR, DEFAULT_LABELS_NAME + current_date)
    else:  # use custom_name
        if not isinstance(custom_name, str):
            raise TypeError('The custom_name parameter must be a string')
        file_path = custom_name

    df = pd.DataFrame(data={'labels': data})
    if not os.path.exists(file_path):  # simple case; just dump there
        df.to_csv(file_path + '.csv', index=False)

    else:  # not-so-simple case, append a unique integer until it can be saved without overwriting
        if not use_default_name:
            raise IOError('A file with the given file name already exists')

        count = 2
        while os.path.exists(file_path + f'_{count}.csv'):
            count += 1
        df.to_csv(file_path + f'_{count}.csv', index=False)
