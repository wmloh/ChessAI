import numpy as np
from parsing.parsing_constant import *

ASCII_a = ord('a')


def tensor_encode(board):
    '''
    Encodes the board state into a np.ndarray

    A simple check for formatting is:
    * assert (tensor.sum(axis=2) == np.ones((8,8)).all()
    * assert tensor.shape == (8,8,13)

    :param board: chess.Board - board to be parsed
    :return: np.ndarray
    '''
    state = board.epd().split(' ', 1)[0]

    tensor_list = list()
    row = list()
    for symbol in state:
        if symbol == '/':  # next row
            tensor_list.append(row)
            row = list()
        elif symbol.isdigit():  # implicit empty symbols
            row.extend([EMPTY] * int(symbol))
        else:  # pieces
            row.append(MAP_SYMBOL[symbol])
    tensor_list.append(row)

    return np.asarray(tensor_list, dtype=np.int8)


def tensor_decode(tensor):
    pass


def get_action_tensor(uci_str):
    '''
    Converts a string of a UCI move object into a bitmap with 2-dimensions
        representing which piece to move and where to move it to respectively

    :param uci_str: str - string obtained from chess.Move.uci()
    :return: np.ndarray - tensor with shape (8, 8, 2) representing action
    '''
    action_tensor = np.zeros((8, 8, 2), dtype=np.int8)
    origin, dest = uci_str[:2], uci_str[2:]

    action_tensor[8 - int(origin[1]), ord(origin[0]) - ASCII_a, 0] = 1
    action_tensor[8 - int(dest[1]), ord(dest[0]) - ASCII_a, 1] = 1

    return action_tensor
