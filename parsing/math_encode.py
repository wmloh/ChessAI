import numpy as np
from parsing.parsing_constant import *

ASCII_a = ord('a')


def tensor_encode(board, flip=False):
    '''
    Encodes the board state into a np.ndarray

    A simple check for formatting is:
    * assert (tensor.sum(axis=2) == np.ones((8,8)).all()
    * assert tensor.shape == (8,8,13)

    :param board: chess.Board - board to be parsed
    :param flip: bool - flip such that the Black and White swaps and the pieces are swapped too
    :return: np.ndarray
    '''
    state = board.epd().split(' ', 1)[0]
    if flip:
        mapper = FLIP_MAP_SYMBOL
    else:
        mapper = MAP_SYMBOL

    tensor_list = list()
    row = list()
    for symbol in state:
        if symbol == '/':  # next row
            tensor_list.append(row)
            row = list()
        elif symbol.isdigit():  # implicit empty symbols
            row.extend([EMPTY] * int(symbol))
        else:  # pieces
            row.append(mapper[symbol])
    tensor_list.append(row)

    if flip:
        tensor_list = list(reversed(tensor_list))

    return np.array(tensor_list, dtype=np.int8)


def tensor_decode(tensor):
    pass


def get_action_tensor(uci_str, flip=False):
    '''
    Converts a string of a UCI move object into a bitmap with 2-dimensions
        representing which piece to move and where to move it to respectively

    :param uci_str: str - string obtained from chess.Move.uci()
    :param flip: bool - flip such that the Black and White swaps and the pieces are swapped too
    :return: np.ndarray - tensor with shape (8, 8, 2) representing action
    '''
    action_tensor = np.zeros((8, 8, 2), dtype=np.int8)
    origin, dest = uci_str[:2], uci_str[2:]

    if flip:
        action_tensor[int(origin[1])-1, ord(origin[0]) - ASCII_a, 0] = 1
        action_tensor[int(dest[1])-1, ord(dest[0]) - ASCII_a, 1] = 1
    else:
        action_tensor[8 - int(origin[1]), ord(origin[0]) - ASCII_a, 0] = 1
        action_tensor[8 - int(dest[1]), ord(dest[0]) - ASCII_a, 1] = 1

    return action_tensor
