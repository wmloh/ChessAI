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


def tensor_decode(state):
    '''
    Decodes the board state into an np.ndarray of characters

    :param state: np.ndarray
    :return: np.ndarray
    '''
    rows = list()
    for tensor_row in state:
        decoded_row = list()
        for piece in tensor_row:
            decoded_row.append(INVERSE_MAP_SYMBOL[tuple(piece)])
        rows.append(decoded_row)
    return np.array(rows)


def tensor_decode_fen(state):
    '''
    Decodes the board state into a FEN formatted string

    :param state: np.ndarray
    :return: string
    '''
    fen = str()
    for tensor_row in state:
        spaces = 0
        for piece in tensor_row:
            piece_char = INVERSE_MAP_SYMBOL[tuple(piece)]
            if piece_char == '.':
                spaces += 1
            elif spaces > 0:
                fen += str(spaces)
                spaces = 0
            if piece_char.isalpha():
                fen += piece_char
        if spaces > 0:
            fen += str(spaces)
        fen += '/'
    fen = fen[:-1]
    # add default info to the end to complete FEN string
    fen += FEN_DEFAULT
    return fen


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
        action_tensor[int(origin[1]) - 1, ord(origin[0]) - ASCII_a, 0] = 1
        action_tensor[int(dest[1]) - 1, ord(dest[0]) - ASCII_a, 1] = 1
    else:
        action_tensor[8 - int(origin[1]), ord(origin[0]) - ASCII_a, 0] = 1
        action_tensor[8 - int(dest[1]), ord(dest[0]) - ASCII_a, 1] = 1

    return action_tensor
