from chess import LegalMoveGenerator, PAWN

from inference.inference_constant import *
from parsing.math_encode import *


def sorted_probs(policy_inference):
    '''
    Returns a sorted list of values corresponding to the order of probabilities of the policy_inference ndarray in
        descending order.
    Values are valued from 0-63 in a 1-dimensional array where the ith element corresponds with the
        (i // BOARD_DIM[0])th row and the (i % BOARD_DIM[1])th column

    Useful assertion for debugging:
        assert policy_inference.shape == BOARD_DIM

    :param policy_inference: np.ndarray - policy inference to sort probabilities of
    :return: list
    '''
    return policy_inference.reshape(1, BOARD_SQUARES).argsort()[0][::-1]


def uci_square_to_coord(uci):
    '''
    Returns the coordinate representation of the square represented by uci

    :param uci: str - the UCI representation of a square (e.g. a4, c5, h3)
    :return: int
    '''
    if uci[0].isupper():
        col = ord(uci[0]) - ord('A')
    else:
        col = ord(uci[0]) - ord('a')
    row = BOARD_DIM[0] - int(uci[1])
    return row * BOARD_DIM[0] + col


def coord_to_uci_square(coord):
    '''
    Returns the UCI representation of the square represented by coord

    :param coord: int - the coordinate representation of a square if the board were 1-dimensional with a8 being 0 and h1
                            being 63
    :return: str
    '''
    row = coord // BOARD_DIM[0]
    col = coord % BOARD_DIM[1]
    uci_row = BOARD_DIM[0] - row
    uci_col = chr(ord('a') + col)
    return uci_col + str(uci_row)


def coords_to_uci_move(source, target, promote):
    '''
    Returns the UCI representation of the move from source coord to target coord. Whenever a promote is true, we promote
        to a queen.

    :param source: int - the coordinate representation of a square if the board were 1-dimensional with a8 being 0 and
                            h1 being 63
    :param target: int - the coordinate representation of a square if the board were 1-dimensional with a8 being 0 and
                            h1 being 63
    :param promote: bool - whether or not a pawn is being promoted
    :return: str
    '''
    uci_source = coord_to_uci_square(source)
    uci_target = coord_to_uci_square(target)
    if promote:
        uci_target += 'q'
    return uci_source + uci_target


def coords_to_chess_square(coord):
    '''
    Returns the python-chess square name of the given co-ordinate.
        Squares in python-chess are represented as integers from 0 - 63, with 0 being A1, 1 being B1, and 63 being H8

    :param coord: int - the coordinate representation of a square
    :return: int
    '''
    row = coord // BOARD_DIM[0]
    col = coord % BOARD_DIM[1]
    # flip the board on the horizontal axis
    if row < BOARD_DIM[0] / 2:
        row = BOARD_DIM[0] - 1 - row
    else:
        row -= BOARD_DIM[0] / 2
        row = BOARD_DIM[0] / 2 - row - 1
    return int(row * BOARD_DIM[0] + col)


def get_action(board, sources, targets):
    '''
    Returns the UCI string notation of the action to take given policy network action
        This is done by:
            * considering all valid moves
            * selecting all moves with a source square with highest probability
            * selecting the target with maximum probability from this subset

    :param sources: - np.ndarray - action policy from PolicyModel.infer
    :param targets - np.ndarray - source policy from PolicyModel.infer
    :param board: chess.Board - current game board
    :return: str
    '''
    sorted_sources = sorted_probs(sources)
    legal_moves = dict()
    for move in LegalMoveGenerator(board):
        uci = move.uci()
        source = uci_square_to_coord(uci[0:2])
        target = uci_square_to_coord(uci[2:4])
        if source not in legal_moves:
            legal_moves[source] = set()
        legal_moves[source].add(target)

    for source in sorted_sources:
        if source in legal_moves:
            suggested_actions = list()
            for target in legal_moves[source]:
                suggested_action = (target, targets.reshape(64, 1)[target][0])
                suggested_actions.append(suggested_action)
            if suggested_actions:
                suggested_actions.sort(key=lambda action: action[1], reverse=True)
                suggested_target = suggested_actions[0]

                # Check for promotion
                promote = False
                square_name = coords_to_chess_square(source)
                if board.piece_type_at(square_name) == PAWN \
                        and suggested_target[0] < BOARD_DIM[0]:
                    promote = True

                return coords_to_uci_move(source, suggested_target[0], promote)
    # No suitable move was found (should not reach here)
    assert False
