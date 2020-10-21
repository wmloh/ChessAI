import numpy as np
from parsing.math_encode import tensor_encode, get_action_tensor
from parsing.parsing_constant import *
from parsing.data_generation import parse_pgn, sample_intermediate_states, generate_dataset
from parsing.data_generation import save_tensor_data, save_labels_data

if __name__ == "__main__":
    np.random.seed(486)

    all_games = parse_pgn('../data/sample_data.pgn', LIMIT=2)
    game_1 = all_games[0]
    tensor_1 = tensor_encode(game_1.board())

    STARTING_STATE = \
        [[ROOK_B, KNHT_B, BISH_B, QUEN_B, KING_B, BISH_B, KNHT_B, ROOK_B],
         [PAWN_B, PAWN_B, PAWN_B, PAWN_B, PAWN_B, PAWN_B, PAWN_B, PAWN_B],
         [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
         [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
         [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
         [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
         [PAWN_W, PAWN_W, PAWN_W, PAWN_W, PAWN_W, PAWN_W, PAWN_W, PAWN_W],
         [ROOK_W, KNHT_W, BISH_W, QUEN_W, KING_W, BISH_W, KNHT_W, ROOK_W]]

    assert tensor_1.shape == (8, 8, 13)
    assert (tensor_1 == STARTING_STATE).all()

    # save_tensor_data(tensors)
    # save_labels_data(labels)

    states, labels, actions = generate_dataset('../data/sample_data.pgn', LIMIT=3)

    assert states.shape[1:] == (8, 8, 13)
    assert actions.shape[1:] == (8, 8, 2)
