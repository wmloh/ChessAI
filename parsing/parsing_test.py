import unittest
import numpy as np

from parsing.math_encode import tensor_encode, tensor_decode, get_action_tensor, tensor_decode_fen
from parsing.parsing_constant import *
from parsing.data_generation import parse_pgn, sample_intermediate_states, generate_dataset
from parsing.data_generation import save_tensor_data, save_labels_data


class TestParsing(unittest.TestCase):
    STARTING_STATE = \
        [[ROOK_B, KNHT_B, BISH_B, QUEN_B, KING_B, BISH_B, KNHT_B, ROOK_B],
         [PAWN_B, PAWN_B, PAWN_B, PAWN_B, PAWN_B, PAWN_B, PAWN_B, PAWN_B],
         [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
         [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
         [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
         [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
         [PAWN_W, PAWN_W, PAWN_W, PAWN_W, PAWN_W, PAWN_W, PAWN_W, PAWN_W],
         [ROOK_W, KNHT_W, BISH_W, QUEN_W, KING_W, BISH_W, KNHT_W, ROOK_W]]

    def test_parses_game(self):
        np.random.seed(486)

        all_games = parse_pgn('../data/sample_data.pgn', LIMIT=2)
        game_1 = all_games[0]
        tensor_1 = tensor_encode(game_1.board())

        self.assertEqual(tensor_1.shape, (8, 8, 13))
        self.assertTrue((tensor_1 == self.STARTING_STATE).all())

        states, labels, sources, targets = generate_dataset('../data/sample_data.pgn', LIMIT=3)

        self.assertEqual(states.shape[1:], (8, 8, 13))
        self.assertEqual(sources.shape[1:], (8, 8))
        self.assertEqual(targets.shape[1:], (8, 8))

        test_state = states[34, ...]

        EXPECTED_FLIP = [[EMPTY, EMPTY, EMPTY, BISH_W, EMPTY, EMPTY, EMPTY, EMPTY],
                         [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, PAWN_W, EMPTY, PAWN_W],
                         [EMPTY, EMPTY, EMPTY, EMPTY, PAWN_W, EMPTY, PAWN_W, PAWN_B],
                         [EMPTY, EMPTY, EMPTY, EMPTY, KING_W, EMPTY, PAWN_B, EMPTY],
                         [BISH_B, EMPTY, EMPTY, PAWN_W, EMPTY, EMPTY, EMPTY, EMPTY],
                         [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, ROOK_B, EMPTY],
                         [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, KING_B, EMPTY],
                         [EMPTY, EMPTY, EMPTY, EMPTY, ROOK_W, EMPTY, EMPTY, EMPTY]]

        self.assertTrue((np.rot90(test_state, 2) == EXPECTED_FLIP).all())



    def test_decodes_state(self):
        test_tensor = np.array(self.STARTING_STATE)
        expected_board = \
            [['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
             ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
             ['.', '.', '.', '.', '.', '.', '.', '.'],
             ['.', '.', '.', '.', '.', '.', '.', '.'],
             ['.', '.', '.', '.', '.', '.', '.', '.'],
             ['.', '.', '.', '.', '.', '.', '.', '.'],
             ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
             ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']]
        actual_board = tensor_decode(test_tensor).tolist()
        self.assertEqual(expected_board, actual_board)

    def test_decodes_state_fen(self):
        test_tensor_1 = np.array(self.STARTING_STATE)
        fen1 = tensor_decode_fen(test_tensor_1)
        self.assertEqual('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0', fen1)

        test_state = \
            [[ROOK_B, KNHT_B, EMPTY, QUEN_B, EMPTY, BISH_B, KNHT_B, ROOK_B],
             [PAWN_B, PAWN_B, PAWN_B, PAWN_B, PAWN_B, PAWN_B, PAWN_B, PAWN_B],
             [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
             [EMPTY, EMPTY, EMPTY, PAWN_B, EMPTY, PAWN_W, EMPTY, EMPTY],
             [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
             [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, KNHT_W],
             [PAWN_W, PAWN_W, PAWN_W, PAWN_W, PAWN_W, PAWN_W, PAWN_W, PAWN_W],
             [ROOK_W, KNHT_W, BISH_W, QUEN_W, KING_W, BISH_W, EMPTY, EMPTY]]
        test_tensor_2 = np.array(test_state)
        fen2 = tensor_decode_fen(test_tensor_2)
        self.assertEqual('rn1q1bnr/pppppppp/8/3p1P2/8/7N/PPPPPPPP/RNBQKB2 w KQkq - 0 0', fen2)


if __name__ == "__main__":
    unittest.main()
