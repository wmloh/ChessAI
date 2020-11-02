import unittest

import chess

from inference.infer_action import *


class TestInference(unittest.TestCase):
    def test_sort_probs_sorts_probs(self):
        test_inference = np.random.random(64).reshape(8, 8)
        sorted_test_inference = sorted_probs(test_inference)
        y1 = sorted_test_inference[0] // BOARD_DIM[0]
        x1 = sorted_test_inference[0] % BOARD_DIM[1]
        self.assertEqual(test_inference.reshape(64, 1).max(), test_inference[y1, x1])
        y2 = sorted_test_inference[63] // BOARD_DIM[0]
        x2 = sorted_test_inference[63] % BOARD_DIM[1]
        self.assertEqual(test_inference.reshape(64, 1).min(), test_inference[y2, x2])

    def test_uci_square_to_coord(self):
        uci1 = 'h1'
        coord1 = uci_square_to_coord(uci1)
        self.assertEqual(63, coord1)
        uci2 = 'a8'
        coord2 = uci_square_to_coord(uci2)
        self.assertEqual(0, coord2)
        uci3 = 'h8'
        coord3 = uci_square_to_coord(uci3)
        self.assertEqual(7, coord3)

    def test_coord_to_uci_square(self):
        coord1 = 63
        uci_square1 = coord_to_uci_square(coord1)
        self.assertEqual('h1', uci_square1)
        coord2 = 0
        uci_square2 = coord_to_uci_square(coord2)
        self.assertEqual('a8', uci_square2)
        coord3 = 7
        uci_square3 = coord_to_uci_square(coord3)
        self.assertEqual('h8', uci_square3)

    def test_coords_to_uci_move(self):
        source1 = 63
        target1 = 55
        uci1 = coords_to_uci_move(source1, target1, False)
        self.assertEqual('h1h2', uci1)
        source2 = 0
        target2 = 8
        uci2 = coords_to_uci_move(source2, target2, False)
        self.assertEqual('a8a7', uci2)
        source3 = 9
        target3 = 1
        uci3 = coords_to_uci_move(source3, target3, True)
        self.assertEqual('b7b8q', uci3)

    def test_coords_to_chess_square(self):
        coord1 = 0
        square1 = coords_to_chess_square(coord1)
        self.assertEqual(56, square1)
        coord2 = 7
        square2 = coords_to_chess_square(coord2)
        self.assertEqual(63, square2)
        coord3 = 8
        square3 = coords_to_chess_square(coord3)
        self.assertEqual(48, square3)
        coord4 = 56
        square4 = coords_to_chess_square(coord4)
        self.assertEqual(0, square4)
        coord5 = 63
        square5 = coords_to_chess_square(coord5)
        self.assertEqual(7, square5)
        coord6 = 48
        square6 = coords_to_chess_square(coord6)
        self.assertEqual(8, square6)

    def test_get_action(self):
        test_board = \
            [[EMPTY, EMPTY, ROOK_B, EMPTY, ROOK_B, EMPTY, KING_B, EMPTY], \
             [EMPTY, QUEN_B, EMPTY, EMPTY, BISH_B, PAWN_B, PAWN_B, PAWN_B], \
             [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY], \
             [EMPTY, PAWN_B, PAWN_B, BISH_B, PAWN_B, EMPTY, EMPTY, EMPTY], \
             [EMPTY, KNHT_B, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY], \
             [EMPTY, EMPTY, EMPTY, PAWN_W, EMPTY, KNHT_W, PAWN_W, EMPTY], \
             [EMPTY, PAWN_W, QUEN_W, BISH_W, PAWN_W, PAWN_W, BISH_W, PAWN_W], \
             [ROOK_W, EMPTY, EMPTY, EMPTY, ROOK_W, EMPTY, KING_W, EMPTY]]

        test_source1 = \
            [[0., 0., 0., 0., 0., 0., 0., 0.], \
             [0., 0., 0., 0., 0., 0., 0., 0.], \
             [0., 0., 0., 0., 0., 0., 0., 0.], \
             [0., 0., 0., 0., 0., 0., 0., 0.], \
             [0., 0., 0., 0., 0., 0., 0., 0.], \
             [0., 0., 0., 0.001, 0., 0.009, 0., 0.], \
             [0., 0.023, 0.845, 0.389, 0.008, 0., 0.001, 0.], \
             [0.005, 0., 0., 0., 0.016, 0., 0., 0.]]
        test_target1 = \
            [[0., 0., 0., 0., 0., 0., 0., 0.], \
             [0.004, 0., 0., 0., 0., 0., 0., 0.], \
             [0., 0., 0., 0., 0., 0., 0., 0.], \
             [0.008, 0., 0., 0.001, 0.003, 0., 0.001, 0.], \
             [0.001, 0.348, 0.002, 0.002, 0.018, 0., 0., 0.001], \
             [0.003, 0.012, 0.241, 0., 0.007, 0., 0., 0.], \
             [0., 0., 0., 0., 0., 0., 0., 0.], \
             [0., 0.581, 0.053, 0.08, 0., 0.002, 0., 0.]]
        test_action1 = np.dstack([np.array(test_source1), np.array(test_target1)])

        fen = tensor_decode_fen(np.array(test_board))
        board = chess.Board(fen)
        suggested_action = get_action(board, test_action1)
        self.assertEqual('c2b1', suggested_action)

        test_source2 = \
            [[0., 0., 0., 0., 0., 0., 0., 0.], \
             [0., 0., 0., 0., 0., 0., 0., 0.], \
             [0., 0., 0., 0., 0., 0., 0., 0.], \
             [0., 0., 0., 0., 0., 0., 0., 0.], \
             [0., 0., 0., 0., 0., 0., 0., 0.], \
             [0., 0., 0., 0.001, 0., 0.009, 0., 0.], \
             [0., 0.023, 0.845, 0.999, 0.008, 0., 0.001, 0.], \
             [0.005, 0., 0., 0., 0.016, 0., 0., 0.]]
        test_target2 = \
            [[0., 0., 0., 0., 0., 0., 0., 0.], \
             [0.004, 0., 0., 0., 0., 0., 0., 0.], \
             [0., 0., 0., 0., 0., 0., 0., 0.], \
             [0.008, 0., 0., 0.001, 0.003, 0., 0.001, 0.], \
             [0.001, 0.348, 0.002, 0.002, 0.018, 0., 0., 0.001], \
             [0.003, 0.012, 0.241, 0., 0.007, 0., 0., 0.], \
             [0., 0., 0., 0., 0., 0., 0., 0.], \
             [0., 0.581, 0.053, 0.08, 0., 0.002, 0., 0.]]
        test_action2 = np.dstack([np.array(test_source2), np.array(test_target2)])

        fen = tensor_decode_fen(np.array(test_board))
        board = chess.Board(fen)
        suggested_action2 = get_action(board, test_action2)
        self.assertEqual('d2b4', suggested_action2)

    def test_get_action_promotion(self):
        test_board = \
            [[EMPTY, EMPTY, EMPTY, EMPTY, ROOK_B, EMPTY, KING_B, EMPTY], \
             [EMPTY, QUEN_B, PAWN_W, EMPTY, BISH_B, PAWN_B, PAWN_B, PAWN_B], \
             [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY], \
             [EMPTY, PAWN_B, PAWN_B, BISH_B, PAWN_B, EMPTY, EMPTY, EMPTY], \
             [EMPTY, KNHT_B, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY], \
             [EMPTY, EMPTY, EMPTY, PAWN_W, EMPTY, KNHT_W, PAWN_W, EMPTY], \
             [EMPTY, PAWN_W, QUEN_W, BISH_W, PAWN_W, PAWN_W, BISH_W, PAWN_W], \
             [ROOK_W, EMPTY, EMPTY, EMPTY, ROOK_W, EMPTY, KING_W, EMPTY]]

        test_source = \
            [[0., 0., 0., 0., 0., 0., 0., 0.], \
             [0., 0., 0.999, 0., 0., 0., 0., 0.], \
             [0., 0., 0., 0., 0., 0., 0., 0.], \
             [0., 0., 0., 0., 0., 0., 0., 0.], \
             [0., 0., 0., 0., 0., 0., 0., 0.], \
             [0., 0., 0., 0.001, 0., 0.009, 0., 0.], \
             [0., 0.023, 0.845, 0.4, 0.008, 0., 0.001, 0.], \
             [0.005, 0., 0., 0., 0.016, 0., 0., 0.]]
        test_target = \
            [[0.8, 0., 0., 0., 0., 0., 0., 0.], \
             [0.004, 0., 0., 0., 0., 0., 0., 0.], \
             [0., 0., 0., 0., 0., 0., 0., 0.], \
             [0.008, 0., 0., 0.001, 0.003, 0., 0.001, 0.], \
             [0.001, 0.348, 0.002, 0.002, 0.018, 0., 0., 0.001], \
             [0.003, 0.012, 0.241, 0., 0.007, 0., 0., 0.], \
             [0., 0., 0., 0., 0., 0., 0., 0.], \
             [0., 0.581, 0.053, 0.08, 0., 0.002, 0., 0.]]
        test_action = np.dstack([np.array(test_source), np.array(test_target)])

        fen_promote = tensor_decode_fen(np.array(test_board))
        board_promote = chess.Board(fen_promote)
        suggested_action = get_action(board_promote, test_action)
        self.assertEqual('c7c8q', suggested_action)


if __name__ == '__main__':
    unittest.main()
