import numpy as np
import chess

from parsing.math_encode import tensor_encode, tensor_decode
from inference.infer_action import get_action


class PlayLoop:
    TRACE_FORMAT = '{:<12}{:<20}{:<12}{}'
    TRACE_HEADER = ('WHITE', 'move', 'BLACK', 'move')

    def __init__(self, policy):
        self.policy = policy
        self.player_white = None
        self.board = None
        self.keep_trace = False
        self.trace = None

    def init_game(self, player_side, keep_trace=False, force_new=False):
        if not force_new and self.board is not None:
            raise RuntimeWarning('Board already initiatized, set force_new=True to force override')

        if player_side == 'w':
            self.player_white = True
        elif player_side == 'b':
            self.player_white = False
        else:
            raise ValueError(f'Expected "w" or "b" for player_side but given {player_side}')

        self.board = chess.Board()
        if keep_trace:
            self.keep_trace = keep_trace
            self.trace = list()

    def loop(self):
        if self.board is None:
            raise RuntimeError('init_game was not called to configure game settings!')

        trace_collector = list()

        if not self.player_white:
            act_uci = self._get_action_from_policy()
            self.board.push(chess.Move.from_uci(act_uci))

            if self.keep_trace:
                self._store_trace(act_uci, trace_collector)

        while not self.board.is_game_over():
            # print(tensor_decode(tensor_encode(self.board, flip=not self.player_white)))
            print(self.board)

            move = self._get_player_move()

            if self.keep_trace:
                self._store_trace(move, trace_collector)

            act_uci = self._get_action_from_policy()
            print(f'\nAI made {act_uci} move\n')
            self.board.push(chess.Move.from_uci(act_uci))

            if self.keep_trace:
                self._store_trace(act_uci, trace_collector)

        print('Game completed')

    def get_trace(self, printable=True):
        if printable:
            return '\n'.join(self.trace)
        return self.trace

    def _get_action_from_policy(self):
        src, tgt, _ = self.policy.infer(tensor_encode(self.board, flip=self.player_white))
        return get_action(self.board, src, tgt)

    def _get_player_move(self):
        while True:  # handles invalid player moves
            try:  # TODO: Use proper validation checks if possible (non-urgent)
                move = input('Enter your move: ')
                self.board.push(chess.Move.from_uci(move))
            except AssertionError as e:
                print(f'ERROR: {e}')
            else:
                break
        return move

    def _store_trace(self, move, trace_collector):
        trace_collector.append(str(self.board))
        trace_collector.append(move)
        if len(trace_collector) == 4:
            self.trace.append(PlayLoop.TRACE_FORMAT.format(*PlayLoop.TRACE_HEADER))
            for b1, b2 in zip(trace_collector[0].split('\n'), trace_collector[2].split('\n')):
                self.trace.append(PlayLoop.TRACE_FORMAT.format(trace_collector[1], b1,
                                                               trace_collector[3], b2))
                trace_collector[1] = ''
                trace_collector[3] = ''
            self.trace.append('\n')
            trace_collector.clear()
