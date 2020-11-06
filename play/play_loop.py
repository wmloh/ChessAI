import numpy as np
import chess

from tkinter.filedialog import asksaveasfilename
from parsing.math_encode import tensor_encode, tensor_decode
from inference.infer_action import get_action


class PlayLoop:
    __doc__ = '''
    An interactive REPL environment for play with a trained chess AI
    '''
    TRACE_FORMAT = '{:<7}{:<18}{:<42}{:<45}{:<7}{:<18}{:<42}{}'
    TRACE_HEADER = ('move', 'WHITE', 'source', 'target', 'move', 'BLACK', 'source', 'target')

    def __init__(self, policy, secondary_policy=None):
        '''
        Constructs a PlayLoop instance

        :param policy: PolicyModel - primary AI agent to simulate
        :param secondary_policy: None/PolicyModel/'same' - AI agent used to replace player moves
            (if None, human is playing; if 'same', secondary_policy=policy)
        '''
        self.policy = policy
        self.player_white = None
        self.board = None
        self.keep_trace = False
        self.trace = None

        if secondary_policy is not None:
            if secondary_policy == 'same':
                self.player_move_func = lambda: self._get_action_from_policy(policy)
            else:
                self.player_move_func = lambda: self._get_action_from_policy(secondary_policy)
        else:
            self.player_move_func = self._get_player_move

    def init_game(self, player_side, keep_trace=True):
        '''
        Sets up a game and indicating the side to play as by the player

        :param player_side: 'w'/'b' - side to play as
        :param keep_trace: bool - if True, accumulates the trace for the entire game
        :return: None
        '''
        if self.board is not None:
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

    def reset(self):
        '''
        Resets the PlayLoop state (except the trace)

        :return: None
        '''
        self.board = None
        self.keep_trace = False

    def loop(self, verbose=True):
        '''
        Runs the loop until the termination of a game

        :param verbose: bool - prints messages if True
        :return: None
        '''
        if self.board is None:
            raise RuntimeError('init_game was not called to configure game settings!')

        trace_collector = list()

        if not self.player_white:
            move, policy = self._get_action_from_policy()
            if self.keep_trace: self._store_trace(move, trace_collector, policy=policy)
            if verbose: print(f'\nAI made {move} move\n')

        while not self.board.is_game_over():
            if verbose: print(self.board)

            # player/secondary_policy move
            move, policy = self.player_move_func()
            if self.keep_trace: self._store_trace(move, trace_collector, policy=policy)
            if verbose: print(f'\nPlayer made {move} move\n')

            if self.board.is_game_over():
                break

            # policy move
            move, policy = self._get_action_from_policy()
            if self.keep_trace: self._store_trace(move, trace_collector, policy=policy)
            if verbose: print(f'\nAI made {move} move\n')

        if len(trace_collector) != 0:
            self._store_trace(move, trace_collector, policy=policy, force_flush=True)

        if verbose: print('Game completed')

    def get_trace(self, printable=True):
        '''
        Returns the trace

        :param printable: bool - If True, returns a printable and formamted string of the trace
        :return: str/list(str)
        '''
        if printable:
            return '\n'.join(self.trace)
        return self.trace

    def save_trace(self, file_path=None, interactive=True):
        '''
        Saves trace in a text file

        Automatically appends ".txt" at the end of the file_path if the suffix is not found

        :param file_path: None/str - file path to save to
        :param interactive: bool - if True, using Tkinter GUI to select file path
        :return: None
        '''
        if interactive:
            file_path = asksaveasfilename(filetypes=[('Text file', '*.txt')])

        if file_path[-4:] != '.txt':
            file_path = file_path + '.txt'
        with open(file_path, 'w') as f:
            f.write(self.get_trace(printable=True))

    def _get_action_from_policy(self, external_policy=None):
        '''
        Gets UCI representation of the move using the policy loaded and pushes the move on the board

        :param external_policy - None/PolicyModel - policy to use (if None, defaults to loaded policy)
        :return: str
        '''
        policy = self.policy
        flip = self.player_white

        if external_policy:  # player is an AI
            policy = external_policy
            flip = not flip

        src, tgt, _ = policy.infer(tensor_encode(self.board, flip=flip))

        if flip:
            src = np.rot90(src, 2)
            tgt = np.rot90(tgt, 2)

        move = get_action(self.board, src, tgt)
        self.board.push(chess.Move.from_uci(move))

        return move, (src, tgt)

    def _get_player_move(self):
        '''
        Obtains the move from the player by command line and pushes the move on the board

        :return: str
        '''
        while True:  # handles invalid player moves
            try:
                move_input = input('Enter your move: ')
                move = chess.Move.from_uci(move_input)
                if move in self.board.legal_moves:
                    self.board.push(move)
                else:
                    raise AssertionError(f'{move_input} is not a valid move')
            except AssertionError as e:
                print(f'ERROR: {e}')
            else:
                break
        return move_input, None

    def _store_trace(self, move, trace_collector, policy=None, force_flush=False):
        '''
        Collects the trace onto trace_collector and once white and black has made the move,
            append to the main trace list

        :param move: str - UCI representation of the move
        :param trace_collector: list(str) - string accumulator
        :param policy: None/tuple(np.ndarray, np.ndarray) - policy output
        :param force_flush: bool - if True, appends incomplete trace
        :return: None
        '''
        trace_collector.append(str(self.board))
        trace_collector.append(move)
        if policy is None:
            trace_collector.append('N/A\n\n\n\n\n\n\n')
            trace_collector.append('N/A\n\n\n\n\n\n\n')
        else:
            trace_collector.append(str(np.around(policy[0][0, ...], 2)).replace('[[', '')
                                   .replace(' [ ', '').replace(' [', '').replace(']', ''))
            trace_collector.append(str(np.around(policy[1][0, ...], 2)).replace('[[', '')
                                   .replace(' [ ', '').replace(' [', '').replace(']', ''))

        if len(trace_collector) == 8:  # two half-moves has been made
            self.trace.append(PlayLoop.TRACE_FORMAT.format(*PlayLoop.TRACE_HEADER))
            for b1, src1, tgt1, b2, src2, tgt2 in zip(trace_collector[0].split('\n'),
                                                      trace_collector[2].split('\n'),
                                                      trace_collector[3].split('\n'),
                                                      trace_collector[4].split('\n'),
                                                      trace_collector[6].split('\n'),
                                                      trace_collector[7].split('\n')):
                self.trace.append(PlayLoop.TRACE_FORMAT.format(trace_collector[1], b1, src1, tgt1,
                                                               trace_collector[5], b2, src2, tgt2))
                trace_collector[1] = ''
                trace_collector[5] = ''
            self.trace.append('\n')
            trace_collector.clear()
        elif force_flush:
            self.trace.append(PlayLoop.TRACE_FORMAT.format(*PlayLoop.TRACE_HEADER))
            for b1, src1, tgt1 in zip(trace_collector[0].split('\n'),
                                    trace_collector[2].split('\n'),
                                    trace_collector[3].split('\n')):
                self.trace.append(PlayLoop.TRACE_FORMAT.format(trace_collector[1], b1, src1, tgt1,
                                                               '', '', '', ''))
                trace_collector[1] = ''
            self.trace.append('\n')
