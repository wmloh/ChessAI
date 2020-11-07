import sys
sys.path.insert(0, '..')

from play.play_loop import PlayLoop
from model.policy_model import PolicyModel

if __name__ == '__main__':

    INTERACTIVE_PLAY = True
    PLAYER_SIDE = 'w'

    if INTERACTIVE_PLAY:
        play_loop = PlayLoop(PolicyModel.load('../model/trained/model_0611.hdf5'))
        VERBOSE = True
    else:
        play_loop = PlayLoop(PolicyModel.load('../model/trained/model_0611.hdf5'),
                             secondary_policy='same')
        VERBOSE = False

    play_loop.init_game(PLAYER_SIDE, keep_trace=True)
    play_loop.loop(verbose=VERBOSE)

    trace = play_loop.get_trace()
    # play_loop.save_trace()


