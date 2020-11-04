import sys
sys.path.insert(0, '..')

from play.play_loop import PlayLoop
from model.policy_model import PolicyModel

if __name__ == '__main__':
    play_loop = PlayLoop(PolicyModel.load('../model/trained/model_0411.hdf5'))
    play_loop.init_game('w', keep_trace=True)

    play_loop.loop()
