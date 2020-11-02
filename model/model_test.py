import numpy as np
from joblib import load
from model.policy_model import PolicyModel

if __name__ == '__main__':
    DATA_PATH = '../data/'

    states = load(DATA_PATH + 'tensor_1710_state_125k.joblib')
    actions = load(DATA_PATH + 'tensor_1710_action_125k.joblib')
    labels = np.genfromtxt(DATA_PATH + 'labels_1710_125k.csv', delimiter=',')

    # flattens the action matrices
    actions = actions.reshape(-1, 128)

    policy = PolicyModel(states.shape[1:], (actions.shape[1], 1))
    policy.construct(12)
    size = states.shape[0]

    policy.train(x=states[0:size + 1, :, :],
                 y=[actions[0:size + 1],
                    labels[0:size]],
                 epochs=1,
                 shuffle=True,
                 batch_size=512,
                 verbose=1)

