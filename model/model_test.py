import numpy as np
import tensorflow as tf
import pickle
from joblib import load
from model.policy_model import PolicyModel

if __name__ == '__main__':
    DATA_PATH = '../data/'

    STATE_SHAPE = (4510672, 8, 8, 13)
    ACTION_SHAPE = (4510672, 8, 8, 2)
    size = STATE_SHAPE[0]

    TRAIN_NEW = False

    TRAINED_MODEL_PATH = 'checkpoint/model-01.hdf5'
    RANDOM_DATA_PATH = DATA_PATH + 'random_data_2.pkl'

    if TRAIN_NEW:
        policy = PolicyModel(STATE_SHAPE[1:], (64, 1))
        policy.construct(10, 4)

        checkpoint_filepath = 'checkpoint/model-{epoch:02d}.hdf5'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_acc',
            save_best_only=False)

        policy.train(x=load(DATA_PATH + 'tensor_0211_state_150k_0.25.joblib')[0:size + 1, :, :],
                     y=[load(DATA_PATH + 'src_150k_0.25.joblib').reshape(-1, 64)[0:size + 1],
                        load(DATA_PATH + 'tgt_150k_0.25.joblib').reshape(-1, 64)[0:size + 1],
                        np.genfromtxt(DATA_PATH + 'labels_0211_150k_0.25.csv', delimiter=',')[0:size]],
                     epochs=1,
                     shuffle=True,
                     batch_size=512,
                     verbose=1,
                     callbacks=[model_checkpoint_callback])

    else:
        policy = PolicyModel.load()
        with open(RANDOM_DATA_PATH, 'rb') as f:
            sample_data = pickle.load(f)

        state, src, tgt, val = sample_data[2]
        psrc, ptgt, pval = policy.infer(state, decimal=3)

