import tensorflow.keras.backend as K

KERNEL_INIT = 'glorot_uniform'


def tanh_loss(y_true, y_pred):
    '''
    Custom cross entropy loss function that supports tanh activation functions.

    Obtained from a user response on StackExchange:
    https://stats.stackexchange.com/questions/221901/can-the-cross-entropy-cost-function-be-used-with-tanh

    :param y_true: Tensor - true labels
    :param y_pred: Tensor - predicted labels
    :return: float - cost value between y_true and y_pred
    '''
    return -0.5 * ((1 - y_true) * K.log(1 - y_pred) + (1 + y_true) * K.log(1 + y_pred)) + K.log(2.)
