import tensorflow as tf
from model.policy_utils import KERNEL_INIT, tanh_loss
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, PReLU, Input, Add, Flatten


class PolicyModel:
    __doc__ = '''
    An abstracted class that encapsulates a Tensorflow Keras implementation of a neural
    network that can learn both the policy and value functions.

    Inspired by: "Mastering the Game of Go without Human Knowledge" - Silver et al.
    '''

    def __init__(self, ground_dim, output_dim):
        '''
        Construct for a PolicyModel

        :param ground_dim: tuple(int, int, int) - 3D dimensions of the board state
        :param output_dim: tuple(int, int) - flattened dimensions of policy output and value output respectively
        '''
        self.model = None
        self.ground_dim = ground_dim
        self.policy_dim = output_dim[0]
        self.value_dim = output_dim[1]

    def construct(self, num_blocks, force_reconstruct=False):
        '''
        Builds the model with the specified number of residual blocks.

        The outputs are passed into the policy and value head.

        :param num_blocks: int - Number of residual blocks to construct
        :param force_reconstruct: bool - if True, overwrites the current model
        :return: tf.keras.models.Model - combined policy and value neural network
        '''

        if self.model is not None and not force_reconstruct:
            raise ValueError('Model has already been constructed.'
                             ' Set force_reconstruct to True to overwrite it')

        x_length, y_length, z_depth = self.ground_dim

        # INPUT
        input_layer = Input(shape=self.ground_dim)
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same',
                       kernel_initializer=KERNEL_INIT)(input_layer)

        # RESIDUAL BLOCKS
        current_residual_block = conv1
        for block in range(num_blocks):
            current_residual_block = PolicyModel.add_residual_block(current_residual_block)

        # POLICY HEAD
        conv_policy = Conv2D(filters=2, kernel_size=(1, 1), strides=1, padding='same',
                             kernel_initializer=KERNEL_INIT)(current_residual_block)
        bn_policy = BatchNormalization()(conv_policy)
        rec_policy1 = PReLU()(bn_policy)
        flatten_policy = Flatten()(rec_policy1)
        dense_policy1 = Dense(x_length * y_length,
                              kernel_initializer=KERNEL_INIT)(flatten_policy)
        rec_policy2 = PReLU()(dense_policy1)
        dense_policy2 = Dense(self.policy_dim,
                              activation='sigmoid',
                              kernel_initializer=KERNEL_INIT,
                              name='policy')(rec_policy2)

        # VALUE HEAD
        conv_value = Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding='same',
                            kernel_initializer=KERNEL_INIT)(current_residual_block)
        bn_value = BatchNormalization()(conv_value)
        rec_value1 = PReLU()(bn_value)
        flatten_value = Flatten()(rec_value1)
        dense_value1 = Dense(x_length * y_length,
                             kernel_initializer=KERNEL_INIT)(flatten_value)
        rec_value2 = PReLU()(dense_value1)
        dense_value2 = Dense(64,
                             kernel_initializer=KERNEL_INIT)(rec_value2)
        rec_value3 = PReLU()(dense_value2)
        dense_value3 = Dense(self.value_dim, activation='tanh',
                             kernel_initializer=KERNEL_INIT,
                             name='value')(rec_value3)

        # encapsulated model
        self.model = Model(inputs=input_layer, outputs=[dense_policy2, dense_value3])

        # compile model
        self.model.compile(optimizer='adam', loss=['categorical_crossentropy', tanh_loss],
                           metrics={'policy': 'categorical_accuracy', 'value': 'accuracy'})

    def train(self, **kwargs):
        '''
        Wrapper function to fit the model
        :param kwargs: dict(String, *) - keyword arguments for self.model.fit
        :return: None
        '''
        self.model.fit(**kwargs)

    def infer(self, ground):
        '''
        Wrapper function to perform a single prediction with the trained model

        :param ground: np.array - ground for prediction (dimensions must be identical to self.ground_dim)
        :return: tuple(POLICY_OUTPUT, VALUE_OUTPUT) - policy and value prediction using the model
        '''
        return tuple(self.model.predict([ground]))

    def save(self, file_path):
        '''
        Calls the model.save method

        :param file_path: str - directory to save the model file to
        :return: None
        '''
        self.model.save(file_path)

    @classmethod
    def load(cls, file_path):
        '''
        Loads a saved tf.keras model and creates a PolicyModel object with it embedded

        :param file_path: str - directory to load model file from
        :return: PolicyModel
        '''
        # load and extract information
        model = tf.keras.models.load_model(file_path, custom_objects={'tanh_loss', tanh_loss})
        input_shape = model.input_shape[1:]
        output_shape = model.output_shape

        # construct a new PolicyModel
        policy = PolicyModel(input_shape, (output_shape[0][1], output_shape[1][1]))
        policy.model = model

        return policy

    @classmethod
    def add_residual_block(cls, residual_input):
        '''
        Builds a single residual block with 2 Conv2D layers (with batch normalization
        and ReLU) and attachs to previous layers

        :param residual_input: tf.keras.layers.* - layer to perform a skip on
        :return: tf.keras.layers.PReLU - output layer for the residual block
        '''
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same',
                       kernel_initializer=KERNEL_INIT)(residual_input)
        bn1 = BatchNormalization()(conv1)
        rec1 = PReLU()(bn1)

        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same',
                       kernel_initializer=KERNEL_INIT)(rec1)
        bn2 = BatchNormalization()(conv2)

        add1 = Add()([bn2, residual_input])
        rec2 = PReLU()(add1)

        return rec2


