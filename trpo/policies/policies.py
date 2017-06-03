"""
Implements base classes for StochasticPolicy and ...? and required stuff 
"""
import collections

import numpy as np
import tensorflow as tf
from gym.utils.ezpickle import EzPickle

from src.py3 import utils as src_utils
from src.py3.algos.trpo.distributions import Distribution
from src.py3.algos.trpo.utils import dtype_X
from src.py3.algos.trpo.utils.keras_utils import batched_shape, glorot_uniform, scaled_glorot_uniform, \
    MultiLayerPerceptron
from src.py3.tf_utils import tf_repeat, define_scope_and_cache


class StochasticPolicy(EzPickle):
    def __init__(self, net, distribution: Distribution):
        super(StochasticPolicy, self).__init__(net, distribution)
        self._act_prob = None
        self._net = net
        self._distribution = distribution

    @property
    def net(self):
        return self._net

    @property
    def distribution(self):
        return self._distribution

    @property
    def trainable_variables(self):
        return self.net.trainable_weights

    @property
    def input(self):
        return self.net.input

    @property
    def output(self):
        return self.net

    def act(self, ob, stochastic=True):
        prob = self._act_prob(ob[None])
        if stochastic:
            return self.distribution.sample(prob)[0], {"prob": prob[0]}
        else:
            return self.distribution.max_prob(prob)[0], {"prob": prob[0]}

    def finalize(self):
        with tf.Session() as sess:
            self._act_prob = sess.run(fetches=self.output,
                                      feed_dict=self.input)


class PolicyNetwork_v1(MultiLayerPerceptron):
    def __init__(self, state_space, action_space, hidden_layer_sizes, activation,
                 model_dir=None, config=None, name='PolicyNetwork'):
        self.state_space, self.action_space = state_space, action_space
        self.input_size = self.state_space.shape[0]
        self._action_space_size = self.action_space.shape[0]
        super().__init__(hidden_layer_sizes, output_size=self.action_space.shape[0],
                         activation=activation, output_activation=activation,
                         model_dir=model_dir, config=config, name=name)

    def __call__(self, input_tensor, *args, **kwargs):
        self.input = super()._create_input_placeholders(input_tensor)
        self.network_output = super()._create_model


class PolicyNetwork:
    _ModelVariables = collections.namedtuple('_ModelVariables', 'network_weights network_biases '
                                                                'parameter_weight parameter_bias '
                                                                'log_std')

    def __init__(self, state_space, action_space, hidden_layer_sizes, activation,
                 model_dir=None, config=None, name=None):
        from keras import activations
        # super(PolicyNetwork, self).__init__(model_dir=model_dir, config=config)
        self.state_space, self.action_space = state_space, action_space
        self.hidden_layer_sizes = src_utils.as_iter(hidden_layer_sizes)
        self.activation_fn = activations.get(activation)
        self.name = name or 'PolicyNetwork'

        self.input_size = self.state_space.shape

        pass

    def __call__(self, input_tensor):
        self.input = input_tensor
        self._create_input_placeholders
        self._dtype = dtype_X(self.input)
        self.trainable_weights = []
        self.output = self._create_network

    @staticmethod
    def _dense_layer(x, W, b, activation_fn=None, name=None):
        if activation_fn is None:
            activation_fn = lambda inp: inp
        with tf.name_scope(name, 'dense_layer', [x, W, b]) as scope:
            product = tf.matmul(x, W, transpose_b=True)
            return activation_fn(tf.add(product, b, name=scope))

    @define_scope_and_cache
    def _create_input_placeholders(self):
        if self.input is None:
            input_shape = batched_shape(self.state_space.shape)

            self.input = tf.placeholder(dtype=dtype_X(self.state_space.low), shape=input_shape,
                                        name='states')

        self.input.shape = self.input.get_shape().as_list()

    @define_scope_and_cache
    def _initialize_weights(self):
        ret_val = self._ModelVariables(network_weights=[], network_biases=[],
                                       parameter_weight=None, parameter_bias=None,
                                       log_std=None)

        action_space_dim = self.action_space.shape[0]
        for w_shape in zip([self.input_size] + self.hidden_layer_sizes, self.hidden_layer_sizes[1:]):
            W = tf.Variable(initial_value=glorot_uniform(shape=w_shape, dtype=self._dtype),
                            name='W')
            b = tf.Variable(initial_value=np.zeros(w_shape[0], self._dtype),
                            name='b')
            self.trainable_weights.extend([W, b])
            ret_val.network_weights.append(W)
            ret_val.network_biases.append(b)

        # Parameters of the Stochastic Policy
        W = tf.Variable(initial_value=scaled_glorot_uniform((self.hidden_layer_sizes[-1], self._dtype)),
                        name='W')
        b = tf.Variable(initial_value=np.zeros(action_space_dim, self._dtype),
                        name='b')
        self.trainable_weights.extend([W, b])
        ret_val.parameter_weight = W
        ret_val.parameter_bias = b

        log_std = tf.Variable(initial_value=np.zeros(action_space_dim, np.float32),
                              name='log_std')
        self.trainable_weights.append(log_std)
        ret_val.log_std = log_std

        return ret_val

    @define_scope_and_cache
    def _create_network(self):
        # assert isinstance(features, tf.Tensor)
        x = self.input
        model_weights = self._initialize_weights
        for W, b in zip(model_weights.network_weights, model_weights.network_biases):
            x = self._dense_layer(x, W, b, activation_fn=self.activation_fn)

        # Parameters of the Stochastic Policy
        mean = self._dense_layer(x, model_weights.parameter_weight, model_weights.parameter_bias,
                                 name='mean')
        batched_log_std = tf_repeat(tf.exp(model_weights.log_std), tf.shape(mean)[0], no_grad=False)
        std = tf.expand_dims(batched_log_std, axis=0, name='batched_log_std')

        return tf.concat_v2([mean, std], axis=1)
