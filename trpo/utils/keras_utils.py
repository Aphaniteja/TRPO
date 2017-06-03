"""
Some utilities for keras. Put into a separate file because this might be removed after refactoring.
"""

import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete
from keras import activations

from src.py3.algos.trpo.distributions import DiagonalGaussian
from src.py3.algos.trpo.policies import StochasticPolicy
from src.py3.algos.trpo.utils import dtype_X
from src.py3.algos.trpo.utils.math_utils import NnVf
from src.py3.tf_utils import tf_repeat


def keras_dense(output_dim, init, graph, activation='linear', init_kwargs=(), name=None, input_shape=None):
    """
    A function which mimics the Dense layer from keras. Returns a tf.Tensor, but return value has additional attributes
    monkey patched on to it - input, trainable_weights, output
    
    BE VERY CAREFUL WHEN USING THIS FUNCTION IN BETWEEN TENSORFLOW COMMANDS TO CREATE A BIGGER GRAPH!
    :param output_dim: 
    :param init: callable
        init(**init_kwargs) -> np.ndarray or array like
    :param activation: 
    :return: 
    """

    def dense(inp):
        _init_kwargs = dict(init_kwargs)
        _input_shape = inp.shape if input_shape is None else input_shape
        _graph = graph or tf.get_default_graph()
        assert isinstance(_graph, tf.Graph)
        with _graph.as_default():
            with tf.name_scope(name, 'Dense', [input]) as scope:
                shape = _init_kwargs.get('shape', _input_shape)
                dtype = _init_kwargs.get('dtype', dtype_X(inp))
                try:
                    del _init_kwargs['shape']
                    del _init_kwargs['dtype']
                except KeyError:
                    pass
                W = tf.Variable(initial_value=init(shape=(output_dim,) + shape[1:], dtype=dtype, **_init_kwargs), name='W')
                b = tf.Variable(initial_value=np.zeros(output_dim, dtype_X(inp)), name='b')

                product = tf.matmul(inp, W, transpose_b=True)

                ret_val = activations.get(activation)(tf.add(product, b, name=scope))
                ret_val.input = inp.input if hasattr(inp, 'input') else inp
                ret_val.shape = (shape[0], output_dim)
                inp_trainable_weights = inp.trainable_weights if hasattr(inp, 'trainable_weights') else []
                ret_val.trainable_weights = [W, b] + inp_trainable_weights

                return ret_val

    return dense


def batched_shape(shape):
    return tuple([None] + [x for x in shape])


def get_fans(shape, dim_ordering='tf'):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4 or len(shape) == 5:
        # assuming convolution kernels (2D or 3D).
        # TH kernel shape: (depth, input_depth, ...)
        # TF kernel shape: (..., input_depth, depth)
        if dim_ordering == 'th':
            fan_in = np.prod(shape[1:])
            fan_out = shape[0]
        elif dim_ordering == 'tf':
            fan_in = np.prod(shape[:-1])
            fan_out = shape[-1]
        else:
            raise Exception('Invalid dim_ordering: ' + dim_ordering)
    else:
        # no specific assumptions
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out


def glorot_uniform(shape, dtype='float32'):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6. / (fan_in + fan_out))
    return np.random.uniform(low=-s, high=s, size=shape).astype(dtype)


def scaled_glorot_uniform(shape, dtype='float32'):
    gl_uniform = glorot_uniform(shape, dtype)
    return gl_uniform * 0.1


def make_mlp(input_placeholder, hidden_layer_sizes: list, activation: str, graph: tf.Graph, name: str = None):
    """
    
    :param input_placeholder: NOT a tf.placeholder or tf.Tensor! monkey patched versions of those with attribute 'shape' 
    :param hidden_layer_sizes: 
    :param activation: 
    :param graph:
    :param name: 
    :return: 
    """
    graph = graph or tf.get_default_graph()
    assert isinstance(graph, tf.Graph)
    with graph.as_default():
        with tf.name_scope(name, 'make_mlp', [input_placeholder]):
            output = input_placeholder  # Just masking it with the name output for convenience in the for loop
            for output_dim in hidden_layer_sizes:
                output = keras_dense(output_dim, glorot_uniform, graph=graph, activation=activation)(output)

            return output


def make_mlp_for_continuous_action_space(state_space, action_space, graph,
                                         hidden_layer_sizes: list, activation: str,
                                         name=None):
    graph = graph or tf.get_default_graph()
    assert isinstance(graph, tf.Graph)
    with graph.as_default():
        with tf.name_scope(name, 'MultiLayerPerceptron_ContinuousActionSpace') as scope:
            output_dim = action_space.shape[0]
            input_shape = batched_shape(state_space.shape)
            input_placeholder = tf.placeholder(dtype=dtype_X(state_space.low), shape=input_shape,
                                               name='states')
            input_placeholder.shape = input_shape
            # Make mlp:
            mlp_tensor = make_mlp(input_placeholder, hidden_layer_sizes, graph=graph, activation=activation)
            mean = keras_dense(output_dim, init=scaled_glorot_uniform, graph=graph)(mlp_tensor)

            # Add standard deviation parameter to our network
            log_std = tf.Variable(np.zeros(output_dim, np.float32), name='log_std')
            batched_log_std = tf_repeat(tf.exp(log_std), tf.shape(mean)[0], no_grad=False)
            std = tf.expand_dims(batched_log_std, axis=0, name='batched_log_std')
            ret_val = tf.concat_v2([mean, std], axis=1, name=scope)
            ret_val.input = mean.input
            ret_val.trainable_weights = mean.trainable_weights + [log_std]

            return ret_val


def make_mlp_for_discrete_action_space(action_space, hidden_layer_sizes):
    raise NotImplementedError


def make_mlps(ob_space, ac_space, cfg, graph):
    assert isinstance(ob_space, Box)
    graph = graph or tf.get_default_graph()
    assert isinstance(graph, tf.Graph)
    hid_sizes = cfg['hid_sizes']
    if isinstance(ac_space, Box):
        mlp = make_mlp_for_continuous_action_space(state_space=ob_space, action_space=ac_space,
                                                   graph=graph,
                                                   hidden_layer_sizes=hid_sizes, activation=cfg['activation'])
    elif isinstance(ac_space, Discrete):
        mlp = make_mlp_for_discrete_action_space(action_space=ac_space, hidden_layer_sizes=hid_sizes)
    else:
        raise NotImplementedError

    probtype = DiagonalGaussian(ac_space.shape[0], graph=graph)
    policy = StochasticPolicy(mlp, probtype)
    with graph.as_default():
        with tf.name_scope('vf_net'):
            input_shape = (None, ob_space.shape[0] + 1)
            vf_net_input_placeholder = tf.placeholder(dtype_X(ob_space.low), input_shape, name='input')
            vf_net_input_placeholder.shape = input_shape
            vf_net = make_mlp(vf_net_input_placeholder, hid_sizes, cfg['activation'], graph=graph, name='hidden_part')
            vf_net = keras_dense(1, init=glorot_uniform, name='output', graph=graph)(vf_net)
    # vf_net = output
    # vf_net.input = vf_net_input_placeholder  # Todo: This is a very bad monkey patch!
    # vf_net.output = output
    baseline = NnVf(vf_net, cfg["timestep_limit"], graph, dict(mixfrac=0.1))
    return policy, baseline

# class ConcatFixedStd(Layer):
#     input_ndim = 2
#
#     def __init__(self, **kwargs):
#         super(ConcatFixedStd, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         input_dim = input_shape[1]
#         self.logstd = tf.Variable(initial_value=np.zeros(input_dim, dtype=np.float32),
#                                   name='{}_logstd'.format(self.name))
#         self.trainable_weights = [self.logstd]
#
#     def get_output_shape_for(self, input_shape):
#         return input_shape[0], input_shape[1] * 2
#
#     def call(self, x, mask=None):
#         mean = x
#         print('here')
#         std = tf_repeat(tf.exp(self.logstd)[None, :], tf.shape(mean)[0], axis=0)
#         return tf.concat_v2([mean, std], axis=1)
