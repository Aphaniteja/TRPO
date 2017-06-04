"""
Convenience functions for tensorflow
"""
import functools

import numpy as np
import tensorflow as tf

from trpo.utils.python_utils import doublewrap

# from src.py3.utils import as_iter

DEFAULT_DTYPE = np.float32  # Tensorflow accepts numpy dtypes for dtypes too!


def tf_discount(x, gamma, name=None):
    """
    Compute discounted sum of future values. 
    y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
                where k = len(x) - t - 1
    """
    with tf.name_scope(name, 'tf_discount', [x, gamma]):
        x = tf.convert_to_tensor(x, dtype=x.dtype)
        gamma = tf.convert_to_tensor(gamma, dtype=DEFAULT_DTYPE)
        raise NotImplementedError


def tf_isclose(a, b, tol=1e-7):
    """
    tensorflow equivalent of np.isclose
    :param a: 
    :param b: 
    :param tol: 
    :return: 
    """
    diff = tf.abs(tf.subtract(a, b))
    return tf.less_equal(diff, tol)


def tf_allclose(a, b, tol=1e-7):
    """
    tensorflow equivalent of np.allclose
    :param a: 
    :param b: 
    :param tol: 
    :return: 
    """
    return tf.reduce_all(tf_isclose(a, b, tol))


def tf_vector_product(a, b, name=None):
    """
    Returns vector product of a and b
    :param a: 
    :param b: 
    :return: 
    """
    with tf.name_scope(name, 'tf_vector_product', [a, b]):
        return tf.reduce_sum(tf.multiply(a, b))


def tf_repeat(a, repeats, axis=None, name=None, no_grad=True):
    """
    tensorflow equivalent of numpy repeat
    :param a:
    :param repeats:
    :param axis:
    :param name:
    :param no_grad: bool
        Does not compute gradient. 
    :return: tf.Tensor of same dtype as a
    """
    with tf.name_scope(name, "repeat", [a, repeats]) as scope:
        a = tf.convert_to_tensor(a, name='a')
        repeats = tf.convert_to_tensor(repeats, name='repeats')
        if no_grad:
            tf.py_func(np.repeat, [a, repeats, axis], a.dtype, name=scope)
        idx = tf.reshape(a, (-1, 1))  # Convert to a len(a) x 1 matrix.
        idx = tf.tile(idx, tf.cast(tf.convert_to_tensor((1, repeats)), tf.int32))  # Create multiple columns.
        return tf.reshape(idx, [-1], name=scope)  # Convert back to a vector.


def print_val(x, data=None, summarize=None, message=None, name=None):
    """
    Just a convenience wrapper for tf.Print so that I don't have to write such a long piece of code every time!
    :param x:
    :param data:
    :param summarize:
    :param message:
    :param name:
    :return:
    """
    data = [x] if data is None else data
    summarize = 20 if summarize is None else summarize
    return tf.Print(x, data, summarize=summarize, message=message, name=name)


# def tf_flatten(tensor, name=None):
#     """
#     Returns the tensor reshaped to a single dimension.
#     :param tensor: tf.Tensor or tf.Variable
#     :param name: name of the op.
#     :return: same as input
#     """
#     with tf.name_scope(name, "tf_flatten", values=as_iter(tensor)) as scope:
#         return tf.reshape(tensor, (-1,), name=scope)


def update_variable(variable: tf.Variable, expression: tf.Tensor, inputs: list, name=None):
    """
    Built to replicate theano.function(inps=inputs, [], updates=updates)

    Updates the value of variable with the expression by substituting the value of passed in tensor in expression
    :param inputs: iterable of [ndarrays or array like objects]
        value which will be fed into expression to update variable
    :param variable: tf.Variable
        WILL BE MODIFIED BY FUNCTION!
    :param expression: tf.Tensor 
        expression / graph with variable and tensor as inputs 
    :return: None 
    """
    with tf.name_scope(name, 'update_variable', [variable, expression]):
        with tf.Session() as sess:
            upd_variable_val = sess.run(expression, feed_dict=inputs)
        variable.assign(upd_variable_val)


def update_variables(inputs: list, updates: list or dict, name=None) -> None:
    """
    Built to replicate theano.function(inps=inputs, [], updates=updates)

    Updates the value of variable with the expression by substituting the value of passed in tensor in expression
    :param inputs: iterable of [ndarrays or array like objects]
        value which will be fed into expression to update variable
    :param updates: {tf.Variable: tf.Tensor, ...} or [(tf.Variable, tf.Tensor), ...]
        VARIABLES WILL BE MODIFIED!
    :return: None 
    """
    try:
        var_list, update_tensors = updates.keys(), updates.items()
    except AttributeError:
        var_list, update_tensors = list(zip(*updates))
    with tf.name_scope(name, 'update_variables', [updates]):  # What's the point of using this here?
        with tf.Session() as sess:
            upd_variable_vals = sess.run(update_tensors, feed_dict=inputs)
        for v, upd in zip(var_list, upd_variable_vals):
            v.assign(upd)


@doublewrap
def define_scope_and_cache(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator
