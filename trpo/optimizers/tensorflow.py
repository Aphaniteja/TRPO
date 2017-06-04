import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface

DEFAULT_CONJUGATE_GRADIENTS_OPTIMIZER_ITER = 10
DEFAULT_CONJUGATE_GRADIENTS_OPTIMIZER_RESIDUAL = 1e-10
DEFAULT_LINE_SEARCH_BFGS_MAX_ITER = 10
DEFAULT_LINE_SEARCH_BACK_TRACKING_ACCEPT_RATIO = 0.1
DEFAULT_LINE_SEARCH_BACK_TRACKING_MAX_ITER = 10
DEFAULT_LINE_SEARCH_BACK_TRACKING_STEP_INTERVAL = 0.5


def line_search_general(
        objective_func, var_list=None, equalities=None,
        inequalities=None, methods=None,
        maxiter=DEFAULT_LINE_SEARCH_BFGS_MAX_ITER):
    """
    Line Search Optimizer for non-convex optimization problem:
    L-BFGS by default.
    """
    if methods == 'backtracking':
        # TODO
        raise NotImplementedError
    elif methods is not None:
        return ScipyOptimizerInterface(
            objective_func, var_list, equalities,
            inequalities, method=methods, options={'maxiter': maxiter})
    else:
        return ScipyOptimizerInterface(
            objective_func, var_list, equalities,
            inequalities, options={'maxiter': maxiter})


def conjugate_gradients(
        f_x, b, total_iter=DEFAULT_CONJUGATE_GRADIENTS_OPTIMIZER_ITER,
        residual_tol=DEFAULT_CONJUGATE_GRADIENTS_OPTIMIZER_RESIDUAL,
        **kwargs):
    """
    Conjugate Gradient Optimizer is to solve the Linear Equation Ax = b, in
    which A is a Semi-Positive Definite matrix with dimension n * n, b is a
    vector with dim n, and x is the variable to be solved.
    Args:
        f_x: The function f(x) to perform A * x
        b: Vector for Linear Equation.
        total_iter: The total optimization iteration.
        residual_tol: The termination error condition for optimization.
        **kwargs: Optional arguments passed to f_x.

    Returns:
        x for Ax = b.
    """

    p = tf.Variable(tf.zeros(b.get_shape()), trainable=False)
    r = tf.Variable(tf.zeros(b.get_shape()), trainable=False)
    x = tf.Variable(tf.zeros(b.get_shape()), trainable=False)
    i = tf.Variable(0, trainable=False)
    p_copy = tf.assign(p, b)
    r_copy = tf.assign(r, b)
    init_i_ops = tf.variables_initializer([i, x])

    def condition(i, p, r, r_dot_r, x):
        return tf.logical_and(i < total_iter, r_dot_r >= residual_tol)

    def body(i, p, r, r_dot_r, x):
        A_p = f_x(p, **kwargs)
        alpha = r_dot_r / tf.reduce_sum(tf.multiply(p, A_p))
        x += alpha * p
        r -= alpha * A_p

        with tf.control_dependencies([r]):
            r_dot_r_update = tf.reduce_sum(tf.multiply(r, r))

        with tf.control_dependencies([r_dot_r_update]):
            mu = r_dot_r_update / r_dot_r
            r_dot_r = r_dot_r_update

        with tf.control_dependencies([mu, r]):
            p = r + mu * p

        i = tf.add(i, 1)
        return i, p, r, r_dot_r, x

    with tf.control_dependencies([init_i_ops, p_copy, r_copy]):
        r_dot_r = tf.reduce_sum(tf.multiply(r, r))
        _, _, _, _, x = tf.while_loop(
            condition, body, [i, p, r, r_dot_r, x])
    return x
