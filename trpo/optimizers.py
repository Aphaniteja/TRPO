"""
Implements optimizers used in TRPOptimizer. Hope is that in future, we can implement other algorithms here and try them 
out with TRPOptimizer.
"""
from collections import namedtuple, OrderedDict
import numpy as np
import scipy.optimize
import tensorflow as tf

from src.py3.algos.trpo import utils as trpo_utils

LineSearchResult = namedtuple('LineSearchResult', 'converged x_new')


def line_search(f, x0, s, max_step_size, gf_x0, max_iter, f_args=(), **kwargs):
    """
    Minimizes the function f along the step direction s0 using line search or one of it's variants.
    
    Written as a wrapper function that enforces an interface for all optimizers to be used in TRPOptimizer
    :param f: callable f(x0, *f_args) 
        Objective function.
    :param x0: ndarray 
        starting point of objective function.
    :param s: ndarray 
        search direction
    :param max_step_size: float
        maximum step size
    :param gf_x0: ndarray
        gradient of f evaluated at x0
    :param max_iter: int 
        maximum number of iterations for which the algorithm will be run
    :param f_args: tuple
        additional arguments passed to the objective function
    :param kwargs: additional keyword arguments passed to the algorithm
        allowed keywords : 
            accept_ratio = 0.1 : float
            Todo: fill this doc for accept_ratio
    :return: namedtuple 
        converged:bool
            Indicates whether algorithm converged or not
        x_new:float or None 
            value of x for which objective is minimized or None if algorithm doesn't converge
    """
    return LineSearchResult(backtracking_line_search(f, x0, s, max_step_size, gf_x0,
                                                     max_iter, f_args, kwargs['accept_ratio']))


def backtracking_line_search(f, x0, s, beta, gf_x0, max_iter, f_args, accept_ratio=0.1):
    step_lengths = 0.5 ** np.arange(max_iter)  # exponentially decaying
    for i, delta in enumerate(step_lengths):
        x_new = x0 + delta * (beta * s)
        delta_f = f(x0, *f_args) - f(x_new, *f_args)
        expected_delta_f = gf_x0 * delta
        ratio = delta_f / expected_delta_f

        if ratio > accept_ratio and delta_f > 0:
            return True, x_new

    return False, None


class LbfgsOptimizer(trpo_utils.EzFlat):
    def __init__(self, loss, params, symb_args, extra_losses=None, maxiter=25):
        trpo_utils.EzFlat.__init__(self, params)
        self.all_losses = OrderedDict()
        self.all_losses["loss"] = loss
        self.symb_args = symb_args
        self.loss = loss
        self.params = params
        if extra_losses is not None:
            self.all_losses.update(extra_losses)
        self.maxiter = maxiter

    def f_lossgrad(self, *args):
        with tf.Session() as sess:
            return sess.run([self.loss, trpo_utils.flat_gradient(self.loss, self.params)], feed_dict=args)

    def f_losses(self, *args):
        with tf.Session() as sess:
            return sess.run(self.all_losses.values(), feed_dict=args)

    def update(self, *args):
        thprev = self.get_params_flat()

        def lossandgrad(th):
            self.set_params_flat(th)
            l, g = self.f_lossgrad(*args)
            g = g.astype('float64')
            return l, g

        losses_before = self.f_losses(*args)
        theta, _, opt_info = scipy.optimize.fmin_l_bfgs_b(lossandgrad, thprev, maxiter=self.maxiter)
        del opt_info['grad']
        self.set_params_flat(theta)
        losses_after = self.f_losses(*args)
        info = OrderedDict()
        for (name, lossbefore, lossafter) in zip(self.all_losses.keys(), losses_before, losses_after):
            info[name + "_before"] = lossbefore
            info[name + "_after"] = lossafter
        return info
