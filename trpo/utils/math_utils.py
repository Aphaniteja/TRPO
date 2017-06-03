import numpy as np
import scipy.signal
import tensorflow as tf
from gym.utils import EzPickle

from src.py3.algos.trpo import optimizers


def discount(x, gamma):
    """
    computes discounted sums along 0th dimension of x.

    inputs
    ------
    x: ndarray
    gamma: float

    outputs
    -------
    y: ndarray with same shape as x, satisfying

        y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
                where k = len(x) - t - 1

    """
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def explained_variance(ypred, y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


def explained_variance_2d(ypred, y):
    assert y.ndim == 2 and ypred.ndim == 2
    vary = np.var(y, axis=0)
    out = 1 - np.var(y - ypred) / vary
    out[vary < 1e-10] = 0
    return out


class NnRegression(EzPickle):
    def __init__(self, net, graph: tf.Graph, mixfrac=1.0, maxiter=25):
        EzPickle.__init__(self, net, mixfrac, maxiter)
        self.net = net
        self.mixfrac = mixfrac

        self.x_nx = net.input

        ypred_ny = net
        with graph.as_default():
            ytarg_ny = tf.placeholder(dtype=ypred_ny.dtype, shape=(None, None))  # T.matrix("ytarg")
        var_list = net.trainable_weights
        l2 = 1e-3 * tf.add_n([tf.reduce_sum(tf.square(v)) for v in var_list])
        N = tf.cast(tf.shape(self.x_nx)[0], dtype=tf.float32)
        mse = tf.reduce_sum(tf.square(ytarg_ny - ypred_ny)) / N
        symb_args = [self.x_nx, ytarg_ny]
        loss = mse + l2
        self.opt = optimizers.LbfgsOptimizer(loss, var_list, symb_args, maxiter=maxiter,
                                             extra_losses={"mse": mse, "l2": l2})

    @property
    def predict(self, *args):
        with tf.Session() as sess:
            return sess.run(self.net.output, feed_dict=args)

    def fit(self, x_nx, ytarg_ny):
        nY = ytarg_ny.shape[1]
        ypredold_ny = self.predict(x_nx)
        out = self.opt.update(x_nx, ytarg_ny * self.mixfrac + ypredold_ny * (1 - self.mixfrac))
        yprednew_ny = self.predict(x_nx)
        out["PredStdevBefore"] = ypredold_ny.std()
        out["PredStdevAfter"] = yprednew_ny.std()
        out["TargStdev"] = ytarg_ny.std()
        if nY == 1:
            out["EV_before"] = explained_variance_2d(ypredold_ny, ytarg_ny)[0]
            out["EV_after"] = explained_variance_2d(yprednew_ny, ytarg_ny)[0]
        else:
            out["EV_avg"] = explained_variance(yprednew_ny.ravel(), ytarg_ny.ravel())
        return out


class NnVf(object):
    def __init__(self, net, timestep_limit, graph: tf.Graph, regression_params):
        self.reg = NnRegression(net, graph=graph, **regression_params)
        self.timestep_limit = timestep_limit

    def predict(self, path):
        ob_no = self.preproc(path["observation"])
        return self.reg.predict(ob_no)[:, 0]

    def fit(self, paths):
        ob_no = np.concatenate([self.preproc(path["observation"]) for path in paths], axis=0)
        vtarg_n1 = np.concatenate([path["return"] for path in paths]).reshape(-1, 1)
        return self.reg.fit(ob_no, vtarg_n1)

    def preproc(self, ob_no):
        return np.concatenate([ob_no, np.arange(len(ob_no)).reshape(-1, 1) / float(self.timestep_limit)], axis=1)
