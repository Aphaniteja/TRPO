"""
Implements base classes for StochasticPolicy and ...? and required stuff 
"""
from gym.utils.ezpickle import EzPickle
import tensorflow as tf

from src.py3.algos.trpo.distributions import Distribution


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
            return self.distribution.maxprob(prob)[0], {"prob": prob[0]}

    def finalize(self):
        with tf.Session() as sess:
            self._act_prob = sess.run(fetches=self.output,
                                      feed_dict=self.input)
