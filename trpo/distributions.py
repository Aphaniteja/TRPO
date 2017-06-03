"""
A bunch of distributions.
# Todo: replace these with tf Distributions and tf.contrib.crf
"""
import numpy as np
import tensorflow as tf
from tensorflow.contrib import distributions


class Distribution:
    def likelihood(self, a, prob):
        raise NotImplementedError

    def log_likelihood(self, a, prob):
        raise NotImplementedError

    def kl_divergence(self, prob0, prob1):
        raise NotImplementedError

    def entropy(self, prob):
        raise NotImplementedError

    def max_prob(self, prob):
        raise NotImplementedError


class Categorical(distributions.Categorical, Distribution):
    def __init__(self, *args, **kwargs):
        distributions.Categorical.__init__(*args, **kwargs)

    def sampled_variable(self):
        return tf.placeholder(dtype=tf.int32, shape=(None, 1), name='sampled_action')

    def prob_variable(self):
        # return self.p
        return tf.placeholder(dtype=(None, None), name='transition_probability')

    def likelihood(self, a, prob):
        return prob[tf.range(tf.shape(prob)[0]), a]

    def log_likelihood(self, a, prob):
        return tf.log(self.likelihood(a, prob), name='log_likelihood')

    def kl_divergence(self, prob0, prob1):
        return tf.reduce_sum(distributions.kl(prob0, prob1, name='kl_divergence'), axis=1)

    def entropy(self, prob=None):
        if prob is None:
            return distributions.Categorical.entropy(self, name='tf_entropy')
        return -tf.reduce_sum((prob * tf.log(prob)), axis=1)

    # def sample(self, prob):
    #     return distributions.categorical_sample(prob)
    def max_prob(self, prob=None):
        if prob is None:
            return tf.argmax(self.p, name='tf_max_prob')
        return tf.argmax(prob, axis=1, name='max_prob')


class DiagonalGaussian(Distribution):
    def __init__(self, d, graph:tf.Graph):
        self._graph = graph
        self.d = d

    def sampled_variable(self):
        with self._graph.as_default():
            return tf.placeholder(dtype=tf.float32, shape=(None, None), name='sampled_action')

    def prob_variable(self):
        with self._graph.as_default():
            return tf.placeholder(dtype=tf.float32, shape=(None, None), name='prob_variable')

    def log_likelihood(self, a, prob):
        mean0 = prob[:, :self.d]
        std0 = prob[:, self.d:]

        term_1 = - tf.reduce_sum(0.5 * tf.square((a - mean0) / std0), axis=1)
        term_2 = - 0.5 * tf.log(2.0 * np.pi) * self.d
        term_3 = tf.reduce_sum(tf.log(std0), axis=1)
        return term_1 + term_2 - term_3

    def likelihood(self, a, prob):
        return tf.exp(self.log_likelihood(a, prob))

    def kl_divergence(self, prob0, prob1):
        mean0 = prob0[:, :self.d]
        std0 = prob0[:, self.d:]
        mean1 = prob1[:, :self.d]
        std1 = prob1[:, self.d:]
        from functools import partial
        sum_ax_1 = partial(tf.reduce_sum, axis=1)

        term_1 = sum_ax_1(tf.log(std1 / std0))

        variance0 = tf.square(std0)
        squared_dissimilarity = tf.square(mean0 - mean1)  # Mahalanobis distance
        numerator = (variance0 + squared_dissimilarity)
        denominator = (2.0 * tf.square(std1))
        term_2 = sum_ax_1(numerator / denominator)

        return term_1 + term_2 - 0.5 * self.d

    def entropy(self, prob):
        std_nd = prob[:, self.d:]
        term_1 = tf.reduce_sum(tf.log(std_nd), axis=1)
        term_2 = .5 * np.log(2 * np.pi * np.e) * self.d
        return term_1 + term_2

    def sample(self, prob):
        mean_nd = prob[:, :self.d]
        std_nd = prob[:, self.d:]
        return np.random.randn(prob.shape[0], self.d).astype(np.float32) * std_nd + mean_nd

    def max_prob(self, prob):
        return prob[:, :self.d]
