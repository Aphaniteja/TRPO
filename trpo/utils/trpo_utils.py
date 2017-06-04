import numpy as np
from trpo.utils import tf_utils


class TRPOSampleMetrics:
    def __init__(self, reward, baseline):
        self.reward = reward
        self.baseline = baseline

    @property
    def discounted_reward(self):
        return tf_utils.tf_discount(self.reward)

    @property
    def advantage(self):
        return self.reward - self.baseline


class TRPOEpisodeMetrics:
    def __init__(self, samples):
        self.samples = samples
        advantage_estimate = np.fromiter((sample.advantage for sample in samples), dtype=tf_utils.DEFAULT_DTYPE)
        self.generalized_advantage_estimate = ((advantage_estimate - advantage_estimate.mean()) /
                                               (advantage_estimate.std() + 1e-8))
