import backend as b
import trpo.utils.math_utils as mutils
from trpo.utils.tf_utils import define_scope_and_cache


class TRPO_GAE:
    def __init__(self):
        self.init_params()
        self.old_policy, self.current_policy = [None] * 2
        for i in range(self.n_episdoes):
            self.train_step()

    @define_scope_and_cache
    def placeholders(self):
        pass

    def train_step(self):
        sample_data = self.simulate(self.current_policy)
        advantage_estimates = self.compute_advantage_estimates(self.current_value_function)
        gae = self.compute_generalized_advantage_estimate(advantage_estimates)
        self.update_policy(gae, sample_data)
        self.update_value_function()

    def update_policy(self, gae, sample_data, epsilon=1e-8):
        # Compute loss
        # # log difference is ratio
        log_likelihood_ratio = self.current_policy.log_likelihood() - self.old_policy.log_likelihood()
        likelihood_ratio = b.exp(log_likelihood_ratio)
        loss = -b.reduce_mean(likelihood_ratio * gae)

        # Calculate constraints
        kl_div = mutils.kl_divergence(self.old_policy, self.current_policy) / self.batch_size_float

        policy_gradient = self.current_policy.gradient(loss)
        policy_gradient_feeds = [self.current_policy.inputs + self.gae_placeholder + self.placeholders]
        policy_gradient_feed_dict.update()
        self.session.run(policy_gradient, feed_dict=policy_gradient_feed_dict)

