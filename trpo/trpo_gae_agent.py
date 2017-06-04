from itertools import chain

import tensorflow as tf
from tensorflow.contrib.distributions import kl as kl_divergence

from trpo.utils import tf_utils
from trpo.optimizers.tensorflow import conjugate_gradients, backtracking_linesearch


class TRPO_GAE:
    def __init__(self):
        # Initialize all variables used in this class
        self.old_policy, self.current_policy = [None] * 2
        self.current_value_function = None
        self.batch_size_float = None
        self.args = None

        self.init_params()

    @tf_utils.define_scope_and_cache
    def placeholders(self):
        # Todo
        raise NotImplementedError

    def fit(self, n_episodes, collect_statistics=False):
        for i in range(n_episodes):
            sample_data = self.simulate(self.current_policy)  # TODO: Check that sample_data is not changed in the
            # # subsequent functions during this iteration.
            gae = self.compute_generalized_advantage_estimate(sample_data)
            self.update_policy(gae, sample_data)
            self.update_value_function(sample_data)

    def compute_generalized_advantage_estimate(self, sample_data):
        # TODO: Currently working on this.
        raise NotImplementedError

    def update_policy(self, gae, sample_data, epsilon=1e-8):
        # Compute loss
        # # log difference is ratio
        log_likelihood_ratio = self.current_policy.log_likelihood() - self.old_policy.log_likelihood()
        likelihood_ratio = tf.exp(log_likelihood_ratio)
        loss = -tf.reduce_mean(likelihood_ratio * gae)

        policy_gradient = self.current_policy.gradient(loss)  # gradient wrt parameters of policy
        if tf_utils.tf_allclose(policy_gradient, 0):
            print("Got zero gradient, not updating!!!")  # TODO: better logging

        # Solve Ax = g where A is Fisher information matrix and g is the gradient of parameters of policy to get
        # step direction for descent as step_dir = A_inverse * g
        step_dir = conjugate_gradients(self.fisher_vector_product, -policy_gradient)
        sd_fvp_sd = tf_utils.tf_vector_product(step_dir, self.fisher_vector_product(step_dir))
        max_step_len = tf.sqrt(sd_fvp_sd / self.args.max_kl)
        full_step = step_dir / max_step_len
        expected_improve_rate = tf_utils.tf_vector_product(-policy_gradient, step_dir) / max_step_len
        theta_new = backtracking_linesearch(loss, self.old_policy.parameters, full_step, expected_improve_rate)

        # Update policies now
        self.old_policy = self.current_policy.copy()  # TODO: Do I need to make a copy here?
        self.current_policy.set_parameters(theta_new)  # TODO: maybe change this to instantiating a new policy instead?

    def update_value_function(self, sample_data):
        self.current_value_function.fit(sample_data)

    def fisher_vector_product(self, vector):
        # Todo
        raise NotImplementedError
