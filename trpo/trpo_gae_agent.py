from itertools import chain

import tensorflow as tf
import trpo.utils.math_utils as mutils
from trpo.utils import tf_utils
from trpo.optimizers.tensorflow import conjugate_gradients, backtracking_linesearch


class TRPO_GAE:
    def __init__(self):
        self.init_params()
        self.old_policy, self.current_policy = [None] * 2
        for i in range(self.n_episdoes):
            self.train_step()

    @tf_utils.define_scope_and_cache
    def placeholders(self):
        # Todo
        raise NotImplementedError

    def fisher_vector_product(self, vector):
        # Todo
        raise NotImplementedError

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
        likelihood_ratio = tf.exp(log_likelihood_ratio)
        loss = -tf.reduce_mean(likelihood_ratio * gae)

        # Calculate constraints
        kl_div = mutils.kl_divergence(self.old_policy, self.current_policy) / self.batch_size_float

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


