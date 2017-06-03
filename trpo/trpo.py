import tensorflow as tf


class TRPOParams:
    def __init__(self, cg_damping=1e-3, max_kl=1e-2, **kwargs):
        self.cg_damping = cg_damping  # Add multiple of the identity to Fisher matrix during CG
        self.max_kl = max_kl  # KL divergence between old and new policy (averaged over state-space)

    def as_list(self):
        return [(key, type(value), value, '') for key, value in self.__dict__.items()]


class TRPO:
    def __init__(self, stoch_pol):
        self.stoch_pol = stoch_pol

    def update_policy(self):
        pass


def fisher_vector_product(pn_params):
    """
    # The actual Fisher-vector product operation, where the gradients are
    # taken w.r.t. the "loss" function `gvp`. I _think_ the `grads` from
    # above computes the first derivatives, and then the `gvp` is computing
    # the second derivatives. But what about hessian_vector_product?
    :param pn_params: 
    :return: 
    """
    # Do elementwise g*tangent then sum components, then add everything at the end.
    # John Schulman used T.add(*[...]). The TF equivalent seems to be tf.add_n.
    assert len(grads) == len(tangents)
    gradient_vector_product = tf.add_n(inputs=[tf.reduce_sum(g * tangent) for (g, tangent) in zip(grads, tangents)])

    # The actual Fisher-vector product operation, where the gradients are
    # taken w.r.t. the "loss" function `gvp`. I _think_ the `grads` from
    # above computes the first derivatives, and then the `gvp` is computing
    # the second derivatives. But what about hessian_vector_product?
    return flatgrad(gradient_vector_product, pn_params)
