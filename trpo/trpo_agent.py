"""
In this codebase, the "Agent" is a container with the policy, value function, etc.
This file contains a bunch of agents
"""
import pickle

import tensorflow as tf

from src.py3.algos.trpo.trpo import TRPOptimizer, TRPOParams
from src.py3.algos.trpo.utils import ZFilter, update_default_config
from src.py3.algos.trpo.utils.keras_utils import make_mlps


def comma_sep_ints(s):
    if s:
        return map(int, s.split(","))
    else:
        return []


MLP_OPTIONS = [
    ("hid_sizes", comma_sep_ints, [64, 64], "Sizes of hidden layers of MLP"),
    ("activation", str, "tanh", "nonlinearity")
]

# def make_deterministic_mlp(ob_space, ac_space, cfg):
#     assert isinstance(ob_space, Box)
#     hid_sizes = cfg["hid_sizes"]
#     if isinstance(ac_space, Box):
#         outdim = ac_space.shape[0]
#         probtype = DiagonalGaussian(outdim)
#     elif isinstance(ac_space, Discrete):
#         outdim = ac_space.n
#         probtype = Categorical(outdim)
#     else:
#         raise NotImplementedError
#     net = Sequential()
#     for (i, layeroutsize) in enumerate(hid_sizes):
#         inshp = dict(input_shape=ob_space.shape) if i == 0 else {}
#         net.add(Dense(layeroutsize, activation="tanh", **inshp))
#     inshp = dict(input_shape=ob_space.shape) if len(hid_sizes) == 0 else {}
#     net.add(Dense(outdim, **inshp))
#     Wlast = net.layers[-1].W
#     Wlast.set_value(Wlast.get_value(borrow=True) * 0.1)
#     policy = StochasticPolicy(net, probtype)
#     return policy


FILTER_OPTIONS = [
    ("filter", int, 1, "Whether to do a running average filter of the incoming observations and rewards")
]

PG_OPTIONS = [
    ("timestep_limit", int, 0, "maximum length of trajectories"),
    ("n_iter", int, 200, "number of batch"),
    ("parallel", int, 0, "collect trajectories in parallel"),
    ("timesteps_per_batch", int, 10000, ""),
    ("gamma", float, 0.99, "discount"),
    ("lam", float, 1.0, "lambda parameter from generalized advantage estimation"),
]


def make_filters(cfg, ob_space):
    if cfg["filter"]:
        obfilter = ZFilter(ob_space.shape, clip=5)
        rewfilter = ZFilter((), demean=False, clip=10)
    else:
        obfilter = lambda x: x
        rewfilter = lambda x: x
    return obfilter, rewfilter


class AgentWithPolicy(object):
    def __init__(self, policy, obfilter, rewfilter):
        self.policy = policy
        self.obfilter = obfilter
        self.rewfilter = rewfilter
        self.stochastic = True

    def set_stochastic(self, stochastic):
        self.stochastic = stochastic

    def act(self, ob_no):
        return self.policy.act(ob_no, stochastic=self.stochastic)

    def get_flat(self):
        return self.policy.get_flat()

    def set_from_flat(self, th):
        return self.policy.set_from_flat(th)

    def obfilt(self, ob):
        return self.obfilter(ob)

    def rewfilt(self, rew):
        return self.rewfilter(rew)


class TrpoAgent(AgentWithPolicy):
    options = MLP_OPTIONS + PG_OPTIONS + TRPOParams().as_list() + FILTER_OPTIONS

    def __init__(self, ob_space, ac_space, usercfg, graph: tf.Graph, restore_path=None):
        cfg = update_default_config(self.options, usercfg)
        if restore_path is None:
            policy, self.baseline = make_mlps(ob_space, ac_space, cfg, graph=graph)
        else:
            with open("{}_policy.pkl".format(restore_path), 'rb') as pf:
                policy = pickle.load(pf)
            with open("{}_baseline.pkl".format(restore_path), 'rb') as bf:
                self.baseline = pickle.load(bf)
        obfilter, rewfilter = make_filters(cfg, ob_space)
        self.updater = TRPOptimizer(policy, graph=graph, cg_damping=cfg['cg_damping'], max_kl=cfg['max_kl'])
        AgentWithPolicy.__init__(self, policy, obfilter, rewfilter)
