import warnings

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as at
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import (
    Distribution,
    SymbolicRandomVariable,
    _moment,
)
from pymc.distributions.logprob import ignore_logprob, logp
from pymc.distributions.shape_utils import (
    _change_dist_size,
    change_dist_size,
    get_support_shape_1d,
)
from pymc.logprob.abstract import _logprob
from pymc.util import check_dist_not_registered
from pytensor.graph.basic import Node
from pytensor.tensor import TensorVariable
from pytensor.tensor.random.op import RandomVariable


class DiscreteMarkovChainRV(SymbolicRandomVariable):
    default_output = 1
    _print_name = ("DiscreteMC", "\\operatorname{DiscreteMC}")

    def update(self, node: Node):
        return {node.inputs[-1]: node.outputs[0]}


class DiscreteMarkovChain(Distribution):
    r"""
    A Discrete Markov Chain is a sequence of random variables
    .. math::
        \{x_t\}_{t=0}^T
    Where transition probability P(x_t | x_{t-1}) depends only on the state of the system at x_{t-1}.

    Parameters
    ----------
    P: tensor
        Matrix of transition probabilities between states. Rows must sum to 1.
        One of P or P_logits must be provided.
    P_logit: tensor, optional
        Matrix of tranisiton logits. Converted to probabilities via Softmax activation.
        One of P or P_logits must be provided.
    steps: tensor, optional
        Length of the markov chain. Only needed if state is not provided.
    init_dist : unnamed distribution, optional
        Vector distribution for initial values. Unnamed refers to distributions
        created with the ``.dist()`` API. Distribution should have shape n_states.
        If not, it will be automatically resized. Defaults to pm.Categorical.dist(p=np.full(n_states, 1/n_states)).
        .. warning:: init_dist will be cloned, rendering it independent of the one passed as input.
    """

    rv_type = DiscreteMarkovChainRV

    def __new__(cls, *args, steps=None, **kwargs):
        # TODO: Allow steps to be None and infer chain length from shape?
        # TODO: Dims breaks the RV

        # Subtract 1 step to account for x0 given, better match user expectation of
        # len(markov_chain) = steps
        if steps is not None:
            steps -= 1

        steps = get_support_shape_1d(
            support_shape=steps,
            shape=None,
            dims=kwargs.get("dims", None),
            observed=kwargs.get("observed", None),
            support_shape_offset=1,
        )

        return super().__new__(cls, *args, steps=steps, **kwargs)

    @classmethod
    def dist(cls, P=None, logit_P=None, steps=None, init_dist=None, **kwargs):

        shape = kwargs.get("shape", None)

        steps = get_support_shape_1d(support_shape=steps, shape=shape, support_shape_offset=1)

        batch_size = None
        if shape is not None:
            batch_size = shape[1:] if shape[0] == steps else shape

        # TODO: Was getting errors with int32 vs int64 mismatches, is there a better way to address this?
        dtype = kwargs.get("dtype", None) or pytensor.config.floatX.replace("float", "int")

        if steps is None:
            raise ValueError("Must specify steps or shape parameter")
        if P is None and logit_P is None:
            raise ValueError("Must specify P or logit_P parameter")
        if P is not None and logit_P is not None:
            raise ValueError("Must specify only one of either P or logit_P parameter")

        if logit_P is not None:
            P = pm.math.softmax(logit_P, axis=1)

        P = at.as_tensor_variable(P)
        steps = at.as_tensor_variable(steps.astype(dtype), ndim=1)

        if init_dist is not None:
            if not isinstance(init_dist, TensorVariable) or not isinstance(
                init_dist.owner.op, (RandomVariable, SymbolicRandomVariable)
            ):
                raise ValueError(
                    f"Init dist must be a distribution created via the `.dist()` API, "
                    f"got {type(init_dist)}"
                )
            check_dist_not_registered(init_dist)
            if init_dist.owner.op.ndim_supp > 1:
                raise ValueError(
                    "Init distribution must have a scalar or vector support dimension, ",
                    f"got ndim_supp={init_dist.owner.op.ndim_supp}.",
                )
        else:
            warnings.warn(
                "Initial distribution not specified, defaulting to "
                "`Categorical.dist(p=at.full((k_states, ), 1/k_states), shape=...)`. You can specify an init_dist "
                "manually to suppress this warning.",
                UserWarning,
            )
            k = P.shape[0]
            init_dist = pm.Categorical.dist(p=at.full((k,), 1 / k), shape=batch_size, dtype=dtype)

        # We can ignore init_dist, as it will be accounted for in the logp term
        init_dist = ignore_logprob(init_dist)

        return super().dist([P, steps, init_dist], **kwargs)

    @classmethod
    def rv_op(cls, P, steps, init_dist, size=None):
        if size is not None:
            batch_size = size
        else:
            batch_size = (1,)

        if init_dist.owner.op.ndim_supp == 0:
            init_dist_size = (*batch_size,)
        else:
            init_dist_size = batch_size

        init_dist = change_dist_size(init_dist, init_dist_size)

        init_dist_ = init_dist.type()
        P_ = P.type()
        steps_ = steps.type()

        state_rng = pytensor.shared(np.random.default_rng())

        def transition(previous_state, transition_probs, old_rng):
            p = transition_probs[previous_state]
            next_rng, next_state = pm.Categorical.dist(p=p, rng=old_rng).owner.outputs
            return next_state, {old_rng: next_rng}

        markov_chain, state_updates = pytensor.scan(
            transition,
            non_sequences=[P_, state_rng],
            outputs_info=[init_dist_],
            n_steps=steps_,
            strict=True,
        )

        (state_next_rng,) = tuple(state_updates.values())

        discrete_mc_ = (
            at.concatenate([init_dist_[None, ...], markov_chain], axis=0)
            .dimshuffle(tuple(range(1, markov_chain.ndim)) + (0,))
            .squeeze()
        )

        discrete_mc_op = DiscreteMarkovChainRV(
            inputs=[P_, init_dist_, steps_],
            outputs=[state_next_rng, discrete_mc_],
            ndim_supp=1,
        )

        discrete_mc = discrete_mc_op(P, init_dist, steps)
        return discrete_mc


@_change_dist_size.register(DiscreteMarkovChainRV)
def change_mc_size(op, dist, new_size, expand=False):
    if expand:
        old_size = dist.shape[:-1]
        new_size = tuple(new_size) + tuple(old_size)

    return DiscreteMarkovChainRV.rv_op(
        *dist.owner.inputs[:-1],
        size=new_size,
    )


@_logprob.register(DiscreteMarkovChainRV)
def discrete_mc_logp(op, values, P, init_dist, steps, state_rng, **kwargs):
    n, k = P.shape

    (value,) = values

    mc_logprob = logp(init_dist, value[..., 0])
    mc_logprob += at.log(P[value[..., :-1], value[..., 1:]]).sum(axis=-1)

    return check_parameters(
        mc_logprob,
        at.eq(n, k),
        at.all(at.allclose(P.sum(axis=1), 1.0)),
        msg="P must be square with rows that sum to 1",
    )


@_moment.register(DiscreteMarkovChainRV)
def discrete_markov_chain_moment(op, rv, P, init_dist, steps, state_rng):
    return at.zeros_like(rv)
