import warnings
from typing import List, Union

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import (
    Distribution,
    SymbolicRandomVariable,
    _moment,
    moment,
)
from pymc.distributions.shape_utils import (
    _change_dist_size,
    change_dist_size,
    get_support_shape_1d,
)
from pymc.logprob.abstract import _logprob
from pymc.logprob.basic import logp
from pymc.pytensorf import intX
from pymc.util import check_dist_not_registered
from pytensor.graph.basic import Node
from pytensor.tensor import TensorVariable
from pytensor.tensor.random.op import RandomVariable


def _make_outputs_info(n_lags: int, init_dist: Distribution) -> List[Union[Distribution, dict]]:
    """
    Two cases are needed for outputs_info in the scans used by DiscreteMarkovRv. If n_lags = 1, we need to throw away
    the first dimension of init_dist_ or else markov_chain will have shape (steps, 1, *batch_size) instead of
    desired (steps, *batch_size)

    Parameters
    ----------
    n_lags: int
        Number of lags the Markov Chain considers when transitioning to the next state
    init_dist: RandomVariable
        Distribution over initial states

    Returns
    -------
    taps: list
        Lags to be fed into pytensor.scan when drawing a markov chain
    """

    if n_lags > 1:
        return [{"initial": init_dist, "taps": list(range(-n_lags, 0))}]
    else:
        return [init_dist[0]]


class DiscreteMarkovChainRV(SymbolicRandomVariable):
    n_lags: int
    default_output = 1
    _print_name = ("DiscreteMC", "\\operatorname{DiscreteMC}")

    def __init__(self, *args, n_lags, **kwargs):
        self.n_lags = n_lags
        super().__init__(*args, **kwargs)

    def update(self, node: Node):
        return {node.inputs[-1]: node.outputs[0]}


class DiscreteMarkovChain(Distribution):
    r"""
    A Discrete Markov Chain is a sequence of random variables

    .. math::

        \{x_t\}_{t=0}^T

    Where transition probability :math:`P(x_t | x_{t-1})` depends only on the state of the system at :math:`x_{t-1}`.

    Parameters
    ----------
    P: tensor
        Matrix of transition probabilities between states. Rows must sum to 1.
        One of P or P_logits must be provided.
    P_logit: tensor, optional
        Matrix of transition logits. Converted to probabilities via Softmax activation.
        One of P or P_logits must be provided.
    steps: tensor, optional
        Length of the markov chain. Only needed if state is not provided.
    init_dist : unnamed distribution, optional
        Vector distribution for initial values. Unnamed refers to distributions
        created with the ``.dist()`` API. Distribution should have shape n_states.
        If not, it will be automatically resized.

        .. warning:: init_dist will be cloned, rendering it independent of the one passed as input.

    Notes
    -----
    The initial distribution will be cloned, rendering it distinct from the one passed as
    input.

    Examples
    --------
     Create a Markov Chain of length 100 with 3 states. The number of states is given by the shape of P,
     3 in this case.

    >>> with pm.Model() as markov_chain:
    >>>     P = pm.Dirichlet("P", a=[1, 1, 1], size=(3,))
    >>>     init_dist = pm.Categorical.dist(p = np.full(3, 1 / 3))
    >>>     markov_chain = pm.DiscreteMarkovChain("markov_chain", P=P, init_dist=init_dist, shape=(100,))

    """

    rv_type = DiscreteMarkovChainRV

    def __new__(cls, *args, steps=None, n_lags=1, **kwargs):
        steps = get_support_shape_1d(
            support_shape=steps,
            shape=None,
            dims=kwargs.get("dims", None),
            observed=kwargs.get("observed", None),
            support_shape_offset=n_lags,
        )

        return super().__new__(cls, *args, steps=steps, n_lags=n_lags, **kwargs)

    @classmethod
    def dist(cls, P=None, logit_P=None, steps=None, init_dist=None, n_lags=1, **kwargs):
        steps = get_support_shape_1d(
            support_shape=steps, shape=kwargs.get("shape", None), support_shape_offset=n_lags
        )

        if steps is None:
            raise ValueError("Must specify steps or shape parameter")
        if P is None and logit_P is None:
            raise ValueError("Must specify P or logit_P parameter")
        if P is not None and logit_P is not None:
            raise ValueError("Must specify only one of either P or logit_P parameter")

        if logit_P is not None:
            P = pm.math.softmax(logit_P, axis=-1)

        P = pt.as_tensor_variable(P)
        steps = pt.as_tensor_variable(intX(steps))

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
                "`Categorical.dist(p=pt.full((k_states, ), 1/k_states), shape=...)`. You can specify an init_dist "
                "manually to suppress this warning.",
                UserWarning,
            )
            k = P.shape[-1]
            init_dist = pm.Categorical.dist(p=pt.full((k,), 1 / k))

        return super().dist([P, steps, init_dist], n_lags=n_lags, **kwargs)

    @classmethod
    def rv_op(cls, P, steps, init_dist, n_lags, size=None):
        if size is not None:
            batch_size = size
        else:
            batch_size = pt.broadcast_shape(
                P[tuple([...] + [0] * (n_lags + 1))], pt.atleast_1d(init_dist)[..., 0]
            )

        init_dist = change_dist_size(init_dist, (n_lags, *batch_size))
        init_dist_ = init_dist.type()
        P_ = P.type()
        steps_ = steps.type()

        state_rng = pytensor.shared(np.random.default_rng())

        def transition(*args):
            *states, transition_probs, old_rng = args
            p = transition_probs[tuple(states)]
            next_rng, next_state = pm.Categorical.dist(p=p, rng=old_rng).owner.outputs
            return next_state, {old_rng: next_rng}

        markov_chain, state_updates = pytensor.scan(
            transition,
            non_sequences=[P_, state_rng],
            outputs_info=_make_outputs_info(n_lags, init_dist_),
            n_steps=steps_,
            strict=True,
        )

        (state_next_rng,) = tuple(state_updates.values())

        discrete_mc_ = pt.moveaxis(pt.concatenate([init_dist_, markov_chain], axis=0), 0, -1)

        discrete_mc_op = DiscreteMarkovChainRV(
            inputs=[P_, steps_, init_dist_],
            outputs=[state_next_rng, discrete_mc_],
            ndim_supp=1,
            n_lags=n_lags,
        )

        discrete_mc = discrete_mc_op(P, steps, init_dist)
        return discrete_mc


@_change_dist_size.register(DiscreteMarkovChainRV)
def change_mc_size(op, dist, new_size, expand=False):
    if expand:
        old_size = dist.shape[:-1]
        new_size = tuple(new_size) + tuple(old_size)

    return DiscreteMarkovChain.rv_op(*dist.owner.inputs[:-1], size=new_size, n_lags=op.n_lags)


@_moment.register(DiscreteMarkovChainRV)
def discrete_mc_moment(op, rv, P, steps, init_dist, state_rng):
    init_dist_moment = moment(init_dist)
    n_lags = op.n_lags

    def greedy_transition(*args):
        *states, transition_probs, old_rng = args
        p = transition_probs[tuple(states)]
        return pt.argmax(p)

    chain_moment, moment_updates = pytensor.scan(
        greedy_transition,
        non_sequences=[P, state_rng],
        outputs_info=_make_outputs_info(n_lags, init_dist),
        n_steps=steps,
        strict=True,
    )
    chain_moment = pt.concatenate([init_dist_moment, chain_moment])
    return chain_moment


@_logprob.register(DiscreteMarkovChainRV)
def discrete_mc_logp(op, values, P, steps, init_dist, state_rng, **kwargs):
    value = values[0]
    n_lags = op.n_lags

    indexes = [value[..., i : -(n_lags - i) if n_lags != i else None] for i in range(n_lags + 1)]

    mc_logprob = logp(init_dist, value[..., :n_lags]).sum(axis=-1)
    mc_logprob += pt.log(P[tuple(indexes)]).sum(axis=-1)

    return check_parameters(
        mc_logprob,
        pt.all(pt.eq(P.shape[-(n_lags + 1) :], P.shape[-1])),
        pt.all(pt.allclose(P.sum(axis=-1), 1.0)),
        pt.eq(pt.atleast_1d(init_dist).shape[0], n_lags),
        msg="Last (n_lags + 1) dimensions of P must be square, "
        "P must sum to 1 along the last axis, "
        "First dimension of init_dist must be n_lags",
    )
