import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as at
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import (
    Discrete,
    Distribution,
    SymbolicRandomVariable,
    _moment,
)
from pymc.distributions.shape_utils import _change_dist_size, get_support_shape_1d
from pymc.logprob.abstract import _logprob
from pymc.pytensorf import intX
from pytensor.graph.basic import Node
from pytensor.tensor import TensorVariable


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
    P_logit: tensor, Optional
        Matrix of tranisiton logits. Converted to probabilities via Softmax activation.
        One of P or P_logits must be provided.
    steps: tensor
        Length of the markov chain
    x0: tensor or RandomVariable
        Intial state of the system. If tensor, treated as deterministic.
    """

    rv_type = DiscreteMarkovChainRV

    def __new__(cls, *args, steps, **kwargs):
        # TODO: Allow steps to be None and infer chain length from shape?
        # TODO: Dims breaks the RV

        # Subtract 1 step to account for x0 given, better match user expectation of
        # len(markov_chain) = steps
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
    def dist(cls, P=None, logit_P=None, steps=None, x0=None, **kwargs):

        steps = get_support_shape_1d(
            support_shape=steps, shape=kwargs.get("shape", None), support_shape_offset=1
        )
        if steps is None:
            raise ValueError("Must specify steps or shape parameter")
        if P is None and logit_P is None:
            raise ValueError("Must specify P or logit_P parameter")
        if P is not None and logit_P is not None:
            raise ValueError("Must specify only one of either P or logit_P parameter")

        if logit_P is not None:
            P = pm.math.softmax(logit_P, axis=1)
        P = at.as_tensor_variable(P)

        # TODO: Can this eval be avoided?
        n_states = P.shape[0].eval()

        if not isinstance(x0, TensorVariable):
            x0 = at.as_tensor_variable(intX(x0))

            # TODO: Can this eval be avoided?
            if not at.all(at.lt(x0, n_states - 1)).eval():
                raise ValueError(
                    "At least one initial state is larger than the number of states in the Markov Chain"
                )

        elif not isinstance(x0.owner.op, Discrete):
            raise ValueError("x0 must be a discrete distribution")

        else:
            x0_probs = x0.owner.inputs[-1].eval()
            n_cats = 1 if x0_probs.ndim == 0 else len(x0_probs)

            if not n_cats <= n_states:
                raise ValueError(
                    "x0 has support over a range of values larger than the number of states in the Markov Chain"
                )

        return super().dist([P, logit_P, steps, x0], **kwargs)

    @classmethod
    def rv_op(cls, P, logit_P, steps, x0, size=None):
        if size is not None:
            batch_size = size
        else:
            batch_size = at.broadcast_shape(x0)

        x0_ = x0.type()
        P_ = P.type()
        steps_ = steps.type()

        state_rng = pytensor.shared(np.random.default_rng())

        def transition(previous_state, transition_probs, old_rng):
            p = transition_probs[previous_state]
            next_rng, next_state = pm.Categorical.dist(p=p, rng=old_rng).owner.outputs
            return intX(next_state), {old_rng: next_rng}

        markov_chain, state_updates = pytensor.scan(
            transition,
            non_sequences=[P_, state_rng],
            outputs_info=[x0_],
            n_steps=steps_,
            strict=True,
        )

        (state_next_rng,) = tuple(state_updates.values())

        discrete_mc_ = (
            at.concatenate([x0_[None, ...], markov_chain], axis=0)
            .dimshuffle(tuple(range(1, markov_chain.ndim)) + (0,))
            .squeeze()
        )

        discrete_mc_op = DiscreteMarkovChainRV(
            inputs=[P_, x0_, steps_],
            outputs=[state_next_rng, discrete_mc_],
            ndim_supp=1,
        )

        discrete_mc = discrete_mc_op(P, x0, steps)
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
def discrete_mc_logp(op, values, P, x0, steps, state_rng, **kwargs):
    n, k = P.shape

    (value,) = values

    mc_logprob = at.log(P[value[..., :-1], value[..., 1:]]).sum(axis=-1)

    return check_parameters(
        mc_logprob,
        at.eq(n, k),
        at.all(at.allclose(P.sum(axis=1), 1.0)),
        msg="P must be square with rows that sum to 1",
    )


@_moment.register(DiscreteMarkovChainRV)
def discrete_markov_chain_moment(op, rv, P, x0, steps, state_rng):
    return at.zeros_like(rv)
