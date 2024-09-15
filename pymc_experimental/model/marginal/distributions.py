from collections.abc import Sequence

import numpy as np

from pymc import Bernoulli, Categorical, DiscreteUniform, SymbolicRandomVariable, logp
from pymc.logprob import conditional_logp
from pymc.logprob.abstract import _logprob
from pymc.pytensorf import constant_fold
from pytensor import Mode, clone_replace, graph_replace, scan
from pytensor import map as scan_map
from pytensor import tensor as pt
from pytensor.graph import vectorize_graph
from pytensor.tensor import TensorType, TensorVariable

from pymc_experimental.distributions import DiscreteMarkovChain


class MarginalRV(SymbolicRandomVariable):
    """Base class for Marginalized RVs"""


class FiniteDiscreteMarginalRV(MarginalRV):
    """Base class for Finite Discrete Marginalized RVs"""


class DiscreteMarginalMarkovChainRV(MarginalRV):
    """Base class for Discrete Marginal Markov Chain RVs"""


def get_domain_of_finite_discrete_rv(rv: TensorVariable) -> tuple[int, ...]:
    op = rv.owner.op
    dist_params = rv.owner.op.dist_params(rv.owner)
    if isinstance(op, Bernoulli):
        return (0, 1)
    elif isinstance(op, Categorical):
        [p_param] = dist_params
        return tuple(range(pt.get_vector_length(p_param)))
    elif isinstance(op, DiscreteUniform):
        lower, upper = constant_fold(dist_params)
        return tuple(np.arange(lower, upper + 1))
    elif isinstance(op, DiscreteMarkovChain):
        P, *_ = dist_params
        return tuple(range(pt.get_vector_length(P[-1])))

    raise NotImplementedError(f"Cannot compute domain for op {op}")


def _add_reduce_batch_dependent_logps(
    marginalized_type: TensorType, dependent_logps: Sequence[TensorVariable]
):
    """Add the logps of dependent RVs while reducing extra batch dims relative to `marginalized_type`."""

    mbcast = marginalized_type.broadcastable
    reduced_logps = []
    for dependent_logp in dependent_logps:
        dbcast = dependent_logp.type.broadcastable
        dim_diff = len(dbcast) - len(mbcast)
        mbcast_aligned = (True,) * dim_diff + mbcast
        vbcast_axis = [i for i, (m, v) in enumerate(zip(mbcast_aligned, dbcast)) if m and not v]
        reduced_logps.append(dependent_logp.sum(vbcast_axis))
    return pt.add(*reduced_logps)


@_logprob.register(FiniteDiscreteMarginalRV)
def finite_discrete_marginal_rv_logp(op, values, *inputs, **kwargs):
    # Clone the inner RV graph of the Marginalized RV
    marginalized_rvs_node = op.make_node(*inputs)
    marginalized_rv, *inner_rvs = clone_replace(
        op.inner_outputs,
        replace={u: v for u, v in zip(op.inner_inputs, marginalized_rvs_node.inputs)},
    )

    # Obtain the joint_logp graph of the inner RV graph
    inner_rv_values = dict(zip(inner_rvs, values))
    marginalized_vv = marginalized_rv.clone()
    rv_values = inner_rv_values | {marginalized_rv: marginalized_vv}
    logps_dict = conditional_logp(rv_values=rv_values, **kwargs)

    # Reduce logp dimensions corresponding to broadcasted variables
    marginalized_logp = logps_dict.pop(marginalized_vv)
    joint_logp = marginalized_logp + _add_reduce_batch_dependent_logps(
        marginalized_rv.type, logps_dict.values()
    )

    # Compute the joint_logp for all possible n values of the marginalized RV. We assume
    # each original dimension is independent so that it suffices to evaluate the graph
    # n times, once with each possible value of the marginalized RV replicated across
    # batched dimensions of the marginalized RV

    # PyMC does not allow RVs in the logp graph, even if we are just using the shape
    marginalized_rv_shape = constant_fold(tuple(marginalized_rv.shape), raise_not_constant=False)
    marginalized_rv_domain = get_domain_of_finite_discrete_rv(marginalized_rv)
    marginalized_rv_domain_tensor = pt.moveaxis(
        pt.full(
            (*marginalized_rv_shape, len(marginalized_rv_domain)),
            marginalized_rv_domain,
            dtype=marginalized_rv.dtype,
        ),
        -1,
        0,
    )

    try:
        joint_logps = vectorize_graph(
            joint_logp, replace={marginalized_vv: marginalized_rv_domain_tensor}
        )
    except Exception:
        # Fallback to Scan
        def logp_fn(marginalized_rv_const, *non_sequences):
            return graph_replace(joint_logp, replace={marginalized_vv: marginalized_rv_const})

        joint_logps, _ = scan_map(
            fn=logp_fn,
            sequences=marginalized_rv_domain_tensor,
            non_sequences=[*values, *inputs],
            mode=Mode().including("local_remove_check_parameter"),
        )

    joint_logps = pt.logsumexp(joint_logps, axis=0)

    # We have to add dummy logps for the remaining value variables, otherwise PyMC will raise
    return joint_logps, *(pt.constant(0),) * (len(values) - 1)


@_logprob.register(DiscreteMarginalMarkovChainRV)
def marginal_hmm_logp(op, values, *inputs, **kwargs):
    marginalized_rvs_node = op.make_node(*inputs)
    inner_rvs = clone_replace(
        op.inner_outputs,
        replace={u: v for u, v in zip(op.inner_inputs, marginalized_rvs_node.inputs)},
    )

    chain_rv, *dependent_rvs = inner_rvs
    P, n_steps_, init_dist_, rng = chain_rv.owner.inputs
    domain = pt.arange(P.shape[-1], dtype="int32")

    # Construct logp in two steps
    # Step 1: Compute the probability of the data ("emissions") under every possible state (vec_logp_emission)

    # First we need to vectorize the conditional logp graph of the data, in case there are batch dimensions floating
    # around. To do this, we need to break the dependency between chain and the init_dist_ random variable. Otherwise,
    # PyMC will detect a random variable in the logp graph (init_dist_), that isn't relevant at this step.
    chain_value = chain_rv.clone()
    dependent_rvs = clone_replace(dependent_rvs, {chain_rv: chain_value})
    logp_emissions_dict = conditional_logp(dict(zip(dependent_rvs, values)))

    # Reduce and add the batch dims beyond the chain dimension
    reduced_logp_emissions = _add_reduce_batch_dependent_logps(
        chain_rv.type, logp_emissions_dict.values()
    )

    # Add a batch dimension for the domain of the chain
    chain_shape = constant_fold(tuple(chain_rv.shape))
    batch_chain_value = pt.moveaxis(pt.full((*chain_shape, domain.size), domain), -1, 0)
    batch_logp_emissions = vectorize_graph(reduced_logp_emissions, {chain_value: batch_chain_value})

    # Step 2: Compute the transition probabilities
    # This is the "forward algorithm", alpha_t = p(y | s_t) * sum_{s_{t-1}}(p(s_t | s_{t-1}) * alpha_{t-1})
    # We do it entirely in logs, though.

    # To compute the prior probabilities of each state, we evaluate the logp of the domain (all possible states)
    # under the initial distribution. This is robust to everything the user can throw at it.
    init_dist_value = init_dist_.type()
    logp_init_dist = logp(init_dist_, init_dist_value)
    # There is a degerate batch dim for lags=1 (the only supported case),
    # that we have to work around, by expanding the batch value and then squeezing it out of the logp
    batch_logp_init_dist = vectorize_graph(
        logp_init_dist, {init_dist_value: batch_chain_value[:, None, ..., 0]}
    ).squeeze(1)
    log_alpha_init = batch_logp_init_dist + batch_logp_emissions[..., 0]

    def step_alpha(logp_emission, log_alpha, log_P):
        step_log_prob = pt.logsumexp(log_alpha[:, None] + log_P, axis=0)
        return logp_emission + step_log_prob

    P_bcast_dims = (len(chain_shape) - 1) - (P.type.ndim - 2)
    log_P = pt.shape_padright(pt.log(P), P_bcast_dims)
    log_alpha_seq, _ = scan(
        step_alpha,
        non_sequences=[log_P],
        outputs_info=[log_alpha_init],
        # Scan needs the time dimension first, and we already consumed the 1st logp computing the initial value
        sequences=pt.moveaxis(batch_logp_emissions[..., 1:], -1, 0),
    )
    # Final logp is just the sum of the last scan state
    joint_logp = pt.logsumexp(log_alpha_seq[-1], axis=0)

    # If there are multiple emission streams, we have to add dummy logps for the remaining value variables. The first
    # return is the joint probability of everything together, but PyMC still expects one logp for each one.
    dummy_logps = (pt.constant(0),) * (len(values) - 1)
    return joint_logp, *dummy_logps
