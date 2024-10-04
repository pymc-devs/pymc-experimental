from collections.abc import Sequence

import numpy as np
import pytensor.tensor as pt

from pymc.distributions import Bernoulli, Categorical, DiscreteUniform
from pymc.logprob.abstract import MeasurableOp, _logprob
from pymc.logprob.basic import conditional_logp, logp
from pymc.pytensorf import constant_fold
from pytensor import Variable
from pytensor.compile.builders import OpFromGraph
from pytensor.compile.mode import Mode
from pytensor.graph import Op, vectorize_graph
from pytensor.graph.replace import clone_replace, graph_replace
from pytensor.scan import map as scan_map
from pytensor.scan import scan
from pytensor.tensor import TensorVariable

from pymc_experimental.distributions import DiscreteMarkovChain


class MarginalRV(OpFromGraph, MeasurableOp):
    """Base class for Marginalized RVs"""

    def __init__(self, *args, dims_connections: tuple[tuple[int | None]], **kwargs) -> None:
        self.dims_connections = dims_connections
        super().__init__(*args, **kwargs)

    @property
    def support_axes(self) -> tuple[tuple[int]]:
        """Dimensions of dependent RVs that belong to the core (non-batched) marginalized variable."""
        marginalized_ndim_supp = self.inner_outputs[0].owner.op.ndim_supp
        support_axes_vars = []
        for dims_connection in self.dims_connections:
            ndim = len(dims_connection)
            marginalized_supp_axes = ndim - marginalized_ndim_supp
            support_axes_vars.append(
                tuple(
                    -i
                    for i, dim in enumerate(reversed(dims_connection), start=1)
                    if (dim is None or dim > marginalized_supp_axes)
                )
            )
        return tuple(support_axes_vars)


class MarginalFiniteDiscreteRV(MarginalRV):
    """Base class for Marginalized Finite Discrete RVs"""


class MarginalDiscreteMarkovChainRV(MarginalRV):
    """Base class for Marginalized Discrete Markov Chain RVs"""


def get_domain_of_finite_discrete_rv(rv: TensorVariable) -> tuple[int, ...]:
    op = rv.owner.op
    dist_params = rv.owner.op.dist_params(rv.owner)
    if isinstance(op, Bernoulli):
        return (0, 1)
    elif isinstance(op, Categorical):
        [p_param] = dist_params
        [p_param_length] = constant_fold([p_param.shape[-1]])
        return tuple(range(p_param_length))
    elif isinstance(op, DiscreteUniform):
        lower, upper = constant_fold(dist_params)
        return tuple(np.arange(lower, upper + 1))
    elif isinstance(op, DiscreteMarkovChain):
        P, *_ = dist_params
        return tuple(range(pt.get_vector_length(P[-1])))

    raise NotImplementedError(f"Cannot compute domain for op {op}")


def reduce_batch_dependent_logps(
    dependent_dims_connections: Sequence[tuple[int | None, ...]],
    dependent_ops: Sequence[Op],
    dependent_logps: Sequence[TensorVariable],
) -> TensorVariable:
    """Combine the logps of dependent RVs and align them with the marginalized logp.

    This requires reducing extra batch dims and transposing when they are not aligned.

       idx = pm.Bernoulli(idx, shape=(3, 2))  # 0, 1
       pm.Normal("dep1", mu=idx.T[..., None] * 2, shape=(3, 2, 5))
       pm.Normal("dep2", mu=idx * 2, shape=(7, 2, 3))

       marginalize(idx)

       The marginalized op will have dims_connections = [(1, 0, None), (None, 0, 1)]
       which tells us we need to reduce the last axis of dep1 logp and the first of dep2 logp,
       as well as transpose the remaining axis of dep1 logp before adding the two element-wise.

    """
    from pymc_experimental.model.marginal.graph_analysis import get_support_axes

    reduced_logps = []
    for dependent_op, dependent_logp, dependent_dims_connection in zip(
        dependent_ops, dependent_logps, dependent_dims_connections
    ):
        if dependent_logp.type.ndim > 0:
            # Find which support axis implied by the MarginalRV need to be reduced
            # Some may have already been reduced by the logp expression of the dependent RV (e.g., multivariate RVs)
            dep_supp_axes = get_support_axes(dependent_op)[0]

            # Dependent RV support axes are already collapsed in the logp, so we ignore them
            supp_axes = [
                -i
                for i, dim in enumerate(reversed(dependent_dims_connection), start=1)
                if (dim is None and -i not in dep_supp_axes)
            ]
            dependent_logp = dependent_logp.sum(supp_axes)

            # Finally, we need to align the dependent logp batch dimensions with the marginalized logp
            dims_alignment = [dim for dim in dependent_dims_connection if dim is not None]
            dependent_logp = dependent_logp.transpose(*dims_alignment)

        reduced_logps.append(dependent_logp)

    reduced_logp = pt.add(*reduced_logps)
    return reduced_logp


def align_logp_dims(dims: tuple[tuple[int, None]], logp: TensorVariable) -> TensorVariable:
    """Align the logp with the order specified in dims."""
    dims_alignment = [dim for dim in dims if dim is not None]
    return logp.transpose(*dims_alignment)


def inline_ofg_outputs(op: OpFromGraph, inputs: Sequence[Variable]) -> tuple[Variable]:
    """Inline the inner graph (outputs) of an OpFromGraph Op.

    Whereas `OpFromGraph` "wraps" a graph inside a single Op, this function "unwraps"
    the inner graph.
    """
    return clone_replace(
        op.inner_outputs,
        replace=tuple(zip(op.inner_inputs, inputs)),
    )


DUMMY_ZERO = pt.constant(0, name="dummy_zero")


@_logprob.register(MarginalFiniteDiscreteRV)
def finite_discrete_marginal_rv_logp(op: MarginalFiniteDiscreteRV, values, *inputs, **kwargs):
    # Clone the inner RV graph of the Marginalized RV
    marginalized_rv, *inner_rvs = inline_ofg_outputs(op, inputs)

    # Obtain the joint_logp graph of the inner RV graph
    inner_rv_values = dict(zip(inner_rvs, values))
    marginalized_vv = marginalized_rv.clone()
    rv_values = inner_rv_values | {marginalized_rv: marginalized_vv}
    logps_dict = conditional_logp(rv_values=rv_values, **kwargs)

    # Reduce logp dimensions corresponding to broadcasted variables
    marginalized_logp = logps_dict.pop(marginalized_vv)
    joint_logp = marginalized_logp + reduce_batch_dependent_logps(
        dependent_dims_connections=op.dims_connections,
        dependent_ops=[inner_rv.owner.op for inner_rv in inner_rvs],
        dependent_logps=[logps_dict[value] for value in values],
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

    joint_logp = pt.logsumexp(joint_logps, axis=0)

    # Align logp with non-collapsed batch dimensions of first RV
    joint_logp = align_logp_dims(dims=op.dims_connections[0], logp=joint_logp)

    # We have to add dummy logps for the remaining value variables, otherwise PyMC will raise
    dummy_logps = (DUMMY_ZERO,) * (len(values) - 1)
    return joint_logp, *dummy_logps


@_logprob.register(MarginalDiscreteMarkovChainRV)
def marginal_hmm_logp(op, values, *inputs, **kwargs):
    chain_rv, *dependent_rvs = inline_ofg_outputs(op, inputs)

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
    reduced_logp_emissions = reduce_batch_dependent_logps(
        dependent_dims_connections=op.dims_connections,
        dependent_ops=[dependent_rv.owner.op for dependent_rv in dependent_rvs],
        dependent_logps=[logp_emissions_dict[value] for value in values],
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

    # Align logp with non-collapsed batch dimensions of first RV
    remaining_dims_first_emission = list(op.dims_connections[0])
    # The last dim of chain_rv was removed when computing the logp
    remaining_dims_first_emission.remove(chain_rv.type.ndim - 1)
    joint_logp = align_logp_dims(remaining_dims_first_emission, joint_logp)

    # If there are multiple emission streams, we have to add dummy logps for the remaining value variables. The first
    # return is the joint probability of everything together, but PyMC still expects one logp for each emission stream.
    dummy_logps = (DUMMY_ZERO,) * (len(values) - 1)
    return joint_logp, *dummy_logps
