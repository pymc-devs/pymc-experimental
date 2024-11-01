import warnings

from collections.abc import Sequence

from pymc.distributions.discrete import Bernoulli, Categorical, DiscreteUniform
from pymc.distributions.transforms import Chain
from pymc.logprob.transforms import IntervalTransform
from pymc.model import Model
from pymc.model.fgraph import ModelFreeRV, ModelValuedVar, fgraph_from_model, model_from_fgraph
from pymc.pytensorf import collect_default_updates, toposort_replace
from pytensor.compile import SharedVariable
from pytensor.graph import FunctionGraph, clone_replace, graph_inputs
from pytensor.tensor import TensorVariable

__all__ = ["MarginalModel", "marginalize"]

from pymc_experimental.distributions import DiscreteMarkovChain
from pymc_experimental.model.marginal.distributions import (
    MarginalDiscreteMarkovChainRV,
    MarginalFiniteDiscreteRV,
)
from pymc_experimental.model.marginal.graph_analysis import (
    find_conditional_dependent_rvs,
    find_conditional_input_rvs,
    is_conditional_dependent,
    subgraph_batch_dim_connection,
)

ModelRVs = TensorVariable | Sequence[TensorVariable] | str | Sequence[str]


class MarginalModel(Model):
    """Subclass of PyMC Model that implements functionality for automatic
    marginalization of variables in the logp transformation

    After defining the full Model, the `marginalize` method can be used to indicate a
    subset of variables that should be marginalized

    Notes
    -----
    Marginalization functionality is still very restricted. Only finite discrete
    variables can be marginalized. Deterministics and Potentials cannot be conditionally
    dependent on the marginalized variables.

    Furthermore, not all instances of such variables can be marginalized. If a variable
    has batched dimensions, it is required that any conditionally dependent variables
    use information from an individual batched dimension. In other words, the graph
    connecting the marginalized variable(s) to the dependent variable(s) must be
    composed strictly of Elemwise Operations. This is necessary to ensure an efficient
    logprob graph can be generated. If you want to bypass this restriction you can
    separate each dimension of the marginalized variable into the scalar components
    and then stack them together. Note that such graphs will grow exponentially in the
    number of  marginalized variables.

    For the same reason, it's not possible to marginalize RVs with multivariate
    dependent RVs.

    Examples
    --------
    Marginalize over a single variable

    .. code-block:: python

        import pymc as pm
        from pymc_experimental import MarginalModel

        with MarginalModel() as m:
            p = pm.Beta("p", 1, 1)
            x = pm.Bernoulli("x", p=p, shape=(3,))
            y = pm.Normal("y", pm.math.switch(x, -10, 10), observed=[10, 10, -10])

            m.marginalize([x])

            idata = pm.sample()

    """

    def __init__(self, *args, **kwargs):
        raise TypeError(
            "MarginalModel was deprecated in favor of `marginalize` which nows returns a PyMC model"
        )


def _warn_interval_transform(rv_to_marginalize, replaced_vars: Sequence[ModelValuedVar]) -> None:
    for replaced_var in replaced_vars:
        if not isinstance(replaced_var.owner.op, ModelValuedVar):
            raise TypeError(f"{replaced_var} is not a ModelValuedVar")

        if not isinstance(replaced_var.owner.op, ModelFreeRV):
            continue

        if replaced_var is rv_to_marginalize:
            continue

        transform = replaced_var.owner.op.transform

        if isinstance(transform, IntervalTransform) or (
            isinstance(transform, Chain)
            and any(isinstance(tr, IntervalTransform) for tr in transform.transform_list)
        ):
            warnings.warn(
                f"The transform {transform} for the variable {replaced_var}, which depends on the "
                f"marginalized {rv_to_marginalize} may no longer work if bounds depended on other variables.",
                UserWarning,
            )


def marginalize(model: Model, rvs_to_marginalize: ModelRVs) -> MarginalModel:
    """Marginalize a subset of variables in a PyMC model.

    This creates a class of `MarginalModel` from an existing `Model`, with the specified
    variables marginalized.

    See documentation for `MarginalModel` for more information.

    Parameters
    ----------
    model : Model
        PyMC model to marginalize. Original variables well be cloned.
    rvs_to_marginalize : Sequence[TensorVariable]
        Variables to marginalize in the returned model.

    Returns
    -------
    marginal_model: MarginalModel
        Marginal model with the specified variables marginalized.
    """
    if not isinstance(rvs_to_marginalize, tuple | list):
        rvs_to_marginalize = (rvs_to_marginalize,)

    rvs_to_marginalize = [model[rv] if isinstance(rv, str) else rv for rv in rvs_to_marginalize]

    for rv_to_marginalize in rvs_to_marginalize:
        if rv_to_marginalize not in model.free_RVs:
            raise ValueError(f"Marginalized RV {rv_to_marginalize} is not a free RV in the model")

        rv_op = rv_to_marginalize.owner.op
        if isinstance(rv_op, DiscreteMarkovChain):
            if rv_op.n_lags > 1:
                raise NotImplementedError(
                    "Marginalization for DiscreteMarkovChain with n_lags > 1 is not supported"
                )
            if rv_to_marginalize.owner.inputs[0].type.ndim > 2:
                raise NotImplementedError(
                    "Marginalization for DiscreteMarkovChain with non-matrix transition probability is not supported"
                )
        elif not isinstance(rv_op, Bernoulli | Categorical | DiscreteUniform):
            raise NotImplementedError(
                f"Marginalization of RV with distribution {rv_to_marginalize.owner.op} is not supported"
            )

    fg, memo = fgraph_from_model(model)
    rvs_to_marginalize = [memo[rv] for rv in rvs_to_marginalize]
    toposort = fg.toposort()

    for rv_to_marginalize in sorted(
        rvs_to_marginalize,
        key=lambda rv: toposort.index(rv.owner),
        reverse=True,
    ):
        all_rvs = [
            rv for rv in fg.variables if rv.owner and isinstance(rv.owner.op, ModelValuedVar)
        ]

        dependent_rvs = find_conditional_dependent_rvs(rv_to_marginalize, all_rvs)
        if not dependent_rvs:
            # TODO: This should at most be a warning, not an error
            raise ValueError(f"No RVs depend on marginalized RV {rv_to_marginalize}")

        # Issue warning for IntervalTransform on dependent RVs
        for dependent_rv in dependent_rvs:
            transform = dependent_rv.owner.op.transform

            if isinstance(transform, IntervalTransform) or (
                isinstance(transform, Chain)
                and any(isinstance(tr, IntervalTransform) for tr in transform.transform_list)
            ):
                warnings.warn(
                    f"The transform {transform} for the variable {dependent_rv}, which depends on the "
                    f"marginalized {rv_to_marginalize} may no longer work if bounds depended on other variables.",
                    UserWarning,
                )

        # Check that no deterministics or potentials depend on the rv to marginalize
        for det in model.deterministics:
            if is_conditional_dependent(memo[det], rv_to_marginalize, all_rvs):
                raise NotImplementedError(
                    f"Cannot marginalize {rv_to_marginalize} due to dependent Deterministic {det}"
                )
        for pot in model.potentials:
            if is_conditional_dependent(memo[pot], rv_to_marginalize, all_rvs):
                raise NotImplementedError(
                    f"Cannot marginalize {rv_to_marginalize} due to dependent Potential {pot}"
                )

        marginalized_rv_input_rvs = find_conditional_input_rvs([rv_to_marginalize], all_rvs)
        other_direct_rv_ancestors = [
            rv
            for rv in find_conditional_input_rvs(dependent_rvs, all_rvs)
            if rv is not rv_to_marginalize
        ]
        input_rvs = list(set((*marginalized_rv_input_rvs, *other_direct_rv_ancestors)))

        replace_finite_discrete_marginal_subgraph(fg, rv_to_marginalize, dependent_rvs, input_rvs)

    return model_from_fgraph(fg, mutate_fgraph=True)


def collect_shared_vars(outputs, blockers):
    return [
        inp
        for inp in graph_inputs(outputs, blockers=blockers)
        if (isinstance(inp, SharedVariable) and inp not in blockers)
    ]


def remove_model_vars(vars):
    """Remove ModelVars from the graph of vars."""
    model_vars = [var for var in vars if isinstance(var.owner.op, ModelValuedVar)]
    replacements = [(model_var, model_var.owner.inputs[0]) for model_var in model_vars]
    fgraph = FunctionGraph(outputs=vars, clone=False)
    toposort_replace(fgraph, replacements)
    return fgraph.outputs


def replace_finite_discrete_marginal_subgraph(
    fgraph, rv_to_marginalize, dependent_rvs, input_rvs
) -> None:
    # If the marginalized RV has multiple dimensions, check that graph between
    # marginalized RV and dependent RVs does not mix information from batch dimensions
    # (otherwise logp would require enumerating over all combinations of batch dimension values)
    try:
        dependent_rvs_dim_connections = subgraph_batch_dim_connection(
            rv_to_marginalize, dependent_rvs
        )
    except (ValueError, NotImplementedError) as e:
        # For the perspective of the user this is a NotImplementedError
        raise NotImplementedError(
            "The graph between the marginalized and dependent RVs cannot be marginalized efficiently. "
            "You can try splitting the marginalized RV into separate components and marginalizing them separately."
        ) from e

    output_rvs = [rv_to_marginalize, *dependent_rvs]
    rng_updates = collect_default_updates(output_rvs, inputs=input_rvs, must_be_shared=False)
    outputs = output_rvs + list(rng_updates.values())
    inputs = input_rvs + list(rng_updates.keys())
    # Add any other shared variable inputs
    inputs += collect_shared_vars(output_rvs, blockers=inputs)

    inner_inputs = [inp.clone() for inp in inputs]
    inner_outputs = clone_replace(outputs, replace=dict(zip(inputs, inner_inputs)))
    inner_outputs = remove_model_vars(inner_outputs)

    if isinstance(rv_to_marginalize.owner.op, DiscreteMarkovChain):
        marginalize_constructor = MarginalDiscreteMarkovChainRV
    else:
        marginalize_constructor = MarginalFiniteDiscreteRV

    marginalization_op = marginalize_constructor(
        inputs=inner_inputs,
        outputs=inner_outputs,  # TODO: Add RNG updates to outputs so this can be used in the generative graph
        dims_connections=dependent_rvs_dim_connections,
    )

    new_outputs = marginalization_op(*inputs)
    for old_output, new_output in zip(outputs, new_outputs):
        new_output.name = old_output.name

    outer_replacements = [
        (
            # Remove the marginalized FreeRV, but keep the dependent ones as Free/ObservedRVs
            (
                old_output
                if (
                    old_output is rv_to_marginalize
                    or not isinstance(old_output.owner.op, ModelValuedVar)
                )
                else old_output.owner.inputs[0]
            ),
            new_output,
        )
        for old_output, new_output in zip(outputs, new_outputs)
    ]
    fgraph.replace_all(outer_replacements)

    if len(dependent_rvs) > 1:
        warnings.warn(
            "There are multiple dependent variables in a FiniteDiscreteMarginalRV. "
            f"Their joint logp terms will be assigned to the first RV: {dependent_rvs[1]}.",
            UserWarning,
        )
