from typing import Any, Dict, List, Sequence, Union

from pymc import Model
from pymc.pytensorf import _replace_vars_in_graphs
from pytensor.tensor import TensorVariable

from pymc_experimental.model_transform.basic import prune_vars_detached_from_observed
from pymc_experimental.utils.model_fgraph import (
    ModelDeterministic,
    ModelFreeRV,
    extract_dims,
    fgraph_from_model,
    model_deterministic,
    model_from_fgraph,
    model_named,
    model_observed_rv,
    toposort_replace,
)
from pymc_experimental.utils.pytensorf import rvs_in_graph


def observe(model: Model, vars_to_observations: Dict[Union["str", TensorVariable], Any]) -> Model:
    """Convert free RVs or Deterministics to observed RVs.

    Parameters
    ----------
    model: PyMC Model
    vars_to_observations: Dict of variable or name to TensorLike
        Dictionary that maps model variables (or names) to observed values.
        Observed values must have a shape and data type that is compatible
        with the original model variable.

    Returns
    -------
    new_model: PyMC model
        A distinct PyMC model with the relevant variables observed.
        All remaining variables are cloned and can be retrieved via `new_model["var_name"]`.

    Examples
    --------

    .. code-block:: python

        import pymc as pm
        from pymc_experimental.model_transform.conditioning import observe

        with pm.Model() as m:
            x = pm.Normal("x")
            y = pm.Normal("y", x)
            z = pm.Normal("z", y)

        m_new = observe(m, {y: 0.5})

    Deterministic variables can also be observed.
    This relies on PyMC ability to infer the logp of the underlying expression

    .. code-block:: python

        import pymc as pm
        from pymc_experimental.model_transform.conditioning import observe

        with pm.Model() as m:
            x = pm.Normal("x")
            y = pm.Normal.dist(x, shape=(5,))
            y_censored = pm.Deterministic("y_censored", pm.math.clip(y, -1, 1))

        new_m = observe(m, {y_censored: [0.9, 0.5, 0.3, 1, 1]})


    """
    vars_to_observations = {
        model[var] if isinstance(var, str) else var: obs
        for var, obs in vars_to_observations.items()
    }

    valid_model_vars = set(model.free_RVs + model.deterministics)
    if any(var not in valid_model_vars for var in vars_to_observations):
        raise ValueError(f"At least one var is not a free variable or deterministic in the model")

    fgraph, memo = fgraph_from_model(model)

    replacements = {}
    for var, obs in vars_to_observations.items():
        model_var = memo[var]

        # Just a sanity check
        assert isinstance(model_var.owner.op, (ModelFreeRV, ModelDeterministic))
        assert model_var in fgraph.variables

        var = model_var.owner.inputs[0]
        var.name = model_var.name
        dims = extract_dims(model_var)
        model_obs_rv = model_observed_rv(var, var.type.filter_variable(obs), *dims)
        replacements[model_var] = model_obs_rv

    toposort_replace(fgraph, tuple(replacements.items()))

    return model_from_fgraph(fgraph)


def replace_vars_in_graphs(graphs: Sequence[TensorVariable], replacements) -> List[TensorVariable]:
    def replacement_fn(var, inner_replacements):
        if var in replacements:
            inner_replacements[var] = replacements[var]

        # Handle root inputs as those will never be passed to the replacement_fn
        for inp in var.owner.inputs:
            if inp.owner is None and inp in replacements:
                inner_replacements[inp] = replacements[inp]

        return [var]

    replaced_graphs, _ = _replace_vars_in_graphs(graphs=graphs, replacement_fn=replacement_fn)
    return replaced_graphs


def do(
    model: Model, vars_to_interventions: Dict[Union["str", TensorVariable], Any], prune_vars=False
) -> Model:
    """Replace model variables by intervention variables.

    Intervention variables will either show up as `Data` or `Deterministics` in the new model,
    depending on whether they depend on other RandomVariables or not.

    Parameters
    ----------
    model: PyMC Model
    vars_to_interventions: Dict of variable or name to TensorLike
        Dictionary that maps model variables (or names) to intervention expressions.
        Intervention expressions must have a shape and data type that is compatible
        with the original model variable.
    prune_vars: bool, defaults to False
        Whether to prune model variables that are not connected to any observed variables,
        after the interventions.

    Returns
    -------
    new_model: PyMC model
        A distinct PyMC model with the relevant variables replaced by the intervention expressions.
        All remaining variables are cloned and can be retrieved via `new_model["var_name"]`.

    Examples
    --------

    .. code-block:: python

        import pymc as pm
        from pymc_experimental.model_transform.conditioning import do

        with pm.Model() as m:
            x = pm.Normal("x", 0, 1)
            y = pm.Normal("y", x, 1)
            z = pm.Normal("z", y + x, 1)

        # Dummy posterior, same as calling `pm.sample`
        idata_m = az.from_dict({rv.name: [pm.draw(rv, draws=500)] for rv in [x, y, z]})

        # Replace `y` by a constant `100.0`
        m_do = do(m, {y: 100.0})
        with m_do:
            idata_do = pm.sample_posterior_predictive(idata_m, var_names="z")

    """
    do_mapping = {}
    for var, obs in vars_to_interventions.items():
        if isinstance(var, str):
            var = model[var]
        try:
            do_mapping[var] = var.type.filter_variable(obs)
        except TypeError as err:
            raise TypeError(
                "Incompatible replacement type. Make sure the shape and datatype of the interventions match the original variables"
            ) from err

    if any(var not in model.named_vars.values() for var in do_mapping):
        raise ValueError(f"At least one var is not a named variable in the model")

    fgraph, memo = fgraph_from_model(model, inlined_views=True)

    # We need the interventions defined in terms of the IR fgraph representation,
    # In case they reference other variables in the model
    ir_interventions = replace_vars_in_graphs(list(do_mapping.values()), replacements=memo)

    replacements = {}
    for var, intervention in zip(do_mapping, ir_interventions):
        model_var = memo[var]

        # Just a sanity check
        assert model_var in fgraph.variables

        intervention.name = model_var.name
        dims = extract_dims(model_var)
        # If there are any RVs in the graph we introduce the intervention as a deterministic
        if rvs_in_graph([intervention]):
            new_var = model_deterministic(intervention.copy(name=intervention.name), *dims)
        # Otherwise as a named variable (Constant or Shared data)
        else:
            new_var = model_named(intervention, *dims)

        replacements[model_var] = new_var

    # Replace variables by interventions
    toposort_replace(fgraph, tuple(replacements.items()))

    model = model_from_fgraph(fgraph)
    if prune_vars:
        return prune_vars_detached_from_observed(model)
    return model
