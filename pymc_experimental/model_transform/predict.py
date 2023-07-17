from typing import List, Optional, Sequence, Union

import pytensor.tensor as pt
from pymc import DiracDelta
from pymc.distributions.censored import CensoredRV
from pymc.distributions.timeseries import AR, AutoRegressiveRV
from pymc.model import Model
from pytensor.graph.basic import Variable

from pymc_experimental.utils.model_fgraph import (
    ModelFreeRV,
    ModelValuedVar,
    fgraph_from_model,
    model_free_rv,
    model_from_fgraph,
    toposort_replace,
)

__all__ = (
    "uncensor",
    "forecast_timeseries",
)


ModelVariable = Union[Variable, str]
SequenceModelVariables = Union[ModelVariable, Sequence[ModelVariable]]


def parse_vars(model: Model, vars: SequenceModelVariables) -> List[Variable]:
    if not isinstance(vars, (list, tuple)):
        vars = (vars,)
    return [model[var] if isinstance(var, str) else var for var in vars]


def uncensor(model: Model, vars: Optional[SequenceModelVariables] = None) -> Model:
    """Replace censored variables in the model by uncensored ones.

    .. code-block:: python

        import pymc as pm
        from pymc_experimental.model_transform.predict import uncensor

        with pm.Model() as model:
            x = pm.Normal("x")
            dist_raw = pm.Normal.dist(x, sigma=10)
            y = pm.Censored("y", dist=dist_raw, lower=0, upper=10, observed=[0, 5, 10])
            trace = pm.sample()

        with uncensor(model):
            pp = pm.sample_posterior_predictive(trace, var_names=["y"])


    Parameters
    ----------
    model: Model
    vars: optional
        Model variables that should be replaced by uncensored counterparts.
        Defaults to all censored variables.

    Returns
    -------
    uncensored_model: Model
        Model with the censored variables replaced by uncensored versions

    """
    vars = parse_vars(model, vars) if vars is not None else []

    fgraph, memo = fgraph_from_model(model)

    target_vars = {memo[var] for var in vars}
    replacements = {}
    for node in fgraph.apply_nodes:
        if not isinstance(node.op, ModelValuedVar):
            continue

        dummy_rv = node.outputs[0]
        if target_vars and dummy_rv not in target_vars:
            continue

        rv, value, *dims = node.inputs
        if not isinstance(rv.owner.op, (CensoredRV,)):
            if target_vars:
                raise NotImplementedError(f"RV distribution {rv.owner.op} is not censored")
            else:
                continue

        # The first argument is the `dist` RV
        new_rv = rv.owner.inputs[0]

        new_rv.name = rv.name
        new_dummy_rv = model_free_rv(new_rv, new_rv.type(), None, *dims)
        replacements[dummy_rv] = new_dummy_rv

    toposort_replace(fgraph, tuple(replacements.items()))
    return model_from_fgraph(fgraph)


def forecast_timeseries(
    model: Model,
    vars: Optional[SequenceModelVariables] = None,
    *,
    steps: Optional[int] = None,
) -> Model:
    """Replace timeseries variables in the model by forecast that start at the last value.

    .. code-block:: python

        import pymc as pm
        from pymc_experimental.model_transform.predict import forecast_timeseries

        with pm.Model() as model:
            rho = pm.Normal("rho")
            sigma = pm.HalfNormal("sigma")
            init_dist = pm.Normal.dist()
            y = pm.AR("y", init_dist=init_dist, rho=rho, sigma=sigma, observed=[0] * 100)
            trace = pm.sample()

        with forecast_timeseries(model, steps=20):
            pp = pm.sample_posterior_predictive(trace, var_names=["y"], predictions=True)



    Parameters
    ----------
    model: Model
    vars: optional
        Model variables that should be replaced by forecast counterparts.
        Defaults to all timeseries variables.
    steps: int, optional
        Number of steps for the forecast. Defaults to the same as originally

    Returns
    -------
    forecast_model: Model
        Model with the timeseries variables replaced by the forecast versions

    """
    vars = parse_vars(model, vars) if vars is not None else []

    if steps is not None:
        steps = pt.as_tensor_variable(steps, dtype=int)

    fgraph, memo = fgraph_from_model(model)

    target_vars = {memo[var] for var in vars}
    replacements = {}
    for node in fgraph.apply_nodes:

        if not isinstance(node.op, ModelValuedVar):
            continue

        dummy_rv = node.outputs[0]
        if target_vars and dummy_rv not in target_vars:
            continue

        rv, value, *dims = node.inputs
        if not isinstance(rv.owner.op, (AutoRegressiveRV,)):
            if target_vars:
                raise NotImplementedError(f"RV distribution {rv.owner.op} can't be forecasted")
            else:
                continue

        # For free RVs we use the RV as the starting value
        # For observedRVs we use the actual value as the starting value
        if isinstance(node.op, ModelFreeRV):
            value = rv

        if isinstance(rv.owner.op, AutoRegressiveRV):
            init_dist = DiracDelta.dist(value[-1])
            rhos, sigma, _, old_steps, _ = rv.owner.inputs
            new_rv = AR.rv_op(
                rhos,
                sigma,
                init_dist,
                steps=steps or old_steps,
                ar_order=rv.owner.op.ar_order,
                constant_term=rv.owner.op.constant_term,
            )

        new_rv.name = rv.name
        new_dummy_rv = model_free_rv(new_rv, new_rv.type(), None, *dims)
        replacements[dummy_rv] = new_dummy_rv

    toposort_replace(fgraph, tuple(replacements.items()))
    return model_from_fgraph(fgraph)
