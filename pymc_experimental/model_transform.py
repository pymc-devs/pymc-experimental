from pymc import DiracDelta
from pymc.distributions.censored import CensoredRV
from pymc.distributions.timeseries import AR, AutoRegressiveRV
from pymc.model import Model
from pytensor import shared
from pytensor.graph import FunctionGraph, node_rewriter
from pytensor.graph.basic import get_var_by_name
from pytensor.graph.rewriting.basic import in2out

from pymc_experimental.utils.model_fgraph import (
    ModelObservedRV,
    ModelValuedVar,
    fgraph_from_model,
    model_free_rv,
    model_from_fgraph,
    model_named,
)

__all__ = (
    "forecast_timeseries",
    "uncensor",
)


@node_rewriter(tracks=[ModelValuedVar])
def uncensor_node_rewrite(fgraph, node):
    """Rewrite that replaces censored variables by uncensored ones"""

    (
        censored_rv,
        value,
        *dims,
    ) = node.inputs
    if not isinstance(censored_rv.owner.op, CensoredRV):
        return

    model_rv = node.outputs[0]
    base_rv = censored_rv.owner.inputs[0]
    uncensored_rv = node.op.make_node(base_rv, value, *dims).default_output()
    uncensored_rv.name = f"{model_rv.name}_uncensored"
    return [uncensored_rv]


uncensor_rewrite = in2out(uncensor_node_rewrite)


def uncensor(model: Model) -> Model:
    """Replace censored variables in the model by uncensored equivalent.

    Replaced variables have the same name as original ones with an additional "_uncensored" suffix.

    .. code-block:: python

        import arviz as az
        import pymc as pm
        from pymc_experimental.model_transform import uncensor

        with pm.Model() as model:
            x = pm.Normal("x")
            dist_raw = pm.Normal.dist(x)
            y = pm.Censored("y", dist=dist_raw, lower=-1, upper=1, observed=[-1, 0.5, 1, 1, 1])
            idata = pm.sample()

        with uncensor(model):
            idata_pp = pm.sample_posterior_predictive(idata, var_names=["y_uncensored"])

        az.summary(idata_pp)
    """
    fg = fgraph_from_model(model)

    (_, nodes_changed, *_) = uncensor_rewrite.apply(fg)
    if not nodes_changed:
        raise RuntimeError("No censored variables were replaced by uncensored counterparts")

    return model_from_fgraph(fg)


@node_rewriter(tracks=[ModelValuedVar])
def forecast_timeseries_node_rewrite(fgraph: FunctionGraph, node):
    """Rewrite that replaces timeseries variables by new ones starting at the last timepoint(s)."""

    (
        timeseries_rv,
        value,
        *dims,
    ) = node.inputs
    if not isinstance(timeseries_rv.owner.op, AutoRegressiveRV):
        return

    forecast_steps = get_var_by_name(fgraph.inputs, "forecast_steps_")
    if len(forecast_steps) != 1:
        return False

    forecast_steps = forecast_steps[0]

    op = timeseries_rv.owner.op
    model_rv = node.outputs[0]

    # We cannot reference the variable we are planning to replace
    # Or it will introduce circularities in the graph
    # FIXME: This special logic shouldn't be needed for ObservedRVs
    # but PyMC does not allow one to not resample observed.
    # We hack around by conditioning on the value variable directly,
    # even though that should not be part of the generative graph...
    if isinstance(node.op, ModelObservedRV):
        init_dist = DiracDelta.dist(value[..., -op.ar_order :])
    else:
        cloned_model_rv = model_rv.owner.clone().default_output()
        fgraph.add_output(cloned_model_rv, import_missing=True)
        init_dist = DiracDelta.dist(cloned_model_rv[..., -op.ar_order :])

    if isinstance(timeseries_rv.owner.op, AutoRegressiveRV):
        rhos, sigma, *_ = timeseries_rv.owner.inputs
        new_timeseries_rv = AR.rv_op(
            rhos=rhos,
            sigma=sigma,
            init_dist=init_dist,
            steps=forecast_steps,
            ar_order=op.ar_order,
            constant_term=op.constant_term,
        )

    new_name = f"{model_rv.name}_forecast"
    new_value = new_timeseries_rv.type(name=new_name)
    new_timeseries_rv = model_free_rv(new_timeseries_rv, new_value, transform=None)
    new_timeseries_rv.name = new_name

    # Import new variables into fgraph (value and RNG)
    fgraph.import_var(new_timeseries_rv, import_missing=True)

    return [new_timeseries_rv]


forecast_timeseries_rewrite = in2out(forecast_timeseries_node_rewrite, ignore_newtrees=True)


def forecast_timeseries(model: Model, forecast_steps: int) -> Model:
    """Replace timeseries variables in the model by forecast that start at the last value.

    Replaced variables have the same name as original ones with an additional "_forecast" suffix.

    The function will fail if any variables with fixed static shape depend on the timeseries being replaced,
    and forecast_steps differs from the original timeseries steps.

    .. code-block:: python

        import pymc as pm
        from pymc_experimental.model_transform import forecast_timeseries

        with pm.Model() as model:
            rho = pm.Normal("rho")
            sigma = pm.HalfNormal("sigma")
            init_dist = pm.Normal.dist()
            y = pm.AR("y", init_dist=init_dist, rho=rho, sigma=sigma, observed=np.zeros(100,))
            idata = pm.sample()

        forecast_model = forecast_timeseries(mode, forecast_steps=20)
        with forecast_model:
            idata_pp = pm.sample_posterior_predictive(idata, var_names=["y_forecast"])

        az.summary(idata_pp)
    """

    fg = fgraph_from_model(model)

    forecast_steps_sh = shared(forecast_steps, name="forecast_steps_")
    forecast_steps_sh = model_named(forecast_steps_sh)
    fg.add_output(forecast_steps_sh, import_missing=True)

    (_, nodes_changed, *_) = forecast_timeseries_rewrite.apply(fg)
    if not nodes_changed:
        raise RuntimeError("No timeseries were replaced by forecast counterparts")

    res = model_from_fgraph(fg)
    return res
