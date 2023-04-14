import arviz as az
import numpy as np
import pymc as pm
import pytest

from pymc_experimental.model_transform import forecast_timeseries, uncensor


@pytest.mark.parametrize(
    "transform, kwargs",
    [
        (uncensor, dict()),
        (forecast_timeseries, dict(forecast_steps=20)),
    ],
)
def test_transform_error(transform, kwargs):
    """Test informative error is raised when the transform is not applicable to a model."""
    with pm.Model() as model:
        x = pm.Normal("x")
        y = pm.Normal("y", x, observed=[0, 5, 10])

    with pytest.raises(RuntimeError, match="No .* were replaced by .* counterparts"):
        transform(model, **kwargs)


def test_uncensor():
    with pm.Model() as model:
        x = pm.Normal("x")
        dist_raw = pm.Normal.dist(x)
        y = pm.Censored("y", dist=dist_raw, lower=-1, upper=1, observed=[0, 5, 10])
        det = pm.Deterministic("det", y * 2)

    idata = az.from_dict({"x": np.zeros((2, 500))})

    with uncensor(model):
        pp = pm.sample_posterior_predictive(
            idata,
            var_names=["y_uncensored", "det"],
            random_seed=18,
        ).posterior_predictive

    assert np.any(np.abs(pp["y_uncensored"]) > 1)
    np.testing.assert_allclose(pp["y_uncensored"] * 2, pp["det"])


@pytest.mark.parametrize("observed", (True, False))
@pytest.mark.parametrize("ar_order", (1, 2))
def test_forecast_timeseries_ar(observed, ar_order):
    data_steps = 3
    data = np.hstack((np.zeros(ar_order), (np.arange(data_steps) + 1) * 100.0))
    with pm.Model() as model:
        rho = pm.Normal("rho", shape=(ar_order,))
        sigma = pm.HalfNormal("sigma")
        init_dist = pm.Normal.dist(0, 1e-3)
        y = pm.AR(
            "y",
            init_dist=init_dist,
            rho=rho,
            sigma=sigma,
            observed=data if observed else None,
            steps=data_steps,
        )
        det = pm.Deterministic("det", y * 2)

    draws = (2, 50)
    # These rhos mean that all steps will be data[-1] for ar_order > 1
    idata_dict = {
        "rho": np.full((*draws, ar_order), (0.1,) + (0,) * (ar_order - 1)),
        "sigma": np.full(draws, 1e-5),
    }
    if observed:
        idata = az.from_dict(idata_dict, observed_data={"y": data})
    else:
        idata_dict["y"] = np.full((*draws, len(data)), data)
        idata = az.from_dict(idata_dict)

    forecast_steps = 5
    with forecast_timeseries(model, forecast_steps=forecast_steps):
        pp = pm.sample_posterior_predictive(
            idata,
            var_names=["y_forecast", "det"],
            random_seed=50,
        ).posterior_predictive

    expected = data[-1] / np.logspace(0, forecast_steps, forecast_steps + 1)
    expected = np.hstack((data[-ar_order:-1], expected))
    np.testing.assert_allclose(
        pp["y_forecast"].values, np.full((*draws, forecast_steps + ar_order), expected), rtol=0.01
    )
    np.testing.assert_allclose(pp["y_forecast"] * 2, pp["det"])
