from itertools import product

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose

from pymc_experimental.statespace import BayesianARIMA
from pymc_experimental.tests.statespace.utilities.shared_fixtures import (  # pylint: disable=unused-import
    rng,
)
from pymc_experimental.tests.statespace.utilities.test_helpers import (
    fast_eval,
    load_nile_test_data,
)

floatX = pytensor.config.floatX
ps = [0, 1, 2, 3]
ds = [0, 1, 2]
qs = [0, 1, 2, 3]
orders = list(product(ps, ds, qs))[1:]
ids = [f"p={x[0]}d={x[1]}q={x[2]}" for x in orders]


@pytest.fixture
def data():
    return load_nile_test_data()


@pytest.fixture(scope="session")
def arima_mod():
    return BayesianARIMA(
        order=(2, 0, 1), stationary_initialization=True, verbose=False, filter_type="univariate"
    )


@pytest.fixture(scope="session")
def pymc_mod(arima_mod):
    data = load_nile_test_data()

    with pm.Model(coords=arima_mod.coords) as pymc_mod:
        # x0  = pm.Normal('x0', dims=['state'])
        # P0_diag = pm.Gamma('P0_diag', alpha=2, beta=1, dims=['state'])
        # P0 = pm.Deterministic('P0', pt.diag(P0_diag), dims=['state', 'state_aux'])
        ar_params = pm.Normal("ar_params", sigma=0.1, dims=["ar_lag"])
        ma_params = pm.Normal("ma_params", sigma=1, dims=["ma_lag"])
        sigma_state = pm.Exponential("sigma_state", 0.5)
        arima_mod.build_statespace_graph(data=data)

    return pymc_mod


@pytest.mark.parametrize("order", orders, ids=ids)
@pytest.mark.parametrize("matrix", ["transition", "selection", "state_cov", "obs_cov", "design"])
def test_SARIMAX_init_matches_statsmodels(data, order, matrix):
    p, d, q = order

    mod = BayesianARIMA(order=(p, d, q), verbose=False)
    sm_sarimax = sm.tsa.SARIMAX(data, order=(p, d, q))

    assert_allclose(fast_eval(mod.ssm[matrix]), sm_sarimax.ssm[matrix])


@pytest.mark.parametrize("order", orders, ids=ids)
@pytest.mark.parametrize("matrix", ["transition", "selection", "state_cov", "obs_cov", "design"])
@pytest.mark.filterwarnings(
    "ignore:Non-invertible starting MA parameters found.",
    "ignore:Non-stationary starting autoregressive parameters found",
)
def test_SARIMAX_update_matches_statsmodels(data, order, matrix, rng):
    p, d, q = order

    sm_sarimax = sm.tsa.SARIMAX(data, order=(p, d, q))

    param_names = sm_sarimax.param_names
    param_d = {name: rng.normal(scale=0.1) ** 2 for name in param_names}

    res = sm_sarimax.fit_constrained(param_d)
    mod = BayesianARIMA(order=(p, d, q), verbose=False)

    with pm.Model() as pm_mod:
        x0 = pm.Deterministic("x0", pt.zeros(mod.k_states))
        ma_params = pm.Deterministic(
            "ma_params",
            pt.as_tensor_variable(np.array([param_d[k] for k in param_d if k.startswith("ma.")])),
        )
        ar_params = pm.Deterministic(
            "ar_params",
            pt.as_tensor_variable(np.array([param_d[k] for k in param_d if k.startswith("ar.")])),
        )
        state_cov = pm.Deterministic(
            "sigma_state", pt.as_tensor_variable(np.array([[param_d["sigma2"]]]))
        )
        mod.build_statespace_graph(data=data)

    assert_allclose(fast_eval(mod.ssm[matrix]), sm_sarimax.ssm[matrix])


@pytest.mark.parametrize("filter_output", ["filtered", "predicted", "smoothed"])
def test_all_prior_covariances_are_PSD(filter_output, pymc_mod, rng):
    rv = pymc_mod[f"{filter_output}_covariance"]
    cov_mats = pm.draw(rv, 100, random_seed=rng)
    w, v = np.linalg.eig(cov_mats)
    assert not np.any(w < 0)
