from itertools import combinations, product

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose, assert_array_less

from pymc_experimental.statespace import BayesianARIMA
from pymc_experimental.statespace.utils.constants import (
    SARIMAX_STATE_STRUCTURES,
    SHORT_NAME_TO_LONG,
)
from pymc_experimental.tests.statespace.utilities.shared_fixtures import (  # pylint: disable=unused-import
    rng,
)
from pymc_experimental.tests.statespace.utilities.test_helpers import (
    load_nile_test_data,
    simulate_from_numpy_model,
)

floatX = pytensor.config.floatX
ps = [0, 1, 2, 3]
ds = [0, 1, 2]
qs = [0, 1, 2, 3]
orders = list(product(ps, ds, qs))[1:]
ids = [f"p={x[0]}d={x[1]}q={x[2]}" for x in orders]

ATOL = 1e-8 if floatX.endswith("64") else 1e-6
RTOL = 0 if floatX.endswith("64") else 1e-6


@pytest.fixture
def data():
    return load_nile_test_data()


@pytest.fixture(scope="session")
def arima_mod():
    return BayesianARIMA(order=(2, 0, 1), stationary_initialization=True, verbose=False)


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


@pytest.fixture(scope="session")
def arima_mod_interp():
    return BayesianARIMA(
        order=(3, 0, 3),
        stationary_initialization=False,
        verbose=False,
        state_structure="interpretable",
        measurement_error=True,
    )


@pytest.fixture(scope="session")
def pymc_mod_interp(arima_mod_interp):
    data = load_nile_test_data()

    with pm.Model(coords=arima_mod_interp.coords) as pymc_mod:
        x0 = pm.Normal("x0", dims=["state"])
        P0_sigma = pm.Exponential("P0_sigma", 1)
        P0 = pm.Deterministic(
            "P0", pt.eye(arima_mod_interp.k_states) * P0_sigma, dims=["state", "state_aux"]
        )
        ar_params = pm.Normal("ar_params", sigma=0.1, dims=["ar_lag"])
        ma_params = pm.Normal("ma_params", sigma=1, dims=["ma_lag"])
        sigma_state = pm.Exponential("sigma_state", 0.5)
        sigma_obs = pm.Exponential("sigma_obs", 0.1)
        arima_mod_interp.build_statespace_graph(data=data)

    return pymc_mod


@pytest.mark.parametrize("order", orders, ids=ids)
@pytest.mark.filterwarnings(
    "ignore:Non-invertible starting MA parameters found.",
    "ignore:Non-stationary starting autoregressive parameters found",
)
def test_SARIMAX_update_matches_statsmodels(data, order, rng):
    p, d, q = order
    sm_sarimax = sm.tsa.SARIMAX(data, order=(p, d, q))

    param_names = sm_sarimax.param_names
    param_d = {name: getattr(np, floatX)(rng.normal(scale=0.1) ** 2) for name in param_names}

    res = sm_sarimax.fit_constrained(param_d)
    mod = BayesianARIMA(order=(p, d, q), verbose=False, stationary_initialization=False)

    with pm.Model() as pm_mod:
        x0 = pm.Normal("x0", shape=(mod.k_states,))
        P0 = pm.Deterministic("P0", pt.eye(mod.k_states, dtype=floatX))

        if q > 0:
            pm.Deterministic(
                "ma_params",
                pt.as_tensor_variable(
                    np.array([param_d[k] for k in param_d if k.startswith("ma.")])
                ),
            )
        if p > 0:
            pm.Deterministic(
                "ar_params",
                pt.as_tensor_variable(
                    np.array([param_d[k] for k in param_d if k.startswith("ar.")])
                ),
            )
        pm.Deterministic("sigma_state", pt.as_tensor_variable(np.array([param_d["sigma2"]])))

        mod._insert_random_variables()
        matrices = pm.draw(mod.subbed_ssm)
        matrix_dict = dict(zip(SHORT_NAME_TO_LONG.values(), matrices))

    for matrix in ["transition", "selection", "state_cov", "obs_cov", "design"]:
        assert_allclose(matrix_dict[matrix], sm_sarimax.ssm[matrix], err_msg=f"{matrix} not equal")


@pytest.mark.parametrize("filter_output", ["filtered", "predicted", "smoothed"])
def test_all_prior_covariances_are_PSD(filter_output, pymc_mod, rng):
    rv = pymc_mod[f"{filter_output}_covariance"]
    cov_mats = pm.draw(rv, 100, random_seed=rng)
    w, v = np.linalg.eig(cov_mats)
    assert_array_less(0, w, err_msg=f"Smallest eigenvalue: {min(w.ravel())}")


def test_interpretable_raises_if_d_nonzero():
    with pytest.raises(
        ValueError, match="Cannot use interpretable state structure with statespace differencing"
    ):
        BayesianARIMA(
            order=(2, 1, 1),
            stationary_initialization=True,
            verbose=False,
            state_structure="interpretable",
        )


def test_interpretable_states_are_interpretable(arima_mod_interp, pymc_mod_interp):
    with pymc_mod_interp:
        prior = pm.sample_prior_predictive(samples=10)

    prior_outputs = arima_mod_interp.sample_unconditional_prior(prior)
    ar_lags = prior.prior.coords["ar_lag"].values - 1
    ma_lags = prior.prior.coords["ma_lag"].values - 1

    # Check the first p states are lags of the previous state
    for (t, tm1) in zip(ar_lags[1:], ar_lags[:-1]):
        assert_allclose(
            prior_outputs.prior_latent.isel(state=t).values[1:],
            prior_outputs.prior_latent.isel(state=tm1).values[:-1],
            err_msg=f"State {tm1} is not a lagged version of state {t} (AR lags)",
        )

    # Check the next p+q states are lags of the innovations
    n = len(ar_lags)
    for (t, tm1) in zip(ma_lags[1:], ma_lags[:-1]):
        assert_allclose(
            prior_outputs.prior_latent.isel(state=n + t).values[1:],
            prior_outputs.prior_latent.isel(state=n + tm1).values[:-1],
            err_msg=f"State {n + tm1} is not a lagged version of state {n+t} (MA lags)",
        )


def test_representations_are_equivalent(rng):

    test_values = {}
    shared_params = {
        "ar_params": np.array([0.95, 0.5, -0.5], dtype=floatX),
        "ma_params": np.array([1.1, -0.6, 2.4], dtype=floatX),
        "sigma_state": np.array([0.8], dtype=floatX),
    }

    for representation in SARIMAX_STATE_STRUCTURES:
        rng = np.random.default_rng(sum(map(ord, "representation test")))
        mod = BayesianARIMA(
            order=(3, 0, 3),
            stationary_initialization=False,
            verbose=False,
            state_structure=representation,
        )
        params = shared_params.copy()
        params["x0"] = np.zeros(mod.k_states, dtype=floatX)
        params["initial_state_cov"] = np.eye(mod.k_states, dtype=floatX)

        x, y = simulate_from_numpy_model(mod, rng, params)
        test_values[representation] = y

    all_pairs = combinations(SARIMAX_STATE_STRUCTURES, r=2)
    for rep_1, rep_2 in all_pairs:
        assert_allclose(
            test_values[rep_1],
            test_values[rep_2],
            err_msg=f"{rep_1} and {rep_2} are not the same!",
            atol=ATOL,
            rtol=RTOL,
        )
