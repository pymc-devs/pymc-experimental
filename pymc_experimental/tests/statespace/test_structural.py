import numpy as np
import pytensor
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from scipy import linalg

from pymc_experimental.statespace import structural as st
from pymc_experimental.statespace.utils.constants import (
    MATRIX_NAMES,
    SHORT_NAME_TO_LONG,
)
from pymc_experimental.tests.statespace.utilities.shared_fixtures import (  # pylint: disable=unused-import
    rng,
)

floatX = pytensor.config.floatX
ATOL = 1e-8 if floatX.endswith("64") else 1e-6
RTOL = 0 if floatX.endswith("64") else 1e-6


def unpack_statespace(ssm):
    return [ssm[SHORT_NAME_TO_LONG[x]] for x in MATRIX_NAMES]


def unpack_symbolic_matrices_with_params(mod, param_dict):
    f_matrices = pytensor.function(
        list(mod._name_to_variable.values()), unpack_statespace(mod.ssm), on_unused_input="ignore"
    )
    x0, P0, c, d, T, Z, R, H, Q = f_matrices(**param_dict)
    return x0, P0, c, d, T, Z, R, H, Q


def simulate_from_numpy_model(mod, rng, param_dict, steps=100):
    """
    Helper function to visualize the components outside of a PyMC model context
    """
    x0, P0, c, d, T, Z, R, H, Q = unpack_symbolic_matrices_with_params(mod, param_dict)
    k_states = mod.k_states
    k_posdef = mod.k_posdef

    x = np.zeros((steps, k_states))
    y = np.zeros(steps)

    x[0] = x0
    y[0] = Z @ x0

    if not np.allclose(H, 0):
        y[0] += rng.multivariate_normal(mean=np.zeros(1), cov=H)

    for t in range(1, steps):
        if k_posdef > 0:
            shock = rng.multivariate_normal(mean=np.zeros(k_posdef), cov=Q)
            innov = R @ shock
        else:
            innov = 0

        if not np.allclose(H, 0):
            error = rng.multivariate_normal(mean=np.zeros(1), cov=H)
        else:
            error = 0

        x[t] = c + T @ x[t - 1] + innov
        y[t] = d + Z @ x[t] + error

    return x, y


def test_deterministic_constant_model(rng):
    mod = st.LevelTrendComponent(order=1, innovations_order=0)
    params = {"initial_trend": [1.0]}
    x, y = simulate_from_numpy_model(mod, rng, params)

    assert_allclose(y, 1, atol=ATOL, rtol=RTOL)


def test_deterministic_slope_model(rng):
    mod = st.LevelTrendComponent(order=2, innovations_order=0)
    params = {"initial_trend": [0.0, 1.0]}
    x, y = simulate_from_numpy_model(mod, rng, params)

    assert_allclose(np.diff(y), 1, atol=ATOL, rtol=RTOL)


def test_model_addition():
    # TODO: Lame test, improve
    ll = st.LevelTrendComponent(order=1, innovations_order=1)
    me = st.MeasurementError("data")

    mod = ll + me
    assert "sigma_data" in mod.param_names
    assert "sigma_data" in mod.param_info.keys()
    assert "sigma_data" in mod.param_dims.keys()


@pytest.mark.parametrize("order", [1, 2, [1, 0, 1]], ids=["AR1", "AR2", "AR(1,0,1)"])
def test_autoregressive_model(order, rng):
    # TODO: Improve this test
    ar = st.AutoregressiveComponent(order=order)
    params = {
        "ar_params": np.full((sum(ar.order),), 0.95, dtype=floatX),
        "sigma_ar": np.array([0.1], dtype=floatX),
    }
    x, y = simulate_from_numpy_model(ar, rng, params)


@pytest.mark.parametrize("s", [10, 25, 50])
@pytest.mark.parametrize("innovations", [True, False])
def test_time_seasonality(s, innovations, rng):
    mod = st.TimeSeasonality(season_length=s, innovations=innovations, name="season")
    x0 = np.zeros(mod.k_states, dtype=floatX)
    x0[0] = 1

    params = {"season_coefs": x0}
    if mod.innovations:
        params["sigma_season"] = np.array([0.0], dtype=floatX)

    x, y = simulate_from_numpy_model(mod, rng, params)
    y = y.ravel()
    if not innovations:
        assert_allclose(
            y[:s], y[s : s * 2], err_msg="seasonal pattern does not repeat", atol=ATOL, rtol=RTOL
        )

    mod2 = sm.tsa.UnobservedComponents(
        endog=rng.normal(size=100),
        seasonal=s,
        stochastic_seasonal=innovations,
        # Silence a warning about no innovations when innovations = False
        irregular=True,
    )
    x0, P0, c, d, T, Z, R, H, Q = unpack_symbolic_matrices_with_params(mod, params)

    for name, matrix in zip(["T", "R", "Z", "Q"], [T, R, Z, Q]):
        long_name = SHORT_NAME_TO_LONG[name]
        assert_allclose(
            mod2.ssm[long_name],
            matrix,
            err_msg=f"matrix {name} does not match statsmodels",
            atol=ATOL,
            rtol=RTOL,
        )


def get_shift_factor(s):
    s_str = str(s)
    if "." not in s_str:
        return 1
    _, decimal = s_str.split(".")
    return 10 ** len(decimal)


@pytest.mark.parametrize("n", np.arange(1, 6, dtype="int").tolist() + [None])
@pytest.mark.parametrize("s", [5, 10, 25, 25.2])
def test_frequency_seasonality(n, s, rng):
    mod = st.FrequencySeasonality(season_length=s, n=n, name="season")
    x0 = rng.normal(size=mod.n_coefs).astype(floatX)
    params = {"season": x0, "sigma_season": np.array([0.0], dtype=floatX)}
    k = get_shift_factor(s)
    T = int(s * k)

    x, y = simulate_from_numpy_model(mod, rng, params, 2 * T)
    assert_allclose(
        np.diff(y.reshape(-1, T), axis=0),
        0,
        err_msg="seasonal pattern does not repeat",
        atol=ATOL,
        rtol=RTOL,
    )

    init_dict = {"period": s}
    if n is not None:
        init_dict["harmonics"] = n

    mod2 = sm.tsa.UnobservedComponents(endog=rng.normal(size=100), freq_seasonal=[init_dict])
    mod2.initialize_default()
    x0, P0, c, d, T, Z, R, H, Q = unpack_symbolic_matrices_with_params(mod, params)

    for name, matrix in zip(["T", "Z", "R", "Q"], [T, Z, R, Q]):
        name = SHORT_NAME_TO_LONG[name]
        assert_allclose(
            mod2.ssm[name],
            matrix,
            err_msg=f"matrix {name} does not match statsmodels",
        )

    assert mod2.initialization.constant.shape == mod.ssm["initial_state"].type.shape


@pytest.mark.parametrize("s,n", [(10, 5), pytest.param(10.2, 5, marks=pytest.mark.xfail)])
def test_state_removed_when_freq_seasonality_is_saturated(s, n):
    mod = st.FrequencySeasonality(season_length=s, n=n, name="test")
    init_params = mod._name_to_variable["test"]

    assert init_params.type.shape[0] == (n * 2 - 1)


def test_add_components():
    ll = st.LevelTrendComponent(order=2)
    se = st.TimeSeasonality(name="seasonal", season_length=12)
    mod = ll + se

    ll_params = {
        "initial_trend": np.zeros(2, dtype=floatX),
        "sigma_trend": np.ones(2, dtype=floatX),
    }
    se_params = {
        "seasonal_coefs": np.ones(11, dtype=floatX),
        "sigma_seasonal": np.ones(1, dtype=floatX),
    }
    all_params = ll_params.copy() | se_params.copy()

    (ll_x0, ll_P0, ll_c, ll_d, ll_T, ll_Z, ll_R, ll_H, ll_Q) = unpack_symbolic_matrices_with_params(
        ll, ll_params
    )
    (se_x0, se_P0, se_c, se_d, se_T, se_Z, se_R, se_H, se_Q) = unpack_symbolic_matrices_with_params(
        se, se_params
    )
    x0, P0, c, d, T, Z, R, H, Q = unpack_symbolic_matrices_with_params(mod, all_params)

    for property in ["param_names", "shock_names", "param_info", "coords", "param_dims"]:
        assert [x in getattr(mod, property) for x in getattr(ll, property)]
        assert [x in getattr(mod, property) for x in getattr(se, property)]

    ll_mats = [ll_T, ll_R, ll_Q]
    se_mats = [se_T, se_R, se_Q]
    all_mats = [T, R, Q]

    for (ll_mat, se_mat, all_mat) in zip(ll_mats, se_mats, all_mats):
        assert_allclose(all_mat, linalg.block_diag(ll_mat, se_mat), atol=ATOL, rtol=RTOL)

    ll_mats = [ll_x0, ll_c, ll_Z]
    se_mats = [se_x0, se_c, se_Z]
    all_mats = [x0, c, Z]
    axes = [0, 0, 1]

    for (ll_mat, se_mat, all_mat, axis) in zip(ll_mats, se_mats, all_mats, axes):
        assert_allclose(all_mat, np.concatenate([ll_mat, se_mat], axis=axis), atol=ATOL, rtol=RTOL)
