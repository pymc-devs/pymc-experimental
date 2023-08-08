import numpy as np
import pytensor
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose

from pymc_experimental.statespace import structural as st
from pymc_experimental.statespace.utils.constants import (
    MATRIX_NAMES,
    SHORT_NAME_TO_LONG,
)
from pymc_experimental.tests.statespace.utilities.shared_fixtures import (  # pylint: disable=unused-import
    rng,
)

floatX = pytensor.config.floatX
ATOL = 1e-8 if floatX.endswith("64") else 1e-4
RTOL = 0


def unpack_statespace(ssm):
    return [ssm[SHORT_NAME_TO_LONG[x]] for x in MATRIX_NAMES]


def simulate_from_numpy_model(mod, rng, param_dict, steps=100):
    """
    Helper function to visualize the components outside of a PyMC model context
    """

    f_matrices = pytensor.function(
        list(mod._name_to_variable.values()), unpack_statespace(mod.ssm), on_unused_input="ignore"
    )
    x0, P0, c, d, T, Z, R, H, Q = f_matrices(**param_dict)

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

    assert_allclose(y, 1)


def test_deterministic_slope_model(rng):
    mod = st.LevelTrendComponent(order=2, innovations_order=0)
    params = {"initial_trend": [0.0, 1.0]}
    x, y = simulate_from_numpy_model(mod, rng, params)

    assert_allclose(np.diff(y), 1)


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
    params = {"season_coefs": x0, "sigma_season": np.array([0.1], dtype=floatX)}
    x, y = simulate_from_numpy_model(mod, rng, params)

    assert_allclose(y[:s], y[s : s * 2])

    mod2 = sm.tsa.UnobservedComponents(
        endog=rng.normal(size=100),
        seasonal=s,
        stochastic_seasonal=innovations,
        # Silence a warning about no innovations when innovations = False
        irregular=True,
    )

    for matrix in ["T", "R", "Z", "Q"]:
        name = SHORT_NAME_TO_LONG[matrix]
        assert_allclose(mod2.ssm[name], getattr(mod, matrix), err_msg=name)


def get_shift_factor(s):
    s_str = str(s)
    if "." not in s_str:
        return 1
    _, decimal = s_str.split(".")
    return 10 ** len(decimal)


@pytest.mark.parametrize("n", np.arange(1, 6, dtype="int").tolist() + [None])
@pytest.mark.parametrize("s", [5, 10, 25, 25.2])
def test_frequency_seasonality(n, s, rng):
    mod = st.FrequencySeasonality(season_length=s, n=n)
    x0 = rng.normal(size=mod.k_states).astype(floatX)
    params = {mod.param_names[0]: x0}
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

    for matrix in ["T", "Z", "R", "Q"]:
        name = SHORT_NAME_TO_LONG[matrix]
        assert_allclose(
            mod2.ssm[name],
            getattr(mod, matrix),
            err_msg=f"matrix {name} does not match statsmodels",
        )

    assert mod2.initialization.constant.shape == mod.ssm["initial_state"].type.shape


@pytest.mark.parametrize("s,n", [(10, 5), pytest.param(10.2, 5, marks=pytest.mark.xfail)])
def test_state_removed_when_freq_seasonality_is_saturated(s, n):
    mod = st.FrequencySeasonality(season_length=s, n=n, name="test")
    init_params = mod._name_to_variable["test"]

    assert init_params.type.shape[0] == (n * 2 - 1)
