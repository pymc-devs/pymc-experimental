import numpy as np
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose

from pymc_experimental.statespace import structural as st
from pymc_experimental.statespace.utils.constants import SHORT_NAME_TO_LONG


@pytest.fixture
def rng():
    return np.random.default_rng(1337)


def simulate_from_numpy_model(mod, rng):
    x0, P0, c, d, T, Z, R, H, Q = mod.x0, mod.P0, mod.c, mod.d, mod.T, mod.Z, mod.R, mod.H, mod.Q
    k_states = mod.k_states
    k_posdef = mod.k_posdef

    x = np.zeros((100, k_states))
    y = np.zeros(100)

    x[0] = x0
    y[0] = Z @ x0

    for t in range(1, 100):
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
    mod.x0[0] = 1
    x, y = simulate_from_numpy_model(mod, rng)

    assert_allclose(y, 1)


def test_deterministic_slope_model(rng):
    mod = st.LevelTrendComponent(order=2, innovations_order=0)
    mod.x0[1] = 1
    x, y = simulate_from_numpy_model(mod, rng)

    assert_allclose(np.diff(y), 1)


def test_model_addition():
    # TODO: Lame test, improve
    ll = st.LevelTrendComponent(order=1, innovations_order=1)
    me = st.MeasurementError("data")

    mod = ll + me
    assert "sigma_data" in mod.param_counts.keys()
    assert "sigma_data" in mod.param_names
    assert "sigma_data" in mod.param_info.keys()
    assert "sigma_data" in mod.param_dims.keys()


@pytest.mark.parametrize("order", [1, 2, [1, 0, 1]], ids=["AR1", "AR2", "AR(1,0,1)"])
def test_autoregressive_model(order, rng):
    # TODO: Lame test, improve
    ar = st.AutoregressiveComponent(order=order)
    ar.T[ar.param_indices["ar_params"][1:]] = 0.9
    x, y = simulate_from_numpy_model(ar, rng)

    k = np.sum(order)
    assert x.shape == (100, k)


@pytest.mark.parametrize("s", [10, 25, 50])
@pytest.mark.parametrize("innovations", [True, False])
def test_time_seasonality(s, innovations, rng):
    mod = st.TimeSeasonality(season_length=s, innovations=innovations)
    mod.x0[0] = 1
    x, y = simulate_from_numpy_model(mod, rng)

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


@pytest.mark.parametrize("n", np.arange(1, 6, dtype="int"))
@pytest.mark.parametrize("s", [10, 25, 50])
def test_frequency_seasonality(n, s, rng):
    mod = st.FrequencySeasonality(season_length=s, n=n)
    mod.x0[0] = 1
    x, y = simulate_from_numpy_model(mod, rng)

    assert_allclose(y[:s], y[s : s * 2])

    mod2 = sm.tsa.UnobservedComponents(
        endog=rng.normal(size=100), freq_seasonal=[{"period": s, "harmonics": n}]
    )

    for matrix in ["T", "R", "Z", "Q"]:
        name = SHORT_NAME_TO_LONG[matrix]
        assert_allclose(mod2.ssm[name], getattr(mod, matrix), err_msg=name)
