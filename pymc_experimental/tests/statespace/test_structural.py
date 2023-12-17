import functools as ft
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from scipy import linalg

from pymc_experimental.statespace import structural as st
from pymc_experimental.statespace.utils.constants import SHORT_NAME_TO_LONG
from pymc_experimental.tests.statespace.utilities.shared_fixtures import (  # pylint: disable=unused-import
    rng,
)
from pymc_experimental.tests.statespace.utilities.test_helpers import (
    assert_pattern_repeats,
    simulate_from_numpy_model,
    unpack_symbolic_matrices_with_params,
)

floatX = pytensor.config.floatX
ATOL = 1e-8 if floatX.endswith("64") else 1e-4
RTOL = 0 if floatX.endswith("64") else 1e-6


def _assert_all_statespace_matrices_match(mod, params, sm_mod):
    x0, P0, c, d, T, Z, R, H, Q = unpack_symbolic_matrices_with_params(mod, params)

    for name, matrix in zip(["T", "R", "Z", "Q"], [T, R, Z, Q]):
        long_name = SHORT_NAME_TO_LONG[name]
        if np.any([x == 0 for x in matrix.shape]):
            continue
        assert_allclose(
            sm_mod.ssm[long_name],
            matrix,
            err_msg=f"matrix {name} does not match statsmodels",
            atol=ATOL,
            rtol=RTOL,
        )


def create_structural_model_and_equivalent_statsmodel(
    rng,
    level: Optional[bool] = False,
    trend: Optional[bool] = False,
    seasonal: Optional[int] = None,
    freq_seasonal: Optional[list[dict]] = None,
    cycle: bool = False,
    autoregressive: Optional[int] = None,
    exog: Optional[np.ndarray] = None,
    irregular: Optional[bool] = False,
    stochastic_level: Optional[bool] = True,
    stochastic_trend: Optional[bool] = False,
    stochastic_seasonal: Optional[bool] = True,
    stochastic_freq_seasonal: Optional[list[bool]] = None,
    stochastic_cycle: Optional[bool] = False,
    damped_cycle: Optional[bool] = False,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = ft.partial(
            sm.tsa.UnobservedComponents,
            level=level,
            trend=trend,
            seasonal=seasonal,
            freq_seasonal=freq_seasonal,
            cycle=cycle,
            autoregressive=autoregressive,
            exog=exog,
            irregular=irregular,
            stochastic_level=stochastic_level,
            stochastic_trend=stochastic_trend,
            stochastic_seasonal=stochastic_seasonal,
            stochastic_freq_seasonal=stochastic_freq_seasonal,
            stochastic_cycle=stochastic_cycle,
            damped_cycle=damped_cycle,
            mle_regression=False,
        )

    params = {}
    sm_params = {}
    components = []

    if irregular:
        sigma = np.abs(rng.normal(size=(1,)))
        params["sigma_irregular"] = sigma
        sm_params["sigma2.irregular"] = sigma.item()
        comp = st.MeasurementError("irregular")
        components.append(comp)

    level_trend_order = [0, 0]
    level_trend_innov_order = [0, 0]

    if level:
        level_trend_order[0] = 1
        if stochastic_level:
            level_trend_innov_order[0] = 1

    if trend:
        level_trend_order[1] = 1
        if stochastic_trend:
            level_trend_innov_order[1] = 1

    if level or trend:
        level_value = np.where(
            level_trend_order,
            rng.normal(
                size=2,
            ),
            np.zeros(
                2,
            ),
        )
        sigma_level_value = np.abs(rng.normal(size=(2,)))[
            np.array(level_trend_innov_order, dtype="bool")
        ]
        max_order = np.flatnonzero(level_value)[-1].item() + 1
        level_trend_order = level_trend_order[:max_order]

        params["initial_trend"] = level_value[:max_order]
        if sum(level_trend_innov_order) > 0:
            params["sigma_trend"] = sigma_level_value

        sigma_level_value = sigma_level_value.tolist()
        if stochastic_level:
            sigma = sigma_level_value.pop(0)
            sm_params["sigma2.level"] = sigma
        if stochastic_trend:
            sigma = sigma_level_value.pop(0)
            sm_params[f"sigma2.trend"] = sigma

        comp = st.LevelTrendComponent(
            name="level", order=level_trend_order, innovations_order=level_trend_innov_order
        )
        components.append(comp)

    if seasonal is not None:
        params["seasonal_coefs"] = rng.normal(size=(seasonal - 1,))

        if stochastic_seasonal:
            sigma = np.abs(rng.normal(size=(1,)))
            params["sigma_seasonal"] = sigma
            sm_params["sigma2.seasonal"] = sigma

        comp = st.TimeSeasonality(
            name="seasonal", season_length=seasonal, innovations=stochastic_seasonal
        )
        components.append(comp)

    if freq_seasonal is not None:
        for d, has_innov in zip(freq_seasonal, stochastic_freq_seasonal):
            n = d["harmonics"]
            s = d["period"]

            last_state_not_identified = s / n == 2.0

            params[f"seasonal_{s}"] = rng.normal(size=(2 * n - int(last_state_not_identified)))
            if has_innov:
                sigma = np.abs(rng.normal(size=(1,)))
                params[f"sigma_seasonal_{s}"] = sigma
                sm_params[f"sigma2.freq_seasonal_{s}({n})"] = sigma

            comp = st.FrequencySeasonality(
                name=f"seasonal_{s}", season_length=s, n=n, innovations=has_innov
            )
            components.append(comp)

    if cycle:
        cycle_length = pm.math.floatX(np.random.choice(np.arange(2, 12)))

        # Statsmodels takes the frequency not the cycle length, so convert it.
        sm_params["frequency.cycle"] = 2.0 * np.pi / cycle_length
        params["cycle_cycle_length"] = np.atleast_1d(cycle_length)
        params["initial_cycle"] = np.ones((1,))

        if stochastic_cycle:
            sigma = np.abs(rng.normal(size=(1,)))
            params["sigma_cycle"] = sigma
            sm_params["sigma2.cycle"] = sigma

        if damped_cycle:
            rho = rng.beta(1, 1, size=(1,))
            params["cycle_dampening_factor"] = rho
            sm_params["damping.cycle"] = rho

        comp = st.CycleComponent(
            name="cycle",
            dampen=damped_cycle,
            innovations=stochastic_cycle,
            estimate_cycle_length=True,
        )

        components.append(comp)

    if autoregressive is not None:
        ar_params = rng.normal(size=(autoregressive,))
        sigma = np.abs(rng.normal(size=(1,)))
        params["ar_params"] = ar_params
        params["sigma_ar"] = sigma

        sm_params["sigma2.ar"] = sigma
        for i, rho in enumerate(ar_params):
            sm_params[f"ar.L{i + 1}"] = rho

        comp = st.AutoregressiveComponent(name="ar", order=autoregressive)
        components.append(comp)

    if exog is not None:
        names = [f"x{i + 1}" for i in range(exog.shape[1])]
        betas = rng.normal(size=(exog.shape[1],))
        params["beta_exog"] = betas
        params["data_exog"] = exog

        for i, beta in enumerate(betas):
            sm_params[f"beta.x{i + 1}"] = beta
        comp = st.RegressionComponent(name="exog", state_names=names)
        components.append(comp)

    st_mod = components.pop(0)
    for comp in components:
        st_mod += comp
    return mod, st_mod, params, sm_params


@pytest.mark.parametrize(
    "level, trend, stochastic_level, stochastic_trend, irregular",
    [
        (False, False, False, False, True),
        (True, True, True, True, True),
        (True, True, False, True, False),
    ],
)
@pytest.mark.parametrize("autoregressive", [None, 3])
@pytest.mark.parametrize("seasonal, stochastic_seasonal", [(None, False), (12, False), (12, True)])
@pytest.mark.parametrize(
    "freq_seasonal, stochastic_freq_seasonal",
    [
        (None, None),
        ([{"period": 12, "harmonics": 2}], [False]),
        ([{"period": 12, "harmonics": 6}], [True]),
    ],
)
@pytest.mark.parametrize(
    "cycle, damped_cycle, stochastic_cycle",
    [(False, False, False), (True, False, True), (True, True, True)],
)
@pytest.mark.filterwarnings("ignore::statsmodels.tools.sm_exceptions.ConvergenceWarning")
@pytest.mark.filterwarnings("ignore::statsmodels.tools.sm_exceptions.SpecificationWarning")
def test_structural_model_against_statsmodels(
    level,
    trend,
    stochastic_level,
    stochastic_trend,
    irregular,
    autoregressive,
    seasonal,
    stochastic_seasonal,
    freq_seasonal,
    stochastic_freq_seasonal,
    cycle,
    damped_cycle,
    stochastic_cycle,
    rng,
):
    f_sm_mod, mod, params, sm_params = create_structural_model_and_equivalent_statsmodel(
        rng,
        level=level,
        trend=trend,
        seasonal=seasonal,
        freq_seasonal=freq_seasonal,
        cycle=cycle,
        damped_cycle=damped_cycle,
        autoregressive=autoregressive,
        irregular=irregular,
        stochastic_level=stochastic_level,
        stochastic_trend=stochastic_trend,
        stochastic_seasonal=stochastic_seasonal,
        stochastic_freq_seasonal=stochastic_freq_seasonal,
        stochastic_cycle=stochastic_cycle,
    )

    data = rng.normal(size=(100,))
    sm_mod = f_sm_mod(data)
    sm_mod.initialize_default()

    if len(sm_params) > 0:
        param_array = np.concatenate(
            [np.atleast_1d(sm_params[k]).ravel() for k in sm_mod.param_names]
        )
        sm_mod.update(param_array, transformed=True)

    _assert_all_statespace_matrices_match(mod, params, sm_mod)


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


@pytest.mark.parametrize("order", [1, 2, [1, 0, 1]], ids=["AR1", "AR2", "AR(1,0,1)"])
def test_autoregressive_model(order, rng):
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
        assert_pattern_repeats(y, s, atol=ATOL, rtol=RTOL)


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
    assert_pattern_repeats(y, T, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("s,n", [(10, 5), pytest.param(10.2, 5, marks=pytest.mark.xfail)])
def test_state_removed_when_freq_seasonality_is_saturated(s, n):
    mod = st.FrequencySeasonality(season_length=s, n=n, name="test")
    init_params = mod._name_to_variable["test"]

    assert init_params.type.shape[0] == (n * 2 - 1)


cycle_test_vals = zip([None, None, 3, 5, 10], [False, True, True, False, False])


@pytest.mark.parametrize("innovations", [True, False], ids=[f"innov={x}" for x in [True, False]])
@pytest.mark.parametrize("dampen", [True, False], ids=[f"dampen={x}" for x in [True, False]])
@pytest.mark.parametrize(
    "cycle_length, estimate_cycle_length",
    list(cycle_test_vals),
    ids=[f"cycle_len={a}-est_cycle_len={b}" for a, b in cycle_test_vals],
)
def test_cycle_component(innovations, dampen, cycle_length, estimate_cycle_length, rng):
    if estimate_cycle_length and (cycle_length is not None):
        with pytest.raises(ValueError, match="Cannot specify cycle_length"):
            st.CycleComponent(
                name="cycle",
                cycle_length=cycle_length,
                estimate_cycle_length=estimate_cycle_length,
                dampen=dampen,
                innovations=innovations,
            )

    elif not estimate_cycle_length and (cycle_length is None):
        with pytest.raises(ValueError, match="Must specify cycle_length"):
            st.CycleComponent(
                name="cycle",
                cycle_length=cycle_length,
                estimate_cycle_length=estimate_cycle_length,
                dampen=dampen,
                innovations=innovations,
            )

    else:
        cycle = st.CycleComponent(
            name="cycle",
            cycle_length=cycle_length,
            estimate_cycle_length=estimate_cycle_length,
            dampen=dampen,
            innovations=innovations,
        )

        params = {"initial_cycle": rng.normal(size=(1,)).astype(floatX)}

        if estimate_cycle_length:
            params["cycle_cycle_length"] = np.array([7], dtype=floatX)
        cycle_len = params.get("cycle_cycle_length", cycle_length)
        fit_params = {"frequency.cycle": cycle_len, "sigma2.irregular": 0.0}
        if dampen:
            params["cycle_dampening_factor"] = np.array([0.95], dtype=floatX)
            fit_params["damping.cycle"] = 0.95
        if innovations:
            params["sigma_cycle"] = np.ones((1,), dtype=floatX)
            fit_params["sigma2.cycle"] = 1.0

        x, y = simulate_from_numpy_model(cycle, rng, params)

        if not innovations and not estimate_cycle_length:
            if dampen:
                # undo the dampening so it's all constant
                y = y / 0.95 ** np.arange(y.shape[0])
            assert_pattern_repeats(y, cycle_length, atol=ATOL, rtol=RTOL)


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
    all_params = ll_params.copy()
    all_params.update(se_params)

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


def test_adding_exogenous_component(rng):
    data = rng.normal(size=(100, 2)).astype(floatX)
    reg = st.RegressionComponent(state_names=["a", "b"], name="exog")
    ll = st.LevelTrendComponent(name="level")

    seasonal = st.FrequencySeasonality(name="annual", season_length=12, n=4)
    mod = reg + ll + seasonal

    assert mod.ssm["design"].eval({"data_exog": data}).shape == (100, 1, 2 + 2 + 8)
    assert_allclose(mod.ssm["design", 5, 0, :2].eval({"data_exog": data}), data[5])


def test_filter_scans_time_varying_design_matrix(rng):
    time_idx = pd.date_range(start="2000-01-01", freq="D", periods=100)
    data = pd.DataFrame(rng.normal(size=(100, 2)), columns=["a", "b"], index=time_idx)

    y = pd.DataFrame(rng.normal(size=(100, 1)), columns=["data"], index=time_idx)

    reg = st.RegressionComponent(state_names=["a", "b"], name="exog")
    mod = reg.build(verbose=False)

    with pm.Model(coords=mod.coords) as m:
        data_exog = pm.MutableData("data_exog", data.values)

        x0 = pm.Normal("x0", dims=["state"])
        P0 = pm.Deterministic("P0", pt.eye(mod.k_states), dims=["state", "state_aux"])
        beta_exog = pm.Normal("beta_exog", dims=["exog_state"])

        mod.build_statespace_graph(y)
        prior = pm.sample_prior_predictive(samples=10)

    prior_Z = prior.prior.Z.values
    assert prior_Z.shape == (1, 10, 100, 1, 2)
    assert_allclose(prior_Z[0, :, :, 0, :], data.values[None].repeat(10, axis=0))


@pytest.mark.skipif(floatX.endswith("32"), reason="Prior covariance not PSD at half-precision")
def test_extract_components_from_idata(rng):
    time_idx = pd.date_range(start="2000-01-01", freq="D", periods=100)
    data = pd.DataFrame(rng.normal(size=(100, 2)), columns=["a", "b"], index=time_idx)

    y = pd.DataFrame(rng.normal(size=(100, 1)), columns=["data"], index=time_idx)

    ll = st.LevelTrendComponent()
    season = st.FrequencySeasonality(name="seasonal", season_length=12, n=2, innovations=False)
    reg = st.RegressionComponent(state_names=["a", "b"], name="exog")
    me = st.MeasurementError("obs")
    mod = (ll + season + reg + me).build(verbose=False)

    with pm.Model(coords=mod.coords) as m:
        data_exog = pm.MutableData("data_exog", data.values)

        x0 = pm.Normal("x0", dims=["state"])
        P0 = pm.Deterministic("P0", pt.eye(mod.k_states), dims=["state", "state_aux"])
        beta_exog = pm.Normal("beta_exog", dims=["exog_state"])
        initial_trend = pm.Normal("initial_trend", dims=["trend_state"])
        sigma_trend = pm.Exponential("sigma_trend", 1, dims=["trend_shock"])
        seasonal_coefs = pm.Normal("seasonal", dims=["seasonal_initial_state"])
        sigma_obs = pm.Exponential("sigma_obs", 1)

        mod.build_statespace_graph(y)
        prior = pm.sample_prior_predictive(samples=10)

    filter_prior = mod.sample_conditional_prior(prior)
    comp_prior = mod.extract_components_from_idata(filter_prior)
    comp_states = comp_prior.filtered_prior.coords["state"].values
    expected_states = ["LevelTrend[level]", "LevelTrend[trend]", "seasonal", "exog[a]", "exog[b]"]
    missing = set(comp_states) - set(expected_states)

    assert len(missing) == 0, missing
