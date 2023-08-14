from typing import List

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import statsmodels.api as sm
from numpy.testing import assert_allclose
from pymc import modelcontext

from pymc_experimental.statespace.filters.kalman_smoother import KalmanSmoother
from pymc_experimental.statespace.utils.constants import (
    MATRIX_NAMES,
    SHORT_NAME_TO_LONG,
)
from pymc_experimental.tests.statespace.utilities.statsmodel_local_level import (
    LocalLinearTrend,
)

floatX = pytensor.config.floatX


def load_nile_test_data():
    nile = pd.read_csv("pymc_experimental/tests/statespace/test_data/nile.csv", dtype={"x": floatX})
    nile.index = pd.date_range(start="1871-01-01", end="1970-01-01", freq="AS-Jan")
    nile.rename(columns={"x": "height"}, inplace=True)
    nile = (nile - nile.mean()) / nile.std()
    nile = nile.astype(floatX)

    return nile


def initialize_filter(kfilter, mode=None):
    ksmoother = KalmanSmoother()
    data = pt.matrix(name="data", dtype=floatX)
    a0 = pt.vector(name="a0", dtype=floatX)
    P0 = pt.matrix(name="P0", dtype=floatX)
    c = pt.vector(name="c", dtype=floatX)
    d = pt.vector(name="d", dtype=floatX)
    Q = pt.matrix(name="Q", dtype=floatX)
    H = pt.matrix(name="H", dtype=floatX)
    T = pt.matrix(name="T", dtype=floatX)
    R = pt.matrix(name="R", dtype=floatX)
    Z = pt.matrix(name="Z", dtype=floatX)

    inputs = [data, a0, P0, c, d, T, Z, R, H, Q]

    (
        filtered_states,
        predicted_states,
        observed_states,
        filtered_covs,
        predicted_covs,
        observed_covs,
        ll_obs,
    ) = kfilter.build_graph(*inputs, mode=mode)

    smoothed_states, smoothed_covs = ksmoother.build_graph(T, R, Q, filtered_states, filtered_covs)

    outputs = [
        filtered_states,
        predicted_states,
        smoothed_states,
        filtered_covs,
        predicted_covs,
        smoothed_covs,
        ll_obs.sum(),
        ll_obs,
    ]

    return inputs, outputs


def add_missing_data(data, n_missing, rng):
    n = data.shape[0]
    missing_idx = rng.choice(n, n_missing, replace=False)
    data[missing_idx] = np.nan

    return data


def make_test_inputs(p, m, r, n, rng, missing_data=None, H_is_zero=False):
    data = np.arange(n * p, dtype=floatX).reshape(-1, p)
    if missing_data is not None:
        data = add_missing_data(data, missing_data, rng)

    a0 = np.zeros(m, dtype=floatX)
    P0 = np.eye(m, dtype=floatX)
    c = np.zeros(m, dtype=floatX)
    d = np.zeros(p, dtype=floatX)
    Q = np.eye(r, dtype=floatX)
    H = np.zeros((p, p), dtype=floatX) if H_is_zero else np.eye(p, dtype=floatX)
    T = np.eye(m, k=-1, dtype=floatX)
    T[0, :] = 1 / m
    R = np.eye(m, dtype=floatX)[:, :r]
    Z = np.eye(m, dtype=floatX)[:p, :]

    data, a0, P0, c, d, T, Z, R, H, Q = map(
        np.ascontiguousarray, [data, a0, P0, c, d, T, Z, R, H, Q]
    )

    return data, a0, P0, c, d, T, Z, R, H, Q


def get_expected_shape(name, p, m, r, n):
    if name == "log_likelihood":
        return ()
    elif name == "ll_obs":
        return (n,)
    filter_type, variable = name.split("_")
    if variable == "states":
        return n, m
    if variable == "covs":
        return n, m, m


def get_sm_state_from_output_name(res, name):
    if name == "log_likelihood":
        return res.llf
    elif name == "ll_obs":
        return res.llf_obs

    filter_type, variable = name.split("_")
    sm_states = getattr(res, "states")

    if variable == "states":
        return getattr(sm_states, filter_type)
    if variable == "covs":
        m = res.filter_results.k_states
        # remove the "s" from "covs"
        return getattr(sm_states, name[:-1]).reshape(-1, m, m)


def nile_test_test_helper(rng, n_missing=0):
    a0 = np.zeros(2, dtype=floatX)
    P0 = np.eye(2, dtype=floatX) * 1e6
    c = np.zeros(2, dtype=floatX)
    d = np.zeros(1, dtype=floatX)
    Q = np.eye(2, dtype=floatX) * np.array([0.5, 0.01], dtype=floatX)
    H = np.eye(1, dtype=floatX) * 0.8
    T = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=floatX)
    R = np.eye(2, dtype=floatX)
    Z = np.array([[1.0, 0.0]], dtype=floatX)

    data = load_nile_test_data().values
    if n_missing > 0:
        data = add_missing_data(data, n_missing, rng)

    sm_model = LocalLinearTrend(
        endog=data,
        initialization="known",
        initial_state_cov=P0,
        initial_state=a0.ravel(),
    )

    res = sm_model.fit_constrained(
        constraints={
            "sigma2.measurement": 0.8,
            "sigma2.level": 0.5,
            "sigma2.trend": 0.01,
        }
    )

    inputs = [data, a0, P0, c, d, T, Z, R, H, Q]

    return res, inputs


def fast_eval(var):
    return pytensor.function([], var, mode="FAST_COMPILE")()


def delete_rvs_from_model(rv_names: List[str]) -> None:
    """Remove all model mappings referring to rv

    This can be used to "delete" an RV from a model
    """
    mod = modelcontext(None)
    all_rvs = mod.basic_RVs + mod.deterministics
    all_rv_names = [x.name for x in all_rvs]

    for name in rv_names:
        assert name in all_rv_names, f"{name} is not part of the Model: {all_rv_names}"

        rv_idx = all_rv_names.index(name)
        rv = all_rvs[rv_idx]

        mod.named_vars.pop(name)
        if name in mod.named_vars_to_dims:
            mod.named_vars_to_dims.pop(name)

        if rv in mod.deterministics:
            mod.deterministics.remove(rv)
            continue

        value = mod.rvs_to_values.pop(rv)
        mod.values_to_rvs.pop(value)
        mod.rvs_to_transforms.pop(rv)
        if rv in mod.free_RVs:
            mod.free_RVs.remove(rv)
            mod.rvs_to_initial_values.pop(rv)
        else:
            mod.observed_RVs.remove(rv)


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


def assert_pattern_repeats(y, T, atol, rtol):
    val = np.diff(y.reshape(-1, T), axis=0)
    if floatX.endswith("64"):
        # Round this before going into the test, otherwise it behaves poorly (atol = inf)
        n_digits = len(str(1 / atol))
        val = np.round(val, n_digits)

    assert_allclose(
        val,
        0,
        err_msg="seasonal pattern does not repeat",
        atol=atol,
        rtol=rtol,
    )


def make_stationary_params(data, p, d, q, P, D, Q, S):
    sm_sarimax = sm.tsa.SARIMAX(data, order=(p, d, q), seasonal_order=(P, D, Q, S))
    res = sm_sarimax.fit(disp=False)

    param_dict = dict(
        ar_params=[], ma_params=[], seasonal_ar_params=[], seasonal_ma_params=[], sigma_state=[]
    )

    for name, param in zip(res.param_names, res.params):
        if name.startswith("ar.S"):
            param_dict["seasonal_ar_params"].append(param)
        elif name.startswith("ma.S"):
            param_dict["seasonal_ma_params"].append(param)
        elif name.startswith("ar."):
            param_dict["ar_params"].append(param)
        elif name.startswith("ma."):
            param_dict["ma_params"].append(param)
        else:
            param_dict["sigma_state"].append(param)

    param_dict = {k: np.array(v, dtype=floatX) for k, v in param_dict.items() if len(v) > 0}
    return param_dict
