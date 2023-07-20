import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt

from pymc_experimental.statespace.filters.kalman_smoother import KalmanSmoother
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


def initialize_filter(kfilter):
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
    ) = kfilter.build_graph(*inputs)

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


def add_missing_data(data, n_missing):
    n = data.shape[0]
    missing_idx = np.random.choice(n, n_missing, replace=False)
    data[missing_idx] = np.nan

    return data


def make_test_inputs(p, m, r, n, missing_data=None, H_is_zero=False):
    data = np.arange(n * p, dtype=floatX).reshape(-1, p)
    if missing_data is not None:
        data = add_missing_data(data, missing_data)

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
    if filter_type == "predicted":
        n += 1
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


def nile_test_test_helper(n_missing=0):
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
        data = add_missing_data(data, n_missing)

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
