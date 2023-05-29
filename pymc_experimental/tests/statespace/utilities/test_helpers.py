import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt

from pymc_experimental.statespace.filters.kalman_smoother import KalmanSmoother
from pymc_experimental.tests.statespace.utilities.statsmodel_local_level import (
    LocalLinearTrend,
)

ROOT = Path(__file__).parent.parent.absolute()
nile_data = pd.read_csv(os.path.join(ROOT, "test_data/nile.csv"))
nile_data["x"] = nile_data["x"].astype(pytensor.config.floatX)


def initialize_filter(kfilter):
    ksmoother = KalmanSmoother()
    data = pt.tensor(name="data", dtype=pytensor.config.floatX, shape=(None, None, 1))
    a0 = pt.matrix(name="a0", dtype=pytensor.config.floatX)
    P0 = pt.matrix(name="P0", dtype=pytensor.config.floatX)
    Q = pt.matrix(name="Q", dtype=pytensor.config.floatX)
    H = pt.matrix(name="H", dtype=pytensor.config.floatX)
    T = pt.matrix(name="T", dtype=pytensor.config.floatX)
    R = pt.matrix(name="R", dtype=pytensor.config.floatX)
    Z = pt.matrix(name="Z", dtype=pytensor.config.floatX)

    inputs = [data, a0, P0, T, Z, R, H, Q]

    (
        filtered_states,
        predicted_states,
        filtered_covs,
        predicted_covs,
        log_likelihood,
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
        log_likelihood,
        ll_obs,
    ]

    return inputs, outputs


def add_missing_data(data, n_missing):
    n = data.shape[0]
    missing_idx = np.random.choice(n, n_missing, replace=False)
    data[missing_idx] = np.nan

    return data


def make_test_inputs(p, m, r, n, missing_data=None, H_is_zero=False):
    data = np.arange(n * p, dtype=pytensor.config.floatX).reshape(-1, p, 1)
    if missing_data is not None:
        data = add_missing_data(data, missing_data)

    a0 = np.zeros((m, 1), dtype=pytensor.config.floatX)
    P0 = np.eye(m, dtype=pytensor.config.floatX)
    Q = np.eye(r, dtype=pytensor.config.floatX)
    H = (
        np.zeros((p, p), dtype=pytensor.config.floatX)
        if H_is_zero
        else np.eye(p, dtype=pytensor.config.floatX)
    )
    T = np.eye(m, k=-1, dtype=pytensor.config.floatX)
    T[0, :] = 1 / m
    R = np.eye(m, dtype=pytensor.config.floatX)[:, :r]
    Z = np.eye(m, dtype=pytensor.config.floatX)[:p, :]

    return data, a0, P0, T, Z, R, H, Q


def get_expected_shape(name, p, m, r, n):
    if name == "log_likelihood":
        return ()
    elif name == "ll_obs":
        return (n,)
    filter_type, variable = name.split("_")
    if filter_type == "predicted":
        n += 1
    if variable == "states":
        return n, m, 1
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
    a0 = np.zeros((2, 1), dtype=pytensor.config.floatX)
    P0 = np.eye(2, dtype=pytensor.config.floatX) * 1e6
    Q = np.eye(2, dtype=pytensor.config.floatX) * np.array(
        [0.5, 0.01], dtype=pytensor.config.floatX
    )
    H = np.eye(1, dtype=pytensor.config.floatX) * 0.8
    T = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=pytensor.config.floatX)
    R = np.eye(2, dtype=pytensor.config.floatX)
    Z = np.array([[1.0, 0.0]], dtype=pytensor.config.floatX)

    data = nile_data.values.copy().astype(pytensor.config.floatX)
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

    inputs = [data[..., None], a0, P0, T, Z, R, H, Q]

    return res, inputs
