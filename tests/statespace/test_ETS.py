import numpy as np
import pytensor
import pytest
from numpy.testing import assert_allclose
from pytensor.graph.basic import explicit_graph_inputs
from scipy import linalg

from pymc_experimental.statespace.models.ETS import BayesianETS
from tests.statespace.utilities.test_helpers import (
    load_nile_test_data,
    simulate_from_numpy_model,
)


@pytest.fixture(scope="session")
def data():
    return load_nile_test_data()


def tests_invalid_order_raises():
    # Order must be length 3
    with pytest.raises(ValueError, match="Order must be a tuple of three strings"):
        BayesianETS(order=("A", "N"))  # noqa

        # Order must be strings
    with pytest.raises(ValueError, match="Order must be a tuple of three strings"):
        BayesianETS(order=(2, 1, 1))  # noqa

    # Only additive errors allowed
    with pytest.raises(ValueError, match="Only additive errors are supported"):
        BayesianETS(order=("M", "N", "N"))

    # Trend must be A or Ad
    with pytest.raises(ValueError, match="Invalid trend specification"):
        BayesianETS(order=("A", "P", "N"))

    # Seasonal must be A or N
    with pytest.raises(ValueError, match="Invalid seasonal specification"):
        BayesianETS(order=("A", "Ad", "M"))

    # seasonal_periods must be provided if seasonal is requested
    with pytest.raises(ValueError, match="If seasonal is True, seasonal_periods must be provided."):
        BayesianETS(order=("A", "Ad", "A"))


@pytest.mark.parametrize(
    "order, expected_flags",
    [
        (("A", "N", "N"), {"trend": False, "damped_trend": False, "seasonal": False}),
        (("A", "A", "N"), {"trend": True, "damped_trend": False, "seasonal": False}),
        (("A", "Ad", "N"), {"trend": True, "damped_trend": True, "seasonal": False}),
        (("A", "N", "A"), {"trend": False, "damped_trend": False, "seasonal": True}),
        (("A", "A", "A"), {"trend": True, "damped_trend": False, "seasonal": True}),
        (("A", "Ad", "A"), {"trend": True, "damped_trend": True, "seasonal": True}),
    ],
    ids=[
        "Basic",
        "Trend",
        "Damped Trend",
        "Seasonal",
        "Trend and Seasonal",
        "Trend, Damped Trend, Seasonal",
    ],
)
def test_order_flags(order, expected_flags):
    mod = BayesianETS(order=order, seasonal_periods=4)
    for key, value in expected_flags.items():
        assert getattr(mod, key) == value


@pytest.mark.parametrize(
    "order, expected_params",
    [
        (("A", "N", "N"), ["alpha"]),
        (("A", "A", "N"), ["alpha", "beta"]),
        (("A", "Ad", "N"), ["alpha", "beta", "phi"]),
        (("A", "N", "A"), ["alpha", "gamma"]),
        (("A", "A", "A"), ["alpha", "beta", "gamma"]),
        (("A", "Ad", "A"), ["alpha", "beta", "gamma", "phi"]),
    ],
    ids=[
        "Basic",
        "Trend",
        "Damped Trend",
        "Seasonal",
        "Trend and Seasonal",
        "Trend, Damped Trend, Seasonal",
    ],
)
def test_param_info(order: tuple[str, str, str], expected_params):
    mod = BayesianETS(order=order, seasonal_periods=4)

    all_expected_params = [*expected_params, "sigma_state", "x0", "P0"]
    assert all(param in mod.param_names for param in all_expected_params)
    assert all(param in all_expected_params for param in mod.param_names)
    assert all(mod.param_info[param]["dims"] is None for param in expected_params)


@pytest.mark.parametrize(
    "order, expected_params",
    [
        (("A", "N", "N"), ["alpha"]),
        (("A", "A", "N"), ["alpha", "beta"]),
        (("A", "Ad", "N"), ["alpha", "beta", "phi"]),
        (("A", "N", "A"), ["alpha", "gamma"]),
        (("A", "A", "A"), ["alpha", "beta", "gamma"]),
        (("A", "Ad", "A"), ["alpha", "beta", "gamma", "phi"]),
    ],
    ids=[
        "Basic",
        "Trend",
        "Damped Trend",
        "Seasonal",
        "Trend and Seasonal",
        "Trend, Damped Trend, Seasonal",
    ],
)
def test_statespace_matrices(order: tuple[str, str, str], expected_params: list[str]):
    seasonal_periods = np.random.randint(3, 12)
    mod = BayesianETS(order=order, seasonal_periods=seasonal_periods, measurement_error=True)
    expected_states = 2 + int(order[1] != "N") + int(order[2] != "N") * seasonal_periods

    test_values = {
        "alpha": 0.7,
        "beta": 0.15,
        "gamma": 0.15,
        "phi": 0.95,
        "sigma_state": 0.1,
        "sigma_obs": 0.1,
        "x0": np.zeros(expected_states),
        "initial_state_cov": np.eye(expected_states),
    }

    matrices = x0, P0, c, d, T, Z, R, H, Q = mod._unpack_statespace_with_placeholders()

    assert x0.type.shape == (expected_states,)
    assert P0.type.shape == (expected_states, expected_states)
    assert c.type.shape == (expected_states,)
    assert d.type.shape == (1,)
    assert T.type.shape == (expected_states, expected_states)
    assert Z.type.shape == (1, expected_states)
    assert R.type.shape == (expected_states, 1)
    assert H.type.shape == (1, 1)
    assert Q.type.shape == (1, 1)

    inputs = list(explicit_graph_inputs(matrices))
    input_names = [x.name for x in inputs]
    assert all(name in input_names for name in expected_params)

    f_matrices = pytensor.function(inputs, matrices)
    [x0, P0, c, d, T, Z, R, H, Q] = f_matrices(**{name: test_values[name] for name in input_names})

    assert_allclose(H, np.eye(1) * test_values["sigma_obs"] ** 2)
    assert_allclose(Q, np.eye(1) * test_values["sigma_state"] ** 2)

    R_val = np.zeros((expected_states, 1))
    R_val[0] = 1.0
    R_val[1] = test_values["alpha"]

    Z_val = np.zeros((1, expected_states))
    Z_val[0, 0] = 1.0
    Z_val[0, 1] = 1.0

    if order[1] == "N":
        T_val = np.array([[0.0, 0.0], [0.0, 1.0]])
    else:
        R_val[2] = test_values["beta"]
        T_val = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
        Z_val[0, 2] = 1.0

    if order[1] == "Ad":
        T_val[1:, -1] *= test_values["phi"]

    if order[2] == "A":
        R_val[3] = test_values["gamma"]
        S = np.eye(seasonal_periods, k=-1)
        S[0, :] = -1
        Z_val[0, 2 + int(order[1] != "N")] = 1.0
    else:
        S = np.eye(0)

    T_val = linalg.block_diag(T_val, S)

    assert_allclose(T, T_val)
    assert_allclose(R, R_val)
    assert_allclose(Z, Z_val)


def test_deterministic_simulation_matches_statsmodels():
    mod = BayesianETS(order=("A", "Ad", "A"), seasonal_periods=4, measurement_error=False)

    rng = np.random.default_rng()
    test_values = {
        "alpha": 0.7,
        "beta": 0.15,
        "gamma": 0.15,
        "phi": 0.95,
        "sigma_state": 0.0,
        "x0": rng.normal(size=(7,)),
        "initial_state_cov": np.eye(7),
    }
    hidden_states, observed = simulate_from_numpy_model(mod, rng, test_values)
