import numpy as np
import pytensor
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from pytensor.graph.basic import explicit_graph_inputs
from scipy import linalg
from statespace.utils.constants import LONG_MATRIX_NAMES

from pymc_experimental.statespace.models.ETS import BayesianETS
from tests.statespace.utilities.shared_fixtures import rng
from tests.statespace.utilities.test_helpers import load_nile_test_data


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


orders = (
    ("A", "N", "N"),
    ("A", "A", "N"),
    ("A", "Ad", "N"),
    ("A", "N", "A"),
    ("A", "A", "A"),
    ("A", "Ad", "A"),
)
order_names = (
    "Basic",
    "Trend",
    "Damped Trend",
    "Seasonal",
    "Trend and Seasonal",
    "Trend, Damped Trend, Seasonal",
)

order_expected_flags = (
    {"trend": False, "damped_trend": False, "seasonal": False},
    {"trend": True, "damped_trend": False, "seasonal": False},
    {"trend": True, "damped_trend": True, "seasonal": False},
    {"trend": False, "damped_trend": False, "seasonal": True},
    {"trend": True, "damped_trend": False, "seasonal": True},
    {"trend": True, "damped_trend": True, "seasonal": True},
)

order_params = (
    ["alpha", "initial_level"],
    ["alpha", "initial_level", "beta", "initial_trend"],
    ["alpha", "initial_level", "beta", "initial_trend", "phi"],
    ["alpha", "initial_level", "gamma", "initial_seasonal"],
    ["alpha", "initial_level", "beta", "initial_trend", "gamma", "initial_seasonal"],
    ["alpha", "initial_level", "beta", "initial_trend", "gamma", "initial_seasonal", "phi"],
)


@pytest.mark.parametrize(
    "order, expected_flags", zip(orders, order_expected_flags), ids=order_names
)
def test_order_flags(order, expected_flags):
    mod = BayesianETS(order=order, seasonal_periods=4)
    for key, value in expected_flags.items():
        assert getattr(mod, key) == value


@pytest.mark.parametrize("order, expected_params", zip(orders, order_params), ids=order_names)
def test_param_info(order: tuple[str, str, str], expected_params):
    mod = BayesianETS(order=order, seasonal_periods=4)

    all_expected_params = [*expected_params, "sigma_state", "P0"]
    assert all(param in mod.param_names for param in all_expected_params)
    assert all(param in all_expected_params for param in mod.param_names)
    assert all(
        mod.param_info[param]["dims"] is None
        for param in expected_params
        if "seasonal" not in param
    )


@pytest.mark.parametrize("order, expected_params", zip(orders, order_params), ids=order_names)
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
        "initial_level": 3.0,
        "initial_trend": 1.0,
        "initial_seasonal": np.ones(seasonal_periods),
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

    x0_val = np.zeros((expected_states,))
    x0_val[1] = test_values["initial_level"]

    if order[1] == "N":
        T_val = np.array([[0.0, 0.0], [0.0, 1.0]])
    else:
        x0_val[2] = test_values["initial_trend"]
        R_val[2] = test_values["beta"]
        T_val = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])

    if order[1] == "Ad":
        T_val[1:, -1] *= test_values["phi"]

    if order[2] == "A":
        x0_val[2 + int(order[1] != "N") :] = test_values["initial_seasonal"]
        R_val[2 + int(order[1] != "N")] = test_values["gamma"]
        S = np.eye(seasonal_periods, k=-1)
        S[0, -1] = 1.0
        Z_val[0, 2 + int(order[1] != "N")] = 1.0
    else:
        S = np.eye(0)

    T_val = linalg.block_diag(T_val, S)

    assert_allclose(x0, x0_val)
    assert_allclose(T, T_val)
    assert_allclose(R, R_val)
    assert_allclose(Z, Z_val)


@pytest.mark.parametrize("order, params", zip(orders, order_params), ids=order_names)
def test_statespace_matches_statsmodels(rng, order: tuple[str, str, str], params):
    seasonal_periods = rng.integers(3, 12)
    data = rng.normal(size=(100,))
    mod = BayesianETS(order=order, seasonal_periods=seasonal_periods, measurement_error=False)
    sm_mod = sm.tsa.statespace.ExponentialSmoothing(
        data,
        trend=mod.trend,
        damped_trend=mod.damped_trend,
        seasonal=seasonal_periods if mod.seasonal else None,
    )

    simplex_params = ["alpha", "beta", "gamma"]
    test_values = dict(zip(simplex_params, rng.dirichlet(alpha=np.ones(3))))
    test_values["phi"] = rng.beta(1, 1)

    test_values["initial_level"] = rng.normal()
    test_values["initial_trend"] = rng.normal()
    test_values["initial_seasonal"] = rng.normal(size=seasonal_periods)
    test_values["initial_state_cov"] = np.eye(mod.k_states)
    test_values["sigma_state"] = 1.0

    sm_test_values = test_values.copy()
    sm_test_values["smoothing_level"] = test_values["alpha"]
    sm_test_values["smoothing_trend"] = test_values["beta"]
    sm_test_values["smoothing_seasonal"] = test_values["gamma"]
    sm_test_values["damping_trend"] = test_values["phi"]
    sm_test_values["initial_seasonal"] = test_values["initial_seasonal"][0]
    for i in range(1, seasonal_periods):
        sm_test_values[f"initial_seasonal.L{i}"] = test_values["initial_seasonal"][i]

    x0 = np.r_[
        0, *[test_values[name] for name in ["initial_level", "initial_trend", "initial_seasonal"]]
    ]
    mask = [True, True, order[1] != "N", *(order[2] != "N",) * seasonal_periods]

    sm_mod.initialize_known(initial_state=x0[mask], initial_state_cov=np.eye(mod.k_states))
    sm_mod.fit_constrained({name: sm_test_values[name] for name in sm_mod.param_names})

    matrices = mod._unpack_statespace_with_placeholders()
    inputs = list(explicit_graph_inputs(matrices))
    input_names = [x.name for x in inputs]

    f_matrices = pytensor.function(inputs, matrices)
    test_values_subset = {name: test_values[name] for name in input_names}

    matrices = f_matrices(**test_values_subset)
    sm_matrices = [sm_mod.ssm[name] for name in LONG_MATRIX_NAMES[2:]]

    for matrix, sm_matrix, name in zip(matrices[2:], sm_matrices, LONG_MATRIX_NAMES[2:]):
        if name == "selection":
            # statsmodel selection matrix seems to be wrong? They set the first element of the selection matrix to
            # 1 - sum(alpha, beta, gamma), which doesn't match the equations presented in ffp3
            assert_allclose(matrix[1:], sm_matrix[1:], err_msg=f"{name} does not match")
            assert matrix[0] == 1.0
            assert sm_matrix[0] != 1.0
        else:
            assert_allclose(matrix, sm_matrix, err_msg=f"{name} does not match")
