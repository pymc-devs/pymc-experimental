import numpy as np
import pymc as pm
import pytensor
import pytest
from numpy.testing import assert_allclose

from pymc_experimental.statespace.core.statespace import FILTER_FACTORY, PyMCStateSpace
from pymc_experimental.statespace.utils.constants import (
    ALL_STATE_DIM,
    FILTER_OUTPUT_NAMES,
    MATRIX_NAMES,
    OBS_STATE_DIM,
    SMOOTHER_OUTPUT_NAMES,
)
from pymc_experimental.tests.statespace.utilities.shared_fixtures import (  # pylint: disable=unused-import
    rng,
)
from pymc_experimental.tests.statespace.utilities.test_helpers import (
    fast_eval,
    load_nile_test_data,
    make_test_inputs,
)

floatX = pytensor.config.floatX
nile = load_nile_test_data()
ALL_SAMPLE_OUTPUTS = MATRIX_NAMES + FILTER_OUTPUT_NAMES + SMOOTHER_OUTPUT_NAMES


@pytest.fixture(scope="session")
def ss_mod():
    class StateSpace(PyMCStateSpace):
        @property
        def param_names(self):
            return ["rho", "zeta"]

        @property
        def state_names(self):
            return ["a", "b"]

        @property
        def observed_states(self):
            return ["a"]

        @property
        def shock_names(self):
            return ["a"]

        def update(self, theta, mode=None):
            self.ssm["transition", 0, :] = theta

    T = np.zeros((2, 2)).astype(floatX)
    T[1, 0] = 1.0
    Z = np.array([[1.0, 0.0]], dtype=floatX)
    R = np.array([[1.0], [0.0]], dtype=floatX)
    H = np.array([[0.1]], dtype=floatX)
    Q = np.array([[0.8]], dtype=floatX)
    P0 = np.eye(2, dtype=floatX) * 1e6

    ss_mod = StateSpace(
        k_endog=nile.shape[1], k_states=2, k_posdef=1, filter_type="standard", verbose=False
    )
    for X, name in zip(
        [T, Z, R, H, Q, P0],
        ["transition", "design", "selection", "obs_cov", "state_cov", "initial_state_cov"],
    ):
        ss_mod.ssm[name] = X

    return ss_mod


@pytest.fixture(scope="session")
def pymc_mod(ss_mod):
    with pm.Model(coords={ALL_STATE_DIM: ["a", "b"], OBS_STATE_DIM: ["a"]}) as pymc_mod:
        rho = pm.Beta("rho", 1, 1)
        zeta = pm.Deterministic("zeta", 1 - rho)
        ss_mod.build_statespace_graph(data=nile, include_smoother=True)

    return pymc_mod


@pytest.fixture(scope="session")
def idata(pymc_mod, rng):
    with pymc_mod:
        idata = pm.sample(draws=10, tune=0, chains=1, random_seed=rng)
        idata_prior = pm.sample_prior_predictive(samples=10, random_seed=rng)

    idata.extend(idata_prior)
    return idata


def test_invalid_filter_name_raises():
    msg = "The following are valid filter types: " + ", ".join(list(FILTER_FACTORY.keys()))
    with pytest.raises(NotImplementedError, match=msg):
        mod = PyMCStateSpace(k_endog=1, k_states=5, k_posdef=1, filter_type="invalid_filter")


def test_singleseriesfilter_raises_if_k_endog_gt_one():
    msg = 'Cannot use filter_type = "single" with multiple observed time series'
    with pytest.raises(ValueError, match=msg):
        mod = PyMCStateSpace(k_endog=10, k_states=5, k_posdef=1, filter_type="single")


def test_unpack_matrices(rng):
    p, m, r, n = 2, 5, 1, 10
    data, *inputs = make_test_inputs(p, m, r, n, rng, missing_data=0)
    mod = PyMCStateSpace(k_endog=p, k_states=m, k_posdef=r, filter_type="standard", verbose=False)

    outputs = mod.unpack_statespace()
    for x, y in zip(inputs, outputs):
        assert_allclose(np.zeros_like(x), fast_eval(y))


def test_param_names_raises_on_base_class():
    mod = PyMCStateSpace(k_endog=1, k_states=5, k_posdef=1, filter_type="standard", verbose=False)
    with pytest.raises(NotImplementedError):
        x = mod.param_names


def test_update_raises_on_base_class():
    mod = PyMCStateSpace(k_endog=1, k_states=5, k_posdef=1, filter_type="standard", verbose=False)
    theta = np.zeros(4)
    with pytest.raises(NotImplementedError):
        mod.update(theta)


def test_gather_pymc_variables(ss_mod):
    with pm.Model() as mod:
        rho = pm.Normal("rho")
        zeta = pm.Deterministic("zeta", 1 - rho)
        theta = ss_mod._gather_required_random_variables()

    assert_allclose(fast_eval(pm.math.stack([rho, zeta])), fast_eval(theta))


def test_gather_raises_if_variable_missing(ss_mod):
    with pm.Model() as mod:
        rho = pm.Normal("rho")
        msg = "The following required model parameters were not found in the PyMC model: zeta"
        with pytest.raises(ValueError, match=msg):
            theta = ss_mod._gather_required_random_variables()


def test_build_statespace_graph(pymc_mod):
    for name in [
        "filtered_state",
        "predicted_state",
        "predicted_covariance",
        "filtered_covariance",
    ]:
        assert name in [x.name for x in pymc_mod.deterministics]


def test_build_smoother_graph(ss_mod, pymc_mod):
    names = ["smoothed_state", "smoothed_covariance"]
    for name in names:
        assert name in [x.name for x in pymc_mod.deterministics]


@pytest.mark.parametrize("group", ["posterior", "prior"])
@pytest.mark.parametrize("matrix", ALL_SAMPLE_OUTPUTS)
def test_no_nans_in_sampling_output(group, matrix, idata):
    assert not np.any(np.isnan(idata[group][matrix].values))


@pytest.mark.parametrize("group", ["posterior", "prior"])
@pytest.mark.parametrize("kind", ["conditional", "unconditional"])
def test_sampling_methods(group, kind, ss_mod, idata, rng):
    f = getattr(ss_mod, f"sample_{kind}_{group}")
    test_idata = f(idata, random_seed=rng)

    if kind == "conditional":
        for output in ["filtered", "predicted", "smoothed"]:
            assert f"{output}_{group}" in test_idata
            assert not np.any(np.isnan(test_idata[f"{output}_{group}"].values))
    if kind == "unconditional":
        for output in ["latent", "observed"]:
            assert f"{group}_{output}" in test_idata
            assert not np.any(np.isnan(test_idata[f"{group}_{output}"].values))
