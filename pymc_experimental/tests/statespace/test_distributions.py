import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose
from pymc.model_graph import fast_eval
from scipy.stats import multivariate_normal

from pymc_experimental.statespace import BayesianLocalLevel
from pymc_experimental.tests.statespace.utilities.test_helpers import (
    load_nile_test_data,
)

floatX = pytensor.config.floatX

# TODO: This needs to be VERY large for float32 to pass, is there a way to put scipy into float32 computation
# to get an apples-to-apples comparison?
ATOL = 1e-8 if floatX.endswith("64") else 0.1

filter_names = [
    "standard",
    "cholesky",
    "univariate",
    "single",
    "steady_state",
]


@pytest.fixture()
def data():
    return load_nile_test_data()


@pytest.fixture()
def pymc_model(data):
    with pm.Model() as mod:
        data = pm.ConstantData("data", data.values)
        x0 = pm.Normal("x0", mu=[900, 0], sigma=[100, 1])
        P0_diag = pm.Exponential("P0_diag", 0.01, shape=2)
        P0 = pm.Deterministic("P0", pt.diag(P0_diag))
        sigma_state = pm.Exponential("sigma_state", 2)
        sigma_obs = pm.Exponential("sigma_obs", 1)

    return mod


@pytest.mark.parametrize("kfilter", filter_names, ids=filter_names)
def test_loglike_vectors_agree(kfilter, pymc_model):
    ss_mod = BayesianLocalLevel(verbose=False, filter_type=kfilter)
    with pymc_model:
        theta = ss_mod.gather_required_random_variables()
        ss_mod.update(theta)
        matrices = ss_mod.unpack_statespace(include_constants=True)

        filter_outputs = ss_mod.kalman_filter.build_graph(pymc_model["data"], *matrices)
        filter_mus, pred_mus, obs_mu, filter_covs, pred_covs, obs_cov, ll = filter_outputs

    test_ll = fast_eval(ll)

    # TODO: BUG: Why does fast eval end up with a 2d output when filter is "single"?
    obs_mu_np = obs_mu.eval()
    obs_cov_np = fast_eval(obs_cov)
    data_np = fast_eval(pymc_model["data"])

    scipy_lls = []
    for y, mu, cov in zip(data_np, obs_mu_np, obs_cov_np):
        scipy_lls.append(multivariate_normal.logpdf(y, mean=mu, cov=cov))
    assert_allclose(test_ll, np.array(scipy_lls).ravel(), atol=ATOL)
