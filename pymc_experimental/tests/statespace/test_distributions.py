import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose

from pymc_experimental.statespace import BayesianLocalLevel
from pymc_experimental.statespace.filters import (
    CholeskyFilter,
    SingleTimeseriesFilter,
    StandardFilter,
    SteadyStateFilter,
    UnivariateFilter,
)
from pymc_experimental.statespace.filters.distributions import SequenceMvNormal
from pymc_experimental.tests.statespace.utilities.test_helpers import (
    load_nile_test_data,
)

floatX = pytensor.config.floatX
ATOL = 1e-8 if floatX.endswith("64") else 1e-4

filter_names = [
    "StandardFilter",
    "CholeskyFilter",
    "UnivariateFilter",
    "SingleTimeSeriesFilter",
    "SteadyStateFilter",
]

filters = [
    StandardFilter,
    CholeskyFilter,
    UnivariateFilter,
    SingleTimeseriesFilter,
    SteadyStateFilter,
]


@pytest.fixture()
def data():
    return load_nile_test_data()


@pytest.fixture()
def pymc_model(data):
    with pm.Model() as mod:
        data = pm.ConstantData("data", data.values)
        x0 = pm.Normal("x0", mu=[900, 0], sigma=[100, 1], shape=2)
        P0_diag = pm.Exponential("P0_diag", 0.01, shape=2)
        P0 = pm.Deterministic("P0", pt.diag(P0_diag))
        sigma_state = pm.Exponential("sigma_state", 1)
        sigma_obs = pm.Exponential("sigma_obs", 1)

    return mod


@pytest.mark.parametrize("kfilter", filters, ids=filter_names)
def test_loglike_vectors_agree(kfilter, pymc_model):
    ss_mod = BayesianLocalLevel(verbose=False)
    with pymc_model:
        theta = ss_mod.gather_required_random_variables()
        ss_mod.update(theta)
        matrices = ss_mod.unpack_statespace(include_constants=True)

        filter_outputs = kfilter().build_graph(pymc_model["data"], *matrices)
        filter_mus, pred_mus, obs_mu, filter_covs, pred_covs, obs_cov, ll = filter_outputs

        obs = SequenceMvNormal(
            "obs", mus=obs_mu, covs=obs_cov, logp=ll, observed=pymc_model["data"]
        )

    test_ll = ll.eval()
    obs_ll = pm.logp(obs, pymc_model["data"]).eval()

    assert_allclose(test_ll, obs_ll, atol=ATOL)
