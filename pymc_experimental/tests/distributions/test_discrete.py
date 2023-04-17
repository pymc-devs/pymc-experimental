#   Copyright 2023 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import scipy.stats
from pymc.logprob.utils import ParameterValueError
from pymc.testing import (
    BaseTestDistributionRandom,
    Domain,
    Rplus,
    assert_moment_is_expected,
    discrete_random_tester,
)
from pytensor import config

from pymc_experimental.distributions import GeneralizedPoisson


class TestGeneralizedPoisson:
    class TestRandomVariable(BaseTestDistributionRandom):
        pymc_dist = GeneralizedPoisson
        pymc_dist_params = {"mu": 4.0, "lam": 1.0}
        expected_rv_op_params = {"mu": 4.0, "lam": 1.0}
        tests_to_run = [
            "check_pymc_params_match_rv_op",
            "check_rv_size",
        ]

        def test_random_matches_poisson(self):
            discrete_random_tester(
                dist=self.pymc_dist,
                paramdomains={"mu": Rplus, "lam": Domain([0], edges=(None, None))},
                ref_rand=lambda mu, lam, size: scipy.stats.poisson.rvs(mu, size=size),
            )

        @pytest.mark.parametrize("mu", (2.5, 20, 50))
        def test_random_lam_expected_moments(self, mu):
            lam = np.array([-0.9, -0.7, -0.2, 0, 0.2, 0.7, 0.9])
            dist = self.pymc_dist.dist(mu=mu, lam=lam, size=(10_000, len(lam)))
            draws = dist.eval()

            expected_mean = mu / (1 - lam)
            np.testing.assert_allclose(draws.mean(0), expected_mean, rtol=1e-1)

            expected_std = np.sqrt(mu / (1 - lam) ** 3)
            np.testing.assert_allclose(draws.std(0), expected_std, rtol=1e-1)

    def test_logp_matches_poisson(self):
        # We are only checking this distribution for lambda=0 where it's equivalent to Poisson.
        mu = pt.scalar("mu")
        lam = pt.scalar("lam")
        value = pt.vector("value", dtype="int64")

        logp = pm.logp(GeneralizedPoisson.dist(mu, lam), value)
        logp_fn = pytensor.function([value, mu, lam], logp)

        test_value = np.array([0, 1, 2, 30])
        for test_mu in (0.01, 0.1, 0.9, 1, 1.5, 20, 100):
            np.testing.assert_allclose(
                logp_fn(test_value, test_mu, lam=0),
                scipy.stats.poisson.logpmf(test_value, test_mu),
                rtol=1e-7 if config.floatX == "float64" else 1e-5,
            )

        # Check out-of-bounds values
        value = pt.scalar("value")
        logp = pm.logp(GeneralizedPoisson.dist(mu, lam), value)
        logp_fn = pytensor.function([value, mu, lam], logp)

        logp_fn(-1, mu=5, lam=0) == -np.inf
        logp_fn(9, mu=5, lam=-1) == -np.inf

        # Check mu/lam restrictions
        with pytest.raises(ParameterValueError):
            logp_fn(1, mu=1, lam=2)

        with pytest.raises(ParameterValueError):
            logp_fn(1, mu=0, lam=0)

        with pytest.raises(ParameterValueError):
            logp_fn(1, mu=1, lam=-1)

    def test_logp_lam_expected_moments(self):
        mu = 30
        lam = np.array([-0.9, -0.7, -0.2, 0, 0.2, 0.7, 0.9])
        with pm.Model():
            x = GeneralizedPoisson("x", mu=mu, lam=lam)
            trace = pm.sample(chains=1, draws=10_000, random_seed=96).posterior

        expected_mean = mu / (1 - lam)
        np.testing.assert_allclose(trace["x"].mean(("chain", "draw")), expected_mean, rtol=1e-1)

        expected_std = np.sqrt(mu / (1 - lam) ** 3)
        np.testing.assert_allclose(trace["x"].std(("chain", "draw")), expected_std, rtol=1e-1)

    @pytest.mark.parametrize(
        "mu, lam, size, expected",
        [
            (50, [-0.6, 0, 0.6], None, np.floor(50 / (1 - np.array([-0.6, 0, 0.6])))),
            ([5, 50], -0.1, (4, 2), np.full((4, 2), np.floor(np.array([5, 50]) / 1.1))),
        ],
    )
    def test_moment(self, mu, lam, size, expected):
        with pm.Model() as model:
            GeneralizedPoisson("x", mu=mu, lam=lam, size=size)
        assert_moment_is_expected(model, expected)
