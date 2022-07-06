#   Copyright 2020 The PyMC Developers
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

# general imports
import scipy.stats.distributions as ssd

# test support imports from pymc
from pymc.tests.test_distributions import (
    R,
    Rplus,
    Domain,
    TestMatchesScipy,
)

# the distributions to be tested
from pymc_experimental.distributions import (
    GenExtreme,
)


class TestMatchesScipyX(TestMatchesScipy):
    """
    Wrapper class so that tests of experimental additions can be dropped into
    PyMC directly on adoption.
    """

    def test_genextreme(self):
        self.check_logp(
            GenExtreme,
            R,
            {"mu": R, "sigma": Rplus, "xi": Domain([-1, -1, -0.5, 0, 0.5, 1, 1])},
            lambda value, mu, sigma, xi: ssd.genextreme.logpdf(value, c=-xi, loc=mu, scale=sigma),
        )
        self.check_logcdf(
            GenExtreme,
            R,
            {"mu": R, "sigma": Rplus, "xi": Domain([-1, -1, -0.5, 0, 0.5, 1, 1])},
            lambda value, mu, sigma, xi: ssd.genextreme.logcdf(value, c=-xi, loc=mu, scale=sigma),
        )
