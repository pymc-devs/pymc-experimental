import pytest
import aesara
import scipy.stats
import scipy.stats.distributions as sp
from pymc_experimental.distributions import GeneralizedGamma
from pymc.tests.test_distributions import Rplus, Rplusbig, check_logp, check_logcdf

class TestMatchesScipy:
    
    def test_generalized_gamma_logp(self):
        check_logp(
            GeneralizedGamma,
            Rplus,
            {"alpha": Rplusbig, "p": Rplusbig, "lambd": Rplusbig},
            lambda value, alpha, p, lambd: sp.gengamma.logpdf(value, a=alpha / p, c=p, scale=lambd),
        )

    @pytest.mark.skipif(
        condition=(aesara.config.floatX == "float32"),
        reason="Fails on float32 due to numerical issues",
    )
    def test_generalized_gamma_logcdf(self):
        check_logcdf(
            GeneralizedGamma,
            Rplus,
            {"alpha": Rplusbig, "p": Rplusbig, "lambd": Rplusbig},
            lambda value, alpha, p, lambd: sp.gengamma.logcdf(value, a=alpha / p, c=p, scale=lambd),
        )