import numpy as np
import pytest

from pymc.distributions import Beta, Binomial, HalfNormal, InverseGamma, Normal
from pymc.model.core import Model
from pymc.sampling.mcmc import sample
from pymc.step_methods import Slice

from pymc_experimental import opt_sample


def test_custom_step_raises():
    with Model() as m:
        a = InverseGamma("a", 1, 1)
        b = InverseGamma("b", 1, 1)
        p = Beta("p", a, b)
        y = Binomial("y", n=100, p=p, observed=99)

        with pytest.raises(
            ValueError, match="The `step` argument is not supported in `opt_sample`"
        ):
            opt_sample(step=Slice([a, b]))


def test_sample_opt_summary_stats(capsys):
    rng = np.random.default_rng(3)
    y_data = rng.normal(loc=1, scale=0.5, size=(1000,))

    with Model() as m:
        mu = Normal("mu")
        sigma = HalfNormal("sigma")
        y = Normal("y", mu=mu, sigma=sigma, observed=y_data)

        sample_kwargs = dict(
            chains=1, tune=500, draws=500, compute_convergence_checks=False, progressbar=False
        )
        idata = sample(**sample_kwargs)
        # TODO: Make extract_data more robust to avoid this warning/error
        #  Or alternatively extract data on the original model, not the optimized one
        with pytest.warns(UserWarning, match="Could not extract data from symbolic observation"):
            opt_idata = opt_sample(**sample_kwargs, verbose=True)

    captured_out = capsys.readouterr().out
    assert "Applied optimization: summary_stats_normal 1x" in captured_out

    assert opt_idata.posterior.sizes["chain"] == 1
    assert opt_idata.posterior.sizes["draw"] == 500
    np.testing.assert_allclose(
        idata.posterior["mu"].mean(), opt_idata.posterior["mu"].mean(), rtol=1e-2
    )
    np.testing.assert_allclose(
        idata.posterior["sigma"].mean(), opt_idata.posterior["sigma"].mean(), rtol=1e-2
    )
    assert idata.sample_stats.sampling_time > opt_idata.sample_stats.sampling_time
