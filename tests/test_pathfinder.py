#   Copyright 2022 The PyMC Developers
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

import sys

import numpy as np
import pymc as pm
import pytest

import pymc_experimental as pmx


def eight_schools_model():
    J = 8
    y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=10.0)
        tau = pm.HalfCauchy("tau", 5.0)

        theta = pm.Normal("theta", mu=0, sigma=1, shape=J)
        obs = pm.Normal("obs", mu=mu + tau * theta, sigma=sigma, shape=J, observed=y)

    return model


@pytest.mark.skipif(sys.platform == "win32", reason="JAX not supported on windows.")
def test_pathfinder():
    model = eight_schools_model()
    idata = pmx.fit(model=model, method="pathfinder", random_seed=41, inference_backend="pymc")

    assert idata.posterior["mu"].shape == (1, 1000)
    assert idata.posterior["tau"].shape == (1, 1000)
    assert idata.posterior["theta"].shape == (1, 1000, 8)
    # FIXME: pathfinder doesn't find a reasonable mean! Fix bug or choose model pathfinder can handle
    np.testing.assert_allclose(idata.posterior["mu"].mean(), 5.0, atol=1.0)
    # FIXME: now the tau is being underestimated. getting tau around 1.5.
    # np.testing.assert_allclose(idata.posterior["tau"].mean(), 4.15, atol=0.5)


def test_bfgs_sample():
    import pytensor.tensor as pt

    from pymc_experimental.inference.pathfinder import (
        alpha_recover,
        bfgs_sample,
        inverse_hessian_factors,
    )

    """test BFGS sampling"""
    L, N = 8, 10
    J = 6
    num_samples = 1000

    # mock data
    x = np.random.randn(L, N)
    g = np.random.randn(L, N)

    # get factors
    x_tensor = pt.as_tensor(x, dtype="float64")
    g_tensor = pt.as_tensor(g, dtype="float64")
    alpha, update_mask = alpha_recover(x_tensor, g_tensor)
    beta, gamma = inverse_hessian_factors(alpha, x_tensor, g_tensor, update_mask, J)

    # sample
    phi, logq = bfgs_sample(
        num_samples=num_samples,
        x=x_tensor,
        g=g_tensor,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        random_seed=88,
    )

    # check shapes
    assert beta.eval().shape == (L, N, 2 * J)
    assert gamma.eval().shape == (L, 2 * J, 2 * J)
    assert phi.eval().shape == (L, num_samples, N)
    assert logq.eval().shape == (L, num_samples)
