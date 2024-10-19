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
import xarray as xr

import pymc_experimental as pmx

from pymc_experimental.inference.pathfinder import fit_pathfinder


def build_eight_schools_model():
    # Data of the Eight Schools Model
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
    model = build_eight_schools_model()

    with model:
        idata = pmx.fit(method="pathfinder", random_seed=41)

    assert idata.posterior["mu"].shape == (1, 1000)
    assert idata.posterior["tau"].shape == (1, 1000)
    assert idata.posterior["theta"].shape == (1, 1000, 8)
    # FIXME: pathfinder doesn't find a reasonable mean! Fix bug or choose model pathfinder can handle
    # np.testing.assert_allclose(idata.posterior["mu"].mean(), 5.0)
    np.testing.assert_allclose(idata.posterior["tau"].mean(), 4.15, atol=0.5)


def test_pathfinder_pmx_equivalence():
    model = build_eight_schools_model()
    with model:
        idata_pmx = pmx.fit(method="pathfinder", random_seed=41)
        idata_pmx = idata_pmx[-1]

    ntests = 2
    runs = dict()
    for k in range(ntests):
        runs[k] = {}
        (
            runs[k]["pathfinder_state"],
            runs[k]["pathfinder_info"],
            runs[k]["pathfinder_samples"],
            runs[k]["pathfinder_idata"],
        ) = fit_pathfinder(model=model, random_seed=41)

        runs[k]["finite_idx"] = (
            np.argwhere(np.isfinite(runs[k]["pathfinder_info"].path.elbo)).ravel()[-1] + 1
        )

    np.testing.assert_allclose(
        runs[0]["pathfinder_info"].path.elbo[: runs[0]["finite_idx"]],
        runs[1]["pathfinder_info"].path.elbo[: runs[1]["finite_idx"]],
    )

    np.testing.assert_allclose(
        runs[0]["pathfinder_info"].path.alpha,
        runs[1]["pathfinder_info"].path.alpha,
    )

    np.testing.assert_allclose(
        runs[0]["pathfinder_info"].path.beta,
        runs[1]["pathfinder_info"].path.beta,
    )

    np.testing.assert_allclose(
        runs[0]["pathfinder_info"].path.gamma,
        runs[1]["pathfinder_info"].path.gamma,
    )

    np.testing.assert_allclose(
        runs[0]["pathfinder_info"].path.position,
        runs[1]["pathfinder_info"].path.position,
    )

    np.testing.assert_allclose(
        runs[0]["pathfinder_info"].path.grad_position,
        runs[1]["pathfinder_info"].path.grad_position,
    )

    xr.testing.assert_allclose(
        idata_pmx.posterior,
        runs[0]["pathfinder_idata"].posterior,
    )

    xr.testing.assert_allclose(
        idata_pmx.posterior,
        runs[1]["pathfinder_idata"].posterior,
    )
