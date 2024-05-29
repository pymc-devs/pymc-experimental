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


import numpy as np
import pymc as pm
import pytest

import pymc_experimental as pmx


@pytest.mark.filterwarnings(
    "ignore:Model.model property is deprecated. Just use Model.:FutureWarning",
    "ignore:hessian will stop negating the output in a future version of PyMC.\n"
    + "To suppress this warning set `negate_output=False`:FutureWarning",
)
def test_quadratic():

    y = np.array([2642, 3503, 4358], dtype=np.float64)
    n = y.size
    draws = 100000

    with pm.Model() as m:
        logsigma = pm.Uniform("logsigma", 1, 100)
        mu = pm.Uniform("mu", -10000, 10000)
        yobs = pm.Normal("y", mu=mu, sigma=pm.math.exp(logsigma), observed=y)
        idata = pmx.fit(
            method="quadratic", vars=[mu, logsigma], model=m, draws=draws, random_seed=173300
        )

    assert idata.posterior["mu"].shape == (1, draws)
    assert idata.posterior["logsigma"].shape == (1, draws)
    assert idata.observed_data["y"].shape == (3,)

    bda_map = [y.mean(), np.log(y.std())]

    assert np.allclose(idata.posterior["mu"].mean(), bda_map[0], atol=1.5)
    assert np.allclose(idata.posterior["logsigma"].mean(), bda_map[1], atol=0.25)

    bda_cov = np.array([[y.var() / n, 0], [0, 1 / (2 * n)]])
    # Extract the samples for the parameters from the trace
    mu_samples = idata.posterior["mu"][0]
    logsigma_samples = idata.posterior["logsigma"][0]

    # Stack the samples into a single array
    samples = np.vstack((mu_samples, logsigma_samples)).T

    # Compute the covariance matrix of samples
    cov_matrix = np.cov(samples, rowvar=False)

    print(bda_cov)
    print(cov_matrix)
    # assert np.allclose(cov_matrix, bda_cov, atol=1e-2)
