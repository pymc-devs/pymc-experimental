#   Copyright 2024 The PyMC Developers
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
def test_laplace():

    # Example originates from Bayesian Data Analyses, 3rd Edition
    # By Andrew Gelman, John Carlin, Hal Stern, David Dunson,
    # Aki Vehtari, and Donald Rubin.
    # See section. 4.1

    y = np.array([2642, 3503, 4358], dtype=np.float64)
    n = y.size
    draws = 100000

    with pm.Model() as m:
        logsigma = pm.Uniform("logsigma", 1, 100)
        mu = pm.Uniform("mu", -10000, 10000)
        yobs = pm.Normal("y", mu=mu, sigma=pm.math.exp(logsigma), observed=y)
        vars = [mu, logsigma]

    idata = pmx.fit(
        method="laplace",
        vars=vars,
        model=m,
        draws=draws,
        random_seed=173300,
    )

    assert idata.posterior["mu"].shape == (1, draws)
    assert idata.posterior["logsigma"].shape == (1, draws)
    assert idata.observed_data["y"].shape == (n,)
    assert idata.fit["mean_vector"].shape == (len(vars),)
    assert idata.fit["covariance_matrix"].shape == (len(vars), len(vars))

    bda_map = [y.mean(), np.log(y.std())]
    bda_cov = np.array([[y.var() / n, 0], [0, 1 / (2 * n)]])

    assert np.allclose(idata.fit["mean_vector"].values, bda_map)
    assert np.allclose(idata.fit["covariance_matrix"].values, bda_cov, atol=1e-4)


@pytest.mark.filterwarnings(
    "ignore:Model.model property is deprecated. Just use Model.:FutureWarning",
    "ignore:hessian will stop negating the output in a future version of PyMC.\n"
    + "To suppress this warning set `negate_output=False`:FutureWarning",
)
def test_laplace_only_fit():

    # Example originates from Bayesian Data Analyses, 3rd Edition
    # By Andrew Gelman, John Carlin, Hal Stern, David Dunson,
    # Aki Vehtari, and Donald Rubin.
    # See section. 4.1

    y = np.array([2642, 3503, 4358], dtype=np.float64)
    n = y.size

    with pm.Model() as m:
        logsigma = pm.Uniform("logsigma", 1, 100)
        mu = pm.Uniform("mu", -10000, 10000)
        yobs = pm.Normal("y", mu=mu, sigma=pm.math.exp(logsigma), observed=y)
        vars = [mu, logsigma]

    idata = pmx.fit(
        method="laplace",
        vars=vars,
        model=m,
        random_seed=173300,
    )

    assert idata.fit["mean_vector"].shape == (len(vars),)
    assert idata.fit["covariance_matrix"].shape == (len(vars), len(vars))

    bda_map = [y.mean(), np.log(y.std())]
    bda_cov = np.array([[y.var() / n, 0], [0, 1 / (2 * n)]])

    assert np.allclose(idata.fit["mean_vector"].values, bda_map)
    assert np.allclose(idata.fit["covariance_matrix"].values, bda_cov, atol=1e-4)
