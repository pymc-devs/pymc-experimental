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
import jax
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import scipy
from numpy import dtype
from xarray.core.utils import Frozen

from pymc_experimental.inference.smc.sampling import (
    arviz_from_particles,
    blackjax_particles_from_pymc_population,
    get_jaxified_loglikelihood,
    get_jaxified_logprior,
    sample_smc_blackjax,
)


def two_gaussians_model():
    n = 4
    mu1 = np.ones(n) * 0.5
    mu2 = -mu1

    stdev = 0.1
    sigma = np.power(stdev, 2) * np.eye(n)
    isigma = np.linalg.inv(sigma)
    dsigma = np.linalg.det(sigma)

    w1 = stdev
    w2 = 1 - stdev

    def two_gaussians(x):
        """
        Mixture of gaussians likelihood
        """
        log_like1 = (
            -0.5 * n * pt.log(2 * np.pi)
            - 0.5 * pt.log(dsigma)
            - 0.5 * (x - mu1).T.dot(isigma).dot(x - mu1)
        )
        log_like2 = (
            -0.5 * n * pt.log(2 * np.pi)
            - 0.5 * pt.log(dsigma)
            - 0.5 * (x - mu2).T.dot(isigma).dot(x - mu2)
        )
        return pt.log(w1 * pt.exp(log_like1) + w2 * pt.exp(log_like2))

    with pm.Model() as m:
        X = pm.Uniform("X", lower=-2, upper=2.0, shape=n)
        llk = pm.Potential("muh", two_gaussians(X))

    return m, mu1


def fast_model():
    with pm.Model() as m:
        x = pm.Normal("x", 0, 1)
        y = pm.Normal("y", x, 1, observed=0)
    return m


@pytest.mark.parametrize(
    "kernel, check_for_integration_steps, inner_kernel_params",
    [
        ("HMC", True, {"step_size": 0.1, "integration_steps": 11}),
        ("NUTS", False, {"step_size": 0.1}),
    ],
)
def test_sample_smc_blackjax(kernel, check_for_integration_steps, inner_kernel_params):
    """
    When running the two gaussians model
    with BlackJax SMC, we sample them correctly,
    the shape of a posterior variable is (1, particles, dimension)
    and the inference_data has the right attributes.

    """
    model, muref = two_gaussians_model()
    iterations_to_diagnose = 2
    n_particles = 1000
    with model:
        inference_data = sample_smc_blackjax(
            n_particles=n_particles,
            kernel=kernel,
            inner_kernel_params=inner_kernel_params,
            iterations_to_diagnose=iterations_to_diagnose,
        )

    x = inference_data.posterior["X"]

    assert x.to_numpy().shape == (1, n_particles, 4)
    mu1d = np.abs(x).mean(axis=0).mean(axis=0)
    np.testing.assert_allclose(muref, mu1d, rtol=0.0, atol=0.03)

    for attribute, value in [
        ("particles", n_particles),
        ("step_size", 0.1),
        ("num_mcmc_steps", 10),
        ("iterations_to_diagnose", iterations_to_diagnose),
        ("sampler", f"Blackjax SMC with {kernel} kernel"),
    ]:
        assert inference_data.posterior.attrs[attribute] == value

    for diagnostic in ["lambda_evolution", "log_likelihood_increments"]:
        assert inference_data.posterior.attrs[diagnostic].shape == (iterations_to_diagnose,)

    for diagnostic in ["ancestors_evolution", "weights_evolution"]:
        assert inference_data.posterior.attrs[diagnostic].shape == (
            iterations_to_diagnose,
            n_particles,
        )

    for attribute in ["running_time_seconds", "iterations"]:
        assert attribute in inference_data.posterior.attrs

    if check_for_integration_steps:
        assert inference_data.posterior.attrs["integration_steps"] == 11


def test_blackjax_particles_from_pymc_population_univariate():
    model = fast_model()
    population = {"x": np.array([2, 3, 4])}
    blackjax_particles = blackjax_particles_from_pymc_population(model, population)
    jax.tree.map(np.testing.assert_allclose, blackjax_particles, [np.array([[2], [3], [4]])])


def test_blackjax_particles_from_pymc_population_multivariate():
    with pm.Model() as model:
        x = pm.Normal("x", 0, 1)
        z = pm.Normal("z", 0, 1)
        y = pm.Normal("y", x + z, 1, observed=0)

    population = {"x": np.array([0.34614613, 1.09163261, -0.44526825]), "z": np.array([1, 2, 3])}
    blackjax_particles = blackjax_particles_from_pymc_population(model, population)
    jax.tree.map(
        np.testing.assert_allclose,
        blackjax_particles,
        [np.array([[0.34614613], [1.09163261], [-0.44526825]]), np.array([[1], [2], [3]])],
    )


def simple_multivariable_model():
    """
    A simple model that has a multivariate variable,
    a has more than one variable (multivariable)
    """
    with pm.Model() as model:
        x = pm.Normal("x", 0, 1, shape=2)
        z = pm.Normal("z", 0, 1)
        y = pm.Normal("y", z, 1, observed=0)
    return model


def test_blackjax_particles_from_pymc_population_multivariable():
    model = simple_multivariable_model()
    population = {"x": np.array([[2, 3], [5, 6], [7, 9]]), "z": np.array([11, 12, 13])}
    blackjax_particles = blackjax_particles_from_pymc_population(model, population)

    jax.tree.map(
        np.testing.assert_allclose,
        blackjax_particles,
        [np.array([[2, 3], [5, 6], [7, 9]]), np.array([[11], [12], [13]])],
    )


def test_arviz_from_particles():
    model = simple_multivariable_model()
    particles = [np.array([[2, 3], [5, 6], [7, 9]]), np.array([[11], [12], [13]])]
    with model:
        inference_data = arviz_from_particles(model, particles)

    assert inference_data.posterior.sizes == Frozen({"chain": 1, "draw": 3, "x_dim_0": 2})
    assert inference_data.posterior.data_vars.dtypes == Frozen(
        {"x": dtype("float64"), "z": dtype("float64")}
    )


def test_get_jaxified_logprior():
    """
    Given a model with a Normal prior
    for a RV, the jaxified logprior
    indeed calculates that number,
    and can be jax.vmap'ed
    """
    logprior = get_jaxified_logprior(fast_model())
    for point in [-0.5, 0.0, 0.5]:
        jax.tree.map(
            np.testing.assert_allclose,
            jax.vmap(logprior)([np.array([point])]),
            np.log(scipy.stats.norm(0, 1).pdf(point)),
        )


def test_get_jaxified_loglikelihood():
    """
    Given a model with a Normal Likelihood, a single observation
    0 and std=1, the only free parameter of that function is the mean.
    When computing the logliklikelihood
    Then the function can be jax.vmap'ed, and the calculation matches the likelihood.
    """
    loglikelihood = get_jaxified_loglikelihood(fast_model())
    for point in [-0.5, 0.0, 0.5]:
        jax.tree.map(
            np.testing.assert_allclose,
            jax.vmap(loglikelihood)([np.array([point])]),
            np.log(scipy.stats.norm(point, 1).pdf(0)),
        )
