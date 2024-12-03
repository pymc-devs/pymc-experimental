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

from pymc_experimental.inference.find_map import find_MAP
from pymc_experimental.inference.laplace import fit_laplace, fit_mvn_to_MAP, laplace


@pytest.fixture(scope="session")
def rng():
    seed = sum(map(ord, "test_laplace"))
    return np.random.default_rng(seed)


@pytest.mark.filterwarnings(
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
        mu = pm.Uniform("mu", -10000, 10000)
        logsigma = pm.Uniform("logsigma", 1, 100)

        yobs = pm.Normal("y", mu=mu, sigma=pm.math.exp(logsigma), observed=y)
        vars = [mu, logsigma]

        idata = pmx.fit(
            method="laplace", optimize_method="trust-ncg", draws=draws, random_seed=173300, chains=1
        )

    assert idata.posterior["mu"].shape == (1, draws)
    assert idata.posterior["logsigma"].shape == (1, draws)
    assert idata.observed_data["y"].shape == (n,)
    assert idata.fit["mean_vector"].shape == (len(vars),)
    assert idata.fit["covariance_matrix"].shape == (len(vars), len(vars))

    bda_map = [y.mean(), np.log(y.std())]
    bda_cov = np.array([[y.var() / n, 0], [0, 1 / (2 * n)]])

    np.testing.assert_allclose(idata.fit["mean_vector"].values, bda_map)
    np.testing.assert_allclose(idata.fit["covariance_matrix"].values, bda_cov, atol=1e-4)


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
            optimize_method="BFGS",
            progressbar=True,
            use_jax_gradients=True,
            compile_kwargs={"mode": "JAX"},
            optimizer_kwargs=dict(maxiter=100_000, gtol=1e-100),
            random_seed=173300,
        )

    assert idata.fit["mean_vector"].shape == (len(vars),)
    assert idata.fit["covariance_matrix"].shape == (len(vars), len(vars))

    bda_map = [np.log(y.std()), y.mean()]
    bda_cov = np.array([[1 / (2 * n), 0], [0, y.var() / n]])

    np.testing.assert_allclose(idata.fit["mean_vector"].values, bda_map)
    np.testing.assert_allclose(idata.fit["covariance_matrix"].values, bda_cov, atol=1e-4)


@pytest.mark.parametrize(
    "transform_samples",
    [True, False],
    ids=["transformed", "untransformed"],
)
@pytest.mark.parametrize("mode", ["JAX", None], ids=["jax", "pytensor"])
def test_fit_laplace_coords(rng, transform_samples, mode):
    coords = {"city": ["A", "B", "C"], "obs_idx": np.arange(100)}
    with pm.Model(coords=coords) as model:
        mu = pm.Normal("mu", mu=3, sigma=0.5, dims=["city"])
        sigma = pm.Exponential("sigma", 1, dims=["city"])
        obs = pm.Normal(
            "obs",
            mu=mu,
            sigma=sigma,
            observed=rng.normal(loc=3, scale=1.5, size=(100, 3)),
            dims=["obs_idx", "city"],
        )

        optimized_point = find_MAP(
            method="trust-ncg",
            use_grad=True,
            use_hessp=True,
            progressbar=False,
            compile_kwargs=dict(mode=mode),
            use_jax_gradients=mode == "JAX",
        )

        for value in optimized_point.values():
            assert value.shape == (3,)

        mu, H_inv = fit_mvn_to_MAP(
            optimized_point=optimized_point,
            model=model,
            transform_samples=transform_samples,
        )

        idata = laplace(mu=mu, H_inv=H_inv, model=model, transform_samples=transform_samples)

    np.testing.assert_allclose(np.mean(idata.posterior.mu, axis=1), np.full((2, 3), 3), atol=0.5)
    np.testing.assert_allclose(
        np.mean(idata.posterior.sigma, axis=1), np.full((2, 3), 1.5), atol=0.3
    )

    suffix = "_log__" if transform_samples else ""
    assert idata.fit.rows.values.tolist() == [
        "mu[A]",
        "mu[B]",
        "mu[C]",
        f"sigma{suffix}[A]",
        f"sigma{suffix}[B]",
        f"sigma{suffix}[C]",
    ]


def test_fit_laplace_ragged_coords(rng):
    coords = {"city": ["A", "B", "C"], "feature": [0, 1], "obs_idx": np.arange(100)}
    with pm.Model(coords=coords) as ragged_dim_model:
        X = pm.Data("X", np.ones((100, 2)), dims=["obs_idx", "feature"])
        beta = pm.Normal(
            "beta", mu=[[-100.0, 100.0], [-100.0, 100.0], [-100.0, 100.0]], dims=["city", "feature"]
        )
        mu = pm.Deterministic(
            "mu", (X[:, None, :] * beta[None]).sum(axis=-1), dims=["obs_idx", "city"]
        )
        sigma = pm.Normal("sigma", mu=1.5, sigma=0.5, dims=["city"])

        obs = pm.Normal(
            "obs",
            mu=mu,
            sigma=sigma,
            observed=rng.normal(loc=3, scale=1.5, size=(100, 3)),
            dims=["obs_idx", "city"],
        )

        idata = fit_laplace(
            optimize_method="Newton-CG",
            progressbar=False,
            use_grad=True,
            use_hessp=True,
            use_jax_gradients=True,
            compile_kwargs={"mode": "JAX"},
        )

    assert idata["posterior"].beta.shape[-2:] == (3, 2)
    assert idata["posterior"].sigma.shape[-1:] == (3,)

    # Check that everything got unraveled correctly -- feature 0 should be strictly negative, feature 1
    # strictly positive
    assert (idata["posterior"].beta.sel(feature=0).to_numpy() < 0).all()
    assert (idata["posterior"].beta.sel(feature=1).to_numpy() > 0).all()


@pytest.mark.parametrize(
    "transform_samples",
    [True, False],
    ids=["transformed", "untransformed"],
)
def test_fit_laplace(transform_samples):
    with pm.Model() as simp_model:
        mu = pm.Normal("mu", mu=3, sigma=0.5)
        sigma = pm.Exponential("sigma", 1)
        obs = pm.Normal(
            "obs",
            mu=mu,
            sigma=sigma,
            observed=np.random.default_rng().normal(loc=3, scale=1.5, size=(10000,)),
        )

        idata = fit_laplace(
            optimize_method="trust-ncg",
            use_grad=True,
            use_hessp=True,
            transform_samples=transform_samples,
            optimizer_kwargs=dict(maxiter=100_000, tol=1e-100),
        )

        np.testing.assert_allclose(np.mean(idata.posterior.mu, axis=1), np.full((2,), 3), atol=0.1)
        np.testing.assert_allclose(
            np.mean(idata.posterior.sigma, axis=1), np.full((2,), 1.5), atol=0.1
        )

        if transform_samples:
            assert idata.fit.rows.values.tolist() == ["mu", "sigma_log__"]
            np.testing.assert_allclose(idata.fit.mean_vector.values, np.array([3.0, 0.4]), atol=0.1)
        else:
            assert idata.fit.rows.values.tolist() == ["mu", "sigma"]
            np.testing.assert_allclose(idata.fit.mean_vector.values, np.array([3.0, 1.5]), atol=0.1)
