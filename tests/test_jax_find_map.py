import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from pymc_experimental.inference.jax_find_map import (
    find_MAP,
    fit_laplace,
    make_jax_funcs_from_graph,
)

pytest.importorskip("jax")


@pytest.fixture(scope="session")
def rng():
    seed = sum(map(ord, "test_fit_map"))
    return np.random.default_rng(seed)


def test_jax_functions_from_graph():
    x = pt.tensor("x", shape=(2,))

    def compute_z(x):
        z1 = x[0] ** 2 + 2
        z2 = x[0] * x[1] + 3
        return z1, z2

    z = pt.stack(compute_z(x))
    f_z, f_grad, f_hess, f_hessp = make_jax_funcs_from_graph(
        z.sum(), use_grad=True, use_hess=True, use_hessp=True
    )

    x_val = np.array([1.0, 2.0])
    expected_z = sum(compute_z(x_val))

    z_jax = f_z(x_val)
    np.testing.assert_allclose(z_jax, expected_z)

    grad_val = np.array(f_grad(x_val))
    np.testing.assert_allclose(grad_val.squeeze(), np.array([2 * x_val[0] + x_val[1], x_val[0]]))

    hess_val = np.array(f_hess(x_val))
    np.testing.assert_allclose(hess_val.squeeze(), np.array([[2, 1], [1, 0]]))

    hessp_val = np.array(f_hessp(x_val, np.array([1.0, 0.0])))
    np.testing.assert_allclose(hessp_val.squeeze(), np.array([2, 1]))


@pytest.mark.parametrize(
    "method, use_grad, use_hess",
    [
        ("nelder-mead", False, False),
        ("powell", False, False),
        ("CG", True, False),
        ("BFGS", True, False),
        ("L-BFGS-B", True, False),
        ("TNC", True, False),
        ("SLSQP", True, False),
        ("dogleg", True, True),
        ("trust-ncg", True, True),
        ("trust-exact", True, True),
        ("trust-krylov", True, True),
        ("trust-constr", True, True),
    ],
)
def test_JAX_map(method, use_grad, use_hess, rng):
    extra_kwargs = {}
    if method == "dogleg":
        # HACK -- dogleg requires that the hessian of the objective function is PSD, so we have to pick a point
        # where this is true
        extra_kwargs = {"initvals": {"mu": 2, "sigma_log__": 1}}

    with pm.Model() as m:
        mu = pm.Normal("mu")
        sigma = pm.Exponential("sigma", 1)
        pm.Normal("y_hat", mu=mu, sigma=sigma, observed=rng.normal(loc=3, scale=1.5, size=100))

        optimized_point = find_MAP(
            method=method, **extra_kwargs, use_grad=use_grad, use_hess=use_hess, progressbar=False
        )
    mu_hat, log_sigma_hat = optimized_point["mu"], optimized_point["sigma_log__"]

    assert np.isclose(mu_hat, 3, atol=0.5)
    assert np.isclose(np.exp(log_sigma_hat), 1.5, atol=0.5)


@pytest.mark.parametrize(
    "transform_samples",
    [True, False],
    ids=["transformed", "untransformed"],
)
def test_fit_laplace_coords(rng, transform_samples):
    coords = {"city": ["A", "B", "C"], "obs_idx": np.arange(100)}
    with pm.Model(coords=coords) as model:
        mu = pm.Normal("mu", mu=3, sigma=0.5, dims=["city"])
        sigma = pm.Normal("sigma", mu=1.5, sigma=0.5, dims=["city"])
        obs = pm.Normal(
            "obs",
            mu=mu,
            sigma=sigma,
            observed=rng.normal(loc=3, scale=1.5, size=(100, 3)),
            dims=["obs_idx", "city"],
        )

        optimized_point = find_MAP(
            method="Newton-CG",
            use_grad=True,
            progressbar=False,
        )

    for value in optimized_point.values():
        assert value.shape == (3,)

    idata = fit_laplace(
        optimized_point,
        model,
        transform_samples=transform_samples,
        progressbar=False,
    )

    np.testing.assert_allclose(np.mean(idata.posterior.mu, axis=1), np.full((2, 3), 3), atol=0.3)
    np.testing.assert_allclose(
        np.mean(idata.posterior.sigma, axis=1), np.full((2, 3), 1.5), atol=0.3
    )


def test_fit_laplace_ragged_coords(rng):
    coords = {"city": ["A", "B", "C"], "feature": [0, 1], "obs_idx": np.arange(100)}
    with pm.Model(coords=coords) as ragged_dim_model:
        X = pm.Data("X", np.ones((100, 2)), dims=["obs_idx", "feature"])
        beta = pm.Normal(
            "beta", mu=[[-100.0, 100.0], [-100.0, 100.0], [-100.0, 100.0]], dims=["city", "feature"]
        )
        mu = pm.Deterministic(
            "mu", (X[:, None, :] * beta[None]).sum(axis=-1), dims=["obs_idx", "feature"]
        )
        sigma = pm.Normal("sigma", mu=1.5, sigma=0.5, dims=["city"])

        obs = pm.Normal(
            "obs",
            mu=mu,
            sigma=sigma,
            observed=rng.normal(loc=3, scale=1.5, size=(100, 3)),
            dims=["obs_idx", "city"],
        )

        optimized_point, _ = find_MAP(
            method="Newton-CG", use_grad=True, progressbar=False, return_raw=True
        )

    idata = fit_laplace(optimized_point, ragged_dim_model, progressbar=False)

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
        sigma = pm.Normal("sigma", mu=1.5, sigma=0.5)
        obs = pm.Normal(
            "obs",
            mu=mu,
            sigma=sigma,
            observed=np.random.default_rng().normal(loc=3, scale=1.5, size=(10000,)),
        )

        optimized_point = find_MAP(
            method="Newton-CG",
            use_grad=True,
            progressbar=False,
        )

    idata = fit_laplace(
        optimized_point, simp_model, transform_samples=transform_samples, progressbar=False
    )

    np.testing.assert_allclose(np.mean(idata.posterior.mu, axis=1), np.full((2,), 3), atol=0.1)
    np.testing.assert_allclose(np.mean(idata.posterior.sigma, axis=1), np.full((2,), 1.5), atol=0.1)
