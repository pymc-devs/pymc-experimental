import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import scipy

from pymc.testing import assert_support_point_is_expected
from pytensor.graph.basic import equal_computations

from pymc_experimental.distributions.multivariate.laplace import MvAsymmetricLaplace, MvLaplace


@pytest.mark.parametrize("dist", [MvLaplace, MvAsymmetricLaplace])
def test_params(dist):
    mu = pt.vector("mu")
    cov = pt.matrix("cov")
    chol = pt.matrix("chol")
    tau = pt.matrix("tau")

    rv = dist.dist(mu=mu, cov=cov)
    mu_param, cov_param = rv.owner.op.dist_params(rv.owner)
    assert mu_param is mu
    assert cov_param is cov

    # Default mu from shape of chol
    rv = dist.dist(chol=chol)
    mu_param, cov_param = rv.owner.op.dist_params(rv.owner)
    cov_expected = chol @ chol.T
    mu_expected, _ = pt.broadcast_arrays(0, cov_expected[..., -1])
    assert equal_computations([mu_param, cov_param], [mu_expected, cov_expected])

    # Broadcast mu to chol (upper triangular)
    rv = dist.dist(mu=mu[0], chol=chol, lower=False)
    mu_param, cov_param = rv.owner.op.dist_params(rv.owner)
    # It's a bit silly but we transpose twice when lower=False
    cov_expected = chol.T @ chol.T.T
    mu_expected, _ = pt.broadcast_arrays(mu[0], cov_expected[..., -1])
    assert equal_computations([mu_param, cov_param], [mu_expected, cov_expected])

    # Mu and tau
    rv = dist.dist(mu=mu, tau=tau)
    mu_param, cov_param = rv.owner.op.dist_params(rv.owner)
    assert equal_computations([mu_param, cov_param], [mu, pt.linalg.inv(tau)])


@pytest.mark.parametrize(
    "mu, cov, size, expected",
    [
        (np.arange(6).reshape(3, 2), np.eye(2), None, np.arange(1, 7).reshape(3, 2)),
        (
            np.arange(2),
            np.broadcast_to(np.eye(2), (3, 2, 2)),
            None,
            np.broadcast_to(np.arange(1, 3), (3, 2)),
        ),
        (0, np.eye(3), (4,), np.ones((4, 3))),
    ],
)
@pytest.mark.parametrize("dist", [MvLaplace, MvAsymmetricLaplace])
def test_support_point(dist, mu, cov, size, expected):
    with pm.Model() as model:
        dist("x", mu=mu, cov=cov, size=size)
    assert_support_point_is_expected(model, expected)


@pytest.mark.parametrize("dist", [MvLaplace, MvAsymmetricLaplace])
def test_mean(dist):
    mu = [np.pi, np.e]

    # Explicit size
    rv = dist.dist(mu=mu, chol=np.eye(2), size=(3,))
    mean = pm.distributions.moments.mean(rv)
    np.testing.assert_allclose(mean.eval(mode="FAST_COMPILE"), np.broadcast_to(mu, (3, 2)))

    # Implicit size from cov
    rv = dist.dist(mu=mu, cov=np.broadcast_to(np.eye(2), (4, 2, 2)))
    mean = pm.distributions.moments.mean(rv)
    np.testing.assert_allclose(mean.eval(mode="FAST_COMPILE"), np.broadcast_to(mu, (4, 2)))


@pytest.mark.parametrize("dist", [MvLaplace, MvAsymmetricLaplace])
def test_random(dist):
    mu = [-1, np.pi, 1]
    cov = [[1, 0.5, 0.25], [0.5, 2, 0.5], [0.25, 0.5, 3]]
    rv = dist.dist(mu=mu, cov=cov, size=10_000)

    samples = pm.draw(rv, random_seed=13)
    assert samples.shape == (10_000, 3)
    np.testing.assert_allclose(np.mean(samples, axis=0), mu, rtol=0.05)

    expected_cov = cov if dist is MvLaplace else cov + np.outer(mu, mu)
    np.testing.assert_allclose(np.cov(samples, rowvar=False), expected_cov, rtol=0.1)


def test_symmetric_matches_univariate_logp():
    # Test MvLaplace matches Univariate Laplace when there's a single entry
    mean = 1.0
    scale = 2.0
    # Variance of Laplace is 2 * scale ** 2
    rv = MvLaplace.dist(mu=[mean], cov=[[2 * scale**2]])
    ref_rv = pm.Laplace.dist(mu=mean, b=scale)

    test_val = np.random.normal(size=(3, 1))
    rv_logp = pm.logp(rv, test_val).eval()
    ref_logp = pm.logp(ref_rv, test_val).squeeze(-1).eval()
    np.testing.assert_allclose(rv_logp, ref_logp)


@pytest.mark.xfail(reason="Not sure about equivalence. Test fails")
def test_asymmetric_matches_univariate_logp():
    # Test MvAsymmetricLaplace matches Univariate AsymmetricLaplace when there's a single entry
    k = 2.0
    # From wikipedia: https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution
    mean = (1 - k**2) / k
    var = ((1 + k**4) / (k**2)) - mean**2
    print(mean, var)
    rv = MvAsymmetricLaplace.dist(mu=[mean], cov=[[var]])
    ref_rv = pm.AsymmetricLaplace.dist(kappa=k, mu=0, b=1)

    test_val = np.random.normal(size=(3, 1))
    rv_logp = pm.logp(rv, test_val).eval()
    ref_logp = pm.logp(ref_rv, test_val).squeeze(-1).eval()
    np.testing.assert_allclose(rv_logp, ref_logp)


def test_asymmetric_matches_symmetric_logp():
    # Test it matches with the symmetric case when mu = 0
    mu = np.zeros(2)
    tau = np.linalg.inv(np.array([[1, -0.5], [-0.5, 2]]))
    rv = MvAsymmetricLaplace.dist(mu=mu, tau=tau)
    ref_rv = MvLaplace.dist(mu=mu, tau=tau)

    rv_val = np.random.normal(size=(3, 2))
    logp_eval = pm.logp(rv, rv_val).eval()
    logp_expected = pm.logp(ref_rv, rv_val).eval()
    np.testing.assert_allclose(logp_eval, logp_expected)


def test_symmetric_logp():
    # Testing against simple bivariate cases described in:
    # https://en.wikipedia.org/wiki/Multivariate_Laplace_distribution#Probability_density_function

    # Zero mean, non-identity covariance case
    mu = np.zeros(2)
    s1 = 0.5
    s2 = 2.0
    r = -0.25
    cov = np.array(
        [
            [s1**2, r * s1 * s2],
            [r * s1 * s2, s2**2],
        ]
    )
    rv = MvLaplace.dist(mu=mu, cov=cov)
    rv_val = np.random.normal(size=(2,))
    logp_eval = pm.logp(rv, rv_val).eval()

    x1, x2 = rv_val
    logp_expected = np.log(
        (1 / (np.pi * s1 * s2 * np.sqrt(1 - r**2)))
        * scipy.special.kv(
            0,
            np.sqrt(
                (2 * ((x1**2 / s1**2) - (2 * r * x1 * x2 / (s1 * s2)) + (x2**2 / s2**2)))
                / (1 - r**2)
            ),
        )
    )
    np.testing.assert_allclose(
        logp_eval,
        logp_expected,
    )

    # Non zero mean, identity covariance case
    mu = np.array([1, 3])
    rv = MvLaplace.dist(mu=mu, cov=np.eye(2))
    rv_val = np.random.normal(size=(2,))
    logp_eval = pm.logp(rv, rv_val).eval()

    logp_expected = np.log(1 / np.pi * scipy.special.kv(0, np.sqrt(2 * np.sum((rv_val - mu) ** 2))))
    np.testing.assert_allclose(
        logp_eval,
        logp_expected,
    )


def test_asymmetric_logp():
    # Testing against trivial univariate case
    mu = 0.5
    cov = np.array([[1.0]])
    rv = MvAsymmetricLaplace.dist(mu=mu, cov=cov)
    rv_val = np.random.normal(size=(1,))
    logp_eval = pm.logp(rv, rv_val).eval()

    [x] = rv_val
    logp_expected = np.log(
        ((2 * np.exp(x * mu)) / np.sqrt(2 * np.pi))
        * np.power(x**2 / (2 + mu**2), 1 / 4)
        * scipy.special.kv(
            1 / 2,
            np.sqrt((2 + mu**2) * x**2),
        )
    )
    np.testing.assert_allclose(
        logp_eval,
        logp_expected,
    )
