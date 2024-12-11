import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from pymc_experimental.gp.pytensor_gp import GP, ExpQuad


def test_exp_quad():
    x = pt.arange(3)[:, None]
    ls = pt.ones(())
    cov = ExpQuad.build_covariance(x, ls).eval()
    expected_distance = np.array([[0.0, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]])

    np.testing.assert_allclose(cov, np.exp(-0.5 * expected_distance))


@pytest.fixture(scope="session")
def marginal_model():
    with pm.Model() as m:
        X = pm.Data("X", np.arange(3)[:, None])
        y = np.full(3, np.pi)
        ls = 1.0
        cov = ExpQuad(X, ls)
        gp = GP("gp", cov=cov)

        sigma = 1.0
        obs = pm.Normal("obs", mu=gp, sigma=sigma, observed=y)

    return m


def test_marginal_sigma_rewrites_to_white_noise_cov(marginal_model):
    obs = marginal_model["obs"]

    # TODO: Bring these checks back after we implement marginalization of the GP RV
    #
    # assert sum(isinstance(var.owner.op, pm.Normal.rv_type)
    #            for var in ancestors([obs])
    #            if var.owner is not None) == 1
    #
    f = pm.compile_pymc([], obs)
    #
    # assert not any(isinstance(node.op, pm.Normal.rv_type) for node in f.maker.fgraph.apply_nodes)

    draws = np.stack([f() for _ in range(10_000)])
    empirical_cov = np.cov(draws.T)

    expected_distance = np.array([[0.0, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]])

    np.testing.assert_allclose(
        empirical_cov, np.exp(-0.5 * expected_distance) + np.eye(3), atol=0.1, rtol=0.1
    )


def test_marginal_gp_logp(marginal_model):
    expected_logps = {"obs": -8.8778}
    point_logps = marginal_model.point_logps(round_vals=4)
    for v1, v2 in zip(point_logps.values(), expected_logps.values()):
        np.testing.assert_allclose(v1, v2, atol=1e-6)
