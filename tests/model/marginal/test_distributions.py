import numpy as np
import pymc as pm
import pytest

from pymc.logprob.abstract import _logprob
from pytensor import tensor as pt
from scipy.stats import norm

from pymc_experimental import MarginalModel
from pymc_experimental.distributions import DiscreteMarkovChain
from pymc_experimental.model.marginal.distributions import FiniteDiscreteMarginalRV


def test_marginalized_bernoulli_logp():
    """Test logp of IR TestFiniteMarginalDiscreteRV directly"""
    mu = pt.vector("mu")

    idx = pm.Bernoulli.dist(0.7, name="idx")
    y = pm.Normal.dist(mu=mu[idx], sigma=1.0, name="y")
    marginal_rv_node = FiniteDiscreteMarginalRV(
        [mu],
        [idx, y],
        ndim_supp=0,
        n_updates=0,
        # Ignore the fact we didn't specify shared RNG input/outputs for idx,y
        strict=False,
    )(mu)[0].owner

    y_vv = y.clone()
    (logp,) = _logprob(
        marginal_rv_node.op,
        (y_vv,),
        *marginal_rv_node.inputs,
    )

    ref_logp = pm.logp(pm.NormalMixture.dist(w=[0.3, 0.7], mu=mu, sigma=1.0), y_vv)
    np.testing.assert_almost_equal(
        logp.eval({mu: [-1, 1], y_vv: 2}),
        ref_logp.eval({mu: [-1, 1], y_vv: 2}),
    )


@pytest.mark.parametrize("batch_chain", (False, True), ids=lambda x: f"batch_chain={x}")
@pytest.mark.parametrize("batch_emission", (False, True), ids=lambda x: f"batch_emission={x}")
def test_marginalized_hmm_normal_emission(batch_chain, batch_emission):
    if batch_chain and not batch_emission:
        pytest.skip("Redundant implicit combination")

    with MarginalModel() as m:
        P = [[0, 1], [1, 0]]
        init_dist = pm.Categorical.dist(p=[1, 0])
        chain = DiscreteMarkovChain(
            "chain", P=P, init_dist=init_dist, steps=3, shape=(3, 4) if batch_chain else None
        )
        emission = pm.Normal(
            "emission", mu=chain * 2 - 1, sigma=1e-1, shape=(3, 4) if batch_emission else None
        )

    m.marginalize([chain])
    logp_fn = m.compile_logp()

    test_value = np.array([-1, 1, -1, 1])
    expected_logp = pm.logp(pm.Normal.dist(0, 1e-1), np.zeros_like(test_value)).sum().eval()
    if batch_emission:
        test_value = np.broadcast_to(test_value, (3, 4))
        expected_logp *= 3
    np.testing.assert_allclose(logp_fn({"emission": test_value}), expected_logp)


@pytest.mark.parametrize(
    "categorical_emission",
    [False, True],
)
def test_marginalized_hmm_categorical_emission(categorical_emission):
    """Example adapted from https://www.youtube.com/watch?v=9-sPm4CfcD0"""
    with MarginalModel() as m:
        P = np.array([[0.5, 0.5], [0.3, 0.7]])
        init_dist = pm.Categorical.dist(p=[0.375, 0.625])
        chain = DiscreteMarkovChain("chain", P=P, init_dist=init_dist, steps=2)
        if categorical_emission:
            emission = pm.Categorical(
                "emission", p=pt.where(pt.eq(chain, 0)[..., None], [0.8, 0.2], [0.4, 0.6])
            )
        else:
            emission = pm.Bernoulli("emission", p=pt.where(pt.eq(chain, 0), 0.2, 0.6))
    m.marginalize([chain])

    test_value = np.array([0, 0, 1])
    expected_logp = np.log(0.1344)  # Shown at the 10m22s mark in the video
    logp_fn = m.compile_logp()
    np.testing.assert_allclose(logp_fn({"emission": test_value}), expected_logp)


@pytest.mark.parametrize("batch_emission1", (False, True))
@pytest.mark.parametrize("batch_emission2", (False, True))
def test_marginalized_hmm_multiple_emissions(batch_emission1, batch_emission2):
    emission1_shape = (2, 4) if batch_emission1 else (4,)
    emission2_shape = (2, 4) if batch_emission2 else (4,)
    with MarginalModel() as m:
        P = [[0, 1], [1, 0]]
        init_dist = pm.Categorical.dist(p=[1, 0])
        chain = DiscreteMarkovChain("chain", P=P, init_dist=init_dist, steps=3)
        emission_1 = pm.Normal("emission_1", mu=chain * 2 - 1, sigma=1e-1, shape=emission1_shape)
        emission_2 = pm.Normal(
            "emission_2", mu=(1 - chain) * 2 - 1, sigma=1e-1, shape=emission2_shape
        )

    with pytest.warns(UserWarning, match="multiple dependent variables"):
        m.marginalize([chain])

    logp_fn = m.compile_logp()

    test_value = np.array([-1, 1, -1, 1])
    multiplier = 2 + batch_emission1 + batch_emission2
    expected_logp = norm.logpdf(np.zeros_like(test_value), 0, 1e-1).sum() * multiplier
    test_value_emission1 = np.broadcast_to(test_value, emission1_shape)
    test_value_emission2 = np.broadcast_to(-test_value, emission2_shape)
    test_point = {"emission_1": test_value_emission1, "emission_2": test_value_emission2}
    np.testing.assert_allclose(logp_fn(test_point), expected_logp)
