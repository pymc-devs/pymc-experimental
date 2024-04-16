import itertools
from contextlib import suppress as does_not_warn

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytest
from arviz import InferenceData, dict_to_dataset
from pymc import ImputationWarning, inputvars
from pymc.distributions import transforms
from pymc.logprob.abstract import _logprob
from pymc.model.fgraph import fgraph_from_model
from pymc.util import UNSET
from scipy.special import log_softmax, logsumexp
from scipy.stats import halfnorm, norm

from pymc_experimental.distributions import DiscreteMarkovChain
from pymc_experimental.model.marginal_model import (
    FiniteDiscreteMarginalRV,
    MarginalModel,
    is_conditional_dependent,
    marginalize,
)
from pymc_experimental.tests.utils import equal_computations_up_to_root


@pytest.fixture
def disaster_model():
    # fmt: off
    disaster_data = pd.Series(
        [4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
         3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
         2, 2, 3, 4, 2, 1, 3, np.nan, 2, 1, 1, 1, 1, 3, 0, 0,
         1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
         0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
         3, 3, 1, np.nan, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]
    )
    # fmt: on
    years = np.arange(1851, 1962)

    with MarginalModel() as disaster_model:
        switchpoint = pm.DiscreteUniform("switchpoint", lower=years.min(), upper=years.max())
        early_rate = pm.Exponential("early_rate", 1.0, initval=3)
        late_rate = pm.Exponential("late_rate", 1.0, initval=1)
        rate = pm.math.switch(switchpoint >= years, early_rate, late_rate)
        with pytest.warns(ImputationWarning), pytest.warns(
            RuntimeWarning, match="invalid value encountered in cast"
        ):
            disasters = pm.Poisson("disasters", rate, observed=disaster_data)

    return disaster_model, years


@pytest.mark.filterwarnings("error")
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


@pytest.mark.filterwarnings("error")
def test_marginalized_basic():
    data = [2] * 5

    with MarginalModel() as m:
        sigma = pm.HalfNormal("sigma")
        idx = pm.Categorical("idx", p=[0.1, 0.3, 0.6])
        mu = pt.switch(
            pt.eq(idx, 0),
            -1.0,
            pt.switch(
                pt.eq(idx, 1),
                0.0,
                1.0,
            ),
        )
        y = pm.Normal("y", mu=mu, sigma=sigma)
        z = pm.Normal("z", y, observed=data)

    m.marginalize([idx])
    assert idx not in m.free_RVs
    assert [rv.name for rv in m.marginalized_rvs] == ["idx"]

    # Test logp
    with pm.Model() as m_ref:
        sigma = pm.HalfNormal("sigma")
        y = pm.NormalMixture("y", w=[0.1, 0.3, 0.6], mu=[-1, 0, 1], sigma=sigma)
        z = pm.Normal("z", y, observed=data)

    test_point = m_ref.initial_point()
    ref_logp = m_ref.compile_logp()(test_point)
    ref_dlogp = m_ref.compile_dlogp([m_ref["y"]])(test_point)

    # Assert we can marginalize and unmarginalize internally non-destructively
    for i in range(3):
        np.testing.assert_almost_equal(
            m.compile_logp()(test_point),
            ref_logp,
        )
        np.testing.assert_almost_equal(
            m.compile_dlogp([m["y"]])(test_point),
            ref_dlogp,
        )


@pytest.mark.filterwarnings("error")
def test_multiple_independent_marginalized_rvs():
    with MarginalModel() as m:
        sigma = pm.HalfNormal("sigma")
        idx1 = pm.Bernoulli("idx1", p=0.75)
        x = pm.Normal("x", mu=idx1, sigma=sigma)
        idx2 = pm.Bernoulli("idx2", p=0.75, shape=(5,))
        y = pm.Normal("y", mu=(idx2 * 2 - 1), sigma=sigma, shape=(5,))

    m.marginalize([idx1, idx2])
    m["x"].owner is not m["y"].owner
    _m = m.clone()._marginalize()
    _m["x"].owner is not _m["y"].owner

    with pm.Model() as m_ref:
        sigma = pm.HalfNormal("sigma")
        x = pm.NormalMixture("x", w=[0.25, 0.75], mu=[0, 1], sigma=sigma)
        y = pm.NormalMixture("y", w=[0.25, 0.75], mu=[-1, 1], sigma=sigma, shape=(5,))

    # Test logp
    test_point = m_ref.initial_point()
    x_logp, y_logp = m.compile_logp(vars=[m["x"], m["y"]], sum=False)(test_point)
    x_ref_log, y_ref_logp = m_ref.compile_logp(vars=[m_ref["x"], m_ref["y"]], sum=False)(test_point)
    np.testing.assert_array_almost_equal(x_logp, x_ref_log.sum())
    np.testing.assert_array_almost_equal(y_logp, y_ref_logp)


@pytest.mark.filterwarnings("error")
def test_multiple_dependent_marginalized_rvs():
    """Test that marginalization works when there is more than one dependent RV"""
    with MarginalModel() as m:
        sigma = pm.HalfNormal("sigma")
        idx = pm.Bernoulli("idx", p=0.75)
        x = pm.Normal("x", mu=idx, sigma=sigma)
        y = pm.Normal("y", mu=(idx * 2 - 1), sigma=sigma, shape=(5,))

    ref_logp_x_y_fn = m.compile_logp([idx, x, y])

    with pytest.warns(UserWarning, match="There are multiple dependent variables"):
        m.marginalize([idx])

    m["x"].owner is not m["y"].owner
    _m = m.clone()._marginalize()
    _m["x"].owner is _m["y"].owner

    tp = m.initial_point()
    ref_logp_x_y = logsumexp([ref_logp_x_y_fn({**tp, **{"idx": idx}}) for idx in (0, 1)])
    logp_x_y = m.compile_logp([x, y])(tp)
    np.testing.assert_array_almost_equal(logp_x_y, ref_logp_x_y)


def test_rv_dependent_multiple_marginalized_rvs():
    """Test when random variables depend on multiple marginalized variables"""
    with MarginalModel() as m:
        x = pm.Bernoulli("x", 0.1)
        y = pm.Bernoulli("y", 0.3)
        z = pm.DiracDelta("z", c=x + y)

    m.marginalize([x, y])
    logp = m.compile_logp()

    np.testing.assert_allclose(np.exp(logp({"z": 0})), 0.9 * 0.7)
    np.testing.assert_allclose(np.exp(logp({"z": 1})), 0.9 * 0.3 + 0.1 * 0.7)
    np.testing.assert_allclose(np.exp(logp({"z": 2})), 0.1 * 0.3)


@pytest.mark.filterwarnings("error")
def test_nested_marginalized_rvs():
    """Test that marginalization works when there are nested marginalized RVs"""

    with MarginalModel() as m:
        sigma = pm.HalfNormal("sigma")

        idx = pm.Bernoulli("idx", p=0.75)
        dep = pm.Normal("dep", mu=pt.switch(pt.eq(idx, 0), -1000.0, 1000.0), sigma=sigma)

        sub_idx = pm.Bernoulli("sub_idx", p=pt.switch(pt.eq(idx, 0), 0.15, 0.95), shape=(5,))
        sub_dep = pm.Normal("sub_dep", mu=dep + sub_idx * 100, sigma=sigma, shape=(5,))

    ref_logp_fn = m.compile_logp(vars=[idx, dep, sub_idx, sub_dep])

    with pytest.warns(UserWarning, match="There are multiple dependent variables"):
        m.marginalize([idx, sub_idx])

    assert set(m.marginalized_rvs) == {idx, sub_idx}

    # Test logp
    test_point = m.initial_point()
    test_point["dep"] = 1000
    test_point["sub_dep"] = np.full((5,), 1000 + 100)

    ref_logp = [
        ref_logp_fn({**test_point, **{"idx": idx, "sub_idx": np.array(sub_idxs)}})
        for idx in (0, 1)
        for sub_idxs in itertools.product((0, 1), repeat=5)
    ]
    logp = m.compile_logp(vars=[dep, sub_dep])(test_point)

    np.testing.assert_almost_equal(
        logp,
        logsumexp(ref_logp),
    )


@pytest.mark.filterwarnings("error")
def test_marginalized_change_point_model(disaster_model):
    m, years = disaster_model

    ip = m.initial_point()
    ip.pop("switchpoint")
    ref_logp_fn = m.compile_logp(
        [m["switchpoint"], m["disasters_observed"], m["disasters_unobserved"]]
    )
    ref_logp = logsumexp([ref_logp_fn({**ip, **{"switchpoint": year}}) for year in years])

    with pytest.warns(UserWarning, match="There are multiple dependent variables"):
        m.marginalize(m["switchpoint"])

    logp = m.compile_logp([m["disasters_observed"], m["disasters_unobserved"]])(ip)
    np.testing.assert_almost_equal(logp, ref_logp)


@pytest.mark.slow
@pytest.mark.filterwarnings("error")
def test_marginalized_change_point_model_sampling(disaster_model):
    m, _ = disaster_model

    rng = np.random.default_rng(211)

    with m:
        before_marg = pm.sample(chains=2, random_seed=rng).posterior.stack(sample=("draw", "chain"))

    with pytest.warns(UserWarning, match="There are multiple dependent variables"):
        m.marginalize([m["switchpoint"]])

    with m:
        after_marg = pm.sample(chains=2, random_seed=rng).posterior.stack(sample=("draw", "chain"))

    np.testing.assert_allclose(
        before_marg["early_rate"].mean(), after_marg["early_rate"].mean(), rtol=1e-2
    )
    np.testing.assert_allclose(
        before_marg["late_rate"].mean(), after_marg["late_rate"].mean(), rtol=1e-2
    )
    np.testing.assert_allclose(
        before_marg["disasters_unobserved"].mean(),
        after_marg["disasters_unobserved"].mean(),
        rtol=1e-2,
    )


def test_recover_marginals_basic():
    with MarginalModel() as m:
        sigma = pm.HalfNormal("sigma")
        p = np.array([0.5, 0.2, 0.3])
        k = pm.Categorical("k", p=p)
        mu = np.array([-3.0, 0.0, 3.0])
        mu_ = pt.as_tensor_variable(mu)
        y = pm.Normal("y", mu=mu_[k], sigma=sigma)

    m.marginalize([k])

    rng = np.random.default_rng(211)

    with m:
        prior = pm.sample_prior_predictive(
            samples=20,
            random_seed=rng,
            return_inferencedata=False,
        )
        idata = InferenceData(posterior=dict_to_dataset(prior))

    idata = m.recover_marginals(idata, return_samples=True)
    post = idata.posterior
    assert "k" in post
    assert "lp_k" in post
    assert post.k.shape == post.y.shape
    assert post.lp_k.shape == post.k.shape + (len(p),)

    def true_logp(y, sigma):
        y = y.repeat(len(p)).reshape(len(y), -1)
        sigma = sigma.repeat(len(p)).reshape(len(sigma), -1)
        return log_softmax(
            np.log(p)
            + norm.logpdf(y, loc=mu, scale=sigma)
            + halfnorm.logpdf(sigma)
            + np.log(sigma),
            axis=1,
        )

    np.testing.assert_almost_equal(
        true_logp(post.y.values.flatten(), post.sigma.values.flatten()),
        post.lp_k[0].values,
    )
    np.testing.assert_almost_equal(logsumexp(post.lp_k, axis=-1), 0)


def test_recover_marginals_coords():
    """Test if coords can be recovered with marginalized value had it originally"""
    with MarginalModel(coords={"year": [1990, 1991, 1992]}) as m:
        sigma = pm.HalfNormal("sigma")
        idx = pm.Bernoulli("idx", p=0.75, dims="year")
        x = pm.Normal("x", mu=idx, sigma=sigma, dims="year")

    m.marginalize([idx])
    rng = np.random.default_rng(211)

    with m:
        prior = pm.sample_prior_predictive(
            samples=20,
            random_seed=rng,
            return_inferencedata=False,
        )
        idata = InferenceData(
            posterior=dict_to_dataset({k: np.expand_dims(prior[k], axis=0) for k in prior})
        )

    idata = m.recover_marginals(idata, return_samples=True)
    post = idata.posterior
    assert post.idx.dims == ("chain", "draw", "year")
    assert post.lp_idx.dims == ("chain", "draw", "year", "lp_idx_dim")


def test_recover_batched_marginal():
    """Test that marginalization works for batched random variables"""
    with MarginalModel() as m:
        sigma = pm.HalfNormal("sigma")
        idx = pm.Bernoulli("idx", p=0.7, shape=(3, 2))
        y = pm.Normal("y", mu=idx, sigma=sigma, shape=(3, 2))

    m.marginalize([idx])

    rng = np.random.default_rng(211)

    with m:
        prior = pm.sample_prior_predictive(
            samples=20,
            random_seed=rng,
            return_inferencedata=False,
        )
        idata = InferenceData(
            posterior=dict_to_dataset({k: np.expand_dims(prior[k], axis=0) for k in prior})
        )

    idata = m.recover_marginals(idata, return_samples=True)
    post = idata.posterior
    assert "idx" in post
    assert "lp_idx" in post
    assert post.idx.shape == post.y.shape
    assert post.lp_idx.shape == post.idx.shape + (2,)


def test_nested_recover_marginals():
    """Test that marginalization works when there are nested marginalized RVs"""

    with MarginalModel() as m:
        idx = pm.Bernoulli("idx", p=0.75)
        sub_idx = pm.Bernoulli("sub_idx", p=pt.switch(pt.eq(idx, 0), 0.15, 0.95))
        sub_dep = pm.Normal("y", mu=idx + sub_idx, sigma=1.0)

    m.marginalize([idx, sub_idx])

    rng = np.random.default_rng(211)

    with m:
        prior = pm.sample_prior_predictive(
            samples=20,
            random_seed=rng,
            return_inferencedata=False,
        )
        idata = InferenceData(posterior=dict_to_dataset(prior))

    idata = m.recover_marginals(idata, return_samples=True)
    post = idata.posterior
    assert "idx" in post
    assert "lp_idx" in post
    assert post.idx.shape == post.y.shape
    assert post.lp_idx.shape == post.idx.shape + (2,)
    assert "sub_idx" in post
    assert "lp_sub_idx" in post
    assert post.sub_idx.shape == post.y.shape
    assert post.lp_sub_idx.shape == post.sub_idx.shape + (2,)

    def true_idx_logp(y):
        idx_0 = np.log(0.85 * 0.25 * norm.pdf(y, loc=0) + 0.15 * 0.25 * norm.pdf(y, loc=1))
        idx_1 = np.log(0.05 * 0.75 * norm.pdf(y, loc=1) + 0.95 * 0.75 * norm.pdf(y, loc=2))
        return log_softmax(np.stack([idx_0, idx_1]).T, axis=1)

    np.testing.assert_almost_equal(
        true_idx_logp(post.y.values.flatten()),
        post.lp_idx[0].values,
    )

    def true_sub_idx_logp(y):
        sub_idx_0 = np.log(0.85 * 0.25 * norm.pdf(y, loc=0) + 0.05 * 0.75 * norm.pdf(y, loc=1))
        sub_idx_1 = np.log(0.15 * 0.25 * norm.pdf(y, loc=1) + 0.95 * 0.75 * norm.pdf(y, loc=2))
        return log_softmax(np.stack([sub_idx_0, sub_idx_1]).T, axis=1)

    np.testing.assert_almost_equal(
        true_sub_idx_logp(post.y.values.flatten()),
        post.lp_sub_idx[0].values,
    )
    np.testing.assert_almost_equal(logsumexp(post.lp_idx, axis=-1), 0)
    np.testing.assert_almost_equal(logsumexp(post.lp_sub_idx, axis=-1), 0)


@pytest.mark.filterwarnings("error")
def test_not_supported_marginalized():
    """Marginalized graphs with non-Elemwise Operations are not supported as they
    would violate the batching logp assumption"""
    mu = pt.constant([-1, 1])

    # Allowed, as only elemwise operations connect idx to y
    with MarginalModel() as m:
        p = pm.Beta("p", 1, 1)
        idx = pm.Bernoulli("idx", p=p, size=2)
        y = pm.Normal("y", mu=pm.math.switch(idx, 0, 1))
        m.marginalize([idx])

    # ALlowed, as index operation does not connext idx to y
    with MarginalModel() as m:
        p = pm.Beta("p", 1, 1)
        idx = pm.Bernoulli("idx", p=p, size=2)
        y = pm.Normal("y", mu=pm.math.switch(idx, mu[0], mu[1]))
        m.marginalize([idx])

    # Not allowed, as index operation  connects idx to y
    with MarginalModel() as m:
        p = pm.Beta("p", 1, 1)
        idx = pm.Bernoulli("idx", p=p, size=2)
        # Not allowed
        y = pm.Normal("y", mu=mu[idx])
        with pytest.raises(NotImplementedError):
            m.marginalize(idx)

    # Not allowed, as index operation  connects idx to y, even though there is a
    # pure Elemwise connection between the two
    with MarginalModel() as m:
        p = pm.Beta("p", 1, 1)
        idx = pm.Bernoulli("idx", p=p, size=2)
        y = pm.Normal("y", mu=mu[idx] + idx)
        with pytest.raises(NotImplementedError):
            m.marginalize(idx)

    # Multivariate dependent RVs not supported
    with MarginalModel() as m:
        x = pm.Bernoulli("x", p=0.7)
        y = pm.Dirichlet("y", a=pm.math.switch(x, [1, 1, 1], [10, 10, 10]))
        with pytest.raises(
            NotImplementedError,
            match="Marginalization with dependent Multivariate RVs not implemented",
        ):
            m.marginalize(x)


@pytest.mark.filterwarnings("error")
def test_marginalized_deterministic_and_potential():
    rng = np.random.default_rng(299)

    with MarginalModel() as m:
        x = pm.Bernoulli("x", p=0.7)
        y = pm.Normal("y", x)
        z = pm.Normal("z", x)
        det = pm.Deterministic("det", y + z)
        pot = pm.Potential("pot", y + z + 1)

    with pytest.warns(UserWarning, match="There are multiple dependent variables"):
        m.marginalize([x])

    y_draw, z_draw, det_draw, pot_draw = pm.draw([y, z, det, pot], draws=5, random_seed=rng)
    np.testing.assert_almost_equal(y_draw + z_draw, det_draw)
    np.testing.assert_almost_equal(det_draw, pot_draw - 1)

    y_value = m.rvs_to_values[y]
    z_value = m.rvs_to_values[z]
    det_value, pot_value = m.replace_rvs_by_values([det, pot])
    assert set(inputvars([det_value, pot_value])) == {y_value, z_value}
    assert det_value.eval({y_value: 2, z_value: 5}) == 7
    assert pot_value.eval({y_value: 2, z_value: 5}) == 8


@pytest.mark.filterwarnings("error")
def test_not_supported_marginalized_deterministic_and_potential():
    with MarginalModel() as m:
        x = pm.Bernoulli("x", p=0.7)
        y = pm.Normal("y", x)
        det = pm.Deterministic("det", x + y)

    with pytest.raises(
        NotImplementedError, match="Cannot marginalize x due to dependent Deterministic det"
    ):
        m.marginalize([x])

    with MarginalModel() as m:
        x = pm.Bernoulli("x", p=0.7)
        y = pm.Normal("y", x)
        pot = pm.Potential("pot", x + y)

    with pytest.raises(
        NotImplementedError, match="Cannot marginalize x due to dependent Potential pot"
    ):
        m.marginalize([x])


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "transform, expected_warning",
    (
        (None, does_not_warn()),
        (UNSET, does_not_warn()),
        (transforms.log, does_not_warn()),
        (transforms.Chain([transforms.log, transforms.logodds]), does_not_warn()),
        (
            transforms.Interval(0, 1),
            pytest.warns(
                UserWarning, match="which depends on the marginalized idx may no longer work"
            ),
        ),
        (
            transforms.Chain([transforms.log, transforms.Interval(0, 1)]),
            pytest.warns(
                UserWarning, match="which depends on the marginalized idx may no longer work"
            ),
        ),
    ),
)
def test_marginalized_transforms(transform, expected_warning):
    w = [0.1, 0.3, 0.6]
    data = [0, 5, 10]
    initval = 0.5  # Value that will be negative on the unconstrained space

    with pm.Model() as m_ref:
        sigma = pm.Mixture(
            "sigma",
            w=w,
            comp_dists=pm.HalfNormal.dist([1, 2, 3]),
            initval=initval,
            transform=transform,
        )
        y = pm.Normal("y", 0, sigma, observed=data)

    with MarginalModel() as m:
        idx = pm.Categorical("idx", p=w)
        sigma = pm.HalfNormal(
            "sigma",
            pt.switch(
                pt.eq(idx, 0),
                1,
                pt.switch(
                    pt.eq(idx, 1),
                    2,
                    3,
                ),
            ),
            initval=initval,
            transform=transform,
        )
        y = pm.Normal("y", 0, sigma, observed=data)

    with expected_warning:
        m.marginalize([idx])

    ip = m.initial_point()
    if transform is not None:
        if transform is UNSET:
            transform_name = "log"
        else:
            transform_name = transform.name
        assert f"sigma_{transform_name}__" in ip
    np.testing.assert_allclose(m.compile_logp()(ip), m_ref.compile_logp()(ip))


def test_is_conditional_dependent_static_shape():
    """Test that we don't consider dependencies through "constant" shape Ops"""
    x1 = pt.matrix("x1", shape=(None, 5))
    y1 = pt.random.normal(size=pt.shape(x1))
    assert is_conditional_dependent(y1, x1, [x1, y1])

    x2 = pt.matrix("x2", shape=(9, 5))
    y2 = pt.random.normal(size=pt.shape(x2))
    assert not is_conditional_dependent(y2, x2, [x2, y2])


def test_data_container():
    """Test that MarginalModel can handle Data containers."""
    with MarginalModel(coords={"obs": [0]}) as marginal_m:
        x = pm.Data("x", 2.5)
        idx = pm.Bernoulli("idx", p=0.7, dims="obs")
        y = pm.Normal("y", idx * x, dims="obs")

    marginal_m.marginalize([idx])

    logp_fn = marginal_m.compile_logp()

    with pm.Model(coords={"obs": [0]}) as m_ref:
        x = pm.Data("x", 2.5)
        y = pm.NormalMixture("y", w=[0.3, 0.7], mu=[0, x], dims="obs")

    ref_logp_fn = m_ref.compile_logp()

    for i, x_val in enumerate((-1.5, 2.5, 3.5), start=1):
        for m in (marginal_m, m_ref):
            m.set_dim("obs", new_length=i, coord_values=tuple(range(i)))
            pm.set_data({"x": x_val}, model=m)

        ip = marginal_m.initial_point()
        np.testing.assert_allclose(logp_fn(ip), ref_logp_fn(ip))


@pytest.mark.parametrize("univariate", (True, False))
def test_vector_univariate_mixture(univariate):

    with MarginalModel() as m:
        idx = pm.Bernoulli("idx", p=0.5, shape=(2,) if univariate else ())

        def dist(idx, size):
            return pm.math.switch(
                pm.math.eq(idx, 0),
                pm.Normal.dist([-10, -10], 1),
                pm.Normal.dist([10, 10], 1),
            )

        pm.CustomDist("norm", idx, dist=dist)

    m.marginalize(idx)
    logp_fn = m.compile_logp()

    if univariate:
        with pm.Model() as ref_m:
            pm.NormalMixture("norm", w=[0.5, 0.5], mu=[[-10, 10], [-10, 10]], shape=(2,))
    else:
        with pm.Model() as ref_m:
            pm.Mixture(
                "norm",
                w=[0.5, 0.5],
                comp_dists=[
                    pm.MvNormal.dist([-10, -10], np.eye(2)),
                    pm.MvNormal.dist([10, 10], np.eye(2)),
                ],
                shape=(2,),
            )
    ref_logp_fn = ref_m.compile_logp()

    for test_value in (
        [-10, -10],
        [10, 10],
        [-10, 10],
        [-10, 10],
    ):
        pt = {"norm": test_value}
        np.testing.assert_allclose(logp_fn(pt), ref_logp_fn(pt))


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
    np.testing.assert_allclose(logp_fn({f"emission": test_value}), expected_logp)


@pytest.mark.parametrize(
    "categorical_emission",
    [
        False,
        # Categorical has a core vector parameter,
        # so it is not possible to build a graph that uses elemwise operations exclusively
        pytest.param(True, marks=pytest.mark.xfail(raises=NotImplementedError)),
    ],
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
    np.testing.assert_allclose(logp_fn({f"emission": test_value}), expected_logp)


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


def test_mutable_indexing_jax_backend():
    pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_logp

    with MarginalModel() as model:
        data = pm.Data(f"data", np.zeros(10))

        cat_effect = pm.Normal("cat_effect", sigma=1, shape=5)
        cat_effect_idx = pm.Data("cat_effect_idx", np.array([0, 1] * 5))

        is_outlier = pm.Bernoulli("is_outlier", 0.4, shape=10)
        pm.LogNormal("y", mu=cat_effect[cat_effect_idx], sigma=1 + is_outlier, observed=data)
    model.marginalize(["is_outlier"])
    get_jaxified_logp(model)


def test_marginal_model_func():
    def create_model(model_class):
        with model_class(coords={"trial": range(10)}) as m:
            idx = pm.Bernoulli("idx", p=0.5, dims="trial")
            mu = pt.where(idx, 1, -1)
            sigma = pm.HalfNormal("sigma")
            y = pm.Normal("y", mu=mu, sigma=sigma, dims="trial", observed=[1] * 10)
        return m

    marginal_m = marginalize(create_model(pm.Model), ["idx"])
    assert isinstance(marginal_m, MarginalModel)

    reference_m = create_model(MarginalModel)
    reference_m.marginalize(["idx"])

    # Check forward graph representation is the same
    marginal_fgraph, _ = fgraph_from_model(marginal_m)
    reference_fgraph, _ = fgraph_from_model(reference_m)
    assert equal_computations_up_to_root(marginal_fgraph.outputs, reference_fgraph.outputs)

    # Check logp graph is the same
    # This fails because OpFromGraphs comparison is broken
    # assert equal_computations_up_to_root([marginal_m.logp()], [reference_m.logp()])
    ip = marginal_m.initial_point()
    np.testing.assert_allclose(
        marginal_m.compile_logp()(ip),
        reference_m.compile_logp()(ip),
    )
