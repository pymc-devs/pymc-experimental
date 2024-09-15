import itertools

from contextlib import suppress as does_not_warn

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytest

from arviz import InferenceData, dict_to_dataset
from pymc.distributions import transforms
from pymc.distributions.transforms import ordered
from pymc.model.fgraph import fgraph_from_model
from pymc.pytensorf import inputvars
from pymc.util import UNSET
from scipy.special import log_softmax, logsumexp
from scipy.stats import halfnorm, norm

from pymc_experimental.model.marginal.marginal_model import (
    MarginalModel,
    marginalize,
)
from tests.utils import equal_computations_up_to_root


def test_basic_marginalized_rv():
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


def test_one_to_one_marginalized_rvs():
    """Test case with multiple, independent marginalized RVs."""
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


def test_one_to_many_marginalized_rvs():
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


def test_one_to_many_unaligned_marginalized_rvs():
    """Test that marginalization works when there is more than one dependent RV with batch dimensions that are not aligned"""

    def build_model(build_batched: bool):
        with MarginalModel() as m:
            if build_batched:
                idx = pm.Bernoulli("idx", p=[0.75, 0.4], shape=(3, 2))
            else:
                idxs = [pm.Bernoulli(f"idx_{i}", p=(0.75 if i % 2 == 0 else 0.4)) for i in range(6)]
                idx = pt.stack(idxs, axis=0).reshape((3, 2))

            x = pm.Normal("x", mu=idx.T[:, :, None], shape=(2, 3, 1))
            y = pm.Normal("y", mu=(idx * 2 - 1), shape=(1, 3, 2))

        return m

    m = build_model(build_batched=True)
    ref_m = build_model(build_batched=False)

    with pytest.warns(UserWarning, match="There are multiple dependent variables"):
        m.marginalize(["idx"])
        ref_m.marginalize([f"idx_{i}" for i in range(6)])

    test_point = m.initial_point()
    np.testing.assert_allclose(
        m.compile_logp()(test_point),
        ref_m.compile_logp()(test_point),
    )


def test_many_to_one_marginalized_rvs():
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


@pytest.mark.parametrize("batched", (False, "left", "right"))
def test_nested_marginalized_rvs(batched):
    """Test that marginalization works when there are nested marginalized RVs"""

    def build_model(build_batched: bool) -> MarginalModel:
        idx_shape = (3,) if build_batched else ()
        sub_idx_shape = (5,) if not build_batched else (5, 3) if batched == "left" else (3, 5)

        with MarginalModel() as m:
            sigma = pm.HalfNormal("sigma")

            idx = pm.Bernoulli("idx", p=0.75, shape=idx_shape)
            dep = pm.Normal("dep", mu=pt.switch(pt.eq(idx, 0), -1000.0, 1000.0), sigma=sigma)

            sub_idx_p = pt.switch(pt.eq(idx, 0), 0.15, 0.95)
            if build_batched and batched == "right":
                sub_idx_p = sub_idx_p[..., None]
                dep = dep[..., None]
            sub_idx = pm.Bernoulli("sub_idx", p=sub_idx_p, shape=sub_idx_shape)
            sub_dep = pm.Normal("sub_dep", mu=dep + sub_idx * 100, sigma=sigma)

        return m

    m = build_model(build_batched=batched)
    with pytest.warns(UserWarning, match="There are multiple dependent variables"):
        m.marginalize(["idx", "sub_idx"])
    assert sorted(m.name for m in m.marginalized_rvs) == ["idx", "sub_idx"]

    # Test logp
    ref_m = build_model(build_batched=False)
    ref_logp_fn = ref_m.compile_logp(
        vars=[ref_m["idx"], ref_m["dep"], ref_m["sub_idx"], ref_m["sub_dep"]]
    )

    test_point = ref_m.initial_point()
    test_point["dep"] = np.full_like(test_point["dep"], 1000)
    test_point["sub_dep"] = np.full_like(test_point["sub_dep"], 1000 + 100)
    ref_logp = logsumexp(
        [
            ref_logp_fn({**test_point, **{"idx": idx, "sub_idx": np.array(sub_idxs)}})
            for idx in (0, 1)
            for sub_idxs in itertools.product((0, 1), repeat=5)
        ]
    )
    if batched:
        ref_logp *= 3

    test_point = m.initial_point()
    test_point["dep"] = np.full_like(test_point["dep"], 1000)
    test_point["sub_dep"] = np.full_like(test_point["sub_dep"], 1000 + 100)
    logp = m.compile_logp(vars=[m["dep"], m["sub_dep"]])(test_point)

    np.testing.assert_almost_equal(logp, ref_logp)


@pytest.mark.parametrize("advanced_indexing", (False, True))
def test_marginalized_index_as_key(advanced_indexing):
    """Test we can marginalize graphs where indexing is used as a mapping."""

    w = [0.1, 0.3, 0.6]
    mu = pt.as_tensor([-1, 0, 1])

    if advanced_indexing:
        y_val = pt.as_tensor([[-1, -1], [0, 1]])
        shape = (2, 2)
    else:
        y_val = -1
        shape = ()

    with MarginalModel() as m:
        x = pm.Categorical("x", p=w, shape=shape)
        y = pm.Normal("y", mu[x].T, sigma=1, observed=y_val)

    m.marginalize(x)

    marginal_logp = m.compile_logp(sum=False)({})[0]
    ref_logp = pm.logp(pm.NormalMixture.dist(w=w, mu=mu.T, sigma=1, shape=shape), y_val).eval()

    np.testing.assert_allclose(marginal_logp, ref_logp)


def test_marginalized_index_as_value_and_key():
    """Test we can marginalize graphs were marginalized_rv is indexed."""

    def build_model(build_batched: bool) -> MarginalModel:
        with MarginalModel() as m:
            if build_batched:
                latent_state = pm.Bernoulli("latent_state", p=0.3, size=(4,))
            else:
                latent_state = pm.math.stack(
                    [pm.Bernoulli(f"latent_state_{i}", p=0.3) for i in range(4)]
                )
            # latent state is used as the indexed variable
            latent_intensities = pt.where(latent_state[:, None], [0.0, 1.0, 2.0], [0.0, 10.0, 20.0])
            picked_intensity = pm.Categorical("picked_intensity", p=[0.2, 0.2, 0.6])
            # picked intensity is used as the indexing variable
            pm.Normal(
                "intensity",
                mu=latent_intensities[:, picked_intensity],
                observed=[0.5, 1.5, 5.0, 15.0],
            )
        return m

    # We compare with the equivalent but less efficient batched model
    m = build_model(build_batched=True)
    ref_m = build_model(build_batched=False)

    m.marginalize(["latent_state"])
    ref_m.marginalize([f"latent_state_{i}" for i in range(4)])
    test_point = {"picked_intensity": 1}
    np.testing.assert_allclose(
        m.compile_logp()(test_point),
        ref_m.compile_logp()(test_point),
    )

    m.marginalize(["picked_intensity"])
    ref_m.marginalize(["picked_intensity"])
    test_point = {}
    np.testing.assert_allclose(
        m.compile_logp()(test_point),
        ref_m.compile_logp()(test_point),
    )


class TestNotSupportedMixedDims:
    """Test lack of support for models where batch dims of marginalized variables are mixed."""

    def test_mixed_dims_via_transposed_dot(self):
        with MarginalModel() as m:
            idx = pm.Bernoulli("idx", p=0.7, shape=2)
            y = pm.Normal("y", mu=idx @ idx.T)
            with pytest.raises(NotImplementedError):
                m.marginalize(idx)

    def test_mixed_dims_via_indexing(self):
        mean = pt.as_tensor([[0.1, 0.9], [0.6, 0.4]])

        with MarginalModel() as m:
            idx = pm.Bernoulli("idx", p=0.7, shape=2)
            y = pm.Normal("y", mu=mean[idx, :] + mean[:, idx])
            with pytest.raises(NotImplementedError):
                m.marginalize(idx)

        with MarginalModel() as m:
            idx = pm.Bernoulli("idx", p=0.7, shape=2)
            y = pm.Normal("y", mu=mean[idx, None] + mean[None, idx])
            with pytest.raises(NotImplementedError):
                m.marginalize(idx)

        with MarginalModel() as m:
            idx = pm.Bernoulli("idx", p=0.7, shape=2)
            mu = pt.specify_broadcastable(mean[:, None][idx], 1) + pt.specify_broadcastable(
                mean[None, :][:, idx], 0
            )
            y = pm.Normal("y", mu=mu)
            with pytest.raises(NotImplementedError):
                m.marginalize(idx)

        with MarginalModel() as m:
            idx = pm.Bernoulli("idx", p=0.7, shape=2)
            y = pm.Normal("y", mu=idx[0] + idx[1])
            with pytest.raises(NotImplementedError):
                m.marginalize(idx)

    def test_mixed_dims_via_vector_indexing(self):
        with MarginalModel() as m:
            idx = pm.Bernoulli("idx", p=0.7, shape=2)
            y = pm.Normal("y", mu=idx[[0, 1, 0, 0]])
            with pytest.raises(NotImplementedError):
                m.marginalize(idx)

        with MarginalModel() as m:
            idx = pm.Categorical("key", p=[0.1, 0.3, 0.6], shape=(2, 2))
            y = pm.Normal("y", pt.as_tensor([[0, 1], [2, 3]])[idx.astype(bool)])
            with pytest.raises(NotImplementedError):
                m.marginalize(idx)

    def test_mixed_dims_via_support_dimension(self):
        with MarginalModel() as m:
            x = pm.Bernoulli("x", p=0.7, shape=3)
            y = pm.Dirichlet("y", a=x * 10 + 1)
            with pytest.raises(NotImplementedError):
                m.marginalize(x)

    def test_mixed_dims_via_nested_marginalization(self):
        with MarginalModel() as m:
            x = pm.Bernoulli("x", p=0.7, shape=(3,))
            y = pm.Bernoulli("y", p=0.7, shape=(2,))
            z = pm.Normal("z", mu=pt.add.outer(x, y), shape=(3, 2))

            with pytest.raises(NotImplementedError):
                m.marginalize([x, y])


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
            default_transform=transform,
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
            default_transform=transform,
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


def test_mutable_indexing_jax_backend():
    pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_logp

    with MarginalModel() as model:
        data = pm.Data("data", np.zeros(10))

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


class TestFullModels:
    @pytest.fixture
    def disaster_model(self):
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
            with pytest.warns(Warning):
                disasters = pm.Poisson("disasters", rate, observed=disaster_data)

        return disaster_model, years

    def test_change_point_model(self, disaster_model):
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
    def test_change_point_model_sampling(self, disaster_model):
        m, _ = disaster_model

        rng = np.random.default_rng(211)

        with m:
            before_marg = pm.sample(chains=2, random_seed=rng).posterior.stack(
                sample=("draw", "chain")
            )

        with pytest.warns(UserWarning, match="There are multiple dependent variables"):
            m.marginalize([m["switchpoint"]])

        with m:
            after_marg = pm.sample(chains=2, random_seed=rng).posterior.stack(
                sample=("draw", "chain")
            )

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

    @pytest.mark.parametrize("univariate", (True, False))
    def test_vector_univariate_mixture(self, univariate):
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

    def test_k_censored_clusters_model(self):
        def build_model(build_batched: bool) -> MarginalModel:
            data = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
            nobs = data.shape[0]
            n_clusters = 5
            coords = {
                "cluster": range(n_clusters),
                "ndim": ("x", "y"),
                "obs": range(nobs),
            }
            with MarginalModel(coords=coords) as m:
                if build_batched:
                    idx = pm.Categorical("idx", p=np.ones(n_clusters) / n_clusters, dims=["obs"])
                else:
                    idx = pm.math.stack(
                        [
                            pm.Categorical(f"idx_{i}", p=np.ones(n_clusters) / n_clusters)
                            for i in range(nobs)
                        ]
                    )

                mu_x = pm.Normal(
                    "mu_x",
                    dims=["cluster"],
                    transform=ordered,
                    initval=np.linspace(-1, 1, n_clusters),
                )
                mu_y = pm.Normal("mu_y", dims=["cluster"])
                mu = pm.math.stack([mu_x, mu_y], axis=-1)  # (cluster, ndim)
                mu_indexed = mu[idx, :]

                sigma = pm.HalfNormal("sigma")

                y = pm.Censored(
                    "y",
                    dist=pm.Normal.dist(mu_indexed, sigma),
                    lower=-3,
                    upper=3,
                    observed=data,
                    dims=["obs", "ndim"],
                )

            return m

        m = build_model(build_batched=True)
        ref_m = build_model(build_batched=False)

        m.marginalize([m["idx"]])
        ref_m.marginalize([n for n in ref_m.named_vars if n.startswith("idx_")])

        test_point = m.initial_point()
        np.testing.assert_almost_equal(
            m.compile_logp()(test_point),
            ref_m.compile_logp()(test_point),
        )


class TestRecoverMarginals:
    def test_basic(self):
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
                draws=20,
                random_seed=rng,
                return_inferencedata=False,
            )
            idata = InferenceData(posterior=dict_to_dataset(prior))

        idata = m.recover_marginals(idata, return_samples=True)
        post = idata.posterior
        assert "k" in post
        assert "lp_k" in post
        assert post.k.shape == post.y.shape
        assert post.lp_k.shape == (*post.k.shape, len(p))

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

    def test_coords(self):
        """Test if coords can be recovered with marginalized value had it originally"""
        with MarginalModel(coords={"year": [1990, 1991, 1992]}) as m:
            sigma = pm.HalfNormal("sigma")
            idx = pm.Bernoulli("idx", p=0.75, dims="year")
            x = pm.Normal("x", mu=idx, sigma=sigma, dims="year")

        m.marginalize([idx])
        rng = np.random.default_rng(211)

        with m:
            prior = pm.sample_prior_predictive(
                draws=20,
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

    def test_batched(self):
        """Test that marginalization works for batched random variables"""
        with MarginalModel() as m:
            sigma = pm.HalfNormal("sigma")
            idx = pm.Bernoulli("idx", p=0.7, shape=(3, 2))
            y = pm.Normal("y", mu=idx.T, sigma=sigma, shape=(2, 3))

        m.marginalize([idx])

        rng = np.random.default_rng(211)

        with m:
            prior = pm.sample_prior_predictive(
                draws=20,
                random_seed=rng,
                return_inferencedata=False,
            )
            idata = InferenceData(
                posterior=dict_to_dataset({k: np.expand_dims(prior[k], axis=0) for k in prior})
            )

        idata = m.recover_marginals(idata, return_samples=True)
        post = idata.posterior
        assert post["y"].shape == (1, 20, 2, 3)
        assert post["idx"].shape == (1, 20, 3, 2)
        assert post["lp_idx"].shape == (1, 20, 3, 2, 2)

    def test_nested(self):
        """Test that marginalization works when there are nested marginalized RVs"""

        with MarginalModel() as m:
            idx = pm.Bernoulli("idx", p=0.75)
            sub_idx = pm.Bernoulli("sub_idx", p=pt.switch(pt.eq(idx, 0), 0.15, 0.95))
            sub_dep = pm.Normal("y", mu=idx + sub_idx, sigma=1.0)

        m.marginalize([idx, sub_idx])

        rng = np.random.default_rng(211)

        with m:
            prior = pm.sample_prior_predictive(
                draws=20,
                random_seed=rng,
                return_inferencedata=False,
            )
            idata = InferenceData(posterior=dict_to_dataset(prior))

        idata = m.recover_marginals(idata, return_samples=True)
        post = idata.posterior
        assert "idx" in post
        assert "lp_idx" in post
        assert post.idx.shape == post.y.shape
        assert post.lp_idx.shape == (*post.idx.shape, 2)
        assert "sub_idx" in post
        assert "lp_sub_idx" in post
        assert post.sub_idx.shape == post.y.shape
        assert post.lp_sub_idx.shape == (*post.sub_idx.shape, 2)

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
