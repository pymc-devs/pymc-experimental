import arviz as az
import numpy as np
import pymc as pm
import pytest
from pymc.variational.minibatch_rv import create_minibatch_rv
from pytensor import config

from pymc_experimental.model_transform.conditioning import do, observe


def test_observe():
    with pm.Model() as m_old:
        x = pm.Normal("x")
        y = pm.Normal("y", x)
        z = pm.Normal("z", y)

    m_new = observe(m_old, {y: 0.5})

    assert len(m_new.free_RVs) == 2
    assert len(m_new.observed_RVs) == 1
    assert m_new["x"] in m_new.free_RVs
    assert m_new["y"] in m_new.observed_RVs
    assert m_new["z"] in m_new.free_RVs

    np.testing.assert_allclose(
        m_old.compile_logp()({"x": 0.9, "y": 0.5, "z": 1.4}),
        m_new.compile_logp()({"x": 0.9, "z": 1.4}),
    )

    # Test two substitutions
    m_new = observe(m_old, {y: 0.5, z: 1.4})

    assert len(m_new.free_RVs) == 1
    assert len(m_new.observed_RVs) == 2
    assert m_new["x"] in m_new.free_RVs
    assert m_new["y"] in m_new.observed_RVs
    assert m_new["z"] in m_new.observed_RVs

    np.testing.assert_allclose(
        m_old.compile_logp()({"x": 0.9, "y": 0.5, "z": 1.4}),
        m_new.compile_logp()({"x": 0.9}),
    )


def test_observe_minibatch():
    data = np.zeros((100,), dtype=config.floatX)
    batch_size = 10
    with pm.Model() as m_old:
        x = pm.Normal("x")
        y = pm.Normal("y", x)
        # Minibatch RVs are usually created with `total_size` kwarg
        z_raw = pm.Normal.dist(y, shape=batch_size)
        mb_z = create_minibatch_rv(z_raw, total_size=data.shape)
        m_old.register_rv(mb_z, name="mb_z")

    mb_data = pm.Minibatch(data, batch_size=batch_size)
    m_new = observe(m_old, {mb_z: mb_data})

    assert len(m_new.free_RVs) == 2
    assert len(m_new.observed_RVs) == 1
    assert m_new["x"] in m_new.free_RVs
    assert m_new["y"] in m_new.free_RVs
    assert m_new["mb_z"] in m_new.observed_RVs

    np.testing.assert_allclose(
        m_old.compile_logp()({"x": 0.9, "y": 0.5, "mb_z": np.zeros(10)}),
        m_new.compile_logp()({"x": 0.9, "y": 0.5}),
    )


def test_observe_deterministic():
    y_censored_obs = np.array([0.9, 0.5, 0.3, 1, 1], dtype=config.floatX)

    with pm.Model() as m_old:
        x = pm.Normal("x")
        y = pm.Normal.dist(x, shape=(5,))
        y_censored = pm.Deterministic("y_censored", pm.math.clip(y, -1, 1))

    m_new = observe(m_old, {y_censored: y_censored_obs})

    with pm.Model() as m_ref:
        x = pm.Normal("x")
        pm.Censored("y_censored", pm.Normal.dist(x), lower=-1, upper=1, observed=y_censored_obs)


def test_observe_dims():
    with pm.Model(coords={"test_dim": range(5)}) as m_old:
        x = pm.Normal("x", dims="test_dim")

    m_new = observe(m_old, {x: np.arange(5, dtype=config.floatX)})
    assert m_new.named_vars_to_dims["x"] == ["test_dim"]


def test_do():
    rng = np.random.default_rng(seed=435)
    with pm.Model() as m_old:
        x = pm.Normal("x", 0, 1e-3)
        y = pm.Normal("y", x, 1e-3)
        z = pm.Normal("z", y + x, 1e-3)

    assert -5 < pm.draw(z, random_seed=rng) < 5

    m_new = do(m_old, {y: x + 100})

    assert len(m_new.free_RVs) == 2
    assert m_new["x"] in m_new.free_RVs
    assert m_new["y"] in m_new.deterministics
    assert m_new["z"] in m_new.free_RVs

    assert 95 < pm.draw(m_new["z"], random_seed=rng) < 105

    # Test two substitutions
    with m_old:
        switch = pm.MutableData("switch", 1)
    m_new = do(m_old, {y: 100 * switch, x: 100 * switch})

    assert len(m_new.free_RVs) == 1
    assert m_new["y"] not in m_new.deterministics
    assert m_new["x"] not in m_new.deterministics
    assert m_new["z"] in m_new.free_RVs

    assert 195 < pm.draw(m_new["z"], random_seed=rng) < 205
    with m_new:
        pm.set_data({"switch": 0})
    assert -5 < pm.draw(m_new["z"], random_seed=rng) < 5


def test_do_posterior_predictive():
    with pm.Model() as m:
        x = pm.Normal("x", 0, 1)
        y = pm.Normal("y", x, 1)
        z = pm.Normal("z", y + x, 1e-3)

    # Dummy posterior
    idata_m = az.from_dict(
        {
            "x": np.full((2, 500), 25),
            "y": np.full((2, 500), np.nan),
            "z": np.full((2, 500), np.nan),
        }
    )

    # Replace `y` by a constant `100.0`
    m_do = do(m, {y: 100.0})
    with m_do:
        idata_do = pm.sample_posterior_predictive(idata_m, var_names="z")

    assert 120 < idata_do.posterior_predictive["z"].mean() < 130


@pytest.mark.parametrize("mutable", (False, True))
def test_do_constant(mutable):
    rng = np.random.default_rng(seed=122)
    with pm.Model() as m:
        x = pm.Data("x", 0, mutable=mutable)
        y = pm.Normal("y", x, 1e-3)

    do_m = do(m, {x: 105})
    assert pm.draw(do_m["y"], random_seed=rng) > 100


def test_do_deterministic():
    rng = np.random.default_rng(seed=435)
    with pm.Model() as m:
        x = pm.Normal("x", 0, 1e-3)
        y = pm.Deterministic("y", x + 105)
        z = pm.Normal("z", y, 1e-3)

    do_m = do(m, {"z": x - 105})
    assert pm.draw(do_m["z"], random_seed=rng) < 100


def test_do_dims():
    coords = {"test_dim": range(10)}
    with pm.Model(coords=coords) as m:
        x = pm.Normal("x", dims="test_dim")
        y = pm.Deterministic("y", x + 5, dims="test_dim")

    do_m = do(
        m,
        {"x": np.zeros(10, dtype=config.floatX)},
    )
    assert do_m.named_vars_to_dims["x"] == ["test_dim"]

    do_m = do(
        m,
        {"y": np.zeros(10, dtype=config.floatX)},
    )
    assert do_m.named_vars_to_dims["y"] == ["test_dim"]


@pytest.mark.parametrize("prune", (False, True))
def test_do_prune(prune):

    with pm.Model() as m:
        x0 = pm.ConstantData("x0", 0)
        x1 = pm.ConstantData("x1", 0)
        y = pm.Normal("y")
        y_det = pm.Deterministic("y_det", y + x0)
        z = pm.Normal("z", y_det)
        llike = pm.Normal("llike", z + x1, observed=0)

    orig_named_vars = {"x0", "x1", "y", "y_det", "z", "llike"}
    assert set(m.named_vars) == orig_named_vars

    do_m = do(m, {y_det: x0 + 5}, prune_vars=prune)
    if prune:
        assert set(do_m.named_vars) == {"x0", "x1", "y_det", "z", "llike"}
    else:
        assert set(do_m.named_vars) == orig_named_vars

    do_m = do(m, {z: 0.5}, prune_vars=prune)
    if prune:
        assert set(do_m.named_vars) == {"x1", "z", "llike"}
    else:
        assert set(do_m.named_vars) == orig_named_vars
