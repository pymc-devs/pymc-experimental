import numpy as np
import pymc as pm
import pytest

from pymc_experimental.model.transforms.autoreparam import vip_reparametrize


@pytest.fixture
def model_c():
    with pm.Model() as mod:
        m = pm.Normal("m")
        s = pm.LogNormal("s")
        pm.Normal("g", m, s, shape=5)
    return mod


@pytest.fixture
def model_nc():
    with pm.Model() as mod:
        m = pm.Normal("m")
        s = pm.LogNormal("s")
        pm.Deterministic("g", pm.Normal("z", shape=5) * s + m)
    return mod


def test_reparametrize_created(model_c: pm.Model):
    model_reparam, vip = vip_reparametrize(model_c, ["g"])
    assert "g" in vip.get_lambda()
    assert "_vip::eps" in model_reparam.named_vars
    assert "_vip::round" in model_reparam.named_vars
    assert "g::lam_logit__" in model_reparam.named_vars
    assert "g::tau_" in model_reparam.named_vars
    vip.set_all_lambda(1)
    assert ~np.isfinite(model_reparam["g::lam_logit__"].get_value()).any()


def test_random_draw(model_c: pm.Model, model_nc):
    model_c = pm.do(model_c, {"m": 3, "s": 2})
    model_nc = pm.do(model_nc, {"m": 3, "s": 2})
    model_v, vip = vip_reparametrize(model_c, ["g"])

    c = pm.draw(model_c["g"], random_seed=42, draws=1000)
    nc = pm.draw(model_nc["g"], random_seed=42, draws=1000)
    vip.set_all_lambda(1)
    v_1 = pm.draw(model_v["g"], random_seed=42, draws=1000)
    vip.set_all_lambda(0)
    v_0 = pm.draw(model_v["g"], random_seed=42, draws=1000)
    vip.set_all_lambda(0.5)
    v_05 = pm.draw(model_v["g"], random_seed=42, draws=1000)
    np.testing.assert_allclose(c.mean(), nc.mean())
    np.testing.assert_allclose(c.mean(), v_0.mean())
    np.testing.assert_allclose(v_05.mean(), v_1.mean())
    np.testing.assert_allclose(v_1.mean(), nc.mean())

    np.testing.assert_allclose(c.std(), nc.std())
    np.testing.assert_allclose(c.std(), v_0.std())
    np.testing.assert_allclose(v_05.std(), v_1.std())
    np.testing.assert_allclose(v_1.std(), nc.std())
