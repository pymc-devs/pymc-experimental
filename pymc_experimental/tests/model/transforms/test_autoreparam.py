import pymc as pm

from pymc_experimental.model.transforms.autoreparam import vip_reparametrize


def test_reparametrize_created():
    with pm.Model() as model:
        m = pm.Normal("m")
        s = pm.LogNormal("s")
        g = pm.Normal("g", m, s, shape=5)

    model_reparam, vip = vip_reparametrize(model, ["g"])
    assert "g" in vip.get_lambda()
    assert "_vip::eps" in model_reparam.named_vars
    assert "_vip::round" in model_reparam.named_vars
    assert "g::lam_logit__" in model_reparam.named_vars
    assert "g::tau_" in model_reparam.named_vars
