import pymc as pm

from pymc_experimental.model_transform.basic import prune_vars_detached_from_observed


def test_prune_vars_detached_from_observed():
    with pm.Model() as m:
        obs_data = pm.MutableData("obs_data", 0)
        a0 = pm.ConstantData("a0", 0)
        a1 = pm.Normal("a1", a0)
        a2 = pm.Normal("a2", a1)
        pm.Normal("obs", a2, observed=obs_data)

        d0 = pm.ConstantData("d0", 0)
        d1 = pm.Normal("d1", d0)

    assert set(m.named_vars.keys()) == {"obs_data", "a0", "a1", "a2", "obs", "d0", "d1"}
    pruned_m = prune_vars_detached_from_observed(m)
    assert set(pruned_m.named_vars.keys()) == {"obs_data", "a0", "a1", "a2", "obs"}
