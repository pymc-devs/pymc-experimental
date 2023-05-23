import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
from pytensor.graph import Constant, FunctionGraph, node_rewriter
from pytensor.graph.rewriting.basic import in2out
from pytensor.tensor.exceptions import NotScalarConstantError

from pymc_experimental.utils.model_fgraph import (
    ModelDeterministic,
    ModelFreeRV,
    ModelNamed,
    ModelObservedRV,
    ModelPotential,
    ModelVar,
    fgraph_from_model,
    model_deterministic,
    model_free_rv,
    model_from_fgraph,
)


def test_basic():
    """Test we can convert from a PyMC Model to a FunctionGraph and back"""
    with pm.Model(coords={"test_dim": range(3)}) as m_old:
        x = pm.Normal("x")
        y = pm.Deterministic("y", x + 1)
        w = pm.HalfNormal("w", pm.math.exp(y))
        z = pm.Normal("z", y, w, observed=[0, 1, 2], dims=("test_dim",))
        pot = pm.Potential("pot", x * 2)

    m_fgraph, memo = fgraph_from_model(m_old)
    assert isinstance(m_fgraph, FunctionGraph)

    assert isinstance(memo[x].owner.op, ModelFreeRV)
    assert isinstance(memo[y].owner.op, ModelDeterministic)
    assert isinstance(memo[w].owner.op, ModelFreeRV)
    assert isinstance(memo[z].owner.op, ModelObservedRV)
    assert isinstance(memo[pot].owner.op, ModelPotential)

    m_new = model_from_fgraph(m_fgraph)
    assert isinstance(m_new, pm.Model)

    assert m_new.coords == {"test_dim": tuple(range(3))}
    assert m_new._dim_lengths["test_dim"].eval() == 3
    assert m_new.named_vars_to_dims == {"z": ["test_dim"]}

    named_vars = {"x", "y", "w", "z", "pot"}
    assert set(m_new.named_vars) == named_vars
    for named_var in named_vars:
        assert m_new[named_var] is not m_old[named_var]
    for value_new, value_old in zip(m_new.rvs_to_values.values(), m_old.rvs_to_values.values()):
        # Constants are not cloned
        if not isinstance(value_new, Constant):
            assert value_new is not value_old
    assert m_new["x"] in m_new.free_RVs
    assert m_new["w"] in m_new.free_RVs
    assert m_new["y"] in m_new.deterministics
    assert m_new["z"] in m_new.observed_RVs
    assert m_new["pot"] in m_new.potentials
    assert m_new.rvs_to_transforms[m_new["x"]] is None
    assert m_new.rvs_to_transforms[m_new["w"]] is pm.distributions.transforms.log
    assert m_new.rvs_to_transforms[m_new["z"]] is None

    # Test random
    new_y_draw, new_z_draw = pm.draw([m_new["y"], m_new["z"]], draws=5, random_seed=1)
    old_y_draw, old_z_draw = pm.draw([m_old["y"], m_old["z"]], draws=5, random_seed=1)
    np.testing.assert_array_equal(new_y_draw, old_y_draw)
    np.testing.assert_array_equal(new_z_draw, old_z_draw)

    # Test logp
    ip = m_new.initial_point()
    np.testing.assert_equal(
        m_new.compile_logp()(ip),
        m_old.compile_logp()(ip),
    )


def test_data():
    """Test shared RNGs, MutableData, ConstantData and Dim lengths are handled correctly.

    Everything should be preserved across new and old models, except for shared RNGs
    """
    with pm.Model(coords_mutable={"test_dim": range(3)}) as m_old:
        x = pm.MutableData("x", [0.0, 1.0, 2.0], dims=("test_dim",))
        y = pm.MutableData("y", [10.0, 11.0, 12.0], dims=("test_dim",))
        b0 = pm.ConstantData("b0", 0.0)
        b1 = pm.Normal("b1")
        mu = pm.Deterministic("mu", b0 + b1 * x, dims=("test_dim",))
        obs = pm.Normal("obs", mu, sigma=1e-5, observed=y, dims=("test_dim",))

    m_fgraph, memo = fgraph_from_model(m_old)
    assert isinstance(memo[x].owner.op, ModelNamed)
    assert isinstance(memo[y].owner.op, ModelNamed)
    assert isinstance(memo[b0].owner.op, ModelNamed)

    m_new = model_from_fgraph(m_fgraph)

    # ConstantData is preserved
    assert m_new["b0"].data == m_old["b0"].data

    # Shared non-rng shared variables are preserved
    assert m_new["x"].container is x.container
    assert m_new["y"].container is y.container
    assert m_new.rvs_to_values[m_new["obs"]] is m_new["y"]

    # Shared rng shared variables are not preserved
    m_new["b1"].owner.inputs[0].container is not m_old["b1"].owner.inputs[0].container

    with m_old:
        pm.set_data({"x": [100.0, 200.0]}, coords={"test_dim": range(2)})

    assert m_new.dim_lengths["test_dim"].eval() == 2
    np.testing.assert_array_almost_equal(pm.draw(m_new["x"]), [100.0, 200.0])


def test_deterministics():
    """Test handling of deterministics.

    We don't want Deterministics in the middle of the FunctionGraph, as they would make rewrites cumbersome
    However we want them in the middle of Model.basic_RVs, so they display nicely in graphviz

    There is one edge case that has to be considered, when a Deterministic is just a copy of a RV.
    In that case we don't bother to reintroduce it in between other Model.basic_RVs
    """
    with pm.Model() as m:
        x = pm.Normal("x")
        mu = pm.Deterministic("mu", pm.math.abs(x))
        sigma = pm.math.exp(x)
        pm.Deterministic("sigma", sigma)
        y = pm.Normal("y", mu, sigma)
        # Special case where the Deterministic
        # is a direct view on another model variable
        y_ = pm.Deterministic("y_", y)
        # Just for kicks, make it a double one!
        y__ = pm.Deterministic("y__", y_)
        z = pm.Normal("z", y__)

    # Deterministic mu is in the graph of x to y but not sigma
    assert m["y"].owner.inputs[3] is m["mu"]
    assert m["y"].owner.inputs[4] is not m["sigma"]

    fg, _ = fgraph_from_model(m)

    # Check that no Deterministics are in graph of x to y and y to z
    x, y, z, det_mu, det_sigma, det_y_, det_y__ = fg.outputs
    # [Det(mu), Det(sigma)]
    mu = det_mu.owner.inputs[0]
    sigma = det_sigma.owner.inputs[0]
    # [FreeRV(y(mu, sigma))] not [FreeRV(y(Det(mu), Det(sigma)))]
    assert y.owner.inputs[0].owner.inputs[3] is mu
    assert y.owner.inputs[0].owner.inputs[4] is sigma
    # [FreeRV(z(y))] not [FreeRV(z(Det(Det(y))))]
    assert z.owner.inputs[0].owner.inputs[3] is y
    # [Det(y), Det(y)], not [Det(y), Det(Det(y))]
    assert det_y_.owner.inputs[0] is y
    assert det_y__.owner.inputs[0] is y
    assert det_y_ is not det_y__

    # Both mu and sigma deterministics are now in the graph of x to y
    m = model_from_fgraph(fg)
    assert m["y"].owner.inputs[3] is m["mu"]
    assert m["y"].owner.inputs[4] is m["sigma"]
    # But not y_* in y to z, since there was no real Op in between
    assert m["z"].owner.inputs[3] is m["y"]
    assert m["y_"].owner.inputs[0] is m["y"]
    assert m["y__"].owner.inputs[0] is m["y"]


def test_context_error():
    """Test that model_from_fgraph fails when called inside a Model context.

    We can't allow it, because the new Model that's returned would be a child of whatever Model context is active.
    """
    with pm.Model() as m:
        x = pm.Normal("x")

        fg = fgraph_from_model(m)

        with pytest.raises(RuntimeError, match="cannot be called inside a PyMC model context"):
            model_from_fgraph(fg)


def test_sub_model_error():
    """Test Error is raised when trying to convert a sub-model to fgraph."""
    with pm.Model() as m:
        x = pm.Beta("x", 1, 1)
        with pm.Model() as sub_m:
            y = pm.Normal("y", x)

    nodes = [v for v in fgraph_from_model(m)[0].toposort() if not isinstance(v.op, ModelVar)]
    assert len(nodes) == 2
    assert isinstance(nodes[0].op, pm.Beta)
    assert isinstance(nodes[1].op, pm.Normal)

    with pytest.raises(ValueError, match="Nested sub-models cannot be converted"):
        fgraph_from_model(sub_m)


@pytest.fixture()
def non_centered_rewrite():
    @node_rewriter(tracks=[ModelFreeRV])
    def non_centered_param(fgraph: FunctionGraph, node):
        """Rewrite that replaces centered normal by non-centered parametrization."""

        rv, value, *dims = node.inputs
        if not isinstance(rv.owner.op, pm.Normal):
            return
        rng, size, dtype, loc, scale = rv.owner.inputs

        # Only apply rewrite if size information is explicit
        if size.ndim == 0:
            return None

        try:
            is_unit = (
                pt.get_underlying_scalar_constant_value(loc) == 0
                and pt.get_underlying_scalar_constant_value(scale) == 1
            )
        except NotScalarConstantError:
            is_unit = False

        # Nothing to do here
        if is_unit:
            return

        raw_norm = pm.Normal.dist(0, 1, size=size, rng=rng)
        raw_norm.name = f"{rv.name}_raw_"
        raw_norm_value = raw_norm.clone()
        fgraph.add_input(raw_norm_value)
        raw_norm = model_free_rv(raw_norm, raw_norm_value, node.op.transform, *dims)

        new_norm = loc + raw_norm * scale
        new_norm.name = rv.name
        new_norm_det = model_deterministic(new_norm, *dims)
        fgraph.add_output(new_norm_det)

        return [new_norm]

    return in2out(non_centered_param)


def test_fgraph_rewrite(non_centered_rewrite):
    """Test we can apply a simple rewrite to a PyMC Model."""

    with pm.Model(coords={"subject": range(10)}) as m_old:
        group_mean = pm.Normal("group_mean")
        group_std = pm.HalfNormal("group_std")
        subject_mean = pm.Normal("subject_mean", group_mean, group_std, dims=("subject",))
        obs = pm.Normal("obs", subject_mean, 1, observed=np.zeros(10), dims=("subject",))

    fg, _ = fgraph_from_model(m_old)
    non_centered_rewrite.apply(fg)

    m_new = model_from_fgraph(fg)
    assert m_new.named_vars_to_dims == {
        "subject_mean": ["subject"],
        "subject_mean_raw_": ["subject"],
        "obs": ["subject"],
    }
    assert set(m_new.named_vars) == {
        "group_mean",
        "group_std",
        "subject_mean_raw_",
        "subject_mean",
        "obs",
    }
    assert {rv.name for rv in m_new.free_RVs} == {"group_mean", "group_std", "subject_mean_raw_"}
    assert {rv.name for rv in m_new.observed_RVs} == {"obs"}
    assert {rv.name for rv in m_new.deterministics} == {"subject_mean"}

    with pm.Model() as m_ref:
        group_mean = pm.Normal("group_mean")
        group_std = pm.HalfNormal("group_std")
        subject_mean_raw = pm.Normal("subject_mean_raw_", 0, 1, shape=(10,))
        subject_mean = pm.Deterministic("subject_mean", group_mean + subject_mean_raw * group_std)
        obs = pm.Normal("obs", subject_mean, 1, observed=np.zeros(10))

    np.testing.assert_array_equal(
        pm.draw(m_new["subject_mean_raw_"], draws=7, random_seed=1),
        pm.draw(m_ref["subject_mean_raw_"], draws=7, random_seed=1),
    )

    ip = m_new.initial_point()
    np.testing.assert_equal(
        m_new.compile_logp()(ip),
        m_ref.compile_logp()(ip),
    )
