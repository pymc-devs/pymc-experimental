from typing import Dict, Optional, Sequence, Tuple

import pytensor
from pymc.logprob.transforms import RVTransform
from pymc.model import Model
from pymc.pytensorf import find_rng_nodes
from pytensor import Variable
from pytensor.graph import Apply, FunctionGraph, Op, node_rewriter
from pytensor.graph.rewriting.basic import out2in
from pytensor.scalar import Identity
from pytensor.tensor.elemwise import Elemwise

from pymc_experimental.utils.pytensorf import StringType


class ModelVar(Op):
    """A dummy Op that describes the purpose of a Model variable and contains
    meta-information as additional inputs (value and dims).
    """

    def make_node(self, rv, *dims):
        assert isinstance(rv, Variable)
        dims = self._parse_dims(rv, *dims)
        return Apply(self, [rv, *dims], [rv.type(name=rv.name)])

    def _parse_dims(self, rv, *dims):
        if dims:
            dims = [pytensor.as_symbolic(dim) for dim in dims]
            assert all(isinstance(dim.type, StringType) for dim in dims)
            assert len(dims) == rv.type.ndim
        return dims

    def infer_shape(self, fgraph, node, inputs_shape):
        return [inputs_shape[0]]

    def do_constant_folding(self, fgraph, node):
        return False

    def perform(self, *args, **kwargs):
        raise RuntimeError("ModelVars should never be in a final graph!")


class ModelValuedVar(ModelVar):

    __props__ = ("transform",)

    def __init__(self, transform: Optional[RVTransform] = None):
        if transform is not None and not isinstance(transform, RVTransform):
            raise TypeError(f"transform must be None or RVTransform type, got {type(transform)}")
        self.transform = transform
        super().__init__()

    def make_node(self, rv, value, *dims):
        assert isinstance(rv, Variable)
        dims = self._parse_dims(rv, *dims)
        if value is not None:
            assert isinstance(value, Variable)
            assert rv.type.in_same_class(value.type)
            return Apply(self, [rv, value, *dims], [rv.type(name=rv.name)])


class ModelFreeRV(ModelValuedVar):
    pass


class ModelObservedRV(ModelValuedVar):
    pass


class ModelPotential(ModelVar):
    pass


class ModelDeterministic(ModelVar):
    pass


class ModelNamed(ModelVar):
    pass


def model_free_rv(rv, value, transform, *dims):
    return ModelFreeRV(transform=transform)(rv, value, *dims)


model_observed_rv = ModelObservedRV()
model_potential = ModelPotential()
model_deterministic = ModelDeterministic()
model_named = ModelNamed()


def toposort_replace(
    fgraph: FunctionGraph, replacements: Sequence[Tuple[Variable, Variable]]
) -> None:
    """Replace multiple variables in topological order."""
    toposort = fgraph.toposort()
    sorted_replacements = sorted(
        replacements, key=lambda pair: toposort.index(pair[0].owner) if pair[0].owner else -1
    )
    fgraph.replace_all(tuple(sorted_replacements), import_missing=True)


@node_rewriter([Elemwise])
def local_remove_identity(fgraph, node):
    if isinstance(node.op.scalar_op, Identity):
        return [node.inputs[0]]


remove_identity_rewrite = out2in(local_remove_identity)


def fgraph_from_model(model: Model) -> Tuple[FunctionGraph, Dict[Variable, Variable]]:
    """Convert Model to FunctionGraph.

    See: model_from_fgraph

    Returns
    -------
    fgraph: FunctionGraph
        FunctionGraph that includes a copy of model variables, wrapped in dummy `ModelVar` Ops.
        It should be possible to reconstruct a valid PyMC model using `model_from_fgraph`.

    memo: Dict
        A dictionary mapping original model variables to the equivalent nodes in the fgraph.
    """

    if any(v is not None for v in model.rvs_to_initial_values.values()):
        raise NotImplementedError("Cannot convert models with non-default initial_values")

    if model.parent is not None:
        raise ValueError(
            "Nested sub-models cannot be converted to fgraph. Convert the parent model instead"
        )

    # Collect PyTensor variables
    rvs_to_values = model.rvs_to_values
    rvs = list(rvs_to_values.keys())
    free_rvs = model.free_RVs
    observed_rvs = model.observed_RVs
    potentials = model.potentials
    # We copy Deterministics (Identity Op) so that they don't show in between "main" variables
    # We later remove these Identity Ops when we have a Deterministic ModelVar Op as a separator
    old_deterministics = model.deterministics
    deterministics = [det.copy(det.name) for det in old_deterministics]
    # Other variables that are in model.named_vars but are not any of the categories above
    # E.g., MutableData, ConstantData, _dim_lengths
    # We use the same trick as deterministics!
    accounted_for = free_rvs + observed_rvs + potentials + old_deterministics
    old_other_named_vars = [var for var in model.named_vars.values() if var not in accounted_for]
    other_named_vars = [var.copy(var.name) for var in old_other_named_vars]
    value_vars = [val for val in rvs_to_values.values() if val not in old_other_named_vars]

    model_vars = rvs + potentials + deterministics + other_named_vars + value_vars

    memo = {}

    # Replace RNG nodes so that seeding does not interfere with old model
    for rng in find_rng_nodes(model_vars):
        new_rng = rng.clone()
        new_rng.set_value(rng.get_value(borrow=False))
        memo[rng] = new_rng

    fgraph = FunctionGraph(
        outputs=model_vars,
        clone=True,
        memo=memo,
        copy_orphans=True,
        copy_inputs=True,
    )
    # Copy model meta-info to fgraph
    fgraph._coords = model._coords.copy()
    fgraph._dim_lengths = model._dim_lengths.copy()

    rvs_to_transforms = model.rvs_to_transforms
    named_vars_to_dims = model.named_vars_to_dims

    # Introduce dummy `ModelVar` Ops
    free_rvs_to_transforms = {memo[k]: tr for k, tr in rvs_to_transforms.items()}
    free_rvs_to_values = {memo[k]: memo[v] for k, v in rvs_to_values.items() if k in free_rvs}
    observed_rvs_to_values = {
        memo[k]: memo[v] for k, v in rvs_to_values.items() if k in observed_rvs
    }
    potentials = [memo[k] for k in potentials]
    deterministics = [memo[k] for k in deterministics]
    other_named_vars = [memo[k] for k in other_named_vars]

    vars = fgraph.outputs
    new_vars = []
    for var in vars:
        dims = named_vars_to_dims.get(var.name, ())
        if var in free_rvs_to_values:
            new_var = model_free_rv(
                var, free_rvs_to_values[var], free_rvs_to_transforms[var], *dims
            )
        elif var in observed_rvs_to_values:
            new_var = model_observed_rv(var, observed_rvs_to_values[var], *dims)
        elif var in potentials:
            new_var = model_potential(var, *dims)
        elif var in deterministics:
            new_var = model_deterministic(var, *dims)
        elif var in other_named_vars:
            new_var = model_named(var, *dims)
        else:
            # Value variables
            new_var = var
        new_vars.append(new_var)

    replacements = tuple(zip(vars, new_vars))
    toposort_replace(fgraph, replacements)

    # Reference model vars in memo
    inverse_memo = {v: k for k, v in memo.items()}
    for var, model_var in replacements:
        if isinstance(
            model_var.owner is not None and model_var.owner.op, (ModelDeterministic, ModelNamed)
        ):
            # Ignore extra identity that will be removed at the end
            var = var.owner.inputs[0]
        original_var = inverse_memo[var]
        memo[original_var] = model_var

    # Remove value variable as outputs, now that they are graph inputs
    first_value_idx = len(fgraph.outputs) - len(value_vars)
    for _ in value_vars:
        fgraph.remove_output(first_value_idx)

    # Now that we have Deterministic dummy Ops, we remove the noisy `Identity`s from the graph
    remove_identity_rewrite.apply(fgraph)

    return fgraph, memo


def model_from_fgraph(fgraph: FunctionGraph) -> Model:
    """Convert FunctionGraph to PyMC model.

    This requires nodes to be properly tagged with `ModelVar` dummy Ops.

    See: fgraph_from_model
    """
    model = Model()
    if model.parent is not None:
        raise RuntimeError("model_to_fgraph cannot be called inside a PyMC model context")
    model._coords = getattr(fgraph, "_coords", {})
    model._dim_lengths = getattr(fgraph, "_dim_lengths", {})

    # Replace dummy `ModelVar` Ops by the underlying variables,
    # Except for Deterministics which could reintroduce the old graphs
    fgraph = fgraph.clone()
    model_dummy_vars = [
        model_node.outputs[0]
        for model_node in fgraph.toposort()
        if isinstance(model_node.op, ModelVar)
    ]
    model_dummy_vars_to_vars = {
        dummy_var: dummy_var.owner.inputs[0]
        for dummy_var in model_dummy_vars
        # Don't include Deterministics!
        if not isinstance(dummy_var.owner.op, ModelDeterministic)
    }
    toposort_replace(fgraph, tuple(model_dummy_vars_to_vars.items()))

    # Populate new PyMC model mappings
    non_det_model_vars = set(model_dummy_vars_to_vars.values())
    for model_var in model_dummy_vars:
        if isinstance(model_var.owner.op, ModelFreeRV):
            var, value, *dims = model_var.owner.inputs
            transform = model_var.owner.op.transform
            model.free_RVs.append(var)
            # PyMC does not allow setting transform when we pass a value_var. Why?
            model.create_value_var(var, transform=None, value_var=value)
            model.rvs_to_transforms[var] = transform
            model.set_initval(var, initval=None)
        elif isinstance(model_var.owner.op, ModelObservedRV):
            var, value, *dims = model_var.owner.inputs
            model.observed_RVs.append(var)
            model.create_value_var(var, transform=None, value_var=value)
        elif isinstance(model_var.owner.op, ModelPotential):
            var, *dims = model_var.owner.inputs
            model.potentials.append(var)
        elif isinstance(model_var.owner.op, ModelDeterministic):
            var, *dims = model_var.owner.inputs
            # Register the original var (not the copy) as the Deterministic
            # So it shows in the expected place in graphviz.
            # unless it's another model var, in which case we need a copy!
            if var in non_det_model_vars:
                var = var.copy()
            model.deterministics.append(var)
        elif isinstance(model_var.owner.op, ModelNamed):
            var, *dims = model_var.owner.inputs
        else:
            raise TypeError(f"Unexpected ModelVar type {type(model_var)}")

        var.name = model_var.name
        dims = [dim.data for dim in dims] if dims else None
        model.add_named_variable(var, dims=dims)

    return model


def clone_model(model: Model) -> Tuple[Model]:
    """Clone a PyMC model.

    Recreates a PyMC model with clones of the original variables.
    Shared variables will point to the same container but be otherwise different objects.
    Constants are not cloned.


    Examples
    --------

        .. code-block:: python

        import pymc as pm
        from pymc_experimental.utils import clone_model

        with pm.Model() as m:
            p = pm.Beta("p", 1, 1)
            x = pm.Bernoulli("x", p=p, shape=(3,))

        with clone_model(m) as clone_m:
            # Access cloned variables by name
            clone_x = clone_m["x"]

            # z will be part of clone_m but not m
            z = pm.Deterministic("z", clone_x + 1)

    """
    return model_from_fgraph(fgraph_from_model(model)[0])
