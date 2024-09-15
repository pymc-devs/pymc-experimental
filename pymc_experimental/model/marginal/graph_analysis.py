from pytensor.compile import SharedVariable
from pytensor.graph import Constant, FunctionGraph, ancestors
from pytensor.tensor import TensorVariable
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.shape import Shape


def static_shape_ancestors(vars):
    """Identify ancestors Shape Ops of static shapes (therefore constant in a valid graph)."""
    return [
        var
        for var in ancestors(vars)
        if (
            var.owner
            and isinstance(var.owner.op, Shape)
            # All static dims lengths of Shape input are known
            and None not in var.owner.inputs[0].type.shape
        )
    ]


def find_conditional_input_rvs(output_rvs, all_rvs):
    """Find conditionally indepedent input RVs."""
    blockers = [other_rv for other_rv in all_rvs if other_rv not in output_rvs]
    blockers += static_shape_ancestors(tuple(all_rvs) + tuple(output_rvs))
    return [
        var
        for var in ancestors(output_rvs, blockers=blockers)
        if var in blockers or (var.owner is None and not isinstance(var, Constant | SharedVariable))
    ]


def is_conditional_dependent(
    dependent_rv: TensorVariable, dependable_rv: TensorVariable, all_rvs
) -> bool:
    """Check if dependent_rv is conditionall dependent on dependable_rv,
    given all conditionally independent all_rvs"""

    return dependable_rv in find_conditional_input_rvs((dependent_rv,), all_rvs)


def find_conditional_dependent_rvs(dependable_rv, all_rvs):
    """Find rvs than depend on dependable"""
    return [
        rv
        for rv in all_rvs
        if (rv is not dependable_rv and is_conditional_dependent(rv, dependable_rv, all_rvs))
    ]


def is_elemwise_subgraph(rv_to_marginalize, other_input_rvs, output_rvs):
    # TODO: No need to consider apply nodes outside the subgraph...
    fg = FunctionGraph(outputs=output_rvs, clone=False)

    non_elemwise_blockers = [
        o
        for node in fg.apply_nodes
        if not (
            isinstance(node.op, Elemwise)
            # Allow expand_dims on the left
            or (
                isinstance(node.op, DimShuffle)
                and not node.op.drop
                and node.op.shuffle == sorted(node.op.shuffle)
            )
        )
        for o in node.outputs
    ]
    blocker_candidates = [rv_to_marginalize, *other_input_rvs, *non_elemwise_blockers]
    blockers = [var for var in blocker_candidates if var not in output_rvs]

    truncated_inputs = [
        var
        for var in ancestors(output_rvs, blockers=blockers)
        if (
            var in blockers
            or (var.owner is None and not isinstance(var, Constant | SharedVariable))
        )
    ]

    # Check that we reach the marginalized rv following a pure elemwise graph
    if rv_to_marginalize not in truncated_inputs:
        return False

    # Check that none of the truncated inputs depends on the marginalized_rv
    other_truncated_inputs = [inp for inp in truncated_inputs if inp is not rv_to_marginalize]
    # TODO: We don't need to go all the way to the root variables
    if rv_to_marginalize in ancestors(
        other_truncated_inputs, blockers=[rv_to_marginalize, *other_input_rvs]
    ):
        return False
    return True
