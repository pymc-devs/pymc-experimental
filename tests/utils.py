from typing import Sequence

from pytensor.compile import SharedVariable
from pytensor.graph import Constant, graph_inputs
from pytensor.graph.basic import Variable, equal_computations
from pytensor.tensor.random.type import RandomType


def equal_computations_up_to_root(
    xs: Sequence[Variable], ys: Sequence[Variable], ignore_rng_values=True
) -> bool:
    # Check if graphs are equivalent even if root variables have distinct identities

    x_graph_inputs = [var for var in graph_inputs(xs) if not isinstance(var, Constant)]
    y_graph_inputs = [var for var in graph_inputs(ys) if not isinstance(var, Constant)]
    if len(x_graph_inputs) != len(y_graph_inputs):
        return False
    for x, y in zip(x_graph_inputs, y_graph_inputs):
        if x.type != y.type:
            return False
        if x.name != y.name:
            return False
        if isinstance(x, SharedVariable):
            if not isinstance(y, SharedVariable):
                return False
            if isinstance(x.type, RandomType) and ignore_rng_values:
                continue
            if not x.type.values_eq(x.get_value(), y.get_value()):
                return False

    return equal_computations(xs, ys, in_xs=x_graph_inputs, in_ys=y_graph_inputs)
