from typing import Literal, Sequence, Tuple, Union

import pytensor
from pymc import SymbolicRandomVariable
from pytensor import Variable
from pytensor.graph import Constant, Type
from pytensor.graph.basic import walk
from pytensor.graph.op import HasInnerGraph
from pytensor.tensor import TensorVariable
from pytensor.tensor.random.op import RandomVariable


class StringType(Type[str]):
    def clone(self, **kwargs):
        return type(self)()

    def filter(self, x, strict=False, allow_downcast=None):
        if isinstance(x, str):
            return x
        else:
            raise TypeError("Expected a string!")

    def __str__(self):
        return "string"

    @staticmethod
    def may_share_memory(a, b):
        return isinstance(a, str) and a is b


stringtype = StringType()


class StringConstant(Constant):
    pass


@pytensor._as_symbolic.register(str)
def as_symbolic_string(x, **kwargs):
    return StringConstant(stringtype, x)


def rvs_in_graph(vars: Sequence[Variable]) -> bool:
    """Check if there are any rvs in the graph of vars"""

    def expand(r):
        owner = r.owner
        if owner:
            inputs = list(reversed(owner.inputs))

            if isinstance(owner.op, HasInnerGraph):
                inputs += owner.op.inner_outputs

            return inputs

    return any(
        node
        for node in walk(vars, expand, False)
        if node.owner and isinstance(node.owner.op, (RandomVariable, SymbolicRandomVariable))
    )


def named_shuffle_pattern(
    input_dims: Tuple[Union[int, str], ...],
    output_dims: Tuple[Union[int, str, None], ...],
) -> Tuple[Union[int, Literal["x"]], ...]:
    if not (set(output_dims) - {None}).issuperset(input_dims):
        raise ValueError(f"Can't arrange {input_dims} to {output_dims}")
    else:
        # mypy complains about None type so I just added it to annotations
        maps: dict[int | str | None, int] = dict(zip(input_dims, range(len(input_dims))))
        return tuple(maps.get(d, "x") for d in output_dims)


def shuffle_named_tensor(
    tensor: TensorVariable,
    input_dims: Tuple[str, ...],
    output_dims: Tuple[str, ...],
) -> TensorVariable:
    """Shuffle tensor using annotated dims.

    Parameters
    ----------
    tensor : pt.TensorVariable
    input_dims : tuple[str, ...]
    output_dims : tuple[str, ...]

    Returns
    -------
    pt.TensorVariable
    """
    int_dims = tuple(range(0, tensor.ndim - len(input_dims)))
    in_dims = int_dims + input_dims
    out_dims = int_dims + output_dims
    pattern = named_shuffle_pattern(in_dims, out_dims)
    return tensor.dimshuffle(*pattern)
