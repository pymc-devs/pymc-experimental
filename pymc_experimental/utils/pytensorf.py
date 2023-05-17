from typing import Sequence

import pytensor
from pymc import SymbolicRandomVariable
from pytensor import Variable
from pytensor.graph import Constant, Type
from pytensor.graph.basic import walk
from pytensor.graph.op import HasInnerGraph
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
