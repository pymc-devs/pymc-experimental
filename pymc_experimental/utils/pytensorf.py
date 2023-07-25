from collections import deque
from itertools import chain
from typing import Iterable, Sequence, Set, Tuple

import pytensor
from pymc import SymbolicRandomVariable
from pytensor import Variable
from pytensor.graph import Constant, FunctionGraph, Type
from pytensor.graph.basic import Apply, walk
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


def _replace_rebuild_all(
    fgraph: FunctionGraph, replacements: Iterable[Tuple[Variable, Variable]], **kwargs
) -> FunctionGraph:
    """Replace variables and rebuild dependent graph if needed.

    Rebuilding allows for replacements that change the semantics of the graph
    (different types), which may not be possible for all Ops.
    """

    def get_client_nodes(vars) -> Set[Apply]:
        nodes = set()
        d = deque(
            chain.from_iterable(fgraph.clients[var] for var in vars if var in fgraph.variables)
        )
        while d:
            node, _ = d.pop()
            if node in nodes or node == "output":
                continue
            nodes.add(node)
            d.extend(chain.from_iterable(fgraph.clients[out] for out in node.outputs))
        return nodes

    repl_dict = {old: new for old, new in replacements}
    root_nodes = {var.owner for var in repl_dict.keys()}

    # Build sorted queue with all nodes that depend on replaced variables
    topo_order = {node: order for order, node in enumerate(fgraph.toposort())}
    client_nodes = get_client_nodes(repl_dict.keys())
    d = deque(sorted(client_nodes, key=lambda node: topo_order[node]))
    while d:
        node = d.popleft()
        if node in root_nodes:
            continue

        new_inputs = [repl_dict.get(i, i) for i in node.inputs]
        if new_inputs == node.inputs:
            continue

        # Either remake the node or do a simple inplace replacement
        # This property is not yet present in PyTensor
        if getattr(node.op, "_output_type_depends_on_input_value", False):
            remake_node = True
        else:
            remake_node = any(
                not inp.type == new_inp.type for inp, new_inp in zip(node.inputs, new_inputs)
            )

        if remake_node:
            new_node = node.clone_with_new_inputs(new_inputs, strict=False)
            fgraph.import_node(new_node, import_missing=True)
            for out, new_out in zip(node.outputs, new_node.outputs):
                repl_dict[out] = new_out
        else:
            replace = list(zip(node.inputs, new_inputs))
            fgraph.replace_all(replace, import_missing=True)

    # We need special logic for the cases where we had to rebuild the output nodes
    for i, (new_output, old_output) in enumerate(
        zip(
            (repl_dict.get(out, out) for out in fgraph.outputs),
            fgraph.outputs,
        )
    ):
        if new_output is old_output:
            continue
        fgraph.outputs[i] = new_output
        fgraph.import_var(new_output, import_missing=True)
        client = ("output", i)
        fgraph.add_client(new_output, client)
        fgraph.remove_client(old_output, client)
        fgraph.execute_callbacks("on_change_input", "output", i, old_output, new_output)


def toposort_replace(
    fgraph: FunctionGraph,
    replacements: Sequence[Tuple[Variable, Variable]],
    reverse: bool = False,
    rebuild: bool = False,
) -> None:
    """Replace multiple variables in topological order."""
    if rebuild and reverse:
        raise NotImplementedError("reverse rebuild not supported")

    toposort = fgraph.toposort()
    sorted_replacements = sorted(
        replacements,
        key=lambda pair: toposort.index(pair[0].owner) if pair[0].owner else -1,
        reverse=reverse,
    )

    if rebuild:
        if len(replacements) > 1:
            # In this case we need to introduce the replacements inside each other
            # To avoid undoing previous changes
            sorted_replacements = [list(pairs) for pairs in sorted_replacements]
            for i in range(1, len(replacements)):
                # Replace-rebuild each successive replacement with the previous replacements (in topological order)
                temp_fgraph = FunctionGraph(
                    outputs=[repl for _, repl in sorted_replacements[i:]], clone=False
                )
                _replace_rebuild_all(temp_fgraph, replacements=sorted_replacements[:i])
                sorted_replacements[i][1] = temp_fgraph.outputs[0]
        _replace_rebuild_all(fgraph, sorted_replacements, import_missing=True)
    else:
        fgraph.replace_all(sorted_replacements, import_missing=True)
