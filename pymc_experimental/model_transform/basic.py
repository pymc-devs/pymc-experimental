from pymc import Model
from pytensor.graph import ancestors

from pymc_experimental.utils.model_fgraph import (
    ModelObservedRV,
    ModelVar,
    fgraph_from_model,
    model_from_fgraph,
)


def prune_vars_detached_from_observed(model: Model) -> Model:
    """Prune model variables that are not related to any observed variable in the Model."""

    # Potentials are ambiguous as whether they correspond to likelihood or prior terms,
    # We simply raise for now
    if model.potentials:
        raise NotImplementedError("Pruning not implemented for models with Potentials")

    fgraph, _ = fgraph_from_model(model, inlined_views=True)
    observed_vars = (
        out
        for node in fgraph.apply_nodes
        if isinstance(node.op, ModelObservedRV)
        for out in node.outputs
    )
    ancestor_nodes = {var.owner for var in ancestors(observed_vars)}
    nodes_to_remove = {
        node
        for node in fgraph.apply_nodes
        if isinstance(node.op, ModelVar) and node not in ancestor_nodes
    }
    for node_to_remove in nodes_to_remove:
        fgraph.remove_node(node_to_remove)
    return model_from_fgraph(fgraph)
