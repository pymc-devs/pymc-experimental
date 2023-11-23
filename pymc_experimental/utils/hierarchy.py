from typing import Dict, List, Optional, Tuple, cast

import formulae.terms
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from pymc_experimental.utils.pytensorf import shuffle_named_tensor


def is_simple_formula(formula: formulae.terms.terms.Model) -> bool:
    for t in formula.terms:
        if isinstance(t, formulae.terms.terms.Intercept):
            continue
        if not isinstance(t, formulae.terms.terms.Term):
            return False
        for c in t.components:
            if not isinstance(c, formulae.terms.variable.Variable):
                return False
    return True


def zerosum_hierarchy(
    formula_string: str,
    dims: Tuple[str, ...],
    named: bool = True,
    importance: Optional[Dict[str, pt.TensorLike]] = None,
    default_importance: pt.TensorLike = 1.0,
) -> pt.TensorVariable:
    """Zero Sum Hierarchy.

    Parameters
    ----------
    formula_string : str
        Wilkinson Notation of interactions, passed to formulae.
    dims : tuple[str, ...]
        Expected dims pattern of the resulting tensor.
    named : bool, optional
        If True, the output is wrapped in Deterministic, by default True
    importance : dict[str, TensorLike] | None, optional
        Dict mapping terms to their importance, by default None. Examples:

        * for term `a`, key is `a`
        * for term `a:b`, key is `a_b`
        * for term `a*b`, possible keys is `a`, `b`, `a_b`

        Raises a ValueError if there were unexpected keys

    default_importance : TensorLike, optional
        If term importance is not provided, the default one is used, by default 1.0

    Returns
    -------
    TensorVariable
        Resulting ZeroSum structured tensor with unit variance.
    """
    formula = cast(formulae.terms.terms.Model, formulae.model_description(formula_string))
    if formula.group_terms:
        raise ValueError("Formula should not have any group terms")
    if not is_simple_formula(formula):
        raise ValueError("Formula should only have Variable terms")
    if formula.response:
        out_name = formula.response.term.name
    else:
        raise ValueError("Formula should have named Response")
    if out_name in dims:
        raise ValueError("Named Response should not be in dims")
    if set(dims) != formula.var_names - {out_name}:
        raise ValueError("Dims should match set of formula Variables")
    base_names: List[str] = []
    interactions: List[Tuple[str, ...]] = []
    for term in formula.terms:
        if isinstance(term, formulae.terms.terms.Intercept):
            continue
        zs_dims = tuple(c.name for c in term.components)
        interactions.append(zs_dims)
        base_names.append("_".join(zs_dims))
    importance = importance or {}
    keys_diff = set(importance) - set(base_names)
    if keys_diff:
        raise ValueError(f"There are unexpected importance keys: {keys_diff}")
    if not interactions:
        raise ValueError("Formula should have at least one Term")
    with pm.Model(name=out_name) as model:
        zed: List[pt.TensorVariable] = []
        alphas: List[pt.TensorLike] = []
        for name, zs_dims in zip(base_names, interactions):
            lengths = pt.prod([model.dim_lengths[c] for c in zs_dims])
            z = pm.ZeroSumNormal(
                f"_{name}",
                (lengths / (lengths - 1)) ** 0.5,
                n_zerosum_axes=len(zs_dims),
                dims=zs_dims,
            )
            imp = importance.get(
                "_".join(zs_dims),
                default_importance,
            )
            zed.append(shuffle_named_tensor(z, zs_dims, dims))
            alphas.append(imp)
        weight_dim_name = model.name_for(f"{out_name}_coord")
        model.add_coord(weight_dim_name, base_names)
        if len(zed) > 1:
            weights = (
                pm.Dirichlet(
                    "weight",
                    alphas,
                    dims=weight_dim_name,
                )
                ** 0.5
            )
        else:
            weights = np.array([1.0])
        z = pt.add(*[c * weights[i] for i, c in enumerate(zed)])
    if named:
        z = pm.Deterministic(out_name, z, dims=dims)
    return z
