from dataclasses import dataclass
from functools import singledispatch
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import scipy.special
from pymc.logprob.transforms import Transform
from pymc.model.fgraph import (
    ModelDeterministic,
    ModelNamed,
    fgraph_from_model,
    model_deterministic,
    model_free_rv,
    model_from_fgraph,
    model_named,
)
from pymc.pytensorf import toposort_replace
from pytensor.graph.basic import Apply, Variable
from pytensor.tensor.random.op import RandomVariable


@dataclass
class VIP:
    _logit_lambda: Dict[str, pytensor.tensor.sharedvar.TensorSharedVariable]
    eps: pytensor.tensor.sharedvar.TensorSharedVariable
    round: pytensor.tensor.sharedvar.TensorSharedVariable

    def set_eps(self, value: float):
        self.eps.set_value(float(value))

    def set_round(self, value: bool):
        self.round.set_value(bool(value))

    def get_eps(self):
        return self.eps.get_value()

    def get_round(self):
        return self.round.get_value()

    def get_lambda(self) -> Dict[str, np.ndarray]:
        return {
            name: scipy.special.expit(shared.get_value())
            for name, shared in self._logit_lambda.items()
        }

    def set_lambda(self, **kwargs: Dict[str, np.ndarray]):
        for key, value in kwargs.items():
            logit_lam = scipy.special.logit(value)
            shared = self._logit_lambda[key]
            fill = np.full(
                shared.type.shape,
                logit_lam,
            )
            shared.set_value(fill)

    def set_all_lambda(self, value: float):
        logit_lam = scipy.special.logit(value)
        for shared in self._logit_lambda.values():
            fill = np.full(
                shared.type.shape,
                logit_lam,
            )
            shared.set_value(fill)

    def fit(self, *args, **kwargs) -> pm.MeanField:
        kwargs.setdefault("obj_optimizer", pm.adagrad_window(learning_rate=0.1))
        return pm.fit(
            *args,
            more_obj_params=list(self._logit_lambda.values()),
            method="advi",
            **kwargs,
        )


def vip_reparam_node(
    op: RandomVariable,
    node: Apply,
    name: str,
    dims: List[Variable],
    transform: Transform,
    eps: ModelNamed,
    round: ModelNamed,
) -> Tuple[ModelDeterministic, ModelNamed]:
    if not isinstance(node.op, RandomVariable):
        raise TypeError("Op should be RandomVariable type")
    size = node.inputs[1]
    if not isinstance(size, pt.TensorConstant):
        raise ValueError("Size should be static for autoreparameterization.")
    return _vip_reparam_node(
        op,
        node=node,
        name=name,
        dims=dims,
        transform=transform,
        eps=eps,
        round=round,
    )


@singledispatch
def _vip_reparam_node(
    op: RandomVariable,
    node: Apply,
    name: str,
    dims: List[Variable],
    transform: Transform,
    eps: ModelNamed,
    round: ModelNamed,
) -> Tuple[ModelDeterministic, ModelNamed]:
    raise NotImplementedError


@_vip_reparam_node.register
def _(
    op: pm.Normal,
    node: Apply,
    name: str,
    dims: List[Variable],
    transform: Transform,
    eps: ModelNamed,
    round: ModelNamed,
) -> Tuple[ModelDeterministic, ModelNamed]:
    rng, size, _, loc, scale = node.inputs
    logit_lam_ = pytensor.shared(
        np.zeros(size.data),
        shape=size.data,
        name=f"{name}::lam_logit__",
    )
    logit_lam = model_named(logit_lam_, *dims)
    lam = pt.sigmoid(logit_lam)
    nc_cond = pt.and_(pt.lt(lam, eps), pt.eq(round, 1))
    c_cond = pt.and_(pt.gt(lam, 1 - eps), pt.eq(round, 1))

    vip_loc_rv = pt.switch(
        nc_cond,
        0,
        pt.switch(
            c_cond,
            loc,
            lam * loc,
        ),
    )
    vip_scale_rv = pt.switch(
        nc_cond,
        1,
        pt.switch(
            c_cond,
            scale,
            scale**lam,
        ),
    )

    vip_rv_ = pm.Normal.dist(
        vip_loc_rv,
        vip_scale_rv,
        size=size,
        rng=rng,
    )
    vip_rv_.name = f"{name}::tau_"

    vip_rv = model_free_rv(
        vip_rv_,
        vip_rv_.clone(),
        transform,
        *dims,
    )

    vip_rep_ = pt.switch(
        nc_cond,
        loc + vip_rv * scale,
        pt.switch(c_cond, vip_rv, loc + scale ** (1 - lam) * (vip_rv - lam * loc)),
    )
    vip_rep_.name = name

    vip_rep = model_deterministic(vip_rep_, *dims)
    return vip_rep, logit_lam


def vip_reparametrize(
    model: pm.Model,
    var_names: Sequence[str],
) -> Tuple[pm.Model, VIP]:
    if "_vip::eps" in model.named_vars:
        raise ValueError(
            "The model seems to be already auto-reparametrized. This action is done once."
        )
    fmodel, memo = fgraph_from_model(model)
    lambda_names = []
    replacements = []
    eps_ = pytensor.shared(np.array(1e-2, dtype=float), name="_vip::eps")
    eps = model_named(eps_)
    round_ = pytensor.shared(np.array(False, dtype=bool), name="_vip::round")
    round = model_named(round_)
    for name in var_names:
        old = memo[model.named_vars[name]]
        rv, _, *dims = old.owner.inputs
        new, lam = vip_reparam_node(
            rv.owner.op,
            rv.owner,
            name=rv.name,
            dims=dims,
            transform=old.owner.op.transform,
            eps=eps,
            round=round,
        )
        replacements.append((old, new))
        lambda_names.append(lam.name)
    toposort_replace(fmodel, replacements, reverse=True)
    reparam_model = model_from_fgraph(fmodel)
    model_lambdas = {n: reparam_model[l] for l, n in zip(lambda_names, var_names)}
    vip = VIP(model_lambdas, reparam_model[eps.name], reparam_model[round.name])
    return reparam_model, vip
