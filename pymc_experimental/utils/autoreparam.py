from dataclasses import dataclass
from typing import Dict

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import scipy.special
from pymc.model.fgraph import (
    ModelNamed,
    fgraph_from_model,
    model_free_rv,
    model_from_fgraph,
    model_named,
)
from pymc.pytensorf import toposort_replace
from pytensor.graph.basic import Apply


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
                shared.get_value(True).shape,
                logit_lam,
            )
            shared.set_value(fill)

    def set_all_lambda(self, value: float):
        logit_lam = scipy.special.logit(value)
        for shared in self._logit_lambda.values():
            fill = np.full(
                shared.get_value(True).shape,
                logit_lam,
            )
            shared.set_value(fill)

    def fit(self, *args, **kwargs):
        kwargs.setdefault(obj_optimizer=pm.adagrad_window(learning_rate=0.1))
        return pm.fit(
            *args,
            more_obj_params=list(self._logit_lambda_.values()),
            **kwargs,
        )


def vip_reparam_node(
    node: Apply,
    eps: ModelNamed,
    round: ModelNamed,
) -> tuple[ModelNamed, pytensor.tensor.sharedvar.TensorSharedVariable]:
    rv, _, *dims = node.inputs
    rng, size, _, loc, scale = rv.owner.inputs
    if not isinstance(size, pt.TensorConstant):
        raise ValueError("Size should be static for autoreparameterization.")
    if not isinstance(rv.owner.op, pm.Normal):
        raise ValueError("RV should be follow a Normal distribution.")
    logit_lam_ = pytensor.shared(
        np.zeros(size.data),
        shape=size.data,
        name=f"{rv.name}::lam_logit__",
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
    vip_rv_.name = f"{rv.name}::tau_"

    vip_rv = model_free_rv(
        vip_rv_,
        vip_rv_.clone(),
        node.op.transform,
        *dims,
    )

    vip_rep_ = pt.switch(
        nc_cond,
        loc + vip_rv * scale,
        pt.switch(c_cond, vip_rv, loc + scale ** (1 - lam) * (vip_rv - lam * loc)),
    )
    vip_rep_.name = rv.name

    vip_rep = model_named(vip_rep_, *dims)
    return vip_rep, logit_lam_


def vip_reparametrize(
    model: pm.Model,
    var_names: list[str],
) -> tuple[pm.Model, VIP]:
    if "_vip::eps" in model.named_vars:
        raise ValueError(
            "The model seems to be already auto-reparametrized. This action is done once."
        )
    fmodel, memo = fgraph_from_model(model)
    lambda_ = {}
    replacements = []
    eps_ = pytensor.shared(np.array(1e-2, dtype=float), name="_vip::eps")
    eps = model_named(eps_)
    round_ = pytensor.shared(np.array(False, dtype=bool), name="_vip::round")
    round = model_named(round_)
    for name in var_names:
        old = memo[model.named_vars[name]]
        new, lam = vip_reparam_node(old.owner, eps=eps, round=round)
        replacements.append((old, new))
        lambda_[name] = lam
    toposort_replace(fmodel, replacements)
    return model_from_fgraph(fmodel), VIP(lambda_, eps_, round_)
