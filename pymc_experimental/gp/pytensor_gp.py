import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from pymc.logprob.abstract import MeasurableVariable, _get_measurable_outputs
from pytensor.graph.op import Apply, Op


class Cov(Op):
    __props__ = ("fn",)

    def __init__(self, fn):
        self.fn = fn

    def make_node(self, ls):
        ls = pt.as_tensor(ls)
        out = pt.matrix(shape=(None, None))

        return Apply(self, [ls], [out])

    def __call__(self, ls=1.0):
        return super().__call__(ls)

    def perform(self, node, inputs, output_storage):
        raise NotImplementedError("You should convert Cov into a TensorVariable expression!")

    def do_constant_folding(self, fgraph, node):
        return False


class GP(Op):
    __props__ = ("approx",)

    def __init__(self, approx):
        self.approx = approx

    def make_node(self, mean, cov):
        mean = pt.as_tensor(mean)
        cov = pt.as_tensor(cov)

        if not (cov.owner and isinstance(cov.owner.op, Cov)):
            raise ValueError("Second argument should be a Cov output.")

        out = pt.vector(shape=(None,))

        return Apply(self, [mean, cov], [out])

    def perform(self, node, inputs, output_storage):
        raise NotImplementedError("You cannot evaluate a GP, not enough RAM in the Universe.")

    def do_constant_folding(self, fgraph, node):
        return False


class PriorFromGP(Op):
    """This Op will be replaced by the right MvNormal."""

    def make_node(self, gp, x, rng):
        gp = pt.as_tensor(gp)
        if not (gp.owner and isinstance(gp.owner.op, GP)):
            raise ValueError("First argument should be a GP output.")

        # TODO: Assert RNG has the right type
        x = pt.as_tensor(x)
        out = x.type()

        return Apply(self, [gp, x, rng], [out])

    def __call__(self, gp, x, rng=None):
        if rng is None:
            rng = pytensor.shared(np.random.default_rng())
        return super().__call__(gp, x, rng)

    def perform(self, node, inputs, output_storage):
        raise NotImplementedError("You should convert PriorFromGP into a MvNormal!")

    def do_constant_folding(self, fgraph, node):
        return False


cov_op = Cov(fn=pm.gp.cov.ExpQuad)
gp_op = GP("vanilla")
# SymbolicRandomVariable.register(type(gp_op))
prior_from_gp = PriorFromGP()

MeasurableVariable.register(type(prior_from_gp))


@_get_measurable_outputs.register(type(prior_from_gp))
def gp_measurable_outputs(op, node):
    return node.outputs
