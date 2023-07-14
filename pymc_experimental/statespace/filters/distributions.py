import pymc as pm
import pytensor
from pymc.distributions.distribution import Distribution, SymbolicRandomVariable
from pymc.distributions.shape_utils import get_support_shape_1d
from pymc.pytensorf import collect_default_updates
from pytensor.graph.basic import Node


class LinearGaussianStateSpaceRV(SymbolicRandomVariable):
    default_output = 1
    _print_name = ("LinearGuassianStateSpace", "\\operatorname{LinearGuassianStateSpace}")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, node: Node):
        return {node.inputs[-1]: node.outputs[0], node.inputs[-2]: node.outputs[1]}


class LinearGaussianStateSpace(Distribution):
    rv_op = LinearGaussianStateSpaceRV

    def __new__(cls, *args, steps=None, **kwargs):
        steps = get_support_shape_1d(
            support_shape=steps,
            shape=None,
            dims=kwargs.get("dims", None),
            observed=kwargs.get("observed", None),
            support_shape_offset=0,
        )

        return super().__new__(cls, *args, steps=steps, **kwargs)

    @classmethod
    def dist(cls, a0, P0, c, d, T, Z, R, H, Q, steps=None, **kwargs):
        steps = get_support_shape_1d(
            support_shape=steps, shape=kwargs.get("shape", None), support_shape_offset=0
        )

        if steps is None:
            raise ValueError("Must specify steps or shape parameter")

        return super().dist([a0, P0, c, d, T, Z, R, H, Q, steps], **kwargs)

    @classmethod
    def rv_op(cls, a0, P0, c, d, T, Z, R, H, Q, steps, size=None):
        if size is not None:
            batch_size = size

        a0_, P0_, c_, d_, T_, Z_, R_, H_, Q_ = map(
            lambda x: x.type(), [a0, P0, c, d, T, Z, R, H, Q]
        )
        steps_ = steps.type()

        def step_fn(*args):
            a, P, c, T, R, Q = args
            a_next = c + T @ a
            P_next = T @ P @ T.T + R @ Q @ R.T

            a_next += pm.MvNormal.dist(mu=0, cov=P_next)
            outputs = (a_next, P_next)

            updates = collect_default_updates(inputs=args, outputs=outputs)

            return outputs, updates

        (statespace, _), updates = pytensor.scan(
            step_fn, outputs_info=[a0_, P0_], non_sequences=[c_, T_, R_, Q_], n_steps=steps_
        )
        (ss_rng,) = tuple(updates.values())
        linear_gaussian_ss_op = LinearGaussianStateSpaceRV(
            inputs=[a0_, P0_, c_, d_, T_, Z_, R_, H_, Q_, steps_],
            outputs=[ss_rng, statespace],
            ndim_supp=1,
        )

        linear_gaussian_ss = linear_gaussian_ss_op(a0, P0, c, d, T, Z, R, H, Q, steps)
        return linear_gaussian_ss
