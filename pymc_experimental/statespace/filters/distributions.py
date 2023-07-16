import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pymc.distributions.distribution import Distribution, SymbolicRandomVariable
from pymc.distributions.shape_utils import get_support_shape_1d
from pymc.gp.util import stabilize
from pymc.logprob.abstract import _logprob
from pytensor.graph.basic import Node


def step_fn(*args):
    mu, cov, rng = args
    next_rng, predicted_state = pm.MvNormal.dist(mu=mu, cov=stabilize(cov), rng=rng).owner.outputs

    return predicted_state, {rng: next_rng}


class LinearGaussianStateSpaceRV(SymbolicRandomVariable):
    default_output = 1
    _print_name = ("LinearGuassianStateSpace", "\\operatorname{LinearGuassianStateSpace}")

    def update(self, node: Node):
        return {node.inputs[-1]: node.outputs[0]}


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
        rng = pytensor.shared(np.random.default_rng())

        def step_fn(*args):
            a, c, T, R, Q, rng = args
            a_next = c + T @ a
            # P_next = T @ P @ T.T + R @ Q @ R.T
            # P_next = R @ Q @ R.T
            next_rng, innovation = pm.MvNormal.dist(mu=0, cov=Q, rng=rng).owner.outputs
            a_next += R @ innovation

            return a_next, {rng: next_rng}

        statespace, updates = pytensor.scan(
            step_fn, outputs_info=[a0_], non_sequences=[c_, T_, R_, Q_, rng], n_steps=steps_
        )

        statespace_ = pt.concatenate([a0_[None], statespace], axis=0)

        (ss_rng,) = tuple(updates.values())
        linear_gaussian_ss_op = LinearGaussianStateSpaceRV(
            inputs=[a0_, P0_, c_, d_, T_, Z_, R_, H_, Q_, steps_],
            outputs=[ss_rng, statespace_],
            ndim_supp=1,
        )

        linear_gaussian_ss = linear_gaussian_ss_op(a0, P0, c, d, T, Z, R, H, Q, steps)
        return linear_gaussian_ss


@_logprob.register(LinearGaussianStateSpaceRV)
def linear_guassian_logp(op, values, *args, **kwargs):
    logp = pt.constant(0, "linear_guassian_logp")
    return logp


class FilteredStateSpaceRV(SymbolicRandomVariable):
    default_output = 1
    _print_name = ("FilteredStateSpaceRV", "\\operatorname{FilteredStateSpace}")

    def update(self, node: Node):
        return {node.inputs[-1]: node.outputs[0]}


class FilteredStateSpace(Distribution):
    rv_op = FilteredStateSpaceRV

    def __new__(cls, *args, **kwargs):
        steps = get_support_shape_1d(
            support_shape=None,
            shape=None,
            dims=kwargs.get("dims", None),
            observed=kwargs.get("observed", None),
            support_shape_offset=0,
        )

        return super().__new__(cls, *args, **kwargs)

    @classmethod
    def dist(cls, mus, covs, **kwargs):
        steps = get_support_shape_1d(
            support_shape=None, shape=kwargs.get("shape", None), support_shape_offset=0
        )

        return super().dist([mus, covs], **kwargs)

    @classmethod
    def rv_op(cls, mus, covs, size=None):
        mus_, covs_ = mus.type(), covs.type()
        rng = pytensor.shared(np.random.default_rng())

        filtered_states, updates = pytensor.scan(
            step_fn, sequences=[mus_, covs_], non_sequences=[rng]
        )

        (filtered_rng,) = tuple(updates.values())
        filtered_state_space_op = LinearGaussianStateSpaceRV(
            inputs=[mus_, covs_],
            outputs=[filtered_rng, filtered_states],
            ndim_supp=1,
        )

        filtered_state_space = filtered_state_space_op(mus, covs)
        return filtered_state_space


@_logprob.register(FilteredStateSpaceRV)
def filtered_ss_logp(op, values, *args, **kwargs):
    logp = pt.constant(0, "filtered_ss_logp")
    return logp


class PredictedStateSpaceRV(SymbolicRandomVariable):
    default_output = 1
    _print_name = ("PredictedStateSpaceRV", "\\operatorname{PredictedStateSpace}")

    def update(self, node: Node):
        return {node.inputs[-1]: node.outputs[0]}


class PredictedStateSpace(Distribution):
    rv_op = PredictedStateSpaceRV

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
    def dist(cls, mus, covs, logp, steps=None, **kwargs):

        steps = get_support_shape_1d(
            support_shape=steps, shape=kwargs.get("shape", None), support_shape_offset=0
        )

        if steps is None:
            raise ValueError("Must specify steps or shape parameter")

        return super().dist([mus, covs, logp, steps], **kwargs)

    @classmethod
    def rv_op(cls, mus, covs, logp, steps, size=None):
        mus_, covs_ = mus.type(), covs.type()
        steps_ = steps.type()
        rng = pytensor.shared(np.random.default_rng())

        predicted_states, updates = pytensor.scan(
            step_fn, sequences=[mus_, covs_], non_sequences=[rng]
        )

        (predicted_rng,) = tuple(updates.values())
        predicted_state_space_op = PredictedStateSpaceRV(
            inputs=[mus_, covs_, steps_],
            outputs=[predicted_rng, predicted_states],
            ndim_supp=1,
        )

        predicted_state_space = predicted_state_space_op(mus, covs, steps)
        return predicted_state_space


@_logprob.register(PredictedStateSpaceRV)
def predicted_ss_logp(op, values, mus, covs, logp, rng, **kwargs):
    return logp


class SmoothedStateSpaceRV(SymbolicRandomVariable):
    default_output = 1
    _print_name = ("SmoothedStateSpaceRV", "\\operatorname{SmoothedStateSpace}")

    def update(self, node: Node):
        return {node.inputs[-1]: node.outputs[0]}
