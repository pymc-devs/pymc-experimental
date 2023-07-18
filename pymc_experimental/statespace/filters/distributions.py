import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import Distribution, SymbolicRandomVariable
from pymc.distributions.shape_utils import get_support_shape, get_support_shape_1d
from pymc.logprob.abstract import _logprob
from pymc.logprob.basic import logp
from pytensor.graph.basic import Node


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


class SequenceMvNormalRV(SymbolicRandomVariable):
    default_output = 1
    _print_name = ("SequenceMvNormal", "\\operatorname{SequenceMvNormal}")

    def update(self, node: Node):
        return {node.inputs[-1]: node.outputs[0]}


class SequenceMvNormal(Distribution):
    rv_op = SequenceMvNormalRV

    def __new__(cls, *args, **kwargs):
        support_shape = get_support_shape(
            support_shape=None,
            shape=None,
            dims=kwargs.get("dims", None),
            observed=None,
            ndim_supp=2,
        )

        return super().__new__(cls, *args, support_shape=support_shape, **kwargs)

    @classmethod
    def dist(cls, mus, covs, support_shape=None, **kwargs):
        support_shape = get_support_shape(
            support_shape=None,
            shape=None,
            dims=kwargs.get("dims", None),
            observed=kwargs.get("observed", None),
            ndim_supp=2,
        )

        if support_shape is None:
            support_shape = pt.as_tensor_variable(())

        return super().dist([mus, covs, support_shape], **kwargs)

    @classmethod
    def rv_op(cls, mus, covs, support_shape, size=None):
        if size is not None:
            batch_size = size
        else:
            batch_size = support_shape

        steps, mvn_ndim = mus.shape[0], mus.shape[1]
        mus_, covs_, support_shape_ = mus.type(), covs.type(), support_shape.type()
        steps_, mvn_ndim_ = steps.type(), mvn_ndim.type()

        rng = pytensor.shared(np.random.default_rng())

        def step(mu, cov, rng):
            new_rng, mvn = pm.MvNormal.dist(mu=mu, cov=cov, rng=rng, size=batch_size).owner.outputs
            return mvn, {rng: new_rng}

        mvn_seq, updates = pytensor.scan(step, sequences=[mus_, covs_], non_sequences=[rng])
        (seq_mvn_rng,) = tuple(updates.values())

        mvn_seq_op = SequenceMvNormalRV(
            inputs=[mus_, covs_, steps_, mvn_ndim_], outputs=[seq_mvn_rng, mvn_seq], ndim_supp=2
        )

        mvn_seq = mvn_seq_op(mus, covs, steps, mvn_ndim)
        return mvn_seq


@_logprob.register(SequenceMvNormalRV)
def sequence_mvnormal_logp(op, values, mus, covs, steps, mvn_ndim, rng, **kwargs):
    def step_logp(x, mu, cov):
        return logp(pm.MvNormal.dist(mu, cov), x)

    logp_values, updates = pytensor.scan(
        step_logp, sequences=[values[0], mus, covs], n_steps=steps, strict=True
    )

    return check_parameters(
        logp_values,
        pt.eq(values[0].shape[0], steps),
        msg="Observed data and parameters must have the same number of timesteps (dimension 0)",
    )
