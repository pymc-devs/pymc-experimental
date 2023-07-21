import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pymc import intX
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import Distribution, SymbolicRandomVariable
from pymc.distributions.shape_utils import get_support_shape, get_support_shape_1d
from pymc.gp.util import stabilize
from pymc.logprob.abstract import _logprob

# from pymc.logprob.basic import logp
from pytensor.graph.basic import Node

JITTER_DEFAULT = 1e-8


class LinearGaussianStateSpaceRV(SymbolicRandomVariable):
    default_output = 1
    _print_name = ("LinearGuassianStateSpace", "\\operatorname{LinearGuassianStateSpace}")

    def update(self, node: Node):
        return {node.inputs[-1]: node.outputs[0]}


class _LinearGaussianStateSpace(Distribution):
    rv_op = LinearGaussianStateSpaceRV

    def __new__(
        cls, name, a0, P0, c, d, T, Z, R, H, Q, init_dist=None, steps=None, k_states=None, **kwargs
    ):
        steps = get_support_shape_1d(
            support_shape=steps,
            shape=None,
            dims=kwargs.get("dims", None),
            observed=kwargs.get("observed", None),
            support_shape_offset=0,
        )

        k_states = cls._get_k_states(T)
        return super().__new__(
            cls, name, a0, P0, c, d, T, Z, R, H, Q, steps=steps, k_states=k_states, **kwargs
        )

    @classmethod
    def dist(cls, a0, P0, c, d, T, Z, R, H, Q, init_dist=None, steps=None, k_states=None, **kwargs):
        steps = get_support_shape_1d(
            support_shape=steps, shape=kwargs.get("shape", None), support_shape_offset=0
        )

        k_states = cls._get_k_states(T)

        if steps is None:
            raise ValueError("Must specify steps or shape parameter")
        steps = pt.as_tensor_variable(intX(steps), ndim=0)

        init_dist = pm.MvNormal.dist(a0, P0)
        init_y = pm.MvNormal.dist(Z @ init_dist, stabilize(H))
        init_dist = pt.concatenate([init_dist, init_y], axis=0)

        return super().dist([a0, P0, c, d, T, Z, R, H, Q, init_dist, steps, k_states], **kwargs)

    @classmethod
    def _get_k_states(cls, T):
        return T.shape[0]

    @classmethod
    def rv_op(cls, a0, P0, c, d, T, Z, R, H, Q, init_dist, steps, k_states, size=None):
        if size is not None:
            batch_size = size

        a0_, P0_, c_, d_, T_, Z_, R_, H_, Q_ = map(
            lambda x: x.type(), [a0, P0, c, d, T, Z, R, H, Q]
        )
        steps_ = steps.type()
        init_dist_ = init_dist.type()
        rng = pytensor.shared(np.random.default_rng())

        def step_fn(*args):
            state, c, d, T, Z, R, H, Q, rng = args
            k = T.shape[0]
            a = state[:k]

            a_next = c + T @ a
            y_next = d + Z @ a_next
            middle_rng, a_innovation = pm.MvNormal.dist(
                mu=0, cov=stabilize(Q), rng=rng
            ).owner.outputs
            next_rng, y_innovation = pm.MvNormal.dist(
                mu=0, cov=stabilize(H), rng=middle_rng
            ).owner.outputs

            a_next += R @ a_innovation
            y_next += y_innovation

            next_state = pt.concatenate([a_next, y_next], axis=0)

            return next_state, {rng: next_rng}

        statespace, updates = pytensor.scan(
            step_fn,
            outputs_info=[init_dist_],
            non_sequences=[c_, d_, T_, Z_, R_, H_, Q_, rng],
            n_steps=steps_,
        )

        statespace_ = pt.concatenate([init_dist_[None], statespace], axis=0)

        (ss_rng,) = tuple(updates.values())
        linear_gaussian_ss_op = LinearGaussianStateSpaceRV(
            inputs=[a0_, P0_, c_, d_, T_, Z_, R_, H_, Q_, init_dist_, steps_],
            outputs=[ss_rng, statespace_],
            ndim_supp=1,
        )

        linear_gaussian_ss = linear_gaussian_ss_op(a0, P0, c, d, T, Z, R, H, Q, init_dist, steps)
        return linear_gaussian_ss


class LinearGaussianStateSpace:
    def __new__(cls, name, a0, P0, c, d, T, Z, R, H, Q, *, init_dist=None, steps=None, **kwargs):
        latent_obs_combined = _LinearGaussianStateSpace(
            f"{name}_combined",
            a0,
            P0,
            c,
            d,
            T,
            Z,
            R,
            H,
            Q,
            init_dist=init_dist,
            steps=steps,
            **kwargs,
        )
        k_states = T.type.shape[0]

        latent_states = latent_obs_combined[..., :k_states]
        obs_states = latent_obs_combined[..., k_states:]

        latent_states = pm.Deterministic(f"{name}_latent", latent_states)
        obs_states = pm.Deterministic(f"{name}_observed", obs_states)

        return latent_states, obs_states

    @classmethod
    def dist(cls, a0, P0, c, d, T, Z, R, H, Q, *, init_dist=None, steps=None, **kwargs):
        latent_obs_combined = _LinearGaussianStateSpace.dist(
            a0, P0, c, d, T, Z, R, H, Q, init_dist=init_dist, steps=steps, **kwargs
        )
        k_states = T.type.shape[0]

        latent_states = latent_obs_combined[..., :k_states]
        obs_states = latent_obs_combined[..., k_states:]

        return latent_states, obs_states


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
    def dist(cls, mus, covs, logp, support_shape=None, **kwargs):
        support_shape = get_support_shape(
            support_shape=None,
            shape=None,
            dims=kwargs.get("dims", None),
            observed=kwargs.get("observed", None),
            ndim_supp=2,
        )

        if support_shape is None:
            support_shape = pt.as_tensor_variable(())

        steps = pm.intX(mus.shape[0])

        return super().dist([mus, covs, logp, steps, support_shape], **kwargs)

    @classmethod
    def rv_op(cls, mus, covs, logp, steps, support_shape, size=None):
        if size is not None:
            batch_size = size
        else:
            batch_size = support_shape

        mus_, covs_, support_shape_ = mus.type(), covs.type(), support_shape.type()
        steps_ = steps.type()
        logp_ = logp.type()

        rng = pytensor.shared(np.random.default_rng())

        def step(mu, cov, rng):
            new_rng, mvn = pm.MvNormal.dist(
                mu=mu, cov=stabilize(cov, JITTER_DEFAULT), rng=rng, size=batch_size
            ).owner.outputs
            return mvn, {rng: new_rng}

        mvn_seq, updates = pytensor.scan(
            step, sequences=[mus_, covs_], non_sequences=[rng], n_steps=steps_, strict=True
        )

        (seq_mvn_rng,) = tuple(updates.values())

        mvn_seq_op = SequenceMvNormalRV(
            inputs=[mus_, covs_, logp_, steps_], outputs=[seq_mvn_rng, mvn_seq], ndim_supp=2
        )

        mvn_seq = mvn_seq_op(mus, covs, logp, steps)
        return mvn_seq


@_logprob.register(SequenceMvNormalRV)
def sequence_mvnormal_logp(op, values, mus, covs, logp, steps, rng, **kwargs):
    # def step_logp(x, mu, cov):
    #     return logp(pm.MvNormal.dist(mu, cov), x)
    #
    # logp_values, updates = pytensor.scan(
    #     step_logp, sequences=[values[0], mus, covs], strict=True
    # )

    return check_parameters(
        logp,
        pt.eq(values[0].shape[0], steps),
        pt.eq(mus.shape[0], steps),
        pt.eq(covs.shape[0], steps),
        msg="Observed data and parameters must have the same number of timesteps (dimension 0)",
    )
