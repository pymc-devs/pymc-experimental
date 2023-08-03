import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pymc import intX
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import Continuous, SymbolicRandomVariable
from pymc.distributions.shape_utils import get_support_shape, get_support_shape_1d
from pymc.gp.util import stabilize
from pymc.logprob.abstract import _logprob
from pytensor.graph.basic import Node

from pymc_experimental.statespace.utils.constants import JITTER_DEFAULT


class LinearGaussianStateSpaceRV(SymbolicRandomVariable):
    default_output = 1
    _print_name = ("LinearGuassianStateSpace", "\\operatorname{LinearGuassianStateSpace}")

    def update(self, node: Node):
        return {node.inputs[-1]: node.outputs[0]}


class _LinearGaussianStateSpace(Continuous):
    rv_op = LinearGaussianStateSpaceRV

    def __new__(
        cls, name, a0, P0, c, d, T, Z, R, H, Q, steps=None, k_states=None, mode=None, **kwargs
    ):
        k_states = cls._get_k_states(T)

        # Ignore dims in support shape because they are just passed along to the "observed" and "latent" distributions
        # created by LinearGaussianStateSpace. This "combined" distribution shouldn't ever be directly used.
        steps = get_support_shape_1d(
            support_shape=steps,
            shape=None,
            dims=None,
            observed=kwargs.get("observed", None),
            support_shape_offset=0,
        )

        return super().__new__(
            cls,
            name,
            a0,
            P0,
            c,
            d,
            T,
            Z,
            R,
            H,
            Q,
            steps=steps,
            k_states=k_states,
            mode=mode,
            **kwargs,
        )

    @classmethod
    def dist(cls, a0, P0, c, d, T, Z, R, H, Q, steps=None, k_states=None, mode=None, **kwargs):
        steps = get_support_shape_1d(
            support_shape=steps, shape=kwargs.get("shape", None), support_shape_offset=0
        )
        k_states = cls._get_k_states(T)

        if steps is None:
            raise ValueError("Must specify steps or shape parameter")

        steps = pt.as_tensor_variable(intX(steps), ndim=0)

        return super().dist([a0, P0, c, d, T, Z, R, H, Q, steps, k_states], mode=mode, **kwargs)

    @classmethod
    def _get_k_states(cls, T):
        return T.shape[0]

    @classmethod
    def rv_op(cls, a0, P0, c, d, T, Z, R, H, Q, steps, k_states, size=None, mode=None):
        if size is not None:
            batch_size = size

        a0_, P0_, c_, d_, T_, Z_, R_, H_, Q_ = map(
            lambda x: x.type(), [a0, P0, c, d, T, Z, R, H, Q]
        )

        steps_ = steps.type()
        rng = pytensor.shared(np.random.default_rng())

        def step_fn(*args):
            state, c, d, T, Z, R, H, Q, rng = args
            k = T.shape[0]
            a = state[:k]

            middle_rng, a_innovation = pm.MvNormal.dist(
                mu=0, cov=stabilize(Q, JITTER_DEFAULT), rng=rng
            ).owner.outputs
            next_rng, y_innovation = pm.MvNormal.dist(
                mu=0, cov=stabilize(H, JITTER_DEFAULT), rng=middle_rng
            ).owner.outputs

            a_next = c + T @ a + R @ a_innovation
            y_next = d + Z @ a_next + y_innovation

            next_state = pt.concatenate([a_next, y_next], axis=0)

            return next_state, {rng: next_rng}

        init_x_ = pm.MvNormal.dist(a0_, stabilize(P0_, JITTER_DEFAULT), rng=rng)
        init_y_ = pm.MvNormal.dist(Z_ @ init_x_, stabilize(H_, JITTER_DEFAULT), rng=rng)
        init_dist_ = pt.concatenate([init_x_, init_y_], axis=0)

        statespace, updates = pytensor.scan(
            step_fn,
            outputs_info=[init_dist_],
            non_sequences=[c_, d_, T_, Z_, R_, H_, Q_, rng],
            n_steps=steps_,
            mode=mode,
            strict=True,
        )

        statespace_ = pt.concatenate([init_dist_[None], statespace], axis=0)

        (ss_rng,) = tuple(updates.values())
        linear_gaussian_ss_op = LinearGaussianStateSpaceRV(
            inputs=[a0_, P0_, c_, d_, T_, Z_, R_, H_, Q_, steps_],
            outputs=[ss_rng, statespace_],
            ndim_supp=1,
        )

        linear_gaussian_ss = linear_gaussian_ss_op(a0, P0, c, d, T, Z, R, H, Q, steps)
        return linear_gaussian_ss


class LinearGaussianStateSpace(Continuous):
    """
    Linear Gaussian Statespace distribution

    """

    def __new__(cls, name, a0, P0, c, d, T, Z, R, H, Q, *, steps=None, mode=None, **kwargs):
        dims = kwargs.pop("dims", None)
        latent_dims = None
        obs_dims = None
        if dims is not None:
            if len(dims) != 3:
                ValueError(
                    "LinearGaussianStateSpace expects 3 dims: time, all_states, and observed_states"
                )
            time_dim, state_dim, obs_dim = dims
            latent_dims = [time_dim, state_dim]
            obs_dims = [time_dim, obs_dim]

        matrices = (a0, P0, c, d, T, Z, R, H, Q)
        latent_obs_combined = _LinearGaussianStateSpace(
            f"{name}_combined", *matrices, steps=steps, mode=mode, **kwargs
        )

        k_states = T.type.shape[0]

        latent_states = latent_obs_combined[..., :k_states]
        obs_states = latent_obs_combined[..., k_states:]

        latent_states = pm.Deterministic(f"{name}_latent", latent_states, dims=latent_dims)
        obs_states = pm.Deterministic(f"{name}_observed", obs_states, dims=obs_dims)

        return latent_states, obs_states

    @classmethod
    def dist(cls, a0, P0, c, d, T, Z, R, H, Q, *, steps=None, **kwargs):
        latent_obs_combined = _LinearGaussianStateSpace.dist(
            a0, P0, c, d, T, Z, R, H, Q, steps=steps, **kwargs
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


class SequenceMvNormal(Continuous):
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
    return check_parameters(
        logp,
        pt.eq(values[0].shape[0], steps),
        pt.eq(mus.shape[0], steps),
        pt.eq(covs.shape[0], steps),
        msg="Observed data and parameters must have the same number of timesteps (dimension 0)",
    )
