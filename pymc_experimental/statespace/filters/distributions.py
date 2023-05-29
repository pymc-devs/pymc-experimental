import numpy as np
import pytensor
from pymc.distributions.distribution import Distribution, SymbolicRandomVariable
from pymc.distributions.shape_utils import get_support_shape_1d
from pymc.logprob.abstract import _logprob
from pytensor.graph.basic import Node
from pytensor.tensor.random.op import RandomVariable
from scipy import stats


class _KalmanFilterBaseRV(RandomVariable):
    name = "_kalmanfilterbase"
    ndim_supp = 0
    ndims_params = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    dtype = "floatX"
    _print_name = ("_kalmanfilterbase", "\\operatorname{_kalmanfilterbase}")

    def make_node(self, rng, x0, P0, c, d, T, Z, R, H, Q, steps, size):
        return super().make_node(rng, x0, P0, c, d, T, Z, R, H, Q, steps, size)

    def rng_fn(self, rng, x0, P0, c, d, T, Z, R, H, Q, steps, size=None):
        n_obs, n_states = Z.shape[:2]
        n_posdef = Q.shape[0]

        state_innovations = stats.multivariate_normal(mean=np.zeros(n_posdef), cov=Q, seed=rng).rvs(
            steps
        )
        measurement_error = stats.multivariate_normal(
            mean=np.zeros(n_obs), cov=H, seed=rng, allow_singular=True
        ).rvs(steps)

        state_innovations = state_innovations[:, :, None]
        measurement_error = measurement_error[:, None, None]

        hidden_states = np.zeros((steps, n_states, 1))
        observed = np.zeros((steps, n_obs, 1))
        hidden_states[0] = x0
        observed[0] = Z @ x0 + measurement_error[0]

        for t in range(1, steps):
            hidden_states[t] = T @ hidden_states[t - 1] + R @ state_innovations[t] + c
            observed[t] = Z @ hidden_states[t] + measurement_error[t] + d

        return observed


#         if size is not None:
#             raise NotImplementedError()

#         (filtered_states, predicted_states, filtered_cov,
#             predicted_cov, log_likelihood) = kalman_filter(data, x0, P0, c, d, T, Z, R, H, Q)

#         return filtered_states, predicted_states, filtered_cov, predicted_cov


_kalman_filter_base = _KalmanFilterBaseRV()


class _KalmanFilterRV(SymbolicRandomVariable):
    default_output = 1
    _print_name = ("KalmanFilter", "\\operatorname{KalmanFilter}")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, node: Node):
        return {node.inputs[0]: node.outputs[0]}


class _KalmanFilter(Distribution):
    rv_type = _KalmanFilterRV

    @classmethod
    def dist(cls, x0, P0, c, d, T, Z, R, H, Q, *args, steps, **kwargs):
        return super().dist([x0, P0, c, d, T, Z, R, H, Q, steps], **kwargs)

    @classmethod
    def rv_op(cls, x0, P0, c, d, T, Z, R, H, Q, steps, size=None):
        rng = pytensor.shared(np.random.default_rng())
        rng_, x0_, P0_, steps_ = rng.type(), x0.type(), P0.type(), steps.type()
        c_, d_, T_, Z_, R_, H_, Q_ = (
            c.type(),
            d.type(),
            T.type(),
            Z.type(),
            R.type(),
            H.type(),
            Q.type(),
        )
        next_rng_, outputs_ = _kalman_filter_base(
            x0=x0_, P0=P0_, c=c_, d=d_, T=T_, Z=Z_, R=R_, H=H_, Q=Q_, steps=steps_, rng=rng_
        ).owner.outputs

        return _KalmanFilterRV(
            inputs=[rng_, x0_, P0_, c_, d_, T_, Z_, R_, H_, Q_, steps_],
            outputs=[next_rng_, outputs_],
            ndim_supp=0,
        )(rng, x0, P0, c, d, T, Z, R, H, Q, steps)


class KalmanFilter:
    def __new__(cls, name, x0, P0, c, d, T, Z, R, H, Q, *args, steps=None, **kwargs):
        steps = get_support_shape_1d(
            support_shape=steps,
            shape=None,  # Shape will be checked in `cls.dist`
            dims=kwargs.get("dims", None),
            observed=kwargs.get("observed", None),
            support_shape_offset=0,
        )
        outputs = _KalmanFilter(name, x0, P0, c, d, T, Z, R, H, Q, *args, steps=steps, **kwargs)
        return outputs

    @classmethod
    def dist(cls, x0, P0, c, d, T, Z, R, H, Q, *args, steps=None, **kwargs):
        outputs = _KalmanFilter.dist(x0, P0, c, d, T, Z, R, H, Q, *args, steps=steps, **kwargs)
        return outputs


@_logprob.register(_KalmanFilterRV)
def _KalmanFilterRV_logp(op, values, x0, P0, c, d, T, Z, R, H, Q, **kwargs):
    raise NotImplementedError
