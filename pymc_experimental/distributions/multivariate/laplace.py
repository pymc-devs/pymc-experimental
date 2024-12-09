import numpy as np
import pytensor.tensor as pt
import scipy

from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import Continuous, SymbolicRandomVariable, _support_point
from pymc.distributions.moments.means import _mean
from pymc.distributions.multivariate import (
    _logdet_from_cholesky,
    nan_lower_cholesky,
    quaddist_chol,
    quaddist_matrix,
    solve_lower,
)
from pymc.distributions.shape_utils import implicit_size_from_params, rv_size_is_none
from pymc.logprob.basic import _logprob
from pymc.pytensorf import normalize_rng_param
from pytensor.gradient import grad_not_implemented
from pytensor.scalar import BinaryScalarOp, upgrade_to_float
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.random.utils import normalize_size_param


class Kv(BinaryScalarOp):
    """
    Modified Bessel function of the second kind of real order v.
    """

    nfunc_spec = ("scipy.special.kv", 2, 1)

    @staticmethod
    def st_impl(v, x):
        return scipy.special.kv(v, x)

    def impl(self, v, x):
        return self.st_impl(v, x)

    def L_op(self, inputs, outputs, output_grads):
        v, x = inputs
        [out] = outputs
        [g_out] = output_grads
        dx = -(v / x) * out - self.kv(v - 1, x)
        return [grad_not_implemented(self, 0, v), g_out * dx]

    def c_code(self, *args, **kwargs):
        raise NotImplementedError()


kv = Elemwise(Kv(upgrade_to_float, name="kv"))


class MvLaplaceRV(SymbolicRandomVariable):
    name = "multivariate_laplace"
    extended_signature = "[rng],[size],(m),(m,m)->[rng],(m)"
    _print_name = ("MultivariateLaplace", "\\operatorname{MultivariateLaplace}")

    @classmethod
    def rv_op(cls, mu, cov, *, size=None, rng=None):
        mu = pt.as_tensor(mu)
        cov = pt.as_tensor(cov)
        rng = normalize_rng_param(rng)
        size = normalize_size_param(size)

        assert mu.type.ndim >= 1
        assert cov.type.ndim >= 2

        if rv_size_is_none(size):
            size = implicit_size_from_params(mu, cov, ndims_params=(1, 2))

        next_rng, e = pt.random.exponential(size=size, rng=rng).owner.outputs
        next_rng, z = pt.random.multivariate_normal(
            mean=pt.zeros(mu.shape[-1]), cov=cov, size=size, rng=next_rng
        ).owner.outputs
        rv = mu + pt.sqrt(e)[..., None] * z

        return cls(
            inputs=[rng, size, mu, cov],
            outputs=[next_rng, rv],
        )(rng, size, mu, cov)


class MvLaplace(Continuous):
    r"""Multivariate (Symmetric) Laplace distribution.

    The pdf of this distribution is

    .. math::

        pdf(x \mid \mu, \Sigma) =
            \frac{2}{(2\pi)^{k/2} |\Sigma|^{1/2}}
            ( \frac{(x-\mu)'\Sigma^{-1}(x-mu)}{2} )^{v/2}
            \K_v (\sqrt{2(x-\mu)' \Sigma^{-1} (x - \mu)}})

    where :math:`v = 1 - k/2` and :math:`\K_v` is the modified Bessel function of the second kind.

    ========  ==========================
    Support   :math:`x \in \mathbb{R}^k`
    Mean      :math:`\mu`
    Variance  :math:`\Sigma`
    ========  ==========================

    Parameters
    ----------
    mu : tensor_like of float
        Location.
    cov : tensor_like of float, optional
        Covariance matrix. Exactly one of cov, tau, or chol is needed.
    tau : tensor_like of float, optional
        Precision matrix. Exactly one of cov, tau, or chol is needed.
    chol : tensor_like of float, optional
        Cholesky decomposition of covariance matrix. Exactly one of cov,
        tau, or chol is needed.
    lower: bool, default=True
        Whether chol is the lower tridiagonal cholesky factor.
    """

    rv_type = MvLaplaceRV
    rv_op = MvLaplaceRV.rv_op

    @classmethod
    def dist(cls, mu=0, cov=None, *, tau=None, chol=None, lower=True, **kwargs):
        cov = quaddist_matrix(cov, chol, tau, lower)

        mu = pt.atleast_1d(pt.as_tensor_variable(mu))
        if mu.type.broadcastable[-1] and not cov.type.broadcastable[-1]:
            mu, _ = pt.broadcast_arrays(mu, cov[..., -1])
        return super().dist([mu, cov], **kwargs)


class MvAsymmetricLaplaceRV(SymbolicRandomVariable):
    name = "multivariate_asymmetric_laplace"
    extended_signature = "[rng],[size],(m),(m,m)->[rng],(m)"
    _print_name = ("MultivariateAsymmetricLaplace", "\\operatorname{MultivariateAsymmetricLaplace}")

    @classmethod
    def rv_op(cls, mu, cov, *, size=None, rng=None):
        mu = pt.as_tensor(mu)
        cov = pt.as_tensor(cov)
        rng = normalize_rng_param(rng)
        size = normalize_size_param(size)

        assert mu.type.ndim >= 1
        assert cov.type.ndim >= 2

        if rv_size_is_none(size):
            size = implicit_size_from_params(mu, cov, ndims_params=(1, 2))

        next_rng, e = pt.random.exponential(size=size, rng=rng).owner.outputs
        next_rng, z = pt.random.multivariate_normal(
            mean=pt.zeros(mu.shape[-1]), cov=cov, size=size, rng=next_rng
        ).owner.outputs
        e = e[..., None]
        rv = e * mu + pt.sqrt(e) * z

        return cls(
            inputs=[rng, size, mu, cov],
            outputs=[next_rng, rv],
        )(rng, size, mu, cov)


class MvAsymmetricLaplace(Continuous):
    r"""Multivariate Asymmetric Laplace distribution.

    The pdf of this distribution is

    .. math::

        pdf(x \mid \mu, \Sigma) =
            \frac{2}{(2\pi)^{k/2} |\Sigma|^{1/2}}
            ( \frac{(x-\mu)'\Sigma^{-1}(x-mu)}{2} )^{v/2}
            \K_v (\sqrt{2(x-\mu)' \Sigma^{-1} (x - \mu)}})

    where :math:`v = 1 - k/2` and :math:`\K_v` is the modified Bessel function of the second kind.

    ========  ==========================
    Support   :math:`x \in \mathbb{R}^k`
    Mean      :math:`\mu`
    Variance  :math:`\Sigma + \mu' \mu`
    ========  ==========================

    Parameters
    ----------
    mu : tensor_like of float
        Location.
    cov : tensor_like of float, optional
        Covariance matrix. Exactly one of cov, tau, or chol is needed.
    tau : tensor_like of float, optional
        Precision matrix. Exactly one of cov, tau, or chol is needed.
    chol : tensor_like of float, optional
        Cholesky decomposition of covariance matrix. Exactly one of cov,
        tau, or chol is needed.
    lower: bool, default=True
        Whether chol is the lower tridiagonal cholesky factor.
    """

    rv_type = MvAsymmetricLaplaceRV
    rv_op = MvAsymmetricLaplaceRV.rv_op

    @classmethod
    def dist(cls, mu=0, cov=None, *, tau=None, chol=None, lower=True, **kwargs):
        cov = quaddist_matrix(cov, chol, tau, lower)

        mu = pt.atleast_1d(pt.as_tensor_variable(mu))
        if mu.type.broadcastable[-1] and not cov.type.broadcastable[-1]:
            mu, _ = pt.broadcast_arrays(mu, cov[..., -1])
        return super().dist([mu, cov], **kwargs)


@_logprob.register(MvLaplaceRV)
def mv_laplace_logp(op, values, rng, size, mu, cov, **kwargs):
    [value] = values
    quaddist, logdet, posdef = quaddist_chol(value, mu, cov)

    k = value.shape[-1].astype("floatX")
    norm = np.log(2) - (k / 2) * np.log(2 * np.pi) - logdet

    v = 1 - (k / 2)
    kernel = ((v / 2) * pt.log(quaddist / 2)) + pt.log(kv(v, pt.sqrt(2 * quaddist)))

    logp_val = norm + kernel
    return check_parameters(logp_val, posdef, msg="posdef scale")


@_logprob.register(MvAsymmetricLaplaceRV)
def mv_asymmetric_laplace_logp(op, values, rng, size, mu, cov, **kwargs):
    [value] = values

    chol_cov = nan_lower_cholesky(cov)
    logdet, posdef = _logdet_from_cholesky(chol_cov)

    # solve_triangular will raise if there are nans
    # (which happens if the cholesky fails)
    chol_cov = pt.switch(posdef[..., None, None], chol_cov, 1)

    solve_x = solve_lower(chol_cov, value, b_ndim=1)
    solve_mu = solve_lower(chol_cov, mu, b_ndim=1)

    x_quaddist = (solve_x**2).sum(-1)
    mu_quaddist = (solve_mu**2).sum(-1)
    x_mu_quaddist = (value * solve_mu).sum(-1)

    k = value.shape[-1].astype("floatX")
    norm = np.log(2) - (k / 2) * np.log(2 * np.pi) - logdet

    v = 1 - (k / 2)
    kernel = (
        x_mu_quaddist
        + ((v / 2) * (pt.log(x_quaddist) - pt.log(2 + mu_quaddist)))
        + pt.log(kv(v, pt.sqrt((2 + mu_quaddist) * x_quaddist)))
    )

    logp_val = norm + kernel
    return check_parameters(logp_val, posdef, msg="posdef scale")


@_mean.register(MvLaplaceRV)
@_mean.register(MvAsymmetricLaplaceRV)
def mv_laplace_mean(op, rv, rng, size, mu, cov):
    if rv_size_is_none(size):
        bcast_mu, _ = pt.random.utils.broadcast_params([mu, cov], ndims_params=[1, 2])
    else:
        bcast_mu = pt.broadcast_to(mu, pt.concatenate([size, [mu.shape[-1]]]))
    return bcast_mu


@_support_point.register(MvLaplaceRV)
@_support_point.register(MvAsymmetricLaplaceRV)
def mv_laplace_support_point(op, rv, rng, size, mu, cov):
    # We have a 0 * inf when value = mu. I assume density is infinite, which isn't a good starting point.
    point = mu + 1
    if rv_size_is_none(size):
        bcast_point, _ = pt.random.utils.broadcast_params([point, cov], ndims_params=[1, 2])
    else:
        bcast_shape = pt.concatenate([size, [point.shape[-1]]])
        bcast_point = pt.broadcast_to(point, bcast_shape)
    return bcast_point
