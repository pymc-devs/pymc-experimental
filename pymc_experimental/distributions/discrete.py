#   Copyright 2023 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
import pymc as pm
from pymc.distributions.dist_math import betaln, check_parameters, factln, logpow
from pymc.distributions.shape_utils import rv_size_is_none
from pytensor import tensor as pt
from pytensor.tensor.random.op import RandomVariable


class GeneralizedPoissonRV(RandomVariable):
    name = "generalized_poisson"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "int64"
    _print_name = ("GeneralizedPoisson", "\\operatorname{GeneralizedPoisson}")

    @classmethod
    def rng_fn(cls, rng, theta, lam, size):
        theta = np.asarray(theta)
        lam = np.asarray(lam)

        if size is not None:
            dist_size = size
        else:
            dist_size = np.broadcast_shapes(theta.shape, lam.shape)

        # A mix of 2 algorithms described by Famoye (1997) is used depending on parameter values
        # 0: Inverse method, computed on the log scale. Used when lam <= 0.
        # 1: Branching method. Used when lambda > 0.
        x = np.empty(dist_size)
        idxs_mask = np.broadcast_to(lam < 0, dist_size)
        if np.any(idxs_mask):
            x[idxs_mask] = cls._inverse_rng_fn(rng, theta, lam, dist_size, idxs_mask=idxs_mask)[
                idxs_mask
            ]
        idxs_mask = ~idxs_mask
        if np.any(idxs_mask):
            x[idxs_mask] = cls._branching_rng_fn(rng, theta, lam, dist_size, idxs_mask=idxs_mask)[
                idxs_mask
            ]
        return x

    @classmethod
    def _inverse_rng_fn(cls, rng, theta, lam, dist_size, idxs_mask):
        # We handle x/0 and log(0) issues with branching
        with np.errstate(divide="ignore", invalid="ignore"):
            log_u = np.log(rng.uniform(size=dist_size))
            pos_lam = lam > 0
            abs_log_lam = np.log(np.abs(lam))
            theta_m_lam = theta - lam
            log_s = -theta
            log_p = log_s.copy()
            x_ = 0
            x = np.zeros(dist_size)
            below_cutpoint = log_s < log_u
            while np.any(below_cutpoint[idxs_mask]):
                x_ += 1
                x[below_cutpoint] += 1
                log_c = np.log(theta_m_lam + lam * x_)
                # Compute log(1 + lam / C)
                log1p_lam_m_C = np.where(
                    pos_lam,
                    np.log1p(np.exp(abs_log_lam - log_c)),
                    pm.math.log1mexp_numpy(abs_log_lam - log_c, negative_input=True),
                )
                log_p = log_c + log1p_lam_m_C * (x_ - 1) + log_p - np.log(x_) - lam
                log_s = np.logaddexp(log_s, log_p)
                below_cutpoint = log_s < log_u
            return x

    @classmethod
    def _branching_rng_fn(cls, rng, theta, lam, dist_size, idxs_mask):
        lam_ = np.abs(lam)  # This algorithm is only valid for positive lam
        y = rng.poisson(theta, size=dist_size)
        x = y.copy()
        higher_than_zero = y > 0
        while np.any(higher_than_zero[idxs_mask]):
            y = rng.poisson(lam_ * y)
            x[higher_than_zero] = x[higher_than_zero] + y[higher_than_zero]
            higher_than_zero = y > 0
        return x


generalized_poisson = GeneralizedPoissonRV()


class GeneralizedPoisson(pm.distributions.Discrete):
    R"""
    Generalized Poisson.
    Used to model count data that can be either overdispersed or underdispersed.
    Offers greater flexibility than the standard Poisson which assumes equidispersion,
    where the mean is equal to the variance.
    The pmf of this distribution is

    .. math:: f(x \mid \mu, \lambda) =
                  \frac{\mu (\mu + \lambda x)^{x-1} e^{-\mu - \lambda x}}{x!}
    ========  ======================================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\frac{\mu}{1 - \lambda}`
    Variance  :math:`\frac{\mu}{(1 - \lambda)^3}`
    ========  ======================================

    Parameters
    ----------
    mu : tensor_like of float
        Mean parameter (mu > 0).
    lam : tensor_like of float
        Dispersion parameter (max(-1, -mu/4) <= lam <= 1).

    Notes
    -----
    When lam = 0, the Generalized Poisson reduces to the standard Poisson with the same mu.
    When lam < 0, the mean is greater than the variance (underdispersion).
    When lam > 0, the mean is less than the variance (overdispersion).

    References
    ----------
    The PMF is taken from [1] and the random generator function is adapted from [2].
    .. [1] Consul, PoC, and Felix Famoye. "Generalized Poisson regression model."
       Communications in Statistics-Theory and Methods 21.1 (1992): 89-109.
    .. [2] Famoye, Felix. "Generalized Poisson random variate generation." American
       Journal of Mathematical and Management Sciences 17.3-4 (1997): 219-237.
    """

    rv_op = generalized_poisson

    @classmethod
    def dist(cls, mu, lam, **kwargs):
        mu = pt.as_tensor_variable(mu)
        lam = pt.as_tensor_variable(lam)
        return super().dist([mu, lam], **kwargs)

    def moment(rv, size, mu, lam):
        mean = pt.floor(mu / (1 - lam))
        if not rv_size_is_none(size):
            mean = pt.full(size, mean)
        return mean

    def logp(value, mu, lam):
        mu_lam_value = mu + lam * value
        logprob = np.log(mu) + logpow(mu_lam_value, value - 1) - mu_lam_value - factln(value)

        # Probability is 0 when value > m, where m is the largest positive integer for
        # which mu + m * lam > 0 (when lam < 0).
        logprob = pt.switch(
            pt.or_(
                mu_lam_value < 0,
                value < 0,
            ),
            -np.inf,
            logprob,
        )

        return check_parameters(
            logprob,
            0 < mu,
            pt.abs(lam) <= 1,
            (-mu / 4) <= lam,
            msg="0 < mu, max(-1, -mu/4)) <= lam <= 1",
        )


class BetaNegativeBinomial(pm.distributions.Discrete):
    R"""
    Beta Negative Binomial distribution.

    The pmf of this distribution is

    .. math::

        f(x \mid \alpha, \beta, r) = \frac{B(r + x, \alpha + \beta)}{B(r, \alpha)} \frac{\Gamma(x + \beta)}{x! \Gamma(\beta)}

    where :math:`B` is the Beta function and :math:`\Gamma` is the Gamma function.

    For more information, see https://en.wikipedia.org/wiki/Beta_negative_binomial_distribution.

    ========  ======================================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`{\begin{cases}{\frac  {r\beta }{\alpha -1}}&{\text{if}}\ \alpha >1\\\infty &{\text{otherwise}}\ \end{cases}}`
    Variance  :math:`{\displaystyle {\begin{cases}{\frac {r\beta (r+\alpha -1)(\beta +\alpha -1)}{(\alpha -2){(\alpha -1)}^{2}}}&{\text{if}}\ \alpha >2\\\infty &{\text{otherwise}}\ \end{cases}}}`
    ========  ======================================

    Parameters
    ----------
    alpha : tensor_like of float
        shape of the beta distribution (alpha > 0).
    beta : tensor_like of float
        shape of the beta distribution (beta > 0).
    r : tensor_like of float
        number of successes until the experiment is stopped (integer but can be extended to real)
    """

    @staticmethod
    def beta_negative_binomial_dist(alpha, beta, r, size):
        p = pm.Beta.dist(alpha, beta, size=size)
        return pm.NegativeBinomial.dist(p, r, size=size)

    @staticmethod
    def beta_negative_binomial_logp(value, alpha, beta, r):
        res = (
            betaln(r + value, alpha + beta)
            - betaln(r, alpha)
            + pt.gammaln(value + beta)
            - factln(value)
            - pt.gammaln(beta)
        )
        res = pt.switch(
            pt.lt(value, 0),
            -np.inf,
            res,
        )

        return check_parameters(
            res,
            alpha > 0,
            beta > 0,
            r > 0,
            msg="alpha > 0, beta > 0, r > 0",
        )

    def __new__(cls, name, alpha, beta, r, **kwargs):
        return pm.CustomDist(
            name,
            alpha,
            beta,
            r,
            dist=cls.beta_negative_binomial_dist,
            logp=cls.beta_negative_binomial_logp,
            class_name="BetaNegativeBinomial",
            **kwargs
        )

    @classmethod
    def dist(cls, alpha, beta, r, **kwargs):
        return pm.CustomDist.dist(
            alpha,
            beta,
            r,
            dist=cls.beta_negative_binomial_dist,
            logp=cls.beta_negative_binomial_logp,
            class_name="BetaNegativeBinomial",
            **kwargs
        )
