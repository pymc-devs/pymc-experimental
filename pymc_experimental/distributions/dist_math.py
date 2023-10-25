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

# coding: utf-8

import numpy as np
import pytensor.tensor as pt
from pymc.distributions.dist_math import SplineWrapper
from scipy.interpolate import UnivariateSpline


def studentt_kld_distance(nu):
    """
    2 * sqrt(KL divergence divergence) between a student t and a normal random variable.  Derived
    by Tang in https://arxiv.org/abs/1811.08042.
    """
    return pt.sqrt(
        1
        + pt.log(2 * pt.reciprocal(nu - 2))
        + 2 * pt.gammaln((nu + 1) / 2)
        - 2 * pt.gammaln(nu / 2)
        - (nu + 1) * (pt.digamma((nu + 1) / 2) - pt.digamma(nu / 2))
    )


def tri_gamma_approx(x):
    """Derivative of the digamma function, or second derivative of the gamma function.  This is a
    series expansion taken from wikipedia: https://en.wikipedia.org/wiki/Trigamma_function.  When
    the trigamma function in pytensor implements a gradient this function can be removed and
    replaced.
    """
    return (
        1 / x
        + (1 / (2 * x**2))
        + (1 / (6 * x**3))
        - (1 / (30 * x**5))
        + (1 / (42 * x**7))
        - (1 / (30 * x**9))
        + (5 / (66 * x**11))
        - (691 / (2730 * x**13))
        + (7 / (6 * x**15))
    )


def pc_prior_studentt_logp(nu, lam):
    """The log probability density function for the PC prior for the degrees of freedom in a
    student t likelihood. Derived by Tang in https://arxiv.org/abs/1811.08042.
    """
    return (
        pt.log(lam)
        + pt.log(
            (1 / (nu - 2))
            + ((nu + 1) / 2) * (tri_gamma_approx((nu + 1) / 2) - tri_gamma_approx(nu / 2))
        )
        - pt.log(4 * studentt_kld_distance(nu))
        - lam * studentt_kld_distance(nu)
        + pt.log(2)
    )


def _make_pct_inv_func():
    """This function constructs a numerical approximation to the inverse of the KLD distance
    function, `studentt_kld_distance`.  It does a spline fit for degrees of freedom values
    from 2 + 1e-6 to 4000.  2 is the smallest valid value for the student t degrees of freedom, and
    values above 4000 don't seem to change much (nearly Gaussian past 30).  It's then wrapped by
    `SplineWrapper` so it can be used as a PyTensor op.
    """
    NU_MIN = 2.0 + 1e-6
    nu = np.concatenate((np.linspace(NU_MIN, 2.4, 2000), np.linspace(2.4 + 1e-4, 4000, 10000)))
    return UnivariateSpline(
        studentt_kld_distance(nu).eval()[::-1],
        nu[::-1],
        ext=3,
        k=3,
        s=0,
    )


pc_prior_studentt_kld_dist_inv_op = SplineWrapper(_make_pct_inv_func())
