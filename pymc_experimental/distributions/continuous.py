from typing import List, Optional, Tuple, Union

import numpy as np
import aesara
import aesara.tensor as at
from pymc.aesaraf import floatX
from aesara.tensor.var import TensorConstant, TensorVariable
from aesara.tensor.random.op import RandomVariable

from aesara.tensor.random.basic import gengamma
from pymc.distributions.continuous import PositiveContinuous
from pymc.distributions.shape_utils import rv_size_is_none
from pymc.distributions.dist_math import check_parameters


class GeneralizedGamma(PositiveContinuous):
    r"""
    Generalized Gamma log-likelihood.
    
    The pdf of this distribution is
    
    .. math::
       
       f(x \mid \alpha, p, \lambda) =
        \frac{ p\lambda^{-1} (x/\lambda)^{\alpha - 1} e^{-(x/\lambda)^p}}
        {\Gamma(\alpha/p)}
    
    .. plot::
        :context: close-figs
        
        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(1, 50, 1000)
        alphas = [1,1,2,2]
        ps = [1, 2, 4, 4]
        lambds = [10., 10., 10., 20.]
        for alpha, p, lambd in zip(alphas, ps, lambds):
            pdf = st.gengamma.pdf(x, alpha/p, p, scale=lambd)
            plt.plot(x, pdf, label=r'$\alpha$ = {}, $p$ = {}, $\lambda$ = {}'.format(alpha, p, lambd))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==========================================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\lambda \frac{\Gamma((\alpha+1)/p)}{\Gamma(\alpha/p)}`
    Variance  :math:`\lambda^2 \left( \frac{\Gamma((\alpha+2)/p)}{\Gamma(\alpha/p)} - \left(\frac{\Gamma((\alpha+1)/p)}{\Gamma(\alpha/p)}\right)^2 \right)`
    ========  ==========================================
    
    Parameters
    ----------
    alpha : tensor_like of float, optional
        Shape parameter :math:`\alpha` (``alpha`` > 0).
        Defaults to 1.
    p : tensor_like of float, optional
        Additional shape parameter `p` (p > 0).
        Defaults to 1.
    lambd : tensor_like of float, optional
        Scale parameter :math:`\lambda` (lambd > 0).
        Defaults to 1.
    
    Examples
    --------
    
    .. code-block:: python
        with pm.Model():
            x = pm.GeneralizedGamma('x', alpha=1, p=2, lambd=5)
    """
    rv_op = gengamma

    @classmethod
    def dist(cls, alpha, p, lambd, **kwargs):
        alpha = at.as_tensor_variable(floatX(alpha))
        p = at.as_tensor_variable(floatX(p))
        lambd = at.as_tensor_variable(floatX(lambd))

        return super().dist([alpha, p, lambd], **kwargs)

    def moment(rv, size, alpha, p, lambd):
        alpha, p, lambd = at.broadcast_arrays(alpha, p, lambd)
        moment = lambd * at.gamma((alpha + 1) / p) / at.gamma(alpha / p)
        if not rv_size_is_none(size):
            moment = at.full(size, moment)
        return moment

    def logp(
        value,
        alpha: TensorVariable,
        p: TensorVariable,
        lambd: TensorVariable,
    ) -> TensorVariable:
        """
        Calculate log-probability of Generalized Gamma distribution at specified value.
        Parameters
        ----------
        value : tensor_like of float
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.
        alpha : tensor_like of float
            Shape parameter (alpha > 0).
        p : tensor_like of float
            Shape parameter (p > 0).
        lambd : tensor_like of float
            Scale parameter (lambd > 0).
        Returns
        -------
        TensorVariable
        """
        logp_expression = (
            at.log(p)
            - at.log(lambd)
            + (alpha - 1) * at.log(value / lambd)
            - (value / lambd) ** p
            - at.gammaln(alpha / p)
        )

        bounded_logp_expression = at.switch(
            at.gt(value, 0),
            logp_expression,
            -np.inf,
        )

        return check_parameters(
            bounded_logp_expression,
            alpha > 0,
            p > 0,
            lambd > 0,
            msg="alpha > 0, p > 0, lambd > 0",
        )

    def logcdf(
        value,
        alpha: Union[float, np.ndarray, TensorVariable],
        p: Union[float, np.ndarray, TensorVariable],
        lambd: Union[float, np.ndarray, TensorVariable],
    ) -> RandomVariable:
        """
        Compute the log of the cumulative distribution function for GeneralizedGamma
        distribution at the specified value.
        Parameters
        ----------
        value : tensor_like of float
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.
        alpha : tensor_like of float
            Shape parameter (alpha > 0).
        p : tensor_like of float
            Shape parameter (p > 0).
        lambd : tensor_like of float
            Scale parameter (lambd > 0).
        Returns
        -------
        TensorVariable
        """
        logcdf_expression = at.log(at.gammainc(alpha / p, (value / lambd) ** p))


        bounded_logcdf_expression = at.switch(
            at.gt(value, 0),
            logcdf_expression,
            -np.inf,
        )

        return check_parameters(
            bounded_logcdf_expression,
            alpha > 0,
            p > 0,
            lambd > 0,
            msg="alpha > 0, p > 0, lambd > 0",
        )

