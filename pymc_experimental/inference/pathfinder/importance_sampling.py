import logging
import warnings

import arviz as az
import numpy as np
import pytensor.tensor as pt

from pytensor.graph import Apply, Op
from pytensor.tensor.variable import TensorVariable

logger = logging.getLogger(__name__)


class PSIS(Op):
    __props__ = ()

    def make_node(self, inputs):
        logweights = pt.as_tensor(inputs)
        psislw = pt.dvector()
        pareto_k = pt.dscalar()
        return Apply(self, [logweights], [psislw, pareto_k])

    def perform(self, node: Apply, inputs, outputs) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message="overflow encountered in exp"
            )
            logweights = inputs[0]
            psislw, pareto_k = az.psislw(logweights)
            outputs[0][0] = psislw
            outputs[1][0] = pareto_k


def psir(
    samples: TensorVariable,
    # logP: TensorVariable,
    # logQ: TensorVariable,
    logiw: TensorVariable,
    num_draws: int = 1000,
    random_seed: int | None = None,
) -> np.ndarray:
    """Pareto Smoothed Importance Resampling (PSIR)
    This implements the Pareto Smooth Importance Resampling (PSIR) method, as described in Algorithm 5 of Zhang et al. (2022). The PSIR follows a similar approach to Algorithm 1 PSIS diagnostic from Yao et al., (2018). However, before computing the the importance ratio r_s, the logP and logQ are adjusted to account for the number multiple estimators (or paths). The process involves resampling from the original sample with replacement, with probabilities proportional to the computed importance weights from PSIS.

    Parameters
    ----------
    samples : np.ndarray
        samples from proposal distribution
    logP : np.ndarray
        log probability of target distribution
    logQ : np.ndarray
        log probability of proposal distribution
    num_draws : int
        number of draws to return where num_draws <= samples.shape[0]
    random_seed : int | None

    Returns
    -------
    np.ndarray
        importance sampled draws

    Future work!
    ----------
    - Implement the 3 sampling approaches and 5 weighting functions from Elvira et al. (2019)
    - Implement Algorithm 2 VSBC marginal diagnostics from Yao et al. (2018)
    - Incorporate these various diagnostics, sampling approaches and weighting functions into VI algorithms.

    References
    ----------
    Elvira, V., Martino, L., Luengo, D., & Bugallo, M. F. (2019). Generalized Multiple Importance Sampling. Statistical Science, 34(1), 129-155. https://doi.org/10.1214/18-STS668

    Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). Yes, but Did It Work?: Evaluating Variational Inference. arXiv:1802.02538 [Stat]. http://arxiv.org/abs/1802.02538

    Zhang, L., Carpenter, B., Gelman, A., & Vehtari, A. (2022). Pathfinder: Parallel quasi-Newton variational inference. Journal of Machine Learning Research, 23(306), 1-49.
    """

    psislw, pareto_k = PSIS()(logiw)
    pareto_k = pareto_k.eval()
    if pareto_k < 0.5:
        pass
    elif 0.5 <= pareto_k < 0.70:
        logger.warning(
            f"Pareto k value ({pareto_k:.2f}) is between 0.5 and 0.7 which indicates an imperfect approximation however still useful."
        )
        logger.info("Consider increasing ftol, gtol or maxcor.")
    elif pareto_k >= 0.7:
        logger.warning(
            f"Pareto k value ({pareto_k:.2f}) exceeds 0.7 which indicates a bad approximation."
        )
        logger.info("Consider increasing ftol, gtol, maxcor or reparametrising the model.")
    else:
        logger.warning(
            f"Received an invalid Pareto k value of {pareto_k:.2f} which indicates the model is seriously flawed."
        )
        logger.info(
            "Consider reparametrising the model all together or ensure the input data are correct."
        )

    p = pt.exp(psislw - pt.logsumexp(psislw)).eval()
    rng = np.random.default_rng(random_seed)
    return rng.choice(samples, size=num_draws, replace=True, p=p, shuffle=False, axis=0)
