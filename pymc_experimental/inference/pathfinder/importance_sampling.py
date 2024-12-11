import logging
import warnings

from typing import Literal

import arviz as az
import numpy as np
import pytensor.tensor as pt

from pytensor.graph import Apply, Op

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


def importance_sampling(
    samples: np.ndarray,
    logP: np.ndarray,
    logQ: np.ndarray,
    num_draws: int,
    method: Literal["psis", "psir", "identity", "none"],
    logiw: np.ndarray | None = None,
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
    method : str, optional
        importance sampling method to use. Options are "psis" (default), "psir", "identity", "none. Pareto Smoothed Importance Sampling (psis) is recommended in many cases for more stable results than Pareto Smoothed Importance Resampling (psir). identity applies the log importance weights directly without resampling. none applies no importance sampling weights and returns the samples as is of size num_draws_per_path * num_paths.
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

    num_paths, num_pdraws, N = samples.shape

    if method == "none":
        logger.warning(
            "importance sampling is disabled. The samples are returned as is which may include samples from failed paths with non-finite logP or logQ values. It is recommended to use importance_sampling='psis' for better stability."
        )
        return samples
    else:
        samples = samples.reshape(-1, N)
        logP = logP.ravel()
        logQ = logQ.ravel()

        # adjust log densities
        log_I = np.log(num_paths)
        logP -= log_I
        logQ -= log_I
        logiw = logP - logQ

        if method == "psis":
            replace = False
            logiw, pareto_k = PSIS()(logiw)
        elif method == "psir":
            replace = True
            logiw, pareto_k = PSIS()(logiw)
        elif method == "identity":
            replace = False
            logiw = logiw
            pareto_k = None
        else:
            raise ValueError(f"Invalid importance sampling method: {method}")

    # NOTE: Pareto k is normally bad for Pathfinder even when the posterior is close to the NUTS posterior or closer to NUTS than ADVI.
    # Pareto k may not be a good diagnostic for Pathfinder.
    if pareto_k is not None:
        pareto_k = pareto_k.eval()
        if pareto_k < 0.5:
            pass
        elif 0.5 <= pareto_k < 0.70:
            logger.info(
                f"Pareto k value ({pareto_k:.2f}) is between 0.5 and 0.7 which indicates an imperfect approximation however still useful."
            )
            logger.info("Consider increasing ftol, gtol, maxcor or num_paths.")
        elif pareto_k >= 0.7:
            logger.info(
                f"Pareto k value ({pareto_k:.2f}) exceeds 0.7 which indicates a bad approximation."
            )
            logger.info(
                "Consider increasing ftol, gtol, maxcor, num_paths or reparametrising the model."
            )
        else:
            logger.info(
                f"Received an invalid Pareto k value of {pareto_k:.2f} which indicates the model is seriously flawed."
            )
            logger.info(
                "Consider reparametrising the model all together or ensure the input data are correct."
            )

        logger.warning(f"Pareto k value: {pareto_k:.2f}")

    p = pt.exp(logiw - pt.logsumexp(logiw)).eval()
    rng = np.random.default_rng(random_seed)
    return rng.choice(samples, size=num_draws, replace=replace, p=p, shuffle=False, axis=0)
