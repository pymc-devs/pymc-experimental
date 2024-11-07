#   Copyright 2022 The PyMC Developers
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

import collections
import logging
import multiprocessing
import platform
import sys

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Literal

import arviz as az
import blackjax
import cloudpickle
import jax
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from packaging import version
from pymc import Model
from pymc.backends.arviz import coords_and_dims_for_inferencedata
from pymc.blocking import DictToArrayBijection, RaveledVars
from pymc.initial_point import make_initial_point_fn
from pymc.model import modelcontext
from pymc.model.core import Point
from pymc.sampling.jax import get_jaxified_graph
from pymc.util import RandomSeed, _get_seeds_per_chain, get_default_varnames

from pymc_experimental.inference.pathfinder.importance_sampling import psir
from pymc_experimental.inference.pathfinder.lbfgs import lbfgs

logger = logging.getLogger(__name__)

REGULARISATION_TERM = 1e-8


class PathfinderResults:
    def __init__(self, num_paths: int, num_draws_per_path: int, num_dims: int):
        self.num_paths = num_paths
        self.num_draws_per_path = num_draws_per_path
        self.paths = {}
        for path_id in range(num_paths):
            self.paths[path_id] = {
                "samples": np.empty((num_draws_per_path, num_dims)),
                "logP": np.empty(num_draws_per_path),
                "logQ": np.empty(num_draws_per_path),
            }

    def add_path_data(self, path_id: int, samples, logP, logQ):
        self.paths[path_id]["samples"][:] = samples
        self.paths[path_id]["logP"][:] = logP
        self.paths[path_id]["logQ"][:] = logQ


def get_jaxified_logp_of_ravel_inputs(
    model: Model,
) -> Callable:
    """
    Get jaxified logp function and ravel inputs for a PyMC model.

    Parameters
    ----------
    model : Model
        PyMC model to jaxify.

    Returns
    -------
    tuple[Callable, DictToArrayBijection]
        A tuple containing the jaxified logp function and the DictToArrayBijection.
    """

    new_logprob, new_input = pm.pytensorf.join_nonshared_inputs(
        model.initial_point(), (model.logp(),), model.value_vars, ()
    )

    logp_func_list = get_jaxified_graph([new_input], new_logprob)

    def logp_func(x):
        return logp_func_list(x)[0]

    return logp_func


def get_logp_dlogp_of_ravel_inputs(
    model: Model,
):  # -> tuple[Callable[..., Any], Callable[..., Any]]:
    initial_points = model.initial_point()
    ip_map = DictToArrayBijection.map(initial_points)
    compiled_logp_func = DictToArrayBijection.mapf(
        model.compile_logp(jacobian=False), initial_points
    )

    def logp_func(x):
        return compiled_logp_func(RaveledVars(x, ip_map.point_map_info))

    compiled_dlogp_func = DictToArrayBijection.mapf(
        model.compile_dlogp(jacobian=False), initial_points
    )

    def dlogp_func(x):
        return compiled_dlogp_func(RaveledVars(x, ip_map.point_map_info))

    return logp_func, dlogp_func


def convert_flat_trace_to_idata(
    samples,
    include_transformed=False,
    postprocessing_backend="cpu",
    model=None,
):
    model = modelcontext(model)
    ip = model.initial_point()
    ip_point_map_info = DictToArrayBijection.map(ip).point_map_info
    trace = collections.defaultdict(list)
    for sample in samples:
        raveld_vars = RaveledVars(sample, ip_point_map_info)
        point = DictToArrayBijection.rmap(raveld_vars, ip)
        for p, v in point.items():
            trace[p].append(v.tolist())

    trace = {k: np.asarray(v)[None, ...] for k, v in trace.items()}

    var_names = model.unobserved_value_vars
    vars_to_sample = list(get_default_varnames(var_names, include_transformed=include_transformed))
    print("Transforming variables...", file=sys.stdout)
    jax_fn = get_jaxified_graph(inputs=model.value_vars, outputs=vars_to_sample)
    result = jax.vmap(jax.vmap(jax_fn))(
        *jax.device_put(list(trace.values()), jax.devices(postprocessing_backend)[0])
    )
    trace = {v.name: r for v, r in zip(vars_to_sample, result)}
    coords, dims = coords_and_dims_for_inferencedata(model)
    idata = az.from_dict(trace, dims=dims, coords=coords)

    return idata


def alpha_recover(x, g, epsilon: float = 1e-11):
    """
    epsilon: float
        value used to filter out large changes in the direction of the update gradient at each iteration l in L. iteration l are only accepted if delta_theta[l] * delta_grad[l] > epsilon * L2_norm(delta_grad[l]) for each l in L.
    """

    def compute_alpha_l(alpha_lm1, s_l, z_l):
        # alpha_lm1: (N,)
        # s_l: (N,)
        # z_l: (N,)
        a = z_l.T @ pt.diag(alpha_lm1) @ z_l
        b = z_l.T @ s_l
        c = s_l.T @ pt.diag(1.0 / alpha_lm1) @ s_l
        inv_alpha_l = (
            a / (b * alpha_lm1)
            + z_l ** 2 / b
            - (a * s_l ** 2) / (b * c * alpha_lm1**2)
        )  # fmt:off
        return 1.0 / inv_alpha_l

    def return_alpha_lm1(alpha_lm1, s_l, z_l):
        return alpha_lm1[-1]

    def scan_body(update_mask_l, s_l, z_l, alpha_lm1):
        return pt.switch(
            update_mask_l,
            compute_alpha_l(alpha_lm1, s_l, z_l),
            return_alpha_lm1(alpha_lm1, s_l, z_l),
        )

    Lp1, N = x.shape
    S = pt.diff(x, axis=0)
    Z = pt.diff(g, axis=0)
    alpha_l_init = pt.ones(N)
    SZ = (S * Z).sum(axis=-1)

    # Q: Line 5 of Algorithm 3 in Zhang et al., (2022) sets SZ < 1e-11 * L2(Z) as opposed to the ">" sign
    update_mask = SZ > epsilon * pt.linalg.norm(Z, axis=-1)

    alpha, _ = pytensor.scan(
        fn=scan_body,
        outputs_info=alpha_l_init,
        sequences=[update_mask, S, Z],
        n_steps=Lp1 - 1,
        strict=True,
    )

    # alpha: (L, N), update_mask: (L, N)
    # alpha = pt.concatenate([pt.ones(N)[None, :], alpha], axis=0)
    # assert np.all(alpha.eval() > 0), "alpha cannot be negative"
    return alpha, S, Z, update_mask


def inverse_hessian_factors(alpha, S, Z, update_mask, J):
    def get_chi_matrix(diff, update_mask, J):
        L, N = diff.shape
        j_last = pt.as_tensor(J - 1)  # since indexing starts at 0

        def chi_update(chi_lm1, diff_l):
            chi_l = pt.roll(chi_lm1, -1, axis=0)
            # z_xi_l = pt.set_subtensor(z_xi_l[j_last], z_l)
            # z_xi_l[j_last] = z_l
            return pt.set_subtensor(chi_l[j_last], diff_l)

        def no_op(chi_lm1, diff_l):
            return chi_lm1

        def scan_body(update_mask_l, diff_l, chi_lm1):
            return pt.switch(update_mask_l, chi_update(chi_lm1, diff_l), no_op(chi_lm1, diff_l))

        # NOTE: removing first index so that L starts at 1
        # update_mask = pt.concatenate([pt.as_tensor([False], dtype="bool"), update_mask], axis=-1)
        # diff = pt.concatenate([pt.zeros((1, N), dtype="float64"), diff], axis=0)

        chi_init = pt.zeros((J, N))
        chi_mat, _ = pytensor.scan(
            fn=scan_body,
            outputs_info=chi_init,
            sequences=[
                update_mask,
                diff,
            ],
        )

        chi_mat = chi_mat.dimshuffle(0, 2, 1)

        return chi_mat

    L, N = alpha.shape
    s_xi = get_chi_matrix(S, update_mask, J)
    z_xi = get_chi_matrix(Z, update_mask, J)

    # (L, J, J)
    sz_xi = pt.matrix_transpose(s_xi) @ z_xi

    # E: (L, J, J)
    # Ij: (L, J, J)
    Ij = pt.repeat(pt.eye(J)[None, ...], L, axis=0)
    E = pt.triu(sz_xi) + Ij * REGULARISATION_TERM

    # eta: (L, J)
    eta, _ = pytensor.scan(lambda e: pt.diag(e), sequences=[E])

    # beta: (L, N, 2J)
    alpha_diag, _ = pytensor.scan(lambda a: pt.diag(a), sequences=[alpha])
    beta = pt.concatenate([alpha_diag @ z_xi, s_xi], axis=-1)

    # more performant and numerically precise to use solve than inverse: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.inv.html

    # E_inv: (L, J, J)
    # TODO: handle compute errors for .linalg.solve. See comments in the _single_pathfinder function.
    E_inv, _ = pytensor.scan(pt.linalg.solve, sequences=[E, Ij])
    eta_diag, _ = pytensor.scan(pt.diag, sequences=[eta])

    # block_dd: (L, J, J)
    block_dd = (
        pt.matrix_transpose(E_inv)
        @ (eta_diag + pt.matrix_transpose(z_xi) @ alpha_diag @ z_xi)
        @ E_inv
    )

    # (L, J, 2J)
    gamma_top = pt.concatenate([pt.zeros((L, J, J)), -E_inv], axis=-1)

    # (L, J, 2J)
    gamma_bottom = pt.concatenate([-pt.matrix_transpose(E_inv), block_dd], axis=-1)

    # (L, 2J, 2J)
    gamma = pt.concatenate([gamma_top, gamma_bottom], axis=1)

    return beta, gamma


def bfgs_sample(
    num_samples: int,
    x,  # position
    g,  # grad
    alpha,
    beta,
    gamma,
    random_seed: RandomSeed | None = None,
):
    # batch: L = 8
    # alpha_l: (N,)         => (L, N)
    # beta_l: (N, 2J)       => (L, N, 2J)
    # gamma_l: (2J, 2J)     => (L, 2J, 2J)
    # Q : (N, N)            => (L, N, N)
    # R: (N, 2J)            => (L, N, 2J)
    # u: (M, N)             => (L, M, N)
    # phi: (M, N)           => (L, M, N)
    # logdensity: (M,)      => (L, M)
    # theta: (J, N)

    rng = pytensor.shared(np.random.default_rng(seed=random_seed))

    def batched(x, g, alpha, beta, gamma):
        var_list = [x, g, alpha, beta, gamma]
        ndims = np.array([2, 2, 2, 3, 3])
        var_ndims = np.array([var.ndim for var in var_list])

        if np.all(var_ndims == ndims):
            return True
        elif np.all(var_ndims == ndims - 1):
            return False
        else:
            raise ValueError("Incorrect number of dimensions.")

    if not batched(x, g, alpha, beta, gamma):
        x = pt.atleast_2d(x)
        g = pt.atleast_2d(g)
        alpha = pt.atleast_2d(alpha)
        beta = pt.atleast_3d(beta)
        gamma = pt.atleast_3d(gamma)

    L, N = x.shape

    (alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag), _ = pytensor.scan(
        lambda a: [pt.diag(a), pt.diag(pt.sqrt(1.0 / a)), pt.diag(pt.sqrt(a))],
        sequences=[alpha],
    )

    qr_input = inv_sqrt_alpha_diag @ beta
    (Q, R), _ = pytensor.scan(fn=pt.nlinalg.qr, sequences=[qr_input])
    IdN = pt.repeat(pt.eye(R.shape[1])[None, ...], L, axis=0)
    Lchol_input = IdN + R @ gamma @ pt.matrix_transpose(R)
    Lchol = pt.linalg.cholesky(Lchol_input)

    logdet = pt.log(pt.prod(alpha, axis=-1)) + 2 * pt.log(pt.linalg.det(Lchol))

    mu = (
        x
        + pt.batched_dot(alpha_diag, g)
        + pt.batched_dot((beta @ gamma @ pt.matrix_transpose(beta)), g)
    )  # fmt: off

    u = pt.random.normal(size=(L, num_samples, N), rng=rng)

    phi = (
        mu[..., None]
        + sqrt_alpha_diag @ (Q @ (Lchol - IdN)) @ (pt.matrix_transpose(Q) @ pt.matrix_transpose(u))
        + pt.matrix_transpose(u)
    ).dimshuffle([0, 2, 1])

    logdensity = -0.5 * (
        logdet[..., None] + pt.sum(u * u, axis=-1) + N * pt.log(2.0 * pt.pi)
    )  # fmt: off

    # phi: (L, M, N)
    # logdensity: (L, M)
    return phi, logdensity


def compute_logp(logp_func, arr):
    logP = np.apply_along_axis(logp_func, axis=-1, arr=arr)
    # replace nan with -inf since np.argmax will return the first index at nan
    return np.where(np.isnan(logP), -np.inf, logP)


def single_pathfinder(
    model,
    num_draws: int,
    maxcor: int | None = None,
    maxiter: int = 1000,
    ftol: float = 1e-10,
    gtol: float = 1e-16,
    maxls: int = 1000,
    num_elbo_draws: int = 10,
    jitter: float = 2.0,
    epsilon: float = 1e-11,
    random_seed: RandomSeed | None = None,
):
    jitter_seed, pathfinder_seed, sample_seed = _get_seeds_per_chain(random_seed, 3)
    logp_func, dlogp_func = get_logp_dlogp_of_ravel_inputs(model)
    ip_map = make_initial_pathfinder_point(model, jitter=jitter, random_seed=jitter_seed)

    def neg_logp_func(x):
        return -logp_func(x)

    def neg_dlogp_func(x):
        return -dlogp_func(x)

    if maxcor is None:
        maxcor = np.ceil(2 * ip_map.data.shape[0] / 3).astype("int32")

    """
    The following excerpt is from Zhang et al., (2022):
    "In some cases, the optimization path terminates at the initialization point and in others it can fail to generate a positive deﬁnite inverse Hessian estimate. In both of these settings, Pathﬁnder essentially fails. Rather than worry about coding exceptions or failure return codes, Pathﬁnder returns the last iteration of the optimization path as a single approximating draw with infinity for the approximate normal log density of the draw. This ensures that failed ﬁts get zero importance weights in the multi-path Pathﬁnder algorithm, which we describe in the next section."
    # TODO: apply the above excerpt to the Pathfinder algorithm.
    """

    lbfgs_history = lbfgs(
        fn=neg_logp_func,
        grad_fn=neg_dlogp_func,
        x0=ip_map.data,
        maxcor=maxcor,
        maxiter=maxiter,
        ftol=ftol,
        gtol=gtol,
        maxls=maxls,
    )

    # x_full, g_full: (L+1, N)
    x_full = pt.as_tensor(lbfgs_history.x, dtype="float64")
    g_full = pt.as_tensor(lbfgs_history.g, dtype="float64")

    # ignore initial point - x, g: (L, N)
    x = x_full[1:]
    g = g_full[1:]

    alpha, S, Z, update_mask = alpha_recover(x_full, g_full, epsilon=epsilon)
    beta, gamma = inverse_hessian_factors(alpha, S, Z, update_mask, J=maxcor)

    phi, logQ_phi = bfgs_sample(
        num_samples=num_elbo_draws,
        x=x,
        g=g,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        random_seed=pathfinder_seed,
    )

    # .vectorize is slower than apply_along_axis
    logP_phi = compute_logp(logp_func, phi.eval())
    logQ_phi = logQ_phi.eval()
    elbo = (logP_phi - logQ_phi).mean(axis=-1)
    lstar = np.argmax(elbo)

    # BUG: elbo may all be -inf for all l in L. So np.argmax(elbo) will return 0 which is wrong. Still, this won't affect the posterior samples in the multipath Pathfinder scenario because of PSIS/PSIR step. However, the user is left unaware of a failed Pathfinder run.
    # TODO: handle this case, e.g. by warning of a failed Pathfinder run and skip the following bfgs_sample step to save time.

    psi, logQ_psi = bfgs_sample(
        num_samples=num_draws,
        x=x[lstar],
        g=g[lstar],
        alpha=alpha[lstar],
        beta=beta[lstar],
        gamma=gamma[lstar],
        random_seed=sample_seed,
    )
    psi = psi.eval()
    logQ_psi = logQ_psi.eval()
    logP_psi = compute_logp(logp_func, psi)
    # psi: (1, M, N)
    # logP_psi: (1, M)
    # logQ_psi: (1, M)
    return psi, logP_psi, logQ_psi


def make_initial_pathfinder_point(
    model,
    jitter: float = 2.0,
    random_seed: RandomSeed | None = None,
) -> DictToArrayBijection:
    """
    create jittered initial point for pathfinder

    Parameters
    ----------
    model : Model
        pymc model
    jitter : float
        initial values in the unconstrained space are jittered by the uniform distribution, U(-jitter, jitter). Set jitter to 0 for no jitter.
    random_seed : RandomSeed | None
        random seed for reproducibility

    Returns
    -------
    DictToArrayBijection
        bijection containing jittered initial point
    """

    # TODO: replace rng.uniform (pseudo random sequence) with scipy.stats.qmc.Sobol (quasi-random sequence)
    # Sobol is a better low discrepancy sequence than uniform.

    ipfn = make_initial_point_fn(
        model=model,
    )
    ip = Point(ipfn(random_seed), model=model)
    ip_map = DictToArrayBijection.map(ip)

    rng = np.random.default_rng(random_seed)
    jitter_value = rng.uniform(-jitter, jitter, size=ip_map.data.shape)
    ip_map = ip_map._replace(data=ip_map.data + jitter_value)
    return ip_map


def _run_single_pathfinder(model, path_id: int, random_seed: RandomSeed, **kwargs):
    """Helper to run single pathfinder instance"""
    try:
        # Handle pickling
        in_out_pickled = isinstance(model, bytes)
        if in_out_pickled:
            model = cloudpickle.loads(model)
            kwargs = {k: cloudpickle.loads(v) for k, v in kwargs.items()}

        # Run pathfinder with explicit random_seed
        samples, logP, logQ = single_pathfinder(model=model, random_seed=random_seed, **kwargs)

        # Return results
        if in_out_pickled:
            return cloudpickle.dumps((samples, logP, logQ))
        return samples, logP, logQ

    except Exception as e:
        logger.error(f"Error in path {path_id}: {e!s}")
        raise


def _get_mp_context(mp_ctx=None):
    """code snippet taken from ParallelSampler in pymc/pymc/sampling/parallel.py"""
    if mp_ctx is None or isinstance(mp_ctx, str):
        if mp_ctx is None and platform.system() == "Darwin":
            if platform.processor() == "arm":
                mp_ctx = "fork"
                logger.debug(
                    "mp_ctx is set to 'fork' for MacOS with ARM architecture. "
                    + "This might cause unexpected behavior with JAX, which is inherently multithreaded."
                )
            else:
                mp_ctx = "forkserver"

        mp_ctx = multiprocessing.get_context(mp_ctx)
    return mp_ctx


def process_multipath_pathfinder_results(
    results: PathfinderResults,
):
    """process pathfinder results to prepare for pareto smoothed importance resampling (PSIR)

    Parameters
    ----------
    results : PathfinderResults
        results from pathfinder

    Returns
    -------
    tuple
        processed samples, logP and logQ arrays
    """
    # path[samples]: (I, M, N)
    N = results.paths[0]["samples"].shape[-1]

    paths_array = np.array([results.paths[i] for i in range(results.num_paths)])
    logP = np.concatenate([path["logP"] for path in paths_array])
    logQ = np.concatenate([path["logQ"] for path in paths_array])
    samples = np.concatenate([path["samples"] for path in paths_array])
    samples = samples.reshape(-1, N, order="F")

    # adjust log densities
    log_I = np.log(results.num_paths)
    logP -= log_I
    logQ -= log_I

    return samples, logP, logQ


def multipath_pathfinder(
    model: Model,
    num_paths: int,
    num_draws: int,
    num_draws_per_path: int,
    maxcor: int | None = None,
    maxiter: int = 1000,
    ftol: float = 1e-10,
    gtol: float = 1e-16,
    maxls: int = 1000,
    num_elbo_draws: int = 10,
    jitter: float = 2.0,
    epsilon: float = 1e-11,
    psis_resample: bool = True,
    random_seed: RandomSeed = None,
    **pathfinder_kwargs,
):
    """Run multiple pathfinder instances in parallel."""
    ctx = _get_mp_context(None)
    seeds = _get_seeds_per_chain(random_seed, num_paths + 1)
    path_seeds = seeds[:-1]
    choice_seed = seeds[-1]

    try:
        num_dims = DictToArrayBijection.map(model.initial_point()).data.shape[0]
        model_pickled = cloudpickle.dumps(model)
        kwargs = {
            "num_draws": num_draws_per_path,  # for single pathfinder only
            "maxcor": maxcor,
            "maxiter": maxiter,
            "ftol": ftol,
            "gtol": gtol,
            "maxls": maxls,
            "num_elbo_draws": num_elbo_draws,
            "jitter": jitter,
            "epsilon": epsilon,
            **pathfinder_kwargs,
        }
        kwargs_pickled = {k: cloudpickle.dumps(v) for k, v in kwargs.items()}
    except Exception as e:
        raise ValueError(
            "Failed to pickle model or kwargs. This might be due to spawn context "
            f"limitations. Error: {e!s}"
        )

    mpf_results = PathfinderResults(num_paths, num_draws_per_path, num_dims)
    with ProcessPoolExecutor(mp_context=ctx) as executor:
        futures = {}
        try:
            for path_id, path_seed in enumerate(path_seeds):
                future = executor.submit(
                    _run_single_pathfinder, model_pickled, path_id, path_seed, **kwargs_pickled
                )
                futures[future] = path_id
                logger.debug(f"Submitted path {path_id} with seed {path_seed}")
        except Exception as e:
            logger.error(f"Failed to submit path {path_id}: {e!s}")
            raise

        failed_paths = []
        for future in as_completed(futures):
            path_id = futures[future]
            try:
                samples, logP, logQ = cloudpickle.loads(future.result())
                mpf_results.add_path_data(path_id, samples, logP, logQ)
            except Exception as e:
                failed_paths.append(path_id)
                logger.error(f"Path {path_id} failed: {e!s}")

    samples, logP, logQ = process_multipath_pathfinder_results(mpf_results)
    if psis_resample:
        return psir(samples, logP=logP, logQ=logQ, num_draws=num_draws, random_seed=choice_seed)
    else:
        return samples


def fit_pathfinder(
    model,
    num_paths: int = 1,  # I
    num_draws: int = 1000,  # R
    num_draws_per_path: int = 1000,  # M
    maxcor: int | None = None,  # J
    maxiter: int = 1000,  # L^max
    ftol: float = 1e-10,
    gtol: float = 1e-16,
    maxls=1000,
    num_elbo_draws: int = 10,  # K
    jitter: float = 2.0,
    epsilon: float = 1e-11,
    psis_resample: bool = True,
    random_seed: RandomSeed | None = None,
    postprocessing_backend: Literal["cpu", "gpu"] = "cpu",
    inference_backend: Literal["pymc", "blackjax"] = "pymc",
    **pathfinder_kwargs,
):
    """
    Fit the Pathfinder Variational Inference algorithm.

    This function fits the Pathfinder algorithm to a given PyMC model, allowing
    for multiple paths and draws. It supports both PyMC and BlackJAX backends.

    Parameters
    ----------
    model : pymc.Model
        The PyMC model to fit the Pathfinder algorithm to.
    num_paths : int
        Number of independent paths to run in the Pathfinder algorithm.
    num_draws : int, optional
        Total number of samples to draw from the fitted approximation (default is 1000).
    num_draws_per_path : int, optional
        Number of samples to draw per path (default is 1000).
    maxcor : int, optional
        Maximum number of variable metric corrections used to define the limited memory matrix.
    maxiter : int, optional
        Maximum number of iterations for the L-BFGS optimisation (default is 1000).
    ftol : float, optional
        Tolerance for the decrease in the objective function (default is 1e-10).
    gtol : float, optional
        Tolerance for the norm of the gradient (default is 1e-16).
    maxls : int, optional
        Maximum number of line search steps for the L-BFGS algorithm (default is 1000).
    num_elbo_draws : int, optional
        Number of draws for the Evidence Lower Bound (ELBO) estimation (default is 10).
    jitter : float, optional
        Amount of jitter to apply to initial points (default is 2.0).
    epsilon: float
        value used to filter out large changes in the direction of the update gradient at each iteration l in L. iteration l are only accepted if delta_theta[l] * delta_grad[l] > epsilon * L2_norm(delta_grad[l]) for each l in L. (default is 1e-11).
    psis_resample : bool, optional
        Whether to apply Pareto Smoothed Importance Sampling Resampling (default is True). If false, the samples are returned as is (i.e. no resampling is applied) of the size num_draws_per_path * num_paths.
    random_seed : RandomSeed, optional
        Random seed for reproducibility.
    postprocessing_backend : str, optional
        Backend for postprocessing transformations, either "cpu" or "gpu" (default is "cpu").
    inference_backend : str, optional
        Backend for inference, either "pymc" or "blackjax" (default is "pymc").
    **pathfinder_kwargs
        Additional keyword arguments for the Pathfinder algorithm.

    Returns
    -------
    arviz.InferenceData
        The inference data containing the results of the Pathfinder algorithm.

    References
    ----------
    Zhang, L., Carpenter, B., Gelman, A., & Vehtari, A. (2022). Pathfinder: Parallel quasi-Newton variational inference. Journal of Machine Learning Research, 23(306), 1-49.
    """
    # Temporarily helper
    if version.parse(blackjax.__version__).major < 1:
        raise ImportError("fit_pathfinder requires blackjax 1.0 or above")

    model = modelcontext(model)

    # TODO: move the initial point jittering outside
    # TODO: Set initial points. PF requires jittering of initial points. See https://github.com/pymc-devs/pymc/issues/7555

    if inference_backend == "pymc":
        pathfinder_samples = multipath_pathfinder(
            model,
            num_paths=num_paths,
            num_draws=num_draws,
            num_draws_per_path=num_draws_per_path,
            maxcor=maxcor,
            maxiter=maxiter,
            ftol=ftol,
            gtol=gtol,
            maxls=maxls,
            num_elbo_draws=num_elbo_draws,
            jitter=jitter,
            epsilon=epsilon,
            psis_resample=psis_resample,
            random_seed=random_seed,
            **pathfinder_kwargs,
        )

    elif inference_backend == "blackjax":
        jitter_seed, pathfinder_seed, sample_seed = _get_seeds_per_chain(random_seed, 3)
        # TODO: extend initial points initialisation to blackjax
        # TODO: extend blackjax pathfinder to multiple paths
        ipfn = make_initial_point_fn(
            model=model,
            jitter_rvs=set(model.free_RVs),
        )
        ip = Point(ipfn(jitter_seed), model=model)
        ip_map = DictToArrayBijection.map(ip)
        if maxcor is None:
            maxcor = np.ceil(2 * ip_map.data.shape[0] / 3).astype("int32")
        logp_func = get_jaxified_logp_of_ravel_inputs(model)
        pathfinder_state, pathfinder_info = blackjax.vi.pathfinder.approximate(
            rng_key=jax.random.key(pathfinder_seed),
            logdensity_fn=logp_func,
            initial_position=ip_map.data,
            num_samples=num_elbo_draws,
            maxiter=maxiter,
            maxcor=maxcor,
            maxls=maxls,
            ftol=ftol,
            gtol=gtol,
            **pathfinder_kwargs,
        )
        pathfinder_samples, _ = blackjax.vi.pathfinder.sample(
            rng_key=jax.random.key(sample_seed),
            state=pathfinder_state,
            num_samples=num_draws,
        )

    else:
        raise ValueError(f"Inference backend {inference_backend} not supported")

    print("Running pathfinder...", file=sys.stdout)

    idata = convert_flat_trace_to_idata(
        pathfinder_samples,
        postprocessing_backend=postprocessing_backend,
        model=model,
    )
    return idata
