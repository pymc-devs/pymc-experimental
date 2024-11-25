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
import functools
import logging
import multiprocessing
import platform

from collections.abc import Callable
from importlib.util import find_spec
from typing import Literal

import arviz as az
import blackjax
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
from pymc.pytensorf import compile_pymc, find_rng_nodes, replace_rng_nodes, reseed_rngs
from pymc.sampling.jax import get_jaxified_graph
from pymc.util import RandomSeed, _get_seeds_per_chain, get_default_varnames
from pytensor.compile.mode import FAST_COMPILE, FAST_RUN
from pytensor.graph import Apply, Op, vectorize_graph
from pytensor.tensor.variable import TensorVariable

from pymc_experimental.inference.pathfinder.importance_sampling import psir
from pymc_experimental.inference.pathfinder.lbfgs import LBFGSInitFailed, LBFGSOp

logger = logging.getLogger(__name__)

REGULARISATION_TERM = 1e-8


def make_seeded_function(
    func: Callable | None = None,
    inputs: list[TensorVariable] | None = [],
    outputs: list[TensorVariable] | None = None,
    compile_kwargs: dict = {},
) -> Callable:
    if (outputs is None) and (func is not None):
        outputs = func(*inputs)
    elif (outputs is None) and (func is None):
        raise ValueError("func must be provided if outputs are not provided")

    if not isinstance(outputs, list | tuple):
        outputs = [outputs]

    # Q: do I need replace_rng_nodes? It still works without it.
    outputs = replace_rng_nodes(outputs)
    default_compile_kwargs = {"mode": FAST_RUN}
    compile_kwargs = default_compile_kwargs | compile_kwargs
    func_compiled = compile_pymc(
        inputs=inputs,
        outputs=outputs,
        on_unused_input="ignore",
        **compile_kwargs,
    )
    rngs = find_rng_nodes(func_compiled.maker.fgraph.outputs)

    @functools.wraps(func_compiled)
    def inner(random_seed=None, *args, **kwargs):
        if random_seed is not None:
            reseed_rngs(rngs, random_seed)
        return func_compiled(*args, **kwargs)

    return inner


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


def get_logp_dlogp_of_ravel_inputs(model: Model, jacobian: bool = False):
    outputs, inputs = pm.pytensorf.join_nonshared_inputs(
        model.initial_point(),
        [model.logp(jacobian=jacobian), model.dlogp(jacobian=jacobian)],
        model.value_vars,
    )

    logp_func = compile_pymc([inputs], outputs[0], mode=FAST_RUN)
    logp_func.trust_input = True

    dlogp_func = compile_pymc([inputs], outputs[1], mode=FAST_RUN)
    dlogp_func.trust_input = True

    return logp_func, dlogp_func


def convert_flat_trace_to_idata(
    samples,
    include_transformed=False,
    postprocessing_backend="cpu",
    inference_backend="pymc",
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
            # instead of .tolist(), use np.asarray(v) since array sizes are known
            trace[p].append(v.tolist())

    trace = {k: np.asarray(v)[None, ...] for k, v in trace.items()}

    var_names = model.unobserved_value_vars
    vars_to_sample = list(get_default_varnames(var_names, include_transformed=include_transformed))
    logger.info("Transforming variables...")

    if inference_backend == "pymc":
        new_shapes = [v.ndim * (None,) for v in trace.values()]
        replace = {
            var: pt.tensor(dtype="float64", shape=new_shapes[i])
            for i, var in enumerate(model.value_vars)
        }

        outputs = vectorize_graph(vars_to_sample, replace=replace)

        fn = pytensor.function(
            inputs=[*list(replace.values())],
            outputs=outputs,
            mode=FAST_COMPILE,
            on_unused_input="ignore",
        )
        fn.trust_input = True
        result = fn(*list(trace.values()))
    elif inference_backend == "blackjax":
        jax_fn = get_jaxified_graph(inputs=model.value_vars, outputs=vars_to_sample)
        result = jax.vmap(jax.vmap(jax_fn))(
            *jax.device_put(list(trace.values()), jax.devices(postprocessing_backend)[0])
        )
    else:
        raise ValueError(f"Invalid inference_backend: {inference_backend}")

    trace = {v.name: r for v, r in zip(vars_to_sample, result)}
    coords, dims = coords_and_dims_for_inferencedata(model)
    idata = az.from_dict(trace, dims=dims, coords=coords)

    return idata


def alpha_recover(x, g, epsilon):
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
    s = pt.diff(x, axis=0)
    z = pt.diff(g, axis=0)
    alpha_l_init = pt.ones(N)
    sz = (s * z).sum(axis=-1)
    update_mask = sz > epsilon * pt.linalg.norm(z, axis=-1)

    alpha, _ = pytensor.scan(
        fn=scan_body,
        outputs_info=alpha_l_init,
        sequences=[update_mask, s, z],
        n_steps=Lp1 - 1,
        allow_gc=False,
    )

    # assert np.all(alpha.eval() > 0), "alpha cannot be negative"
    # alpha: (L, N), update_mask: (L, N)
    return alpha, s, z, update_mask


def inverse_hessian_factors(alpha, s, z, update_mask, J):
    # NOTE: get_chi_matrix_1 is a modified version of get_chi_matrix_2 to closely follow Zhang et al., (2022)
    # NOTE: get_chi_matrix_2 is from blackjax which MAYBE incorrectly implemented

    def get_chi_matrix_1(diff, update_mask, J):
        L, N = diff.shape
        j_last = pt.as_tensor(J - 1)  # since indexing starts at 0

        def chi_update(chi_lm1, diff_l):
            chi_l = pt.roll(chi_lm1, -1, axis=0)
            return pt.set_subtensor(chi_l[j_last], diff_l)

        def no_op(chi_lm1, diff_l):
            return chi_lm1

        def scan_body(update_mask_l, diff_l, chi_lm1):
            return pt.switch(update_mask_l, chi_update(chi_lm1, diff_l), no_op(chi_lm1, diff_l))

        chi_init = pt.zeros((J, N))
        chi_mat, _ = pytensor.scan(
            fn=scan_body,
            outputs_info=chi_init,
            sequences=[
                update_mask,
                diff,
            ],
            allow_gc=False,
        )

        chi_mat = pt.matrix_transpose(chi_mat)

        # (L, N, J)
        return chi_mat

    def get_chi_matrix_2(diff, update_mask, J):
        L, N = diff.shape

        diff_masked = update_mask[:, None] * diff

        # diff_padded: (L+J, N)
        pad_width = pt.zeros(shape=(2, 2), dtype="int32")
        pad_width = pt.set_subtensor(pad_width[0, 0], J)
        diff_padded = pt.pad(diff_masked, pad_width, mode="constant")

        index = pt.arange(L)[:, None] + pt.arange(J)[None, :]
        index = index.reshape((L, J))

        chi_mat = pt.matrix_transpose(diff_padded[index])

        # (L, N, J)
        return chi_mat

    L, N = alpha.shape
    S = get_chi_matrix_1(s, update_mask, J)
    Z = get_chi_matrix_1(z, update_mask, J)

    # E: (L, J, J)
    Ij = pt.eye(J)[None, ...]
    E = pt.triu(pt.matrix_transpose(S) @ Z)
    E += Ij * REGULARISATION_TERM

    # eta: (L, J)
    eta = pt.diagonal(E, axis1=-2, axis2=-1)

    # beta: (L, N, 2J)
    alpha_diag, _ = pytensor.scan(lambda a: pt.diag(a), sequences=[alpha])
    beta = pt.concatenate([alpha_diag @ Z, S], axis=-1)

    # more performant and numerically precise to use solve than inverse: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.inv.html

    # E_inv: (L, J, J)
    E_inv = pt.slinalg.solve_triangular(E, Ij, check_finite=False)
    eta_diag, _ = pytensor.scan(pt.diag, sequences=[eta])

    # block_dd: (L, J, J)
    block_dd = (
        pt.matrix_transpose(E_inv) @ (eta_diag + pt.matrix_transpose(Z) @ alpha_diag @ Z) @ E_inv
    )

    # (L, J, 2J)
    gamma_top = pt.concatenate([pt.zeros((L, J, J)), -E_inv], axis=-1)

    # (L, J, 2J)
    gamma_bottom = pt.concatenate([-pt.matrix_transpose(E_inv), block_dd], axis=-1)

    # (L, 2J, 2J)
    gamma = pt.concatenate([gamma_top, gamma_bottom], axis=1)

    return beta, gamma


def bfgs_sample_dense(
    x,
    g,
    alpha,
    beta,
    gamma,
    alpha_diag,
    inv_sqrt_alpha_diag,
    sqrt_alpha_diag,
    u,
):
    N = x.shape[-1]
    IdN = pt.eye(N)[None, ...]

    # inverse Hessian
    H_inv = (
        sqrt_alpha_diag
        @ (
            IdN
            + inv_sqrt_alpha_diag @ beta @ gamma @ pt.matrix_transpose(beta) @ inv_sqrt_alpha_diag
        )
        @ sqrt_alpha_diag
    )

    Lchol = pt.linalg.cholesky(H_inv, lower=False, check_finite=False, on_error="nan")

    logdet = 2.0 * pt.sum(pt.log(pt.abs(pt.diagonal(Lchol, axis1=-2, axis2=-1))), axis=-1)

    mu = x - pt.batched_dot(H_inv, g)

    phi = pt.matrix_transpose(
        # (L, N, 1)
        mu[..., None]
        # (L, N, M)
        + Lchol @ pt.matrix_transpose(u)
    )  # fmt: off

    return phi, logdet


def bfgs_sample_sparse(
    x,
    g,
    alpha,
    beta,
    gamma,
    alpha_diag,
    inv_sqrt_alpha_diag,
    sqrt_alpha_diag,
    u,
):
    # qr_input: (L, N, 2J)
    qr_input = inv_sqrt_alpha_diag @ beta
    (Q, R), _ = pytensor.scan(fn=pt.nlinalg.qr, sequences=[qr_input], allow_gc=False)
    IdN = pt.eye(R.shape[1])[None, ...]
    Lchol_input = IdN + R @ gamma @ pt.matrix_transpose(R)

    Lchol = pt.linalg.cholesky(Lchol_input, lower=False, check_finite=False, on_error="nan")

    logdet = 2.0 * pt.sum(pt.log(pt.abs(pt.diagonal(Lchol, axis1=-2, axis2=-1))), axis=-1)
    logdet += pt.sum(pt.log(alpha), axis=-1)

    # NOTE: changed the sign from "x + " to "x -" of the expression to match Stan which differs from Zhang et al., (2022). same for dense version.
    mu = x - (
        # (L, N), (L, N) -> (L, N)
        pt.batched_dot(alpha_diag, g)
        # beta @ gamma @ beta.T
        # (L, N, 2J), (L, 2J, 2J), (L, 2J, N) -> (L, N, N)
        # (L, N, N), (L, N) -> (L, N)
        + pt.batched_dot((beta @ gamma @ pt.matrix_transpose(beta)), g)
    )

    phi = pt.matrix_transpose(
        # (L, N, 1)
        mu[..., None]
        # (L, N, N), (L, N, M) -> (L, N, M)
        + sqrt_alpha_diag
        @ (
            # (L, N, 2J), (L, 2J, M) -> (L, N, M)
            # intermediate calcs below
            # (L, N, 2J), (L, 2J, 2J) -> (L, N, 2J)
            (Q @ (Lchol - IdN))
            # (L, 2J, N), (L, N, M) -> (L, 2J, M)
            @ (pt.matrix_transpose(Q) @ pt.matrix_transpose(u))
            # (L, N, M)
            + pt.matrix_transpose(u)
        )
    )  # fmt: off

    return phi, logdet


def bfgs_sample(
    num_samples: int,
    x,  # position
    g,  # grad
    alpha,
    beta,
    gamma,
    index: int | None = None,
    # random_seed: RandomSeed | None = None,
    # rng,
):
    # batch: L = 8
    # alpha_l: (N,)         => (L, N)
    # beta_l: (N, 2J)       => (L, N, 2J)
    # gamma_l: (2J, 2J)     => (L, 2J, 2J)
    # Q : (N, 2J)           => (L, N, 2J)
    # R: (2J, 2J)           => (L, 2J, 2J)
    # u: (M, N)             => (L, M, N)
    # phi: (M, N)           => (L, M, N)
    # logQ_phi: (M,)        => (L, M)
    # Lchol: (2J, 2J)       => (L, 2J, 2J)
    # theta: (J, N)

    if index is not None:
        x = x[index][None, ...]
        g = g[index][None, ...]
        alpha = alpha[index][None, ...]
        beta = beta[index][None, ...]
        gamma = gamma[index][None, ...]

    L, N, JJ = beta.shape

    (alpha_diag, inv_sqrt_alpha_diag, sqrt_alpha_diag), _ = pytensor.scan(
        lambda a: [pt.diag(a), pt.diag(pt.sqrt(1.0 / a)), pt.diag(pt.sqrt(a))],
        sequences=[alpha],
        allow_gc=False,
    )

    u = pt.random.normal(size=(L, num_samples, N))

    sample_inputs = (
        x,
        g,
        alpha,
        beta,
        gamma,
        alpha_diag,
        inv_sqrt_alpha_diag,
        sqrt_alpha_diag,
        u,
    )

    # ifelse is faster than pt.switch I think?
    phi, logdet = pytensor.ifelse(
        JJ >= N,
        bfgs_sample_dense(*sample_inputs),
        bfgs_sample_sparse(*sample_inputs),
    )

    logQ_phi = -0.5 * (
        logdet[..., None]
        + pt.sum(u * u, axis=-1)
        + N * pt.log(2.0 * pt.pi)
    )  # fmt: off

    nan_mask = pt.isnan(logQ_phi)

    # TODO: let users know if there are NaNs in logQ_phi
    # nan values would occur from cholesky (where check_finite=False, raise="nan") and solve_triangular
    logQ_phi = pt.set_subtensor(logQ_phi[nan_mask], pt.inf)
    # phi: (L, M, N)
    # logQ_phi: (L, M)
    return phi, logQ_phi


# TODO: remove make_initial_points function when feature request is implemented: https://github.com/pymc-devs/pymc/issues/7555
def make_initial_points(
    random_seed: RandomSeed | None = None,
    model=None,
    jitter: float = 2.0,
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
    ndarray
        jittered initial point
    """

    ipfn = make_initial_point_fn(
        model=model,
    )
    ip = Point(ipfn(random_seed), model=model)
    ip_map = DictToArrayBijection.map(ip)

    rng = np.random.default_rng(random_seed)
    jitter_value = rng.uniform(-jitter, jitter, size=ip_map.data.shape)
    ip_map = ip_map._replace(data=ip_map.data + jitter_value)
    return ip_map.data


def compute_logp(logp_func, arr):
    # .vectorize is slower than apply_along_axis
    logP = np.apply_along_axis(logp_func, axis=-1, arr=arr)
    # replace nan with -inf since np.argmax will return the first index at nan
    nan_mask = np.isnan(logP)
    logger.info(f"Number of NaNs in logP in a path: {np.sum(nan_mask)}")
    return np.where(nan_mask, -np.inf, logP)


class LogLike(Op):
    __props__ = ()

    def __init__(self, logp_func):
        self.logp_func = logp_func
        super().__init__()

    def make_node(self, inputs):
        # Convert inputs to tensor variables
        inputs = pt.as_tensor(inputs)
        outputs = pt.tensor(dtype="float64", shape=(None, None))
        return Apply(self, [inputs], [outputs])

    def perform(self, node: Apply, inputs, outputs) -> None:
        phi = inputs[0]
        logp = compute_logp(self.logp_func, arr=phi)
        outputs[0][0] = logp


def make_initial_points_fn(model, jitter):
    return functools.partial(make_initial_points, model=model, jitter=jitter)


def make_lbfgs_fn(fn, grad_fn, maxcor, maxiter, ftol, gtol, maxls):
    x0 = pt.dvector("x0")
    lbfgs_op = LBFGSOp(fn, grad_fn, maxcor, maxiter, ftol, gtol, maxls)
    return pytensor.function([x0], lbfgs_op(x0), mode=FAST_RUN)


def make_pathfinder_body(logp_func, num_draws, maxcor, num_elbo_draws, epsilon):
    """Returns a compiled function f where:
    f-inputs:
        seeds:list[int, int],
        x_full: ndarray[L+1, N],
        g_full: ndarray[L+1, N]
    f-outputs:
        psi: ndarray[1, M, N],
        logP_psi: ndarray[1, M],
        logQ_psi: ndarray[1, M]
    """

    # x_full, g_full: (L+1, N)
    x_full = pt.matrix("x", dtype="float64")
    g_full = pt.matrix("g", dtype="float64")

    num_draws = pt.constant(num_draws, "num_draws", dtype="int32")
    num_elbo_draws = pt.constant(num_elbo_draws, "num_elbo_draws", dtype="int32")
    epsilon = pt.constant(epsilon, "epsilon", dtype="float64")
    maxcor = pt.constant(maxcor, "maxcor", dtype="int32")

    alpha, s, z, update_mask = alpha_recover(x_full, g_full, epsilon=epsilon)
    beta, gamma = inverse_hessian_factors(alpha, s, z, update_mask, J=maxcor)

    # ignore initial point - x, g: (L, N)
    x = x_full[1:]
    g = g_full[1:]

    phi, logQ_phi = bfgs_sample(
        num_samples=num_elbo_draws, x=x, g=g, alpha=alpha, beta=beta, gamma=gamma
    )

    loglike = LogLike(logp_func)
    logP_phi = loglike(phi)
    elbo = pt.mean(logP_phi - logQ_phi, axis=-1)
    lstar = pt.argmax(elbo)

    psi, logQ_psi = bfgs_sample(
        num_samples=num_draws,
        x=x,
        g=g,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        index=lstar,
    )
    logP_psi = loglike(psi)

    return make_seeded_function(
        inputs=[x_full, g_full],
        outputs=[psi, logP_psi, logQ_psi],
    )


def make_single_pathfinder_fn(
    model,
    num_draws: int,
    maxcor: int = 5,
    maxiter: int = 1000,
    ftol: float = 1e-10,
    gtol: float = 1e-16,
    maxls: int = 1000,
    num_elbo_draws: int = 10,
    jitter: float = 2.0,
    epsilon: float = 1e-11,
):
    logp_func, dlogp_func = get_logp_dlogp_of_ravel_inputs(model)

    def neg_logp_func(x):
        return -logp_func(x)

    def neg_dlogp_func(x):
        return -dlogp_func(x)

    # initial_point_fn: (jitter_seed) -> x0
    initial_point_fn = make_initial_points_fn(model=model, jitter=jitter)

    # lbfgs_fn: (x0) -> (x, g)
    lbfgs_fn = make_lbfgs_fn(neg_logp_func, neg_dlogp_func, maxcor, maxiter, ftol, gtol, maxls)
    lbfgs_fn.trust_input = True

    # pathfinder_body_fn: (tuple[elbo_draw_seed, num_draws_seed], x, g) -> (psi, logP_psi, logQ_psi)
    pathfinder_body_fn = make_pathfinder_body(logp_func, num_draws, maxcor, num_elbo_draws, epsilon)
    pathfinder_body_fn.trust_input = True

    """
    BUG: elbo may all be -inf for all l in L. So np.argmax(elbo) will return 0 which is wrong. Still, this won't affect the posterior samples in the multipath Pathfinder scenario because of PSIS/PSIR step. However, the user is left unaware of a failed Pathfinder run.
    # TODO: handle this case, e.g. by warning of a failed Pathfinder run and skip the following bfgs_sample step to save time.
    """

    def single_pathfinder_fn(random_seed):
        # pathfinder_body_fn has 2 shared variable RNGs in the graph as bfgs_sample gets called twice.
        jitter_seed, *pathfinder_seed = _get_seeds_per_chain(random_seed, 3)
        x0 = initial_point_fn(jitter_seed)
        x, g, status = lbfgs_fn(x0)
        psi, logP_psi, logQ_psi = pathfinder_body_fn(pathfinder_seed, x, g)
        # psi: (1, M, N)
        # logP_psi: (1, M)
        # logQ_psi: (1, M)
        return psi, logP_psi, logQ_psi

    # single_pathfinder_fn: (random_seed) -> (psi, logP_psi, logQ_psi)
    return single_pathfinder_fn


# keep this in case we need it for multiprocessing
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


def calculate_processes():
    total_cpus = multiprocessing.cpu_count() or 1
    processes = max(2, int(total_cpus * 0.3))
    if processes % 2 != 0:
        processes += 1
    return processes


def multipath_pathfinder(
    model: Model,
    num_paths: int,
    num_draws: int,
    num_draws_per_path: int,
    maxcor: int = 5,
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
    seeds = _get_seeds_per_chain(random_seed, num_paths + 1)
    path_seeds = seeds[:-1]
    choice_seed = seeds[-1]

    single_pathfinder_fn = make_single_pathfinder_fn(
        model,
        num_draws_per_path,
        maxcor,
        maxiter,
        ftol,
        gtol,
        maxls,
        num_elbo_draws,
        jitter,
        epsilon,
    )

    results = []
    num_failed = 0
    num_success = num_paths
    for seed in path_seeds:
        try:
            results.append(single_pathfinder_fn(seed))
        except LBFGSInitFailed:
            num_failed += 1
            continue

    if num_failed > 0:
        logger.warning(f"Number of failed paths: {num_failed} out of {num_paths}")
        num_success -= num_failed
    if num_success == 0:
        raise ValueError(
            "All paths failed. Consider decreasing the jitter or reparameterising the model."
        )

    samples, logP, logQ = zip(*results)
    samples = np.concatenate(samples)
    logP = np.concatenate(logP)
    logQ = np.concatenate(logQ)

    samples = samples.reshape(num_success * num_draws_per_path, -1, order="C")
    logP = logP.reshape(num_success * num_draws_per_path, order="C")
    logQ = logQ.reshape(num_success * num_draws_per_path, order="C")

    # adjust log densities
    log_I = np.log(num_success)
    logP -= log_I
    logQ -= log_I
    logiw = logP - logQ
    if psis_resample:
        return psir(samples, logiw=logiw, num_draws=num_draws, random_seed=choice_seed)
    else:
        return samples


def fit_pathfinder(
    model=None,
    num_paths: int = 6,  # I
    num_draws: int = 1000,  # R
    num_draws_per_path: int = 1000,  # M
    maxcor: int = 5,  # J
    maxiter: int = 1000,  # L^max
    ftol: float = 1e-5,
    gtol: float = 1e-8,
    maxls=1000,
    num_elbo_draws: int = 10,  # K
    jitter: float = 2.0,
    epsilon: float = 1e-8,
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
        Number of independent paths to run in the Pathfinder algorithm. (default is 4)
    num_draws : int, optional
        Total number of samples to draw from the fitted approximation (default is 1000).
    num_draws_per_path : int, optional
        Number of samples to draw per path (default is 1000).
    maxcor : int, optional
        Maximum number of variable metric corrections used to define the limited memory matrix (default is 5).
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
        if find_spec("blackjax") is None:
            raise RuntimeError("Need BlackJAX to use `pathfinder`")
        if version.parse(blackjax.__version__).major < 1:
            raise ImportError("fit_pathfinder requires blackjax 1.0 or above")

        jitter_seed, pathfinder_seed, sample_seed = _get_seeds_per_chain(random_seed, 3)
        # TODO: extend initial points initialisation to blackjax
        # TODO: extend blackjax pathfinder to multiple paths
        ipfn = make_initial_point_fn(
            model=model,
            jitter_rvs=set(model.free_RVs),
        )
        ip = Point(ipfn(jitter_seed), model=model)
        ip_map = DictToArrayBijection.map(ip)
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
        raise ValueError(f"Invalid inference_backend: {inference_backend}")

    logger.info("Transforming variables...")

    idata = convert_flat_trace_to_idata(
        pathfinder_samples,
        postprocessing_backend=postprocessing_backend,
        inference_backend=inference_backend,
        model=model,
    )
    return idata
