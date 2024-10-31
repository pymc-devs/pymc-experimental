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
import sys

from collections.abc import Callable

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
from pymc.sampling.jax import get_jaxified_graph
from pymc.util import RandomSeed, _get_seeds_per_chain, get_default_varnames

from pymc_experimental.inference.lbfgs import lbfgs

REGULARISATION_TERM = 1e-8


def get_jaxified_logp_ravel_inputs(
    model: Model,
    initial_points: dict | None = None,
) -> tuple[Callable, DictToArrayBijection]:
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
        initial_points, (model.logp(),), model.value_vars, ()
    )

    logp_func_list = get_jaxified_graph([new_input], new_logprob)

    def logp_func(x):
        return logp_func_list(x)[0]

    return logp_func, DictToArrayBijection.map(initial_points)


def get_logp_dlogp_ravel_inputs(
    model: Model,
    initial_points: dict | None = None,
):  # -> tuple[Callable[..., Any], Callable[..., Any]]:
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

    return logp_func, dlogp_func, ip_map


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


def _get_delta_x_delta_g(x, g):
    # x or g: (L - 1, N)
    return pt.diff(x, axis=0), pt.diff(g, axis=0)


# TODO: potentially incorrect
def get_s_xi_z_xi(x, g, update_mask, J):
    L, N = x.shape
    S, Z = _get_delta_x_delta_g(x, g)
    # TODO: double check this
    # Z = -Z

    s_masked = update_mask[:, None] * S
    z_masked = update_mask[:, None] * Z

    # s_padded, z_padded: (L-1+J, N)
    s_padded = pt.pad(s_masked, ((J, 0), (0, 0)), mode="constant")
    z_padded = pt.pad(z_masked, ((J, 0), (0, 0)), mode="constant")

    index = pt.arange(L)[:, None] + pt.arange(J)[None, :]
    index = index.reshape((L, J))

    # s_xi, z_xi (L, N, J) # The J-th column needs to have the last update
    s_xi = s_padded[index].dimshuffle(0, 2, 1)
    z_xi = z_padded[index].dimshuffle(0, 2, 1)

    return s_xi, z_xi


def _get_chi_matrix(diff, update_mask, J):
    _, N = diff.shape
    j_last = pt.as_tensor(J - 1)  # since indexing starts at 0

    def z_xi_update(chi_lm1, diff_l):
        chi_l = pt.roll(chi_lm1, -1, axis=0)
        # z_xi_l = pt.set_subtensor(z_xi_l[j_last], z_l)
        # z_xi_l[j_last] = z_l
        return pt.set_subtensor(chi_l[j_last], diff_l)

    def no_op(chi_lm1, diff_l):
        return chi_lm1

    def scan_body(update_mask_l, diff_l, chi_lm1):
        return pt.switch(update_mask_l, z_xi_update(chi_lm1, diff_l), no_op(chi_lm1, diff_l))

    update_mask = pt.concatenate([pt.as_tensor([False], dtype="bool"), update_mask], axis=-1)
    diff = pt.concatenate([pt.zeros((1, N), dtype="float64"), diff], axis=0)

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


def _get_s_xi_z_xi(x, g, update_mask, J):
    L, N = x.shape
    S, Z = _get_delta_x_delta_g(x, g)
    # TODO: double check this
    # Z = -Z

    s_xi = _get_chi_matrix(S, update_mask, J)
    z_xi = _get_chi_matrix(Z, update_mask, J)

    return s_xi, z_xi


def alpha_recover(x, g):
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

    L, N = x.shape
    S, Z = _get_delta_x_delta_g(x, g)
    alpha_l_init = pt.ones(N)
    SZ = (S * Z).sum(axis=-1)
    update_mask = SZ > 1e-11 * pt.linalg.norm(Z, axis=-1)

    alpha, _ = pytensor.scan(
        fn=scan_body,
        outputs_info=alpha_l_init,
        sequences=[update_mask, S, Z],
        n_steps=L - 1,
        strict=True,
    )

    # alpha: (L, N), update_mask: (L-1, N)
    alpha = pt.concatenate([pt.ones(N)[None, :], alpha], axis=0)
    # assert np.all(alpha.eval() > 0), "alpha cannot be negative"
    return alpha, update_mask


def inverse_hessian_factors(alpha, x, g, update_mask, J):
    L, N = alpha.shape
    # s_xi, z_xi = get_s_xi_z_xi(x, g, update_mask, J)
    s_xi, z_xi = _get_s_xi_z_xi(x, g, update_mask, J)

    # (L, J, J)
    sz_xi = pt.matrix_transpose(s_xi) @ z_xi

    # E: (L, J, J)
    # Ij: (L, J, J)
    Ij = pt.repeat(pt.eye(J)[None, ...], L, axis=0)
    E = pt.triu(sz_xi) + Ij * REGULARISATION_TERM

    # eta: (L, J)
    eta, _ = pytensor.scan(pt.diag, sequences=[E])

    # beta: (L, N, 2J)
    alpha_diag, _ = pytensor.scan(lambda a: pt.diag(a), sequences=[alpha])
    beta = pt.concatenate([alpha_diag @ z_xi, s_xi], axis=-1)

    # more performant and numerically precise to use solve than inverse: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.inv.html

    # E_inv: (L, J, J)
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


def _batched(x, g, alpha, beta, gamma):
    var_list = [x, g, alpha, beta, gamma]
    ndims = np.array([2, 2, 2, 3, 3])
    var_ndims = np.array([var.ndim for var in var_list])

    if all(var_ndims == ndims):
        return True
    elif all(var_ndims == ndims - 1):
        return False
    else:
        raise ValueError(
            "All variables must have the same number of dimensions, either matching ndims or ndims - 1."
        )


def bfgs_sample(
    num_samples,
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

    if not _batched(x, g, alpha, beta, gamma):
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


def _pymc_pathfinder(
    model,
    x0: np.float64,
    num_draws: int,
    maxcor: int | None = None,
    maxiter=1000,
    ftol=1e-5,
    gtol=1e-8,
    maxls=1000,
    num_elbo_draws: int = 10,
    random_seed: RandomSeed = None,
):
    # TODO: insert single seed, then use _get_seeds_per_chain inside pymc_pathfinder
    pathfinder_seed, sample_seed = _get_seeds_per_chain(random_seed, 2)
    logp_func, dlogp_func, ip_map = get_logp_dlogp_ravel_inputs(model, initial_points=x0)

    def neg_logp_func(x):
        return -logp_func(x)

    def neg_dlogp_func(x):
        return -dlogp_func(x)

    if maxcor is None:
        maxcor = np.ceil(2 * ip_map.data.shape[0] / 3).astype("int32")

    history = lbfgs(
        neg_logp_func,
        neg_dlogp_func,
        ip_map.data,
        maxcor=maxcor,
        maxiter=maxiter,
        ftol=ftol,
        gtol=gtol,
        maxls=maxls,
    )

    alpha, update_mask = alpha_recover(history.x, history.g)

    beta, gamma = inverse_hessian_factors(alpha, history.x, history.g, update_mask, J=maxcor)

    phi, logq_phi = bfgs_sample(
        num_samples=num_elbo_draws,
        x=history.x,
        g=history.g,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        random_seed=pathfinder_seed,
    )

    # .vectorize is slower than apply_along_axis
    logp_phi = np.apply_along_axis(logp_func, axis=-1, arr=phi.eval())
    logq_phi = logq_phi.eval()
    elbo = (logp_phi - logq_phi).mean(axis=-1)
    lstar = np.argmax(elbo)

    psi, logq_psi = bfgs_sample(
        num_samples=num_draws,
        x=history.x[lstar],
        g=history.g[lstar],
        alpha=alpha[lstar],
        beta=beta[lstar],
        gamma=gamma[lstar],
        random_seed=sample_seed,
    )

    return psi[0].eval(), logq_psi, logp_func


def fit_pathfinder(
    model=None,
    num_draws=1000,
    maxcor=None,
    random_seed: RandomSeed | None = None,
    postprocessing_backend="cpu",
    inference_backend="pymc",
    **pathfinder_kwargs,
):
    """
    Fit the pathfinder algorithm as implemented in blackjax

    Requires the JAX backend

    Parameters
    ----------
    samples : int
        Number of samples to draw from the fitted approximation.
    random_seed : int
        Random seed to set.
    postprocessing_backend : str
        Where to compute transformations of the trace.
        "cpu" or "gpu".
    pathfinder_kwargs:
        kwargs for blackjax.vi.pathfinder.approximate

    Returns
    -------
    arviz.InferenceData

    Reference
    ---------
    https://arxiv.org/abs/2108.03782
    """
    # Temporarily helper
    if version.parse(blackjax.__version__).major < 1:
        raise ImportError("fit_pathfinder requires blackjax 1.0 or above")

    model = modelcontext(model)

    [jitter_seed, pathfinder_seed, sample_seed] = _get_seeds_per_chain(random_seed, 3)

    # set initial points. PF requires jittering of initial points
    ipfn = make_initial_point_fn(
        model=model,
        jitter_rvs=set(model.free_RVs),
        # TODO: add argument for jitter strategy
    )
    ip = Point(ipfn(jitter_seed), model=model)

    # TODO: make better
    if inference_backend == "pymc":
        pathfinder_samples, logq_psi, logp_func = _pymc_pathfinder(
            model,
            ip,
            maxcor=maxcor,
            num_draws=num_draws,
            # TODO: insert single seed, then use _get_seeds_per_chain inside pymc_pathfinder
            random_seed=(pathfinder_seed, sample_seed),
            **pathfinder_kwargs,
        )

    elif inference_backend == "blackjax":
        logp_func, ip_map = get_jaxified_logp_ravel_inputs(model, initial_points=ip)
        pathfinder_state, pathfinder_info = blackjax.vi.pathfinder.approximate(
            rng_key=jax.random.key(pathfinder_seed),
            logdensity_fn=logp_func,
            initial_position=ip_map.data,
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
