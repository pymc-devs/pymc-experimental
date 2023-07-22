from functools import partial

import pymc as pm
import pytensor.tensor as pt


def get_slice_and_move_cursor(cursor, param_count, last_slice=False):
    param_slice = slice(cursor, None if last_slice else cursor + param_count)
    cursor += param_count

    return param_slice, cursor


def _create_lkj_prior(shape, dims=None):
    n = shape[0]
    with pm.modelcontext(None):
        sd_dist = pm.Gamma.dist(alpha=2, beta=1)
        P0_chol, *_ = pm.LKJCholeskyCov("P0_chol", n=n, eta=1, sd_dist=sd_dist)
        P0 = pm.Deterministic("P0", P0_chol @ P0_chol.T, dims=dims)


def _create_diagonal_covariance(shape, dims=None):
    diag_dims = None if dims is None else dims[0]
    diag_shape = None if shape is None else shape[0]

    with pm.modelcontext(None):
        P0_diag = pm.Gamma("P0_diag", alpha=2, beta=1, shape=diag_shape, dims=diag_dims)
        P0 = pm.Deterministic("P0", pt.diag(P0_diag), dims=dims)


def default_prior_factory(shape=None, dims=None):
    return {
        "x0": lambda: pm.Normal("x0", mu=0, sigma=1, shape=shape, dims=dims),
        "P0": partial(_create_diagonal_covariance, shape=shape, dims=dims),
        "sigma_obs": lambda: pm.Exponential("sigma_obs", lam=1, shape=shape, dims=dims),
        "sigma_state": lambda: pm.Exponential("sigma_state", lam=1, shape=shape, dims=dims),
    }
