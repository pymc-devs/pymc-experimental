import logging

from collections.abc import Callable
from typing import Literal, cast

import arviz as az
import jax
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray as xr

from arviz import dict_to_dataset
from better_optimize import minimize
from better_optimize.constants import minimize_method
from pymc.backends.arviz import (
    coords_and_dims_for_inferencedata,
    find_constants,
    find_observations,
)
from pymc.blocking import DictToArrayBijection, RaveledVars
from pymc.initial_point import make_initial_point_fn
from pymc.model.transform.conditioning import remove_value_transforms
from pymc.model.transform.optimization import freeze_dims_and_data
from pymc.sampling.jax import get_jaxified_graph
from pymc.util import get_default_varnames
from pytensor.tensor import TensorVariable
from scipy import stats
from scipy.optimize import OptimizeResult

_log = logging.getLogger(__name__)


def get_near_psd(A: np.ndarray) -> np.ndarray:
    """
    Compute the nearest positive semi-definite matrix to a given matrix.

    This function takes a square matrix and returns the nearest positive
    semi-definite matrix using eigenvalue decomposition. It ensures all
    eigenvalues are non-negative. The "nearest" matrix is defined in terms
    of the Frobenius norm.

    Parameters
    ----------
    A : np.ndarray
        Input square matrix.

    Returns
    -------
    np.ndarray
        The nearest positive semi-definite matrix to the input matrix.
    """
    C = (A + A.T) / 2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval < 0] = 0

    return eigvec @ np.diag(eigval) @ eigvec.T


def _get_unravel_rv_info(optimized_point, variables, model):
    cursor = 0
    slices = {}
    out_shapes = {}

    for i, var in enumerate(variables):
        raveled_shape = np.prod(optimized_point[var.name].shape).astype(int)
        rv = model.values_to_rvs.get(var, var)

        idx = slice(cursor, cursor + raveled_shape)
        slices[rv] = idx
        out_shapes[rv] = tuple(
            [len(model.coords[dim]) for dim in model.named_vars_to_dims.get(rv.name, [])]
        )
        cursor += raveled_shape

    return slices, out_shapes


def _create_transformed_draws(H_inv, slices, out_shapes, posterior_draws, model, chains, draws):
    X = pt.tensor("transformed_draws", shape=(chains, draws, H_inv.shape[0]))
    out = []
    for rv, idx in slices.items():
        f = model.rvs_to_transforms[rv]
        untransformed_X = f.backward(X[..., idx]) if f is not None else X[..., idx]

        if rv in out_shapes:
            new_shape = (chains, draws) + out_shapes[rv]
            untransformed_X = untransformed_X.reshape(new_shape)

        out.append(untransformed_X)

    f_untransform = pytensor.function([X], out, mode="JAX")
    return f_untransform(posterior_draws)


def fit_laplace(
    optimized_point: dict[str, np.ndarray],
    model: pm.Model,
    chains: int = 2,
    draws: int = 500,
    on_bad_cov: Literal["warn", "error", "ignore"] = "ignore",
    transform_samples: bool = True,
    zero_tol: float = 1e-8,
    diag_jitter: float | None = 1e-8,
    progressbar: bool = True,
    mode: str = "JAX",
) -> az.InferenceData:
    """
    Compute the Laplace approximation of the posterior distribution.

    The posterior distribution will be approximated as a Gaussian
    distribution centered at the posterior mode.
    The covariance is the inverse of the negative Hessian matrix of
    the log-posterior evaluated at the mode.

    Parameters
    ----------
    optimized_point : dict[str, np.ndarray]
        Local maximum a posteriori (MAP) point returned from pymc.find_MAP
        or jax_tools.fit_map
    model : Model
        A PyMC model
    chains : int
        The number of sampling chains running in parallel. Default is 2.
    draws : int
        The number of samples to draw from the approximated posterior. Default is 500.
    on_bad_cov : str, one of 'ignore', 'warn', or 'error', default: 'ignore'
        What to do when ``H_inv`` (inverse Hessian) is not positive semi-definite.
        If 'ignore' or 'warn', the closest positive-semi-definite matrix to ``H_inv`` (in L1 norm) will be returned.
        If 'error', an error will be raised.
    transform_samples : bool
        Whether to transform the samples back to the original parameter space. Default is True.
    zero_tol: float
        Value below which an element of the Hessian matrix is counted as 0.
        This is used to stabilize the computation of the inverse Hessian matrix. Default is 1e-8.
    diag_jitter: float | None
        A small value added to the diagonal of the inverse Hessian matrix to ensure it is positive semi-definite.
        If None, no jitter is added. Default is 1e-8.
    progressbar : bool
        Whether or not to display progress bar. Default is True.
    mode : str
        Computation backend mode. Default is "JAX".

    Returns
    -------
    InferenceData
        arviz.InferenceData object storing posterior, observed_data, and constant_data groups.

    """
    frozen_model = freeze_dims_and_data(model)
    if not transform_samples:
        untransformed_model = remove_value_transforms(frozen_model)
        logp = untransformed_model.logp(jacobian=False)
        variables = untransformed_model.continuous_value_vars
    else:
        logp = frozen_model.logp(jacobian=True)
        variables = frozen_model.continuous_value_vars

    mu = np.concatenate(
        [np.atleast_1d(optimized_point[var.name]).ravel() for var in variables], axis=0
    )

    f_logp, f_grad, f_hess, f_hessp = make_jax_funcs_from_graph(
        cast(TensorVariable, logp),
        use_grad=True,
        use_hess=True,
        use_hessp=False,
        inputs=variables,
    )

    H = f_hess(mu)
    H_inv = np.linalg.pinv(np.where(np.abs(H) < zero_tol, 0, -H))

    def stabilize(x, jitter):
        return x + np.eye(x.shape[0]) * jitter

    H_inv = H_inv if diag_jitter is None else stabilize(H_inv, diag_jitter)

    try:
        np.linalg.cholesky(H_inv)
    except np.linalg.LinAlgError:
        if on_bad_cov == "error":
            raise np.linalg.LinAlgError(
                "Inverse Hessian not positive-semi definite at the provided point"
            )
        H_inv = get_near_psd(H_inv)
        if on_bad_cov == "warn":
            _log.warning(
                "Inverse Hessian is not positive semi-definite at the provided point, using the closest PSD "
                "matrix in L1-norm instead"
            )

    posterior_dist = stats.multivariate_normal(mean=mu, cov=H_inv, allow_singular=True)
    posterior_draws = posterior_dist.rvs(size=(chains, draws))
    slices, out_shapes = _get_unravel_rv_info(optimized_point, variables, frozen_model)

    if transform_samples:
        posterior_draws = _create_transformed_draws(
            H_inv, slices, out_shapes, posterior_draws, frozen_model, chains, draws
        )
    else:
        posterior_draws = [
            posterior_draws[..., idx].reshape((chains, draws, *out_shapes.get(rv, ())))
            for rv, idx in slices.items()
        ]

    def make_rv_coords(rv):
        coords = {"chain": range(chains), "draw": range(draws)}
        extra_dims = frozen_model.named_vars_to_dims.get(rv.name)
        if extra_dims is None:
            return coords
        return coords | {dim: list(frozen_model.coords[dim]) for dim in extra_dims}

    def make_rv_dims(rv):
        dims = ["chain", "draw"]
        extra_dims = frozen_model.named_vars_to_dims.get(rv.name)
        if extra_dims is None:
            return dims
        return dims + list(extra_dims)

    idata = {
        rv.name: xr.DataArray(
            data=draws.squeeze(),
            coords=make_rv_coords(rv),
            dims=make_rv_dims(rv),
            name=rv.name,
        )
        for rv, draws in zip(slices.keys(), posterior_draws)
    }

    coords, dims = coords_and_dims_for_inferencedata(frozen_model)
    idata = az.convert_to_inference_data(idata, coords=coords, dims=dims)

    if frozen_model.deterministics:
        idata.posterior = pm.compute_deterministics(
            idata.posterior,
            model=frozen_model,
            merge_dataset=True,
            progressbar=progressbar,
            compile_kwargs={"mode": mode},
        )

    observed_data = dict_to_dataset(
        find_observations(frozen_model),
        library=pm,
        coords=coords,
        dims=dims,
        default_dims=[],
    )

    constant_data = dict_to_dataset(
        find_constants(frozen_model),
        library=pm,
        coords=coords,
        dims=dims,
        default_dims=[],
    )

    idata.add_groups(
        {"observed_data": observed_data, "constant_data": constant_data},
        coords=coords,
        dims=dims,
    )

    return idata


def make_jax_funcs_from_graph(
    graph: TensorVariable,
    use_grad: bool,
    use_hess: bool,
    use_hessp: bool,
    inputs: list[TensorVariable] | None = None,
) -> tuple[Callable, ...]:
    if inputs is None:
        from pymc.pytensorf import inputvars

        inputs = inputvars(graph)
    if not isinstance(inputs, list):
        inputs = [inputs]

    f = cast(Callable, get_jaxified_graph(inputs=inputs, outputs=[graph]))
    input_shapes = [x.type.shape for x in inputs]

    def at_least_tuple(x):
        if isinstance(x, tuple | list):
            return x
        return (x,)

    assert all([xi is not None for x in input_shapes for xi in at_least_tuple(x)])

    def f_jax(x):
        args = []
        cursor = 0
        for shape in input_shapes:
            n_elements = int(np.prod(shape))
            s = slice(cursor, cursor + n_elements)
            args.append(x[s].reshape(shape))
            cursor += n_elements
        return f(*args)[0]

    f_logp = jax.jit(f_jax)

    f_grad = None
    f_hess = None
    f_hessp = None

    if use_grad:
        _f_grad_jax = jax.grad(f_jax)

        def f_grad_jax(x):
            return jax.numpy.stack(_f_grad_jax(x))

        f_grad = jax.jit(f_grad_jax)

    if use_hessp:
        if not use_grad:
            raise ValueError("Cannot ask for Hessian without asking for Gradients")

        def f_hessp_jax(x, p):
            y, u = jax.jvp(f_grad_jax, (x,), (p,))
            return jax.numpy.stack(u)

        f_hessp = jax.jit(f_hessp_jax)

    if use_hess:
        if not use_grad:
            raise ValueError("Cannot ask for Hessian without asking for Gradients")
        _f_hess_jax = jax.jacfwd(f_grad_jax)

        def f_hess_jax(x):
            return jax.numpy.stack(_f_hess_jax(x))

        f_hess = jax.jit(f_hess_jax)

    return f_logp, f_grad, f_hess, f_hessp


def find_MAP(
    model: pm.Model,
    method: minimize_method,
    use_grad: bool | None = None,
    use_hessp: bool | None = None,
    use_hess: bool | None = None,
    initvals: dict | None = None,
    random_seed: int | np.random.Generator | None = None,
    return_raw: bool = False,
    jitter_rvs: list[TensorVariable] | None = None,
    progressbar: bool = True,
    include_transformed: bool = True,
    **optimizer_kwargs,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], OptimizeResult]:
    """
    Fit a PyMC model via maximum a posteriori (MAP) estimation using JAX and scipy.minimize.

    Parameters
    ----------
    model : pm.Model
        The PyMC model to be fitted.
    method : str
        The optimization method to use. See scipy.optimize.minimize documentation for details.
    use_grad : bool | None, optional
        Whether to use gradients in the optimization. Defaults to None, which determines this automatically based on
        the ``method``.
    use_hessp : bool | None, optional
        Whether to use Hessian-vector products in the optimization. Defaults to None, which determines this automatically based on
        the ``method``.
    use_hess : bool | None, optional
        Whether to use the Hessian matrix in the optimization. Defaults to None, which determines this automatically based on
        the ``method``.
    initvals : None | dict, optional
        Initial values for the model parameters, as str:ndarray key-value pairs. Paritial initialization is permitted.
         If None, the model's default initial values are used.
    random_seed : None | int | np.random.Generator, optional
        Seed for the random number generator or a numpy Generator for reproducibility
    return_raw: bool | False, optinal
        Whether to also return the full output of `scipy.optimize.minimize`
    jitter_rvs : list of TensorVariables, optional
        Variables whose initial values should be jittered. If None, all variables are jittered.
    progressbar : bool, optional
        Whether to display a progress bar during optimization. Defaults to True.
    include_transformed: bool, optional
        Whether to include transformed variable values in the returned dictionary. Defaults to True.
    **optimizer_kwargs
        Additional keyword arguments to pass to the ``scipy.optimize.minimize`` function.

    Returns
    -------
    optimizer_result: dict[str, np.ndarray] or tuple[dict[str, np.ndarray], OptimizerResult]
        Dictionary with names of random variables as keys, and optimization results as values. If return_raw is True,
        also returns the object returned by ``scipy.optimize.minimize``.
    """
    frozen_model = freeze_dims_and_data(model)

    if jitter_rvs is None:
        jitter_rvs = []

    ipfn = make_initial_point_fn(
        model=frozen_model,
        jitter_rvs=set(jitter_rvs),
        return_transformed=True,
        overrides=initvals,
    )

    start_dict = ipfn(random_seed)
    vars_dict = {var.name: var for var in frozen_model.continuous_value_vars}
    initial_params = DictToArrayBijection.map(
        {var_name: value for var_name, value in start_dict.items() if var_name in vars_dict}
    )

    inputs = [frozen_model.values_to_rvs[vars_dict[x]] for x in start_dict.keys()]
    inputs = [frozen_model.rvs_to_values[x] for x in inputs]

    logp_factors = frozen_model.logp(sum=False, jacobian=False)
    neg_logp = -pt.sum([pt.sum(factor) for factor in logp_factors])

    f_logp, f_grad, f_hess, f_hessp = make_jax_funcs_from_graph(
        neg_logp, use_grad, use_hess, use_hessp, inputs=inputs
    )

    args = optimizer_kwargs.pop("args", None)

    optimizer_result = minimize(
        f=f_logp,
        x0=cast(np.ndarray[float], initial_params.data),
        args=args,
        jac=f_grad,
        hess=f_hess,
        hessp=f_hessp,
        progressbar=progressbar,
        method=method,
        **optimizer_kwargs,
    )

    initial_point = RaveledVars(optimizer_result.x, initial_params.point_map_info)
    unobserved_vars = get_default_varnames(model.unobserved_value_vars, include_transformed)
    unobserved_vars_values = model.compile_fn(unobserved_vars)(
        DictToArrayBijection.rmap(initial_point, start_dict)
    )
    optimized_point = {
        var.name: value for var, value in zip(unobserved_vars, unobserved_vars_values)
    }

    if return_raw:
        return optimized_point, optimizer_result

    return optimized_point
