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
from pymc.pytensorf import join_nonshared_inputs
from pymc.sampling.jax import get_jaxified_graph
from pymc.util import get_default_varnames
from pytensor.tensor import TensorVariable
from scipy import stats
from scipy.optimize import OptimizeResult

_log = logging.getLogger(__name__)


def get_nearest_psd(A: np.ndarray) -> np.ndarray:
    """
    Compute the nearest positive semi-definite matrix to a given matrix.

    This function takes a square matrix and returns the nearest positive semi-definite matrix using
    eigenvalue decomposition. It ensures all eigenvalues are non-negative. The "nearest" matrix is defined in terms
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


def unobserved_value_vars(model):
    vars = []
    transformed_rvs = []
    for rv in model.free_RVs:
        value_var = model.rvs_to_values[rv]
        transform = model.rvs_to_transforms[rv]
        if transform is not None:
            transformed_rvs.append(rv)
        vars.append(value_var)

    # Remove rvs from untransformed values graph
    untransformed_vars = model.replace_rvs_by_values(transformed_rvs)

    return vars + untransformed_vars


def _unconstrained_vector_to_constrained_rvs(model):
    constrained_rvs, unconstrained_vector = join_nonshared_inputs(
        model.initial_point(), inputs=model.value_vars, outputs=model.unobserved_value_vars
    )

    unconstrained_vector.name = "unconstrained_vector"
    return constrained_rvs, unconstrained_vector


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


def jax_fit_mvn_to_MAP(
    optimized_point: dict[str, np.ndarray],
    model: pm.Model,
    on_bad_cov: Literal["warn", "error", "ignore"] = "ignore",
    transform_samples: bool = True,
    zero_tol: float = 1e-8,
    diag_jitter: float | None = 1e-8,
) -> tuple[RaveledVars, np.ndarray]:
    """
    Create a multivariate normal distribution using the inverse of the negative Hessian matrix of the log-posterior
    evaluated at the MAP estimate. This is the basis of the Laplace approximation.

    Parameters
    ----------
    optimized_point : dict[str, np.ndarray]
        Local maximum a posteriori (MAP) point returned from pymc.find_MAP or jax_tools.fit_map
    model : Model
        A PyMC model
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

    Returns
    -------
    map_estimate: RaveledVars
        The MAP estimate of the model parameters, raveled into a 1D array.

    inverse_hessian: np.ndarray
        The inverse Hessian matrix of the log-posterior evaluated at the MAP estimate.
    """
    frozen_model = freeze_dims_and_data(model)

    if not transform_samples:
        untransformed_model = remove_value_transforms(frozen_model)
        logp = untransformed_model.logp(jacobian=False)
        variables = untransformed_model.continuous_value_vars
    else:
        logp = frozen_model.logp(jacobian=True)
        variables = frozen_model.continuous_value_vars

    variable_names = {var.name for var in variables}
    optimized_free_params = {k: v for k, v in optimized_point.items() if k in variable_names}
    mu = DictToArrayBijection.map(optimized_free_params)

    [neg_logp], flat_inputs = join_nonshared_inputs(
        point=frozen_model.initial_point(), outputs=[-logp], inputs=variables
    )

    f_logp, f_grad, f_hess, f_hessp = make_jax_funcs_from_graph(
        neg_logp, use_grad=True, use_hess=True, use_hessp=False, inputs=[flat_inputs]
    )

    H = -f_hess(mu.data)
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
        H_inv = get_nearest_psd(H_inv)
        if on_bad_cov == "warn":
            _log.warning(
                "Inverse Hessian is not positive semi-definite at the provided point, using the closest PSD "
                "matrix in L1-norm instead"
            )

    return mu, H_inv


def jax_laplace(
    mu: RaveledVars,
    H_inv: np.ndarray,
    model: pm.Model,
    chains: int = 2,
    draws: int = 500,
    transform_samples: bool = True,
    progressbar: bool = True,
) -> az.InferenceData:
    """

    Parameters
    ----------
    mu
    H_inv
    model : Model
        A PyMC model
    chains : int
        The number of sampling chains running in parallel. Default is 2.
    draws : int
        The number of samples to draw from the approximated posterior. Default is 500.
    transform_samples : bool
        Whether to transform the samples back to the original parameter space. Default is True.

    Returns
    -------
    idata: az.InferenceData
        An InferenceData object containing the approximated posterior samples.
    """
    posterior_dist = stats.multivariate_normal(mean=mu.data, cov=H_inv, allow_singular=True)
    posterior_draws = posterior_dist.rvs(size=(chains, draws))

    if transform_samples:
        constrained_rvs, unconstrained_vector = _unconstrained_vector_to_constrained_rvs(model)
        f_constrain = get_jaxified_graph(inputs=[unconstrained_vector], outputs=constrained_rvs)

        posterior_draws = jax.jit(jax.vmap(jax.vmap(f_constrain)))(posterior_draws)

    else:
        info = mu.point_map_info
        flat_shapes = [np.prod(shape).astype(int) for _, shape, _ in info]
        slices = [
            slice(sum(flat_shapes[:i]), sum(flat_shapes[: i + 1])) for i in range(len(flat_shapes))
        ]

        posterior_draws = [
            posterior_draws[..., idx].reshape((chains, draws, *shape)).astype(dtype)
            for idx, (name, shape, dtype) in zip(slices, info)
        ]

    def make_rv_coords(name):
        coords = {"chain": range(chains), "draw": range(draws)}
        extra_dims = model.named_vars_to_dims.get(name)
        if extra_dims is None:
            return coords
        return coords | {dim: list(model.coords[dim]) for dim in extra_dims}

    def make_rv_dims(name):
        dims = ["chain", "draw"]
        extra_dims = model.named_vars_to_dims.get(name)
        if extra_dims is None:
            return dims
        return dims + list(extra_dims)

    idata = {
        name: xr.DataArray(
            data=draws.squeeze(),
            coords=make_rv_coords(name),
            dims=make_rv_dims(name),
            name=name,
        )
        for (name, _, _), draws in zip(mu.point_map_info, posterior_draws)
    }

    coords, dims = coords_and_dims_for_inferencedata(model)
    idata = az.convert_to_inference_data(idata, coords=coords, dims=dims)

    if model.deterministics:
        idata.posterior = pm.compute_deterministics(
            idata.posterior,
            model=model,
            merge_dataset=True,
            progressbar=progressbar,
        )

    observed_data = dict_to_dataset(
        find_observations(model),
        library=pm,
        coords=coords,
        dims=dims,
        default_dims=[],
    )

    constant_data = dict_to_dataset(
        find_constants(model),
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
) -> az.InferenceData:
    mu, H_inv = jax_fit_mvn_to_MAP(
        optimized_point,
        model,
        on_bad_cov,
        transform_samples,
        zero_tol,
        diag_jitter,
    )

    return jax_laplace(mu, H_inv, model, chains, draws, transform_samples, progressbar)


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

    f_tuple = cast(Callable, get_jaxified_graph(inputs=inputs, outputs=[graph]))

    def f(*args, **kwargs):
        return f_tuple(*args, **kwargs)[0]

    f_logp = jax.jit(f)

    f_grad = None
    f_hess = None
    f_hessp = None

    if use_grad:
        _f_grad_jax = jax.grad(f)

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
    method: minimize_method,
    *,
    model: pm.Model | None = None,
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
        The PyMC model to be fit. If None, the current model context is used.
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
    model = pm.modelcontext(model)
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

    [neg_logp], inputs = join_nonshared_inputs(
        point=start_dict, outputs=[-frozen_model.logp()], inputs=frozen_model.continuous_value_vars
    )

    f_logp, f_grad, f_hess, f_hessp = make_jax_funcs_from_graph(
        neg_logp, use_grad, use_hess, use_hessp, inputs=[inputs]
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

    raveled_optimized = RaveledVars(optimizer_result.x, initial_params.point_map_info)
    unobserved_vars = get_default_varnames(model.unobserved_value_vars, include_transformed)
    unobserved_vars_values = model.compile_fn(unobserved_vars)(
        DictToArrayBijection.rmap(raveled_optimized)
    )

    optimized_point = {
        var.name: value for var, value in zip(unobserved_vars, unobserved_vars_values)
    }

    if return_raw:
        return optimized_point, optimizer_result

    return optimized_point
