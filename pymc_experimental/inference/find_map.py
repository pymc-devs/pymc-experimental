import logging

from collections.abc import Callable
from typing import cast

import jax
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from better_optimize import minimize
from better_optimize.constants import MINIMIZE_MODE_KWARGS, minimize_method
from pymc.blocking import DictToArrayBijection, RaveledVars
from pymc.initial_point import make_initial_point_fn
from pymc.model.transform.optimization import freeze_dims_and_data
from pymc.pytensorf import join_nonshared_inputs
from pymc.util import get_default_varnames
from pytensor.compile import Function
from pytensor.tensor import TensorVariable
from scipy.optimize import OptimizeResult

_log = logging.getLogger(__name__)


def set_optimizer_function_defaults(method, use_grad, use_hess, use_hessp):
    method_info = MINIMIZE_MODE_KWARGS[method].copy()

    use_grad = use_grad if use_grad is not None else method_info["uses_grad"]
    use_hess = use_hess if use_hess is not None else method_info["uses_hess"]
    use_hessp = use_hessp if use_hessp is not None else method_info["uses_hessp"]

    if use_hess and use_hessp:
        use_hess = False

    return use_grad, use_hess, use_hessp


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


def _unconstrained_vector_to_constrained_rvs(model):
    constrained_rvs, unconstrained_vector = join_nonshared_inputs(
        model.initial_point(),
        inputs=model.value_vars,
        outputs=get_default_varnames(model.unobserved_value_vars, include_transformed=False),
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


def _compile_jax_gradients(
    f_loss: Function, use_hess: bool, use_hessp: bool
) -> tuple[Callable | None, Callable | None]:
    """
    Compile loss function gradients using JAX.

    Parameters
    ----------
    f_loss: Function
        The loss function to compile gradients for. Expected to be a pytensor function that returns a scalar loss,
        compiled with mode="JAX".
    use_hess: bool
        Whether to compile a function to compute the hessian of the loss function.
    use_hessp: bool
        Whether to compile a function to compute the hessian-vector product of the loss function.

    Returns
    -------
    f_loss_and_grad: Callable
        The compiled loss function and gradient function.
    f_hess: Callable | None
        The compiled hessian function, or None if use_hess is False.
    f_hessp: Callable | None
        The compiled hessian-vector product function, or None if use_hessp is False.
    """
    f_hess = None
    f_hessp = None

    orig_loss_fn = f_loss.vm.jit_fn

    @jax.jit
    def loss_fn_jax_grad(x, *shared):
        return jax.value_and_grad(lambda x: orig_loss_fn(x)[0])(x)

    f_loss_and_grad = loss_fn_jax_grad

    if use_hessp:

        def f_hessp_jax(x, p):
            y, u = jax.jvp(lambda x: f_loss_and_grad(x)[1], (x,), (p,))
            return jax.numpy.stack(u)

        f_hessp = jax.jit(f_hessp_jax)

    if use_hess:
        _f_hess_jax = jax.jacfwd(lambda x: f_loss_and_grad(x)[1])

        def f_hess_jax(x):
            return jax.numpy.stack(_f_hess_jax(x))

        f_hess = jax.jit(f_hess_jax)

    return f_loss_and_grad, f_hess, f_hessp


def _compile_functions(
    loss: TensorVariable,
    inputs: list[TensorVariable],
    compute_grad: bool,
    compute_hess: bool,
    compute_hessp: bool,
    compile_kwargs: dict | None = None,
) -> list[Function] | list[Function, Function | None, Function | None]:
    """
    Compile loss functions for use with scipy.optimize.minimize.

    Parameters
    ----------
    loss: TensorVariable
        The loss function to compile.
    inputs: list[TensorVariable]
        A single flat vector input variable, collecting all inputs to the loss function. Scipy optimize routines
        expect the function signature to be f(x, *args), where x is a 1D array of parameters.
    compute_grad: bool
        Whether to compile a function that computes the gradients of the loss function.
    compute_hess: bool
        Whether to compile a function that computes the Hessian of the loss function.
    compute_hessp: bool
        Whether to compile a function that computes the Hessian-vector product of the loss function.
    compile_kwargs: dict, optional
        Additional keyword arguments to pass to the ``pm.compile_pymc`` function.

    Returns
    -------
    f_loss: Function

    f_hess: Function | None
    f_hessp: Function | None
    """
    loss = pm.pytensorf.rewrite_pregrad(loss)
    f_hess = None
    f_hessp = None

    if compute_grad:
        grads = pytensor.gradient.grad(loss, inputs)
        grad = pt.concatenate([grad.ravel() for grad in grads])
        f_loss_and_grad = pm.compile_pymc(inputs, [loss, grad], **compile_kwargs)
    else:
        f_loss = pm.compile_pymc(inputs, loss, **compile_kwargs)
        return [f_loss]

    if compute_hess:
        hess = pytensor.gradient.jacobian(grad, inputs)[0]
        f_hess = pm.compile_pymc(inputs, hess, **compile_kwargs)

    if compute_hessp:
        p = pt.tensor("p", shape=inputs[0].type.shape)
        hessp = pytensor.gradient.hessian_vector_product(loss, inputs, p)
        f_hessp = pm.compile_pymc([*inputs, p], hessp[0], **compile_kwargs)

    return [f_loss_and_grad, f_hess, f_hessp]


def scipy_optimize_funcs_from_loss(
    loss: TensorVariable,
    inputs: list[TensorVariable],
    initial_point_dict: dict[str, np.ndarray | float | int],
    use_grad: bool,
    use_hess: bool,
    use_hessp: bool,
    use_jax_gradients: bool = False,
    compile_kwargs: dict | None = None,
) -> tuple[Callable, ...]:
    """
    Compile loss functions for use with scipy.optimize.minimize.


    Parameters
    ----------
    loss: TensorVariable
        The loss function to compile.
    inputs: list[TensorVariable]
        The input variables to the loss function.
    initial_point_dict: dict[str, np.ndarray | float | int]
        Dictionary mapping variable names to initial values. Used to determine the shapes of the input variables.
    use_grad: bool
        Whether to compile a function that computes the gradients of the loss function.
    use_hess: bool
        Whether to compile a function that computes the Hessian of the loss function.
    use_hessp: bool
        Whether to compile a function that computes the Hessian-vector product of the loss function.
    use_jax_gradients: bool
        If True, use JAX to compute gradients. This is only possible when ``compile_kwargs["mode"]`` is set to "JAX".
    compile_kwargs:
        Additional keyword arguments to pass to the ``pm.compile_pymc`` function.

    Returns
    -------
    f_loss: Callable
        The compiled loss function.
    f_hess: Callable | None
        The compiled hessian function, or None if use_hess is False.
    f_hessp: Callable | None
        The compiled hessian-vector product function, or None if use_hessp is False.
    """

    compile_kwargs = {} if compile_kwargs is None else compile_kwargs

    if (use_hess or use_hessp) and not use_grad:
        raise ValueError(
            "Cannot compute hessian or hessian-vector product without also computing the gradient"
        )

    use_jax_gradients = use_jax_gradients and use_grad

    mode = compile_kwargs.get("mode", None)
    if mode is None and use_jax_gradients:
        compile_kwargs["mode"] = "JAX"
    elif mode != "JAX" and use_jax_gradients:
        raise ValueError(
            'jax gradients can only be used when ``compile_kwargs["mode"]`` is set to "JAX"'
        )

    if not isinstance(inputs, list):
        inputs = [inputs]

    [loss], flat_input = join_nonshared_inputs(
        point=initial_point_dict, outputs=[loss], inputs=inputs
    )

    compute_grad = use_grad and not use_jax_gradients
    compute_hess = use_hess and not use_jax_gradients
    compute_hessp = use_hessp and not use_jax_gradients

    funcs = _compile_functions(
        loss=loss,
        inputs=[flat_input],
        compute_grad=compute_grad,
        compute_hess=compute_hess,
        compute_hessp=compute_hessp,
        compile_kwargs=compile_kwargs,
    )

    # f_loss here is f_loss_and_grad if compute_grad = True. The name is unchanged to simplify the return values
    f_loss = funcs.pop(0)
    f_hess = funcs.pop(0) if compute_grad else None
    f_hessp = funcs.pop(0) if compute_grad else None

    if use_jax_gradients:
        # f_loss here is f_loss_and_grad; the name is unchanged to simplify the return values
        f_loss, f_hess, f_hessp = _compile_jax_gradients(f_loss, use_hess, use_hessp)

    return f_loss, f_hess, f_hessp


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
    use_jax_gradients: bool = False,
    compile_kwargs: dict | None = None,
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

    jitter_rvs = [] if jitter_rvs is None else jitter_rvs
    compile_kwargs = {} if compile_kwargs is None else compile_kwargs

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
    use_grad, use_hess, use_hessp = set_optimizer_function_defaults(
        method, use_grad, use_hess, use_hessp
    )

    f_logp, f_hess, f_hessp = scipy_optimize_funcs_from_loss(
        loss=-frozen_model.logp(jacobian=False),
        inputs=frozen_model.continuous_value_vars + frozen_model.discrete_value_vars,
        initial_point_dict=start_dict,
        use_grad=use_grad,
        use_hess=use_hess,
        use_hessp=use_hessp,
        use_jax_gradients=use_jax_gradients,
        compile_kwargs=compile_kwargs,
    )

    args = optimizer_kwargs.pop("args", None)

    # better_optimize.minimize will check if f_logp is a fused loss+grad Op, and automatically assign the jac argument
    # if so. That is why it is not set here, regardless of user settings.
    optimizer_result = minimize(
        f=f_logp,
        x0=cast(np.ndarray[float], initial_params.data),
        args=args,
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
