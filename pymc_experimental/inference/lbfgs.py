from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pytensor.tensor as pt

from pytensor.tensor.variable import TensorVariable
from scipy.optimize import fmin_l_bfgs_b


class LBFGSHistory(NamedTuple):
    x: TensorVariable
    f: TensorVariable
    g: TensorVariable


class LBFGSHistoryManager:
    def __init__(self, fn: Callable, grad_fn: Callable, x0: np.ndarray, maxiter: int):
        dim = x0.shape[0]
        maxiter_add_one = maxiter + 1
        # Preallocate arrays to save memory and improve speed
        self.x_history = np.empty((maxiter_add_one, dim), dtype=np.float64)
        self.f_history = np.empty(maxiter_add_one, dtype=np.float64)
        self.g_history = np.empty((maxiter_add_one, dim), dtype=np.float64)
        self.count = 0
        self.fn = fn
        self.grad_fn = grad_fn
        self.add_entry(x0, fn(x0), grad_fn(x0))

    def add_entry(self, x, f, g=None):
        # Store the values directly in preallocated arrays
        self.x_history[self.count] = x
        self.f_history[self.count] = f
        if self.g_history is not None and g is not None:
            self.g_history[self.count] = g
        self.count += 1

    def get_history(self):
        # Return trimmed arrays up to the number of entries actually used
        x = self.x_history[: self.count]
        f = self.f_history[: self.count]
        g = self.g_history[: self.count] if self.g_history is not None else None
        return LBFGSHistory(
            x=pt.as_tensor(x, dtype="float64"),
            f=pt.as_tensor(f, dtype="float64"),
            g=pt.as_tensor(g, dtype="float64"),
        )

    def __call__(self, x):
        self.add_entry(x, self.fn(x), self.grad_fn(x))


def lbfgs(
    fn,
    grad_fn,
    x0: np.ndarray,
    maxcor: int | None = None,
    maxiter=1000,
    ftol=1e-5,
    gtol=1e-8,
    maxls=1000,
):
    def callback(xk):
        lbfgs_history_manager(xk)

    lbfgs_history_manager = LBFGSHistoryManager(
        fn=fn,
        grad_fn=grad_fn,
        x0=x0,
        maxiter=maxiter,
    )

    # options = dict(
    #     maxcor=maxcor,
    #     maxiter=maxiter,
    #     ftol=ftol,
    #     gtol=gtol,
    #     maxls=maxls,
    # )
    # minimize(
    #     fn,
    #     x0,
    #     method="L-BFGS-B",
    #     jac=grad_fn,
    #     options=options,
    #     callback=callback,
    # )
    fmin_l_bfgs_b(
        func=fn,
        fprime=grad_fn,
        x0=x0,
        pgtol=gtol,
        factr=ftol / np.finfo(float).eps,
        maxls=maxls,
        maxiter=maxiter,
        m=maxcor,
        callback=callback,
    )
    return lbfgs_history_manager.get_history()
