from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pytensor.tensor as pt

from pytensor.graph import Apply, Op
from scipy.optimize import minimize


class LBFGSHistory(NamedTuple):
    x: np.ndarray
    g: np.ndarray


class LBFGSHistoryManager:
    def __init__(self, grad_fn: Callable, x0: np.ndarray, maxiter: int):
        dim = x0.shape[0]
        maxiter_add_one = maxiter + 1
        # Pre-allocate arrays to save memory and improve speed
        self.x_history = np.empty((maxiter_add_one, dim), dtype=np.float64)
        self.g_history = np.empty((maxiter_add_one, dim), dtype=np.float64)
        self.count = 0
        self.grad_fn = grad_fn
        self.add_entry(x0, grad_fn(x0))

    def add_entry(self, x, g):
        self.x_history[self.count] = x
        self.g_history[self.count] = g
        self.count += 1

    def get_history(self):
        # Return trimmed arrays up to L << L^max
        x = self.x_history[: self.count]
        g = self.g_history[: self.count]
        return LBFGSHistory(
            x=x,
            g=g,
        )

    def __call__(self, x):
        grad = self.grad_fn(x)
        if np.all(np.isfinite(grad)):
            self.add_entry(x, grad)


def lbfgs(
    fn,
    grad_fn,
    x0: np.ndarray,
    maxcor: int | None = None,
    maxiter=1000,
    ftol=1e-5,
    gtol=1e-8,
    maxls=1000,
    **lbfgs_kwargs,
) -> LBFGSHistory:
    def callback(xk):
        lbfgs_history_manager(xk)

    lbfgs_history_manager = LBFGSHistoryManager(
        grad_fn=grad_fn,
        x0=x0,
        maxiter=maxiter,
    )

    default_lbfgs_options = dict(
        maxcor=maxcor,
        maxiter=maxiter,
        ftol=ftol,
        gtol=gtol,
        maxls=maxls,
    )
    options = lbfgs_kwargs.pop("options", {})
    options = default_lbfgs_options | options

    # TODO: return the status of the lbfgs optimisation to handle the case where the optimisation fails. More details in the _single_pathfinder function.

    minimize(
        fn,
        x0,
        method="L-BFGS-B",
        jac=grad_fn,
        options=options,
        callback=callback,
        **lbfgs_kwargs,
    )
    lbfgs_history = lbfgs_history_manager.get_history()
    return lbfgs_history.x, lbfgs_history.g


class LBFGSOp(Op):
    def __init__(self, fn, grad_fn, maxcor, maxiter=1000, ftol=1e-5, gtol=1e-8, maxls=1000):
        self.fn = fn
        self.grad_fn = grad_fn
        self.maxcor = maxcor
        self.maxiter = maxiter
        self.ftol = ftol
        self.gtol = gtol
        self.maxls = maxls

    def make_node(self, x0):
        x0 = pt.as_tensor_variable(x0)
        x_history = pt.dmatrix()
        g_history = pt.dmatrix()
        return Apply(self, [x0], [x_history, g_history])

    def perform(self, node, inputs, outputs):
        x0 = inputs[0]
        x0 = np.array(x0, dtype=np.float64)

        history_manager = LBFGSHistoryManager(grad_fn=self.grad_fn, x0=x0, maxiter=self.maxiter)

        minimize(
            self.fn,
            x0,
            method="L-BFGS-B",
            jac=self.grad_fn,
            callback=history_manager,
            options={
                "maxcor": self.maxcor,
                "maxiter": self.maxiter,
                "ftol": self.ftol,
                "gtol": self.gtol,
                "maxls": self.maxls,
            },
        )

        # fmin_l_bfgs_b(
        #     func=self.fn,
        #     fprime=self.grad_fn,
        #     x0=x0,
        #     pgtol=self.gtol,
        #     factr=self.ftol / np.finfo(float).eps,
        #     maxls=self.maxls,
        #     maxiter=self.maxiter,
        #     m=self.maxcor,
        #     callback=history_manager,
        # )

        outputs[0][0] = history_manager.get_history().x
        outputs[1][0] = history_manager.get_history().g
