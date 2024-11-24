from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pytensor.tensor as pt

from numpy.typing import NDArray
from pytensor.graph import Apply, Op
from scipy.optimize import minimize


@dataclass(slots=True)
class LBFGSHistory:
    x: NDArray[np.float64]
    g: NDArray[np.float64]

    def __post_init__(self):
        self.x = np.ascontiguousarray(self.x, dtype=np.float64)
        self.g = np.ascontiguousarray(self.g, dtype=np.float64)


@dataclass(slots=True)
class LBFGSHistoryManager:
    grad_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    x0: NDArray[np.float64]
    maxiter: int
    x_history: NDArray[np.float64] = field(init=False)
    g_history: NDArray[np.float64] = field(init=False)
    count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.x_history = np.empty((self.maxiter + 1, self.x0.shape[0]), dtype=np.float64)
        self.g_history = np.empty((self.maxiter + 1, self.x0.shape[0]), dtype=np.float64)

        grad = self.grad_fn(self.x0)
        if not np.all(np.isfinite(grad)):
            self.x_history[0] = self.x0
            self.g_history[0] = grad
            self.count = 1

    def add_entry(self, x: NDArray[np.float64], g: NDArray[np.float64]) -> None:
        self.x_history[self.count] = x
        self.g_history[self.count] = g
        self.count += 1

    def get_history(self) -> LBFGSHistory:
        return LBFGSHistory(x=self.x_history[: self.count], g=self.g_history[: self.count])

    def __call__(self, x: NDArray[np.float64]) -> None:
        grad = self.grad_fn(x)
        if np.all(np.isfinite(grad)) and self.count < self.maxiter + 1:
            self.add_entry(x, grad)


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

        # TODO: return the status of the lbfgs optimisation to handle the case where the optimisation fails. More details in the _single_pathfinder function.

        outputs[0][0] = history_manager.get_history().x
        outputs[1][0] = history_manager.get_history().g
