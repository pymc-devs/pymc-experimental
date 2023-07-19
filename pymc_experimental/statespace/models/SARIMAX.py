from typing import Tuple

import numpy as np
import pytensor.tensor as pt

from pymc_experimental.statespace.core.statespace import PyMCStateSpace
from pymc_experimental.statespace.utils.pytensor_scipy import solve_discrete_lyapunov


class BayesianARMA(PyMCStateSpace):
    def __init__(
        self,
        order: Tuple[int, int],
        stationary_initialization: bool = True,
        filter_type: str = "standard",
        state_structure: str = "fast",
        verbose=True,
    ):

        # Model order
        self.p, self.q = order
        self.stationary_initialization = stationary_initialization
        self.state_structure = state_structure

        # k_states = max(self.p, self.q + 1)
        p_max = max(1, self.p)
        q_max = max(1, self.q)

        if self.state_structure == "fast":
            k_states = max(self.p, self.q + 1)
        elif self.state_structure == "intrepretable":
            k_states = p_max + q_max

        k_posdef = 1
        k_endog = 1

        super().__init__(k_endog, k_states, k_posdef, filter_type, verbose=verbose)

        # Initialize the matrices
        self.ssm["design"] = np.r_[[1.0], np.zeros(k_states - 1)][None]

        if self.state_structure == "fast":
            transition = np.eye(k_states, k=1)
            selection = np.r_[[[1.0]], np.zeros(k_states - 1)[:, None]]

        elif self.state_structure == "interpretable":
            transition = np.eye(k_states, k=-1)
            transition[-q_max, p_max - 1] = 0

            selection = np.r_[[[1.0]], np.zeros(k_states - 1)[:, None]]
            selection[-q_max, 0] = 1

        self.ssm["selection"] = selection
        self.ssm["transition"] = transition

        self.ssm["initial_state"] = np.zeros((k_states,))

        self.ssm["initial_state_cov"] = np.eye(k_states)

        # Cache some indices
        self._state_cov_idx = ("state_cov",) + np.diag_indices(k_posdef)

        if self.state_structure == "fast":
            self._ar_param_idx = ("transition",) + (
                np.arange(self.p, dtype=int),
                np.zeros(self.p, dtype=int),
            )
            self._ma_param_idx = ("selection",) + (
                np.arange(1, self.q + 1, dtype=int),
                np.zeros(self.q, dtype=int),
            )

        elif self.state_structure == "interpretable":
            self._ar_param_idx = ("transition",) + (
                np.zeros(self.p, dtype=int),
                np.arange(self.p, dtype=int),
            )
            self._ma_param_idx = ("transition",) + (
                np.zeros(self.q, dtype=int),
                np.arange(p_max, p_max + self.q, dtype=int),
            )

    @property
    def param_names(self):
        names = ["x0", "P0", "sigma_state", "ar_params", "ma_params"]
        if self.stationary_initialization:
            names.remove("P0")
        if self.p == 0:
            names.remove("ar_params")
        if self.q == 0:
            names.remove("ma_params")

        return names

    @property
    def state_names(self):
        if self.state_structure == "fast":
            states = ["data"] + [f"state_{i+1}" for i in range(self.k_states)]
        else:
            states = ["data"]
            if self.p > 0:
                states += [f"L{i+1}.data" for i in range(self.p - 1)]
            states += ["innovations"]
            if self.q > 0:
                states += [f"L{i+1}.innovations" for i in range(self.q - 1)]
        return states

    @property
    def observed_states(self):
        return self.state_names[0]

    def update(self, theta: pt.TensorVariable) -> None:
        """
        Put parameter values from vector theta into the correct positions in the state space matrices.
        TODO: Can this be done using variable names to avoid the need to ravel and concatenate all RVs in the
              PyMC model?

        Parameters
        ----------
        theta: TensorVariable
            Vector of all variables in the state space model
        """
        cursor = 0

        # initial states
        param_slice = slice(cursor, cursor + self.k_states)
        cursor += self.k_states
        self.ssm["initial_state", :] = theta[param_slice]

        if not self.stationary_initialization:
            # initial covariance
            param_slice = slice(cursor, cursor + self.k_states**2)
            cursor += self.k_states**2
            self.ssm["initial_state_cov"] = theta[param_slice].reshape(
                (self.k_states, self.k_states)
            )

        # State covariance
        param_slice = slice(cursor, cursor + 1)
        cursor += 1
        self.ssm[self._state_cov_idx] = theta[param_slice]

        if self.p > 0:
            # AR parameteres
            param_slice = slice(cursor, cursor + self.p)
            cursor += self.p
            self.ssm[self._ar_param_idx] = theta[param_slice]

        if self.q > 0:
            # MA parameters
            param_slice = slice(cursor, cursor + self.q)
            cursor += self.q
            self.ssm[self._ma_param_idx] = theta[param_slice]

        if self.stationary_initialization:
            # Solve for matrix quadratic for P0
            T = self.ssm["transition"]
            R = self.ssm["selection"]
            Q = self.ssm["state_cov"]

            P0 = solve_discrete_lyapunov(
                T,
                pt.linalg.matrix_dot(R, Q, R.T),
                method="direct" if self.k_states < 5 else "bilinear",
            )

            self.ssm["initial_state_cov", :, :] = P0
