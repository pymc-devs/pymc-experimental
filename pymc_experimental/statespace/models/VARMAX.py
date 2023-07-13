from typing import Tuple

import numpy as np
import pytensor.tensor as at

from pymc_experimental.statespace.core.statespace import PyMCStateSpace
from pymc_experimental.statespace.models.utilities import get_slice_and_move_cursor
from pymc_experimental.statespace.utils.pytensor_scipy import solve_discrete_lyapunov


class BayesianVARMAX(PyMCStateSpace):
    def __init__(
        self,
        k_endog: int,
        order: Tuple[int, int],
        stationary_initialization: bool = True,
        filter_type: str = "standard",
        measurement_error: bool = True,
        verbose=True,
    ):

        self.p, self.q = order
        self.stationary_initialization = stationary_initialization
        self.measurement_error = measurement_error

        k_order = max(self.p, 1) + self.q
        k_states = k_endog * k_order
        k_posdef = k_endog

        super().__init__(k_endog, k_states, k_posdef, filter_type, verbose=verbose)

        # Save counts of the number of parameters in each category
        self.param_counts = {
            "x0": k_states,
            "P0": k_states**2 * (1 - self.stationary_initialization),
            "AR": k_endog**2 * self.p,
            "MA": k_endog**2 * self.q,
            "state_cov": k_posdef**2,
            "obs_cov": k_endog * self.measurement_error,
        }

        # Initialize the matrices
        # Design matrix is a truncated identity (first k_obs states observed)
        self.ssm[("design",) + np.diag_indices(k_endog)] = 1

        # Transition matrix has 4 blocks:
        self.ssm["transition"] = np.zeros((k_states, k_states))

        # UL: AR coefs (k_obs, k_obs * min(p, 1))
        # UR: MA coefs (k_obs, k_obs * q)
        # LL: Truncated identity (k_obs * min(p, 1), k_obs * min(p, 1))
        # LR: Shifted identity (k_obs * p, k_obs * q)
        if self.p > 1:
            idx = (slice(k_endog, k_endog * self.p), slice(0, k_endog * (self.p - 1)))
            self.ssm[("transition",) + idx] = np.eye(k_endog * (self.p - 1))

        if self.q > 1:
            idx = (slice(-k_endog * (self.q - 1), None), slice(-k_endog * self.q, -k_endog))
            self.ssm[("transition",) + idx] = np.eye(k_endog * (self.q - 1))

        # The selection matrix is (k_states, k_obs), with two (k_obs, k_obs) identity
        # matrix blocks inside. One is always on top, the other starts after (k_obs * p) rows
        self.ssm["selection"] = np.zeros((k_states, k_endog))
        self.ssm["selection", slice(0, k_endog), :] = np.eye(k_endog)
        if self.q > 0:
            end = -k_endog * (self.q - 1) if self.q > 1 else None
            self.ssm["selection", slice(k_endog * -self.q, end), :] = np.eye(k_endog)

        # self.ssm["initial_state"] = np.zeros(k_states)[:, None]
        # self.ssm["initial_state_cov"] = np.eye(k_states)
        # self.ssm["state_cov"] = np.eye(k_posdef)
        #
        # if self.measurement_error:
        #     self.ssm['obs_cov'] = np.eye(k_obs)

        # Cache some indices
        self._ar_param_idx = ("transition", slice(0, k_endog), slice(0, k_endog * self.p))
        self._ma_param_idx = (
            "transition",
            slice(0, k_endog),
            slice(k_endog * max(1, self.p), None),
        )
        self._obs_cov_idx = ("obs_cov",) + np.diag_indices(k_endog)

    @property
    def param_names(self):
        names = ["x0", "P0", "ar_params", "ma_params", "state_cov", "obs_cov"]
        if self.stationary_initialization:
            names.remove("P0")
        if not self.measurement_error:
            names.remove("obs_cov")
        if self.p == 0:
            names.remove("ar_params")
        if self.q == 0:
            names.remove("ma_params")
        return names

    def update(self, theta: at.TensorVariable) -> None:
        """
        Put parameter values from vector theta into the correct positions in the state space matrices.

        Parameters
        ----------
        theta: TensorVariable
            Vector of all variables in the state space model
        """

        cursor = 0
        # initial states
        param_slice, cursor = get_slice_and_move_cursor(cursor, self.param_counts["x0"])
        self.ssm["initial_state", :, 0] = theta[param_slice]

        if not self.stationary_initialization:
            # initial covariance
            param_slice, cursor = get_slice_and_move_cursor(cursor, self.param_counts["P0"])
            self.ssm["initial_state_cov", :, :] = theta[param_slice].reshape(
                (self.k_states, self.k_states)
            )

        # AR parameters
        if self.p > 0:
            ar_shape = (self.k_endog, self.k_endog * self.p)
            param_slice, cursor = get_slice_and_move_cursor(cursor, self.param_counts["AR"])
            self.ssm[self._ar_param_idx] = theta[param_slice].reshape(ar_shape)

        # MA parameters
        if self.q > 0:
            ma_shape = (self.k_endog, self.k_endog * self.q)
            param_slice, cursor = get_slice_and_move_cursor(cursor, self.param_counts["MA"])
            self.ssm[self._ma_param_idx] = theta[param_slice].reshape(ma_shape)

        # State covariance
        param_slice, cursor = get_slice_and_move_cursor(
            cursor, self.param_counts["state_cov"], last_slice=not self.measurement_error
        )

        self.ssm["state_cov", :, :] = theta[param_slice].reshape((self.k_posdef, self.k_posdef))

        # Measurement error
        if self.measurement_error:
            param_slice, cursor = get_slice_and_move_cursor(
                cursor, self.param_counts["obs_cov"], last_slice=True
            )
            self.ssm[self._obs_cov_idx] = theta[param_slice]

        if self.stationary_initialization:
            # Solve for matrix quadratic for P0
            T = self.ssm["transition"]
            R = self.ssm["selection"]
            Q = self.ssm["state_cov"]

            P0 = solve_discrete_lyapunov(T, at.linalg.matrix_dot(R, Q, R.T), method="bilinear")
            self.ssm["initial_state_cov", :, :] = P0
