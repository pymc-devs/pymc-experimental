from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pytensor.tensor as pt

from pymc_experimental.statespace.core.statespace import PyMCStateSpace
from pymc_experimental.statespace.models.utilities import (
    get_slice_and_move_cursor,
    make_default_coords,
)
from pymc_experimental.statespace.utils.constants import (
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    AR_PARAM_DIM,
    MA_PARAM_DIM,
    OBS_STATE_AUX_DIM,
    OBS_STATE_DIM,
    SHOCK_AUX_DIM,
    SHOCK_DIM,
)
from pymc_experimental.statespace.utils.pytensor_scipy import solve_discrete_lyapunov


class BayesianVARMAX(PyMCStateSpace):
    def __init__(
        self,
        order: Tuple[int, int],
        endog_names: List[str] = None,
        k_endog: int = None,
        stationary_initialization: bool = True,
        filter_type: str = "standard",
        measurement_error: bool = True,
        verbose=True,
    ):

        if (endog_names is None) and (k_endog is None):
            raise ValueError("Must specify either endog_names or k_endog")
        if (endog_names is not None) and (k_endog is None):
            k_endog = len(endog_names)
        if (endog_names is None) and (k_endog is not None):
            endog_names = [f"state.{i + 1}" for i in range(k_endog)]
        if (endog_names is not None) and (k_endog is not None):
            if len(endog_names) != k_endog:
                raise ValueError("Length of provided endog_names does not match provided k_endog")

        self.endog_names = list(endog_names)
        self.p, self.q = order
        self.stationary_initialization = stationary_initialization
        self.measurement_error = measurement_error

        k_order = max(self.p, 1) + self.q
        k_states = int(k_endog * k_order)
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

    @property
    def param_info(self) -> Dict[str, Dict[str, Any]]:
        info = {
            "x0": {
                "shape": (self.k_states,),
                "constraints": None,
            },
            "P0": {
                "shape": (self.k_states, self.k_states),
                "constraints": "Positive Semi-definite",
            },
            "obs_cov": {
                "shape": (self.k_endog, self.k_endog),
                "constraints": "Positive Semi-definite",
            },
            "state_cov": {
                "shape": (self.k_posdef, self.k_posdef),
                "constraints": "Positive Semi-definite",
            },
            "ar_params": {
                "shape": (self.k_states, self.p, self.k_states),
                "constraints": "None",
            },
            "ma_params": {
                "shape": (self.k_states, self.q, self.k_states),
                "constraints": "None",
            },
        }

        for name in self.param_names:
            info[name]["dims"] = self.param_dims[name]

        return {name: info[name] for name in self.param_names}

    @property
    def state_names(self):
        state_names = self.endog_names.copy()
        state_names += [
            f"L{i + 1}.{state}" for i in range(self.p - 1) for state in self.endog_names
        ]
        state_names += [
            f"L{i + 1}.{state}_innov" for i in range(self.q) for state in self.endog_names
        ]

        return state_names

    @property
    def observed_states(self):
        return self.endog_names

    @property
    def shock_names(self):
        return self.endog_names

    @property
    def default_priors(self):
        raise NotImplementedError

    @property
    def coords(self) -> Dict[str, Sequence]:
        coords = make_default_coords(self)
        if self.p > 0:
            coords.update({AR_PARAM_DIM: list(range(1, self.p + 1))})
        if self.q > 0:
            coords.update({MA_PARAM_DIM: list(range(1, self.q + 1))})

        return coords

    @property
    def param_dims(self):
        coord_map = {
            "x0": (ALL_STATE_DIM,),
            "P0": (ALL_STATE_DIM, ALL_STATE_AUX_DIM),
            "obs_cov": (OBS_STATE_DIM, OBS_STATE_AUX_DIM),
            "state_cov": (SHOCK_DIM, SHOCK_AUX_DIM),
            "ar_params": (OBS_STATE_DIM, AR_PARAM_DIM, OBS_STATE_AUX_DIM),
            "ma_params": (OBS_STATE_DIM, MA_PARAM_DIM, OBS_STATE_AUX_DIM),
        }

        if not self.measurement_error:
            del coord_map["obs_cov"]
        if self.p == 0:
            del coord_map["ar_params"]
        if self.q == 0:
            del coord_map["ma_params"]
        if self.stationary_initialization:
            del coord_map["P0"]

        return coord_map

    def add_default_priors(self):
        raise NotImplementedError

    def update(self, theta: pt.TensorVariable) -> None:
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
        self.ssm["initial_state", :] = theta[param_slice]

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

            P0 = solve_discrete_lyapunov(
                T,
                pt.linalg.matrix_dot(R, Q, R.T),
                method="direct" if self.k_states < 10 else "bilinear",
            )
            self.ssm["initial_state_cov", :, :] = P0
