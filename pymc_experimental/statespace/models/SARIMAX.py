from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pytensor.tensor as pt

from pymc_experimental.statespace.core.statespace import PyMCStateSpace
from pymc_experimental.statespace.models.utilities import make_default_coords
from pymc_experimental.statespace.utils.constants import (
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    AR_PARAM_DIM,
    MA_PARAM_DIM,
    OBS_STATE_DIM,
)
from pymc_experimental.statespace.utils.pytensor_scipy import solve_discrete_lyapunov

STATE_STRUCTURES = ["fast", "interpretable"]


class BayesianARMA(PyMCStateSpace):
    def __init__(
        self,
        order: Tuple[int, int],
        stationary_initialization: bool = True,
        filter_type: str = "standard",
        state_structure: str = "fast",
        measurement_error: bool = False,
        verbose=True,
    ):

        # Model order
        self.p, self.q = order
        self.stationary_initialization = stationary_initialization
        self.measurement_error = measurement_error

        if state_structure not in STATE_STRUCTURES:
            raise ValueError(
                f"Got invalid argument {state_structure} for state structure, expected one of "
                f'{", ".join(STATE_STRUCTURES)}'
            )
        self.state_structure = state_structure

        # k_states = max(self.p, self.q + 1)
        p_max = max(1, self.p)
        q_max = max(1, self.q)

        k_states = None
        if self.state_structure == "fast":
            k_states = max(self.p, self.q + 1)
        elif self.state_structure == "interpretable":
            k_states = p_max + q_max

        k_posdef = 1
        k_endog = 1

        super().__init__(k_endog, k_states, k_posdef, filter_type, verbose=verbose)

        # Initialize the matrices
        self.ssm["design"] = np.r_[[1.0], np.zeros(k_states - 1)][None]

        transition = None
        selection = None

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

        if self.measurement_error:
            self._obs_cov_idx = ("obs_cov",) + np.diag_indices(k_endog)

    @property
    def param_names(self):
        names = ["x0", "P0", "ar_params", "ma_params", "sigma_state", "sigma_obs"]
        if self.stationary_initialization:
            names.remove("P0")
        if self.p == 0:
            names.remove("ar_params")
        if self.q == 0:
            names.remove("ma_params")
        if not self.measurement_error:
            names.remove("sigma_obs")

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
            "sigma_obs": {
                "shape": (self.k_endog,),
                "constraints": "Positive",
            },
            "sigma_state": {
                "shape": (self.k_posdef,),
                "constraints": "Positive",
            },
            "ar_params": {
                "shape": (self.p,),
                "constraints": "None",
            },
            "ma_params": {
                "shape": (self.q,),
                "constraints": "None",
            },
        }

        for name in self.param_names:
            info[name]["dims"] = self.param_dims[name]

        return {name: info[name] for name in self.param_names}

    @property
    def state_names(self):
        if self.state_structure == "fast":
            states = ["data"] + [f"state_{i + 1}" for i in range(self.k_states - 1)]
        else:
            states = ["data"]
            if self.p > 0:
                states += [f"L{i + 1}.data" for i in range(self.p - 1)]
            states += ["innovations"]
            if self.q > 0:
                states += [f"L{i + 1}.innovations" for i in range(self.q - 1)]
        return states

    @property
    def observed_states(self):
        return [self.state_names[0]]

    @property
    def shock_names(self):
        if self.state_structure == "fast":
            shocks = ["innovations"]
            shocks += [f"L{i + 1}.innovations" for i in range(self.q - 1)]
            return shocks

        elif self.state_structure == "interpretable":
            return ["innovations"]

    @property
    def param_dims(self):
        coord_map = {
            "x0": (ALL_STATE_DIM,),
            "P0": (ALL_STATE_DIM, ALL_STATE_AUX_DIM),
            "sigma_obs": (OBS_STATE_DIM,),
            "sigma_state": (OBS_STATE_DIM,),
            "ar_params": (AR_PARAM_DIM,),
            "ma_params": (MA_PARAM_DIM,),
        }

        if not self.measurement_error:
            del coord_map["sigma_obs"]
        if self.p == 0:
            del coord_map["ar_params"]
        if self.q == 0:
            del coord_map["ma_params"]
        if self.stationary_initialization:
            del coord_map["P0"]

        return coord_map

    @property
    def coords(self) -> Dict[str, Sequence]:
        coords = make_default_coords(self)
        if self.p > 0:
            coords.update({AR_PARAM_DIM: list(range(1, self.p + 1))})
        if self.q > 0:
            coords.update({MA_PARAM_DIM: list(range(1, self.q + 1))})

        return coords

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

        # State covariance
        param_slice = slice(cursor, cursor + 1)
        cursor += 1
        self.ssm[self._state_cov_idx] = theta[param_slice]

        if self.measurement_error:
            # Measurement error
            param_slice = slice(cursor, cursor + 1)
            cursor += 1
            self.ssm[self._obs_cov_idx] = theta[param_slice]

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
