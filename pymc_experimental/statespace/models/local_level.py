import logging
from typing import Any, Dict

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from pymc_experimental.statespace.core.statespace import PyMCStateSpace
from pymc_experimental.statespace.models.utilities import default_prior_factory
from pymc_experimental.statespace.utils.constants import (
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    OBS_STATE_DIM,
)

_log = logging.getLogger("pymc.experimental.statespace")


class BayesianLocalLevel(PyMCStateSpace):
    def __init__(self, verbose=False, filter_type: str = "standard"):
        k_states = 2
        k_posdef = 2
        k_endog = 1

        super().__init__(k_endog, k_states, k_posdef, filter_type=filter_type, verbose=verbose)

        # Initialize the matrices
        self.ssm["design"] = np.array([[1.0, 0.0]])
        self.ssm["transition"] = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.ssm["selection"] = np.eye(k_states)

        self.ssm["initial_state"] = np.array([0.0, 0.0])
        self.ssm["initial_state_cov"] = np.array([[1.0, 0.0], [0.0, 1.0]])

        # Cache some indices
        self._state_cov_idx = ("state_cov",) + np.diag_indices(k_posdef)

    @property
    def param_names(self):
        return ["x0", "P0", "sigma_obs", "sigma_state"]

    @property
    def state_names(self):
        return ["level", "trend"]

    @property
    def observed_states(self):
        return ["level"]

    @property
    def shock_names(self):
        return ["level", "trend"]

    @property
    def param_to_coord(self):
        return {
            "x0": (ALL_STATE_DIM,),
            "P0": (ALL_STATE_DIM, ALL_STATE_AUX_DIM),
            "sigma_obs": (OBS_STATE_DIM,),
            "sigma_state": (ALL_STATE_DIM,),
        }

    @property
    def param_info(self) -> Dict[str, Dict[str, Any]]:
        info = {
            "x0": {
                "shape": (self.k_states,),
                "constraints": None,
                "dims": self.param_to_coord["x0"],
            },
            "P0": {
                "shape": (self.k_states, self.k_states),
                "constraints": "Positive Semi-definite",
                "dims": self.param_to_coord["P0"],
            },
            "sigma_obs": {
                "shape": (self.k_endog,),
                "constraints": "Positive",
                "dims": self.param_to_coord["sigma_obs"],
            },
            "sigma_state": {
                "shape": (self.k_posdef,),
                "constraints": "Positive",
                "dims": self.param_to_coord["sigma_state"],
            },
        }
        return {name: info[name] for name in self.param_names}

    def add_default_priors(self):
        use_coords = True
        mod = pm.modelcontext(None)

        all_keys_exist = all([k in mod.coords.keys() for k in self.coords.keys()])
        values_are_expected = all(
            [all([v in mod.coords.get(k, []) for v in self.coords[k]]) for k in self.coords.keys()]
        )

        if not (all_keys_exist and values_are_expected):
            _log.info("Coords not found, default priors will not have labeled dimensions.")
            use_coords = False

        for param_name in self.param_names:
            if not getattr(mod, param_name, False):
                dims = self.param_to_coord[param_name] if use_coords else None
                shape = self.param_info[param_name]["shape"]

                _ = default_prior_factory(shape=shape, dims=dims)[param_name]()

    def update(self, theta: pt.TensorVariable) -> None:
        """
        Put parameter values from vector theta into the correct positions in the state space matrices.

        Parameters
        ----------
        theta: TensorVariable
            Vector of all variables in the state space model
        """
        # initial states
        self.ssm["initial_state"] = theta[:2].reshape((2,))

        # initial covariance
        self.ssm["initial_state_cov"] = theta[2:6].reshape((2, 2))

        # Observation covariance
        self.ssm["obs_cov", 0, 0] = theta[6]

        # State covariance
        self.ssm[self._state_cov_idx] = theta[7:]
