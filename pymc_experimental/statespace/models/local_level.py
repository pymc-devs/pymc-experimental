import logging
from typing import Any, Dict, Sequence

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc import modelcontext

from pymc_experimental.statespace.core.statespace import (
    MATRIX_NAMES,
    PyMCStateSpace,
    find_aux_dim,
)
from pymc_experimental.statespace.models.utilities import default_prior_factory

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
    def default_coords(self) -> Dict[str, Sequence]:
        coords = {
            "all_states": self.state_names,
            "all_states_aux": self.state_names,
            "observed_states": self.observed_states,
            "observed_states_aux": self.observed_states,
        }

        return coords

    @property
    def param_to_coord(self):
        return {
            "x0": ["all_states"],
            "P0": ["all_states", "all_states_aux"],
            "sigma_obs": ["observed_states"],
            "sigma_state": ["all_states"],
        }

    @property
    def param_info(self) -> Dict[str, Dict[str, Any]]:
        info = {
            "x0": {"shape": (self.k_states,), "constraints": None},
            "P0": {
                "shape": (self.k_states, self.k_states),
                "constraints": "Positive Semi-definite",
            },
            "sigma_obs": {"shape": (self.k_endog,), "constraints": "Positive"},
            "sigma_state": {"shape": (self.k_posdef,), "constraints": "Positive"},
        }
        return {name: info[name] for name in self.param_names}

    def _determine_matrix_dims(self):
        pm_mod = modelcontext(None)

        params_to_dims = pm_mod.named_vars_to_dims
        all_state_dim = params_to_dims.get("x0", [None])[0]
        obs_state_dim = params_to_dims.get("sigma_obs", [None])[0]
        state_posdef_dim = params_to_dims.get("sigma_state", [None])[0]

        if all_state_dim is None or obs_state_dim is None or state_posdef_dim is None:
            return dict.fromkeys(MATRIX_NAMES, None)

        all_state_aux_dim = find_aux_dim(all_state_dim)
        obs_state_aux_dim = find_aux_dim(obs_state_dim)

        return {
            "x0": (all_state_dim,),
            "P0": (all_state_dim, all_state_aux_dim),
            "c": (all_state_dim,),
            "d": (obs_state_dim,),
            "T": (all_state_dim, all_state_aux_dim),
            "Z": (obs_state_dim, all_state_dim),
            "R": (all_state_dim, state_posdef_dim),
            "H": (obs_state_dim, obs_state_aux_dim),
            "Q": (all_state_dim, all_state_aux_dim),
        }

    def add_default_priors(self):
        use_coords = True
        mod = pm.modelcontext(None)

        all_keys_exist = all([k in mod.coords.keys() for k in self.default_coords.keys()])
        values_are_expected = all(
            [
                all([v in mod.coords[k] for v in self.default_coords[k]])
                for k in self.default_coords.keys()
            ]
        )

        if not all_keys_exist and values_are_expected:
            _log.info("Default coords not found, default priors will not have labeled dimensions.")
            use_coords = False

        for param_name in self.param_names:
            if not getattr(mod, param_name, False):
                dims = self.param_to_coord[param_name] if use_coords else None
                shape = self.param_info[param_name]["shape"]
                _ = default_prior_factory(mod, shape=shape, dims=dims)[param_name]()

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
