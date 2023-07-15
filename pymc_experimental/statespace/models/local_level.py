import logging
from functools import partial
from typing import Any, Dict, Sequence

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from pymc_experimental.statespace.core.statespace import PyMCStateSpace

_log = logging.getLogger("pymc.experimental.statespace")


class BayesianLocalLevel(PyMCStateSpace):
    def __init__(self):
        k_states = 2
        k_posdef = 2
        k_endog = 1

        super().__init__(k_endog, k_states, k_posdef)

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

    def _default_prior_factory(self, mod, shape=None, dims=None):

        return {
            "x0": lambda: pm.Normal("x0", mu=0, sigma=1, shape=shape, dims=dims),
            "P0": partial(self._create_lkj_prior, shape=shape, dims=dims),
            "sigma_obs": lambda: pm.Exponential("sigma_obs", lam=1, shape=shape, dims=dims),
            "sigma_state": lambda: pm.Exponential("sigma_state", lam=1, shape=shape, dims=dims),
        }

    def _create_lkj_prior(self, shape, dims=None):
        n = shape[0]
        with pm.modelcontext(None):
            sd_dist = pm.Exponential.dist(1)
            P0_chol, *_ = pm.LKJCholeskyCov("P0_chol", n=n, eta=1, sd_dist=sd_dist)
            P0 = pm.Deterministic("P0", P0_chol @ P0_chol.T, dims=dims)

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
                _ = self._default_prior_factory(mod, shape=shape, dims=dims)[param_name]()

    def update(self, theta: pt.TensorVariable) -> None:
        """
        Put parameter values from vector theta into the correct positions in the state space matrices.

        Parameters
        ----------
        theta: TensorVariable
            Vector of all variables in the state space model
        """
        # initial states
        self.ssm["initial_state", :] = theta[:2]

        # initial covariance
        self.ssm["initial_state_cov", :, :] = theta[2:6].reshape((2, 2))

        # Observation covariance
        self.ssm["obs_cov", 0, 0] = theta[6]

        # State covariance
        self.ssm[self._state_cov_idx] = theta[7:]
