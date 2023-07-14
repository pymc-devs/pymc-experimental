import numpy as np
import pytensor.tensor as pt

from pymc_experimental.statespace.core.statespace import PyMCStateSpace


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
