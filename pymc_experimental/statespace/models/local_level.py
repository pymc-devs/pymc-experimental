import numpy as np
import pytensor.tensor as at

from pymc_experimental.statespace.core.statespace import PyMCStateSpace


class BayesianLocalLevel(PyMCStateSpace):
    def __init__(self, data):
        k_states = k_posdef = 2

        super().__init__(data, k_states, k_posdef)

        # Initialize the matrices
        self.ssm["design"] = np.array([[1.0, 0.0]])
        self.ssm["transition"] = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.ssm["selection"] = np.eye(k_states)

        self.ssm["initial_state"] = np.array([[0.0], [0.0]])
        self.ssm["initial_state_cov"] = np.array([[1.0, 0.0], [0.0, 1.0]])

        # Cache some indices
        self._state_cov_idx = ("state_cov",) + np.diag_indices(k_posdef)

    @property
    def param_names(self):
        return ["x0", "P0", "sigma_obs", "sigma_state"]

    def update(self, theta: at.TensorVariable) -> None:
        """
        Put parameter values from vector theta into the correct positions in the state space matrices.
        TODO: Can this be done using variable names to avoid the need to ravel and concatenate all RVs in the
              PyMC model?

        Parameters
        ----------
        theta: TensorVariable
            Vector of all variables in the state space model
        """
        # initial states
        self.ssm["initial_state", :, 0] = theta[:2]

        # initial covariance
        self.ssm["initial_state_cov", :, :] = theta[2:6].reshape((2, 2))

        # Observation covariance
        self.ssm["obs_cov", 0, 0] = theta[6]

        # State covariance
        self.ssm[self._state_cov_idx] = theta[7:]
