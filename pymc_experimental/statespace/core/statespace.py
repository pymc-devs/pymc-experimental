import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from warnings import catch_warnings, simplefilter

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from numpy.typing import ArrayLike
from pymc.model import modelcontext

from pymc_experimental.statespace.core.representation import PytensorRepresentation
from pymc_experimental.statespace.filters import (
    CholeskyFilter,
    KalmanSmoother,
    SingleTimeseriesFilter,
    StandardFilter,
    SteadyStateFilter,
    UnivariateFilter,
)
from pymc_experimental.statespace.filters.distributions import (  # LinearGaussianStateSpace,
    SequenceMvNormal,
)
from pymc_experimental.statespace.utils.simulation import (
    conditional_simulation,
    unconditional_simulations,
)

_log = logging.getLogger("pymc.experimental.statespace")

floatX = pytensor.config.floatX
FILTER_FACTORY = {
    "standard": StandardFilter,
    "univariate": UnivariateFilter,
    "steady_state": SteadyStateFilter,
    "single": SingleTimeseriesFilter,
    "cholesky": CholeskyFilter,
}

MATRIX_NAMES = ["x0_matrix", "P0_matrix", "c", "d", "T", "Z", "R", "H", "Q"]
OUTPUT_NAMES = [
    "filtered_states",
    "" "predicted_states",
    "filtered_covariances",
    "predicted_covariances",
]

SMOOTHER_OUTPUT_NAMES = ["smoothed_states", "smoothed_covariances"]


def get_posterior_samples(posterior_samples, posterior_size):
    if isinstance(posterior_samples, float):
        if posterior_samples > 1.0 or posterior_samples < 0.0:
            raise ValueError(
                "If posterior_samples is a float, it should be between 0 and 1, representing the "
                "fraction of total posterior samples to re-sample."
            )
        posterior_samples = int(np.floor(posterior_samples * posterior_size))

    elif posterior_samples is None:
        posterior_samples = posterior_size

    return posterior_samples


def validate_filter_arg(filter_arg):
    if filter_arg.lower() not in ["filtered", "predicted", "smoothed"]:
        raise ValueError(
            f"filter_output should be one of filtered, predicted, or smoothed, recieved {filter_arg}"
        )


class PyMCStateSpace:
    def __init__(
        self,
        k_endog: int,
        k_states: int,
        k_posdef: int,
        filter_type: str = "standard",
        verbose: bool = True,
    ):
        self.k_endog = k_endog
        self.k_states = k_states
        self.k_posdef = k_posdef

        # All models contain a state space representation and a Kalman filter
        self.ssm = PytensorRepresentation(k_endog, k_states, k_posdef)

        if filter_type.lower() not in FILTER_FACTORY.keys():
            raise NotImplementedError(
                "The following are valid filter types: " + ", ".join(list(FILTER_FACTORY.keys()))
            )

        if filter_type == "single" and self.k_endog > 1:
            raise ValueError('Cannot use filter_type = "single" with multiple observed time series')

        self.kalman_filter = FILTER_FACTORY[filter_type.lower()]()
        self.kalman_smoother = KalmanSmoother()

        if verbose:
            _log.info(
                "Model successfully initialized! The following parameters should be assigned priors inside a PyMC "
                f"model block: \n"
                f"{self._print_prior_requirements()}"
            )

    def _print_prior_requirements(self):
        out = ""
        for param, info in self.param_info.items():
            out += f'\t{param} -- shape: {info["shape"]}, constraints: {info["constraints"]}\n'
        return out.rstrip()

    def unpack_statespace(self, include_constants=False):
        a0 = self.ssm["initial_state"]
        P0 = self.ssm["initial_state_cov"]
        c = self.ssm["state_intercept"]
        d = self.ssm["obs_intercept"]
        T = self.ssm["transition"]
        Z = self.ssm["design"]
        R = self.ssm["selection"]
        H = self.ssm["obs_cov"]
        Q = self.ssm["state_cov"]

        return a0, P0, c, d, T, Z, R, H, Q

    @property
    def param_names(self) -> List[str]:
        raise NotImplementedError

    @property
    def param_info(self) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError

    @property
    def state_names(self):
        raise NotImplementedError

    @property
    def observed_states(self):
        raise NotImplementedError

    @property
    def default_priors(self):
        raise NotImplementedError

    @property
    def default_coords(self) -> Dict[str, Sequence]:
        raise NotImplementedError

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
        raise NotImplementedError

    def gather_required_random_variables(self) -> pt.TensorVariable:
        """
        Iterates through random variables in the model on the context stack, matches their names with the statespace
        model's named parameters, and returns a single vector of parameters to pass to the update method.

        Important point is that the *order of the variables matters*. Update will expect that the theta vector will be
        organized as variables are listed in param_names.

        Returns
        ----------
        theta: TensorVariable
            A size (p,) tensor containing all parameters to be estimated among all state space matrices in the
            system.
        """

        theta = []
        pymc_model = modelcontext(None)
        found_params = []
        with pymc_model:
            for param_name in self.param_names:
                param = getattr(pymc_model, param_name, None)
                if param:
                    found_params.append(param.name)
                    theta.append(param.ravel())

        missing_params = set(self.param_names) - set(found_params)
        if len(missing_params) > 0:
            raise ValueError(
                "The following required model parameters were not found in the PyMC model: "
                + ", ".join(param for param in list(missing_params))
            )
        return pt.concatenate(theta)

    def build_statespace_graph(
        self, data, mode=None, return_updates=False, include_smoother=True
    ) -> None:
        """
        Given parameter vector theta, constructs the full computational graph describing the state space model.
        Must be called inside a PyMC model context.
        """

        modelcontext(None)
        theta = self.gather_required_random_variables()
        self.update(theta)

        matrices = self.unpack_statespace()

        for matrix, name in zip(matrices, MATRIX_NAMES):
            pm.Deterministic(name, matrix)

        filter_outputs = self.kalman_filter.build_graph(
            pt.as_tensor_variable(data),
            *matrices,
            mode=mode,
            return_updates=return_updates,
        )

        if return_updates:
            outputs, updates = filter_outputs
        else:
            outputs = filter_outputs

        logp = outputs.pop(-1)
        states, covs = outputs[:3], outputs[3:]
        filtered_states, predicted_states, observed_states = states
        filtered_covariances, predicted_covariances, observed_covariances = covs

        outputs_to_use = [
            filtered_states,
            predicted_states,
            filtered_covariances,
            predicted_covariances,
        ]
        names_to_use = OUTPUT_NAMES.copy()

        if include_smoother:
            smoothed_states, smoothed_covariances = self._build_smoother_graph(
                filtered_states, filtered_covariances, mode=mode
            )
            outputs_to_use += [smoothed_states, smoothed_covariances]
            names_to_use += SMOOTHER_OUTPUT_NAMES.copy()

        for output, name in zip(outputs_to_use, names_to_use):
            pm.Deterministic(name, output)

        SequenceMvNormal(
            "obs", mus=observed_states, covs=observed_covariances, logp=logp, observed=data
        )

    def _build_smoother_graph(self, filtered_states, filtered_covariances, mode=None):
        pymc_model = modelcontext(None)
        with pymc_model:
            *_, T, Z, R, H, Q = self.unpack_statespace()

            smooth_states, smooth_covariances = self.kalman_smoother.build_graph(
                T, R, Q, filtered_states, filtered_covariances, mode=mode
            )

            return smooth_states, smooth_covariances

    def make_matrix_update_funcs(self, pymc_model):
        rvs_on_graph = []
        with pymc_model:
            for param_name in self.param_names:
                param = getattr(pymc_model, param_name, None)
                if param:
                    rvs_on_graph.append(param)

        # TODO: This is pretty hacky, ask on the forums if there is a better solution
        matrix_update_funcs = [
            pytensor.function(rvs_on_graph, [X], on_unused_input="ignore")
            for X in self.unpack_statespace()
        ]

        return matrix_update_funcs

    @staticmethod
    def sample_conditional_prior(
        filter_output="filtered", n_simulations=100, prior_samples=500
    ) -> ArrayLike:
        """
        Sample from the conditional prior; that is, given parameter draws from the prior distribution, compute kalman
        filtered trajectories. Trajectories are drawn from a single multivariate normal with mean and covariance
        computed via either the kalman filter, smoother, or predictions.

        Parameters
        ----------
        filter_output: string, default = 'filtered'
            One of 'filtered', 'smoothed', or 'predicted'. Corresponds to which Kalman filter output you would like to
            sample from.
        n_simulations: int, default = 100
            The number of simulations to run for each prior parameter sample drawn. Total trajectories returned by this
            function will be n_simulations x prior_samples
        prior_samples: int, default = 500
            The number of samples to draw from the prior distribution, passed to pm.sample_prior_predictive. Defaults
            to the PyMC default of 500.

        Returns
        -------
        simulations: ArrayLike
            A numpy array of shape (n_simulations x prior_samples, n_timesteps, n_states) with simulated trajectories.

        """
        validate_filter_arg(filter_output)
        pymc_model = modelcontext(None)
        with pymc_model, catch_warnings():
            simplefilter("ignore", category=UserWarning)
            cond_prior = pm.sample_prior_predictive(
                samples=prior_samples,
                var_names=[
                    f"{filter_output}_states",
                    f"{filter_output}_covariances",
                ],
            )

        _, n, k, *_ = cond_prior.prior[f"{filter_output}_states"].values.squeeze().shape

        mus = (
            cond_prior.prior[f"{filter_output}_states"]
            .values.squeeze()
            .reshape(-1, n * k)
            .astype(floatX)
        )
        covs = (
            cond_prior.prior[f"{filter_output}_covariances"]
            .values.squeeze()
            .reshape(-1, n, k, k)
            .astype(floatX)
        )

        simulations = conditional_simulation(mus, covs, n, k, n_simulations)

        return simulations

    @staticmethod
    def sample_conditional_posterior(
        trace,
        filter_output: str = "filtered",
        n_simulations: int = 100,
        posterior_samples: Optional[Union[float, int]] = None,
    ):
        """
        Sample from the conditional posterior; that is, given parameter draws from the posterior distribution,
        compute kalman filtered trajectories. Trajectories are drawn from a single multivariate normal with mean and
        covariance computed via either the kalman filter, smoother, or predictions.

        Parameters
        ----------
        trace: xarray
            PyMC trace idata object. Should be an xarray returned by pm.sample() with return_inferencedata = True.
        filter_output: string, default = 'filtered'
            One of 'filtered', 'smoothed', or 'predicted'. Corresponds to which Kalman filter output you would like to
            sample from.
        n_simulations: int, default = 100
            The number of simulations to run for each prior parameter sample drawn. Total trajectories returned by this
            function will be n_simulations x prior_samples
        posterior_samples: int or float, default = None
            A number of subsamples to draw from the posterior trace. If None, all samples in the trace are used. If an
            integer, that number of samples will be drawn with replacement (from among all chains) from the trace. If a
            float between 0 and 1, that fraction of total draws in the trace will be sampled.

        Returns
        -------
        simulations: ArrayLike
            A numpy array of shape (n_simulations x prior_samples, n_timesteps, n_states) with simulated trajectories.

        """
        validate_filter_arg(filter_output)
        chains, draws, n, k, *_ = trace.posterior[f"{filter_output}_states"].shape
        posterior_size = chains * draws
        posterior_samples = get_posterior_samples(posterior_samples, posterior_size)

        resample_idxs = np.random.randint(0, posterior_size, size=posterior_samples)

        mus = (
            trace.posterior[f"{filter_output}_states"]
            .values.squeeze()
            .reshape(-1, n * k)[resample_idxs]
            .astype(floatX)
        )
        covs = (
            trace.posterior[f"{filter_output}_covariances"]
            .values.squeeze()
            .reshape(-1, n, k, k)[resample_idxs]
            .astype(floatX)
        )

        simulations = conditional_simulation(mus, covs, n, k, n_simulations)

        return simulations

    def sample_unconditional_prior(
        self, n_steps=100, n_simulations=100, prior_samples=500
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Draw unconditional sample trajectories according to state space dynamics, using random samples from the prior
        distribution over model parameters. The state space update equations are:

            X[t+1] = T @ X[t] + R @ eta[t], eta ~ N(0, Q)
            Y[t] = Z @ X[t] + nu[t], nu ~ N(0, H)

        Parameters
        ----------
        n_steps: int, default = 100
            Number of time steps to simulate
        n_simulations: int, default = 100
            Number of stochastic simulations to run for each parameter draw.
        prior_samples: int, default = 500
            Number of parameter draws from the prior distribution, passed to pm.sample_prior_predictive. Defaults to
            the PyMC default of 500.

        Returns
        -------
        simulated_states: ArrayLike
            Numpy array of shape (prior_samples * n_simulations, n_steps, n_states), corresponding to the unobserved
            states in the state-space system, X in the equations above
        simulated_data: ArrayLike
            Numpy array of shape (prior_samples * n_simulations, n_steps, n_observed), corresponding to the observed
            states in the state-space system, Y in the equations above.
        """
        pymc_model = modelcontext(None)

        with pymc_model, catch_warnings():
            simplefilter("ignore", category=UserWarning)
            prior_params = pm.sample_prior_predictive(
                var_names=self.param_names, samples=prior_samples
            )

        matrix_update_funcs = self.make_matrix_update_funcs(pymc_model)
        # Take the 0th element to remove the chain dimension
        thetas = [prior_params.prior[var].values[0] for var in self.param_names]
        simulated_states, simulated_data = unconditional_simulations(
            thetas, matrix_update_funcs, n_steps=n_steps, n_simulations=n_simulations
        )

        return simulated_states, simulated_data

    def sample_unconditional_posterior(
        self, trace, n_steps=100, n_simulations=100, posterior_samples=None
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Draw unconditional sample trajectories according to state space dynamics, using random samples from the
        posterior distribution over model parameters. The state space update equations are:

            X[t+1] = T @ X[t] + R @ eta[t], eta ~ N(0, Q)
            Y[t] = Z @ X[t] + nu[t], nu ~ N(0, H)

        Parameters
        ----------
        trace: xarray
            PyMC trace idata object. Should be an xarray returned by pm.sample() with return_inferencedata = True.
        n_steps: int, default = 100
            Number of time steps to simulate
        n_simulations: int, default = 100
            Number of stochastic simulations to run for each parameter draw.
        posterior_samples: int or float, default = None
            A number of subsamples to draw from the posterior trace. If None, all samples in the trace are used. If an
            integer, that number of samples will be drawn with replacement (from among all chains) from the trace. If a
            float between 0 and 1, that fraction of total draws in the trace will be sampled.

        Returns
        -------
        simulations: ArrayLike
            A numpy array of shape (n_simulations x prior_samples, n_timesteps, n_states) with simulated trajectories.

        """

        chains = trace.posterior.dims["chain"]
        draws = trace.posterior.dims["draw"]

        posterior_size = chains * draws
        posterior_samples = get_posterior_samples(posterior_samples, posterior_size)

        resample_idxs = np.random.randint(0, posterior_size, size=posterior_samples)

        pymc_model = modelcontext(None)
        matrix_update_funcs = self.make_matrix_update_funcs(pymc_model)

        thetas = [trace.posterior[var].values for var in self.param_names]
        thetas = [arr.reshape(-1, *arr.shape[2:])[resample_idxs] for arr in thetas]

        simulated_states, simulated_data = unconditional_simulations(
            thetas, matrix_update_funcs, n_steps=n_steps, n_simulations=n_simulations
        )

        return simulated_states, simulated_data
