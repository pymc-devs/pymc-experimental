import logging
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
from numpy.typing import ArrayLike
from pymc.model import modelcontext
from pytensor.compile import get_mode

from pymc_experimental.statespace.core.representation import (
    SHORT_NAME_TO_LONG,
    PytensorRepresentation,
)
from pymc_experimental.statespace.filters import (
    CholeskyFilter,
    KalmanSmoother,
    SingleTimeseriesFilter,
    StandardFilter,
    SteadyStateFilter,
    UnivariateFilter,
)
from pymc_experimental.statespace.filters.distributions import (
    LinearGaussianStateSpace,
    SequenceMvNormal,
)
from pymc_experimental.statespace.utils.constants import (
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    EXTENDED_TIME_DIM,
    FILTER_OUTPUT_DIMS,
    FILTER_OUTPUT_NAMES,
    MATRIX_DIMS,
    MATRIX_NAMES,
    OBS_STATE_DIM,
    SHOCK_AUX_DIM,
    SHOCK_DIM,
    SMOOTHER_OUTPUT_NAMES,
    TIME_DIM,
)
from pymc_experimental.statespace.utils.data_tools import register_data_with_pymc

_log = logging.getLogger("pymc.experimental.statespace")

floatX = pytensor.config.floatX
FILTER_FACTORY = {
    "standard": StandardFilter,
    "univariate": UnivariateFilter,
    "steady_state": SteadyStateFilter,
    "single": SingleTimeseriesFilter,
    "cholesky": CholeskyFilter,
}


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

        self._fit_mode = None
        self._fit_coords = None
        self._prior_mod = pm.Model()

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
            out += f'\t{param} -- shape: {info["shape"]}, constraints: {info["constraints"]}, dims: {info["dims"]}\n'
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
    def shock_names(self):
        raise NotImplementedError

    @property
    def default_priors(self):
        raise NotImplementedError

    @property
    def coords(self) -> Dict[str, Sequence]:
        raise NotImplementedError

    @property
    def param_dims(self):
        raise NotImplementedError

    def add_default_priors(self):
        raise NotImplementedError

    def _get_matrix_shape_and_dims(self, name):
        pm_mod = modelcontext(None)
        dims = MATRIX_DIMS.get(name, None)
        dims = dims if all([dim in pm_mod.coords.keys() for dim in dims]) else None
        shape = self.ssm[SHORT_NAME_TO_LONG[name]].type.shape if dims is None else None

        return shape, dims

    def _get_output_shape_and_dims(self, idata, filter_output):
        mu_dims = None
        cov_dims = None

        mu_shape = idata[f"{filter_output}_state"].values.shape[2:]
        cov_shape = idata[f"{filter_output}_covariance"].values.shape[2:]

        if all(
            [
                dim in self._fit_coords
                for dim in [TIME_DIM, EXTENDED_TIME_DIM, ALL_STATE_DIM, ALL_STATE_AUX_DIM]
            ]
        ):
            time_dim = EXTENDED_TIME_DIM if filter_output == "predicted" else TIME_DIM
            mu_dims = [time_dim, ALL_STATE_DIM]
            cov_dims = [time_dim, ALL_STATE_DIM, ALL_STATE_AUX_DIM]

            mu_shape = None
            cov_shape = None

        return mu_shape, cov_shape, mu_dims, cov_dims

    def update(self, theta: pt.TensorVariable) -> None:
        """
        Put parameter values from vector theta into the correct positions in the state space matrices.

        Parameters
        ----------
        theta: TensorVariable
            Vector of all variables in the state space model
        """
        raise NotImplementedError

    def _gather_required_random_variables(self) -> pt.TensorVariable:
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
        self, data, register_data=True, mode=None, return_updates=False, include_smoother=True
    ) -> None:
        """
        Given parameter vector theta, constructs the full computational graph describing the state space model.
        Must be called inside a PyMC model context.
        """

        pm_mod = modelcontext(None)

        theta = self._gather_required_random_variables()
        self.update(theta)

        matrices = self.unpack_statespace()
        n_obs = self.ssm.shapes["obs_intercept"][0]
        obs_coords = pm_mod.coords.get(OBS_STATE_DIM, None)

        if register_data:
            register_data_with_pymc(data, n_obs=n_obs, obs_coords=obs_coords)

        registered_matrices = []
        for i, (matrix, name) in enumerate(zip(matrices, MATRIX_NAMES)):
            if not getattr(pm_mod, name, None):
                shape, dims = self._get_matrix_shape_and_dims(name)
                x = pm.Deterministic(name, matrix, dims=dims)
                registered_matrices.append(x)
            else:
                registered_matrices.append(matrices[i])

        filter_outputs = self.kalman_filter.build_graph(
            pt.as_tensor_variable(data),
            *registered_matrices,
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
        names_to_use = FILTER_OUTPUT_NAMES.copy()

        if include_smoother:
            smoothed_states, smoothed_covariances = self._build_smoother_graph(
                filtered_states, filtered_covariances, mode=mode
            )
            outputs_to_use += [smoothed_states, smoothed_covariances]
            names_to_use += SMOOTHER_OUTPUT_NAMES.copy()

        for output, name in zip(outputs_to_use, names_to_use):
            dims = FILTER_OUTPUT_DIMS.get(name, None)
            dims = dims if all([dim in pm_mod.coords.keys() for dim in dims]) else None
            pm.Deterministic(name, output, dims=dims)

        obs_dims = FILTER_OUTPUT_DIMS.get("obs", None)
        obs_dims = obs_dims if all([dim in pm_mod.coords.keys() for dim in obs_dims]) else None
        SequenceMvNormal(
            "obs",
            mus=observed_states,
            covs=observed_covariances,
            logp=logp,
            observed=data,
            dims=obs_dims,
        )

        self._fit_coords = pm_mod.coords.copy()
        self._fit_mode = mode

        if return_updates:
            return updates

    def _build_smoother_graph(self, filtered_states, filtered_covariances, mode=None):
        pymc_model = modelcontext(None)
        with pymc_model:
            *_, T, Z, R, H, Q = self.unpack_statespace()

            smooth_states, smooth_covariances = self.kalman_smoother.build_graph(
                T, R, Q, filtered_states, filtered_covariances, mode=mode
            )

            return smooth_states, smooth_covariances

    def _build_dummy_graph(self, skip_matrices=None):
        modelcontext(None)
        if skip_matrices is None:
            skip_matrices = []

        matrices = []
        for name in MATRIX_NAMES:
            if name in skip_matrices:
                continue

            shape, dims = self._get_matrix_shape_and_dims(name)
            x = pm.Flat(name, shape=shape, dims=dims)
            matrices.append(x)

        return matrices

    @staticmethod
    def sample_conditional_prior(idata, filter_output="filtered") -> ArrayLike:
        """
        Sample from the conditional prior; that is, given parameter draws from the prior distribution, compute kalman
        filtered trajectories. Trajectories are drawn from a single multivariate normal with mean and covariance
        computed via either the kalman filter, smoother, or predictions.

        Parameters
        ----------
        idata: InferenceData
            Arviz InfereData object with a prior group
        filter_output: string, default = 'filtered'
            One of 'filtered', 'smoothed', or 'predicted'. Corresponds to which Kalman filter output you would like to
            sample from.
        Returns
        -------
        idata: InferenceData
        """

        validate_filter_arg(filter_output)
        raise NotImplementedError

    def sample_conditional_posterior(self, idata):
        """
        Sample from the conditional posterior; that is, given parameter draws from the posterior distribution,
        compute kalman filtered trajectories. Trajectories are drawn from a single multivariate normal with mean and
        covariance computed via either the kalman filter, smoother, or predictions.

        Parameters
        ----------
        idata: InferenceData
            Arviz InferenceData object with a "posterior" group
        filter_output: string, default = 'filtered'
            One of 'filtered', 'smoothed', or 'predicted'. Corresponds to which Kalman filter output you would like to
            sample from.

        Returns
        -------
        idata: InferenceData


        """
        with pm.Model(coords=self._fit_coords):
            for filter_output in ["filtered", "predicted", "smoothed"]:
                mu_shape, cov_shape, mu_dims, cov_dims = self._get_output_shape_and_dims(
                    idata.posterior, filter_output
                )

                mus = pm.Flat(f"{filter_output}_state", shape=mu_shape, dims=mu_dims)
                covs = pm.Flat(f"{filter_output}_covariance", shape=cov_shape, dims=cov_dims)

                SequenceMvNormal(
                    f"{filter_output}_posterior",
                    mus=mus,
                    covs=covs,
                    logp=pt.zeros(mus.shape[0]),
                    dims=mu_dims,
                )
            idata_post = pm.sample_posterior_predictive(
                idata,
                var_names=["filtered_posterior", "predicted_posterior", "smoothed_posterior"],
                compile_kwargs={"mode": get_mode(self._fit_mode)},
            )
        return idata_post

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
        raise NotImplementedError

    def sample_unconditional_posterior(
        self, idata, steps=None, use_data_time_dim=False
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Draw unconditional sample trajectories according to state space dynamics, using random samples from the
        posterior distribution over model parameters. The state space update equations are:

            X[t+1] = T @ X[t] + R @ eta[t], eta ~ N(0, Q)
            Y[t] = Z @ X[t] + nu[t], nu ~ N(0, H)
            x[0] ~ N(a0, P0)

        Parameters
        ----------
        idata: InferenceData
            Arviz InferenceData object with a posterior group

        Returns
        -------
        idata: InferenceData
        """
        dims = None
        temp_coords = self._fit_coords.copy()

        if not use_data_time_dim:
            temp_coords.update({TIME_DIM: np.arange(1 + steps, dtype="int")})
            steps = len(temp_coords[TIME_DIM]) - 1
        elif steps is not None:
            n_dimsteps = len(temp_coords[TIME_DIM])
            if n_dimsteps != steps:
                raise ValueError(
                    f"Length of time dimension does not match specified number of steps, expected"
                    f" {n_dimsteps} steps, or steps=None."
                )

        if all([dim in self._fit_coords for dim in [TIME_DIM, ALL_STATE_DIM, OBS_STATE_DIM]]):
            dims = [TIME_DIM, ALL_STATE_DIM, OBS_STATE_DIM]

        with pm.Model(coords=temp_coords if dims is not None else None):
            matrices = self._build_dummy_graph()
            _ = LinearGaussianStateSpace(
                "posterior",
                *matrices,
                steps=steps,
                dims=dims,
                mode=self._fit_mode,
            )
            idata_unconditional_post = pm.sample_posterior_predictive(
                idata,
                var_names=["posterior_latent", "posterior_observed"],
                compile_kwargs={"mode": self._fit_mode},
            )

        return idata_unconditional_post

    def forecast(self, idata, start, periods=None, end=None, filter_output="predicted"):
        validate_filter_arg(filter_output)
        if periods is None and end is None:
            raise ValueError("Must specify one of either periods or end")
        if periods is not None and end is not None:
            raise ValueError("Must specify exactly one of either periods or end")

        dims = None
        temp_coords = self._fit_coords.copy()

        filter_time_dim = EXTENDED_TIME_DIM if filter_output == "predicted" else TIME_DIM

        if all([dim in temp_coords for dim in [filter_time_dim, ALL_STATE_DIM, OBS_STATE_DIM]]):
            dims = [TIME_DIM, ALL_STATE_DIM, OBS_STATE_DIM]

        time_index = temp_coords[filter_time_dim]

        if start not in time_index:
            raise ValueError("Start date is not in the provided data")

        is_datetime = isinstance(time_index[0], pd.Timestamp)

        forecast_index = None

        if is_datetime:
            time_index = pd.DatetimeIndex(time_index)
            freq = time_index.inferred_freq

            if end is not None:
                forecast_index = pd.date_range(start, end=end, freq=freq)
            if periods is not None:
                forecast_index = pd.date_range(start, periods=periods, freq=freq)
            t0 = forecast_index[0]

        else:
            if end is not None:
                forecast_index = np.arange(start, end, dtype="int")
            if periods is not None:
                forecast_index = np.arange(start, start + periods, dtype="int")
            t0 = forecast_index[0]

        t0_idx = np.flatnonzero(time_index == t0)[0]
        temp_coords["data_time"] = time_index
        temp_coords[TIME_DIM] = forecast_index

        mu_shape, cov_shape, mu_dims, cov_dims = self._get_output_shape_and_dims(
            idata.posterior, filter_output
        )

        if mu_dims is not None:
            mu_dims = ["data_time"] + mu_dims[1:]
        if cov_dims is not None:
            cov_dims = ["data_time"] + cov_dims[1:]

        with pm.Model(coords=temp_coords):
            c, d, T, Z, R, H, Q = self._build_dummy_graph(skip_matrices=["x0", "P0"])
            mu = pm.Flat(f"{filter_output}_state", shape=mu_shape, dims=mu_dims)
            cov = pm.Flat(f"{filter_output}_covariance", shape=cov_shape, dims=cov_dims)

            x0 = pm.Deterministic(
                "x0_slice", mu[t0_idx], dims=mu_dims[1:] if mu_dims is not None else None
            )
            P0 = pm.Deterministic(
                "P0_slice", cov[t0_idx], dims=cov_dims[1:] if cov_dims is not None else None
            )

            _ = LinearGaussianStateSpace(
                "forecast",
                x0,
                P0,
                c,
                d,
                T,
                Z,
                R,
                H,
                Q,
                steps=len(forecast_index[:-1]),
                dims=dims,
                mode=self._fit_mode,
            )

            idata_forecast = pm.sample_posterior_predictive(
                idata,
                var_names=["forecast_latent", "forecast_observed"],
                compile_kwargs={"mode": self._fit_mode},
            )

            return idata_forecast

    def impulse_response_function(
        self,
        idata,
        steps: int = None,
        shock_size: Sequence[float] = None,
        shock_cov: Sequence[float] = None,
        shock_trajectory: Sequence[float] = None,
        orthogonalize_shocks=False,
    ):

        options = [shock_size, shock_cov, shock_trajectory]
        Q_value = None  # default case -- sample from posterior

        if sum(x is not None for x in options) > 1:
            raise ValueError("Specify exactly 0 or 1 of shock_size, shock_cov, or shock_trajectory")
        elif shock_size is not None:
            Q_value = pt.eye(self.k_posdef) * shock_size
        elif shock_cov is not None:
            Q_value = pt.as_tensor_variable(shock_cov)
        elif shock_trajectory is not None:
            n, k = shock_trajectory.shape
            if k != self.k_posdef:
                raise ValueError(
                    "If shock_trajectory is provided, there must be a trajectory provided for each shock. "
                    f"Model has {self.k_posdef} shocks, but shock_trajectory has only {k} columns"
                )
            if steps is not None and steps != n:
                _log.warning(
                    "Both steps and shock_trajectory were provided but do not agree. Length of "
                    "shock_trajectory will take priority, and steps will be ignored."
                )
                steps = n

            shock_trajectory = pt.as_tensor_variable(shock_trajectory)

        if steps is None:
            steps = 40

        simulation_coords = self._fit_coords.copy()
        simulation_coords[TIME_DIM] = np.arange(steps, dtype="int")

        with pm.Model(coords=simulation_coords):
            x0 = pm.DiracDelta("x0_new", pt.zeros(self.k_states), dims=[ALL_STATE_DIM])
            matrices = self._build_dummy_graph(
                skip_matrices=["x0"] if Q_value is None else ["x0", "Q"]
            )

            if Q_value is None:
                # x0 was not loaded into the model
                P0, c, _, T, _, R, _, Q = matrices
            else:
                # Neither x0 nor Q was loaded into the model
                P0, c, _, T, _, R, _ = matrices
                Q = pm.DiracDelta("Q_new", Q_value, dims=[SHOCK_DIM, SHOCK_AUX_DIM])

            if shock_trajectory is None:
                shock_trajectory = pt.zeros((steps, self.k_posdef))
                if orthogonalize_shocks:
                    Q = pt.linalg.cholesky(Q)
                initial_shock = pm.MvNormal("initial_shock", mu=0, cov=Q, dims=[SHOCK_DIM])
                shock_trajectory = pt.set_subtensor(shock_trajectory[0], initial_shock)

            def irf_step(shock, x, c, T, R):
                next_x = c + T @ x + R @ shock
                return next_x

            irf, updates = pytensor.scan(
                irf_step,
                sequences=[shock_trajectory],
                outputs_info=[x0],
                non_sequences=[c, T, R],
                n_steps=steps,
                strict=True,
                mode=self._fit_mode,
            )
            irf = pm.Deterministic("irf", irf, dims=[TIME_DIM, ALL_STATE_DIM])
            irf_idata = pm.sample_posterior_predictive(
                idata, var_names=["irf"], compile_kwargs={"mode": self._fit_mode}
            )

            return irf_idata
