from collections.abc import Sequence
from typing import Any

import numpy as np
import pytensor.tensor as pt

from pymc_experimental.statespace.core.statespace import PyMCStateSpace, floatX
from pymc_experimental.statespace.models.utilities import make_default_coords
from pymc_experimental.statespace.utils.constants import (
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    ETS_SEASONAL_DIM,
    OBS_STATE_DIM,
)


class BayesianETS(PyMCStateSpace):
    r"""
    Exponential Smoothing State Space Model

    This class can represent a subset of exponential smoothing state space models, specifically those with additive
    errors. Following .. [1], The general form of the model is:

    .. math::

        \begin{align}
        y_t &= l_{t-1} + b_{t-1} + s_{t-m} + \epsilon_t \\
        \epsilon_t &\sim N(0, \sigma)
        \end{align}

    where :math:`l_t` is the level component, :math:`b_t` is the trend component, and :math:`s_t` is the seasonal
    component. These components can be included or excluded, leading to different model specifications. The following
    models are possible:

    * `ETS(A,N,N)`: Simple exponential smoothing

        .. math::

            \begin{align}
            y_t &= l_{t-1} + \epsilon_t \\
            l_t &= l_{t-1} + \alpha \epsilon_t
            \end{align}

    Where :math:`\alpha \in [0, 1]` is a mixing parameter between past observations and current innovations.
    These equations arise by starting from the "component form":

        .. math::

            \begin{align}
            \hat{y}_{t+1 | t} &= l_t \\
            l_t &= \alpha y_t + (1 - \alpha) l_{t-1} \\
            &= l_{t-1} + \alpha (y_t - l_{t-1})
            &= l_{t-1} + \alpha \epsilon_t
            \end{align}

    Where $\epsilon_t$ are the forecast errors, assumed to be IID mean zero and normally distributed. The role of
    :math:`\alpha` is clearest in the second line. The level of the time series at each time is a mixture of
    :math:`\alpha` percent of the incoming data, and :math:`1 - \alpha` percent of the previous level. Recursive
    substitution reveals that the level is a weighted composite of all previous observations; thus the name
    "Exponential Smoothing".

    Additional supposed specifications include:

    * `ETS(A,A,N)`: Holt's linear trend method

        .. math::

            \begin{align}
            y_t &= l_{t-1} + b_{t-1} + \epsilon_t \\
            l_t &= l_{t-1} + b_{t-1} + \alpha \epsilon_t \\
            b_t &= b_{t-1} + \alpha \beta^\star \epsilon_t
            \end{align}

        [1]_ also consider an alternative parameterization with :math:`\beta = \alpha \beta^\star`.

    * `ETS(A,N,A)`: Additive seasonal method

        .. math::

            \begin{align}
            y_t &= l_{t-1} + s_{t-m} + \epsilon_t \\
            l_t &= l_{t-1} + \alpha \epsilon_t \\
            s_t &= s_{t-m} + (1 - \alpha)\gamma^\star \epsilon_t
            \end{align}

        [1]_ also consider an alternative parameterization with :math:`\gamma = (1 - \alpha) \gamma^\star`.

    * `ETS(A,A,A)`: Additive Holt-Winters method

        .. math::

            \begin{align}
            y_t &= l_{t-1} + b_{t-1} + s_{t-m} + \epsilon_t \\
            l_t &= l_{t-1} + \alpha \epsilon_t \\
            b_t &= b_{t-1} + \alpha \beta^\star \epsilon_t \\
            s_t &= s_{t-m} + (1 - \alpha) \gamma^\star \epsilon_t
            \end{align}

        [1]_ also consider an alternative parameterization with :math:`\beta = \alpha \beta^star` and
        :math:`\gamma = (1 - \alpha) \gamma^\star`.

    * `ETS(A, Ad, N)`: Dampened trend method

        .. math::

            \begin{align}
            y_t &= l_{t-1} + b_{t-1} + \epsilon_t \\
            l_t &= l_{t-1} + \alpha \epsilon_t \\
            b_t &= \phi b_{t-1} + \alpha \beta^\star \epsilon_t
            \end{align}

        [1]_ also consider an alternative parameterization with :math:`\beta = \alpha \beta^\star`.

    * `ETS(A, Ad, A)`: Dampened trend with seasonal method

        .. math::

            \begin{align}
            y_t &= l_{t-1} + b_{t-1} + s_{t-m} + \epsilon_t \\
            l_t &= l_{t-1} + \alpha \epsilon_t \\
            b_t &= \phi b_{t-1} + \alpha \beta^\star \epsilon_t \\
            s_t &= s_{t-m} + (1 - \alpha) \gamma^\star \epsilon_t
            \end{align}

        [1]_ also consider an alternative parameterization with :math:`\beta = \alpha \beta^star` and
        :math:`\gamma = (1 - \alpha) \gamma^\star`.


    Parameters
    ----------
    endog: pd.DataFrame
        The observed time series.
    order: tuple of string, Optional
        The exponential smoothing "order". This is a tuple of three strings, each of which should be one of 'A', 'Ad',
        or 'N'.

            - The first element indicates the type of errors to use. Only 'A' is allowed.
            - The second element indicates the type of trend to use. 'A', Ad' or 'N' are allowed.
            - The third element indicates the type of seasonal component to use. 'A' or 'N' are allowed.

        If provided, the model will be initialized from the given order, and the `trend`, `damped_trend`, and `seasonal`
        arguments will be ignored.

    trend: bool
        Whether to include a trend component.
    damped_trend: bool
        Whether to include a damping parameter on the trend component. Ignored if `trend` is `False`.
    seasonal: bool
        Whether to include a seasonal component.
    seasonal_periods: int
        The number of periods in a complete seasonal cycle. Ignored if `seasonal` is `False`.
    measurement_error: bool
        Whether to include a measurement error term in the model. Default is `False`.
    use_transformed_parameterization: bool, default False
        If true, use the :math:`\alpha, \beta, \gamma` parameterization, otherwise use the :math:`\alpha, \beta^\star,
        \gamma^\star` parameterization. This will change the admissible region for the priors.

        - Under the **non-transformed** parameterization, all of :math:`\alpha, \beta^\star, \gamma^\star` should be
          between 0 and 1.
        - Under the **transformed**  parameterization, :math:`\alpha \in (0, 1)`, :math:`\beta \in (0, \alpha)`, and
          :math:`\gamma \in (0, 1 - \alpha)`

        The :meth:`param_info` method will change to reflect the suggested intervals based on the value of this
        argument.
    filter_type: str, default "standard"
        The type of Kalman Filter to use. Options are "standard", "single", "univariate", "steady_state",
        and "cholesky". See the docs for kalman filters for more details.
    verbose: bool, default True
        If true, a message will be logged to the terminal explaining the variable names, dimensions, and supports.

    References
    ----------
    .. [1] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles and practice. OTexts, 2018.
    """

    def __init__(
        self,
        order: tuple[str, str, str] | None = None,
        trend: bool = True,
        damped_trend: bool = False,
        seasonal: bool = False,
        seasonal_periods: int | None = None,
        measurement_error: bool = False,
        use_transformed_parameterization: bool = False,
        filter_type: str = "standard",
        verbose: bool = True,
    ):
        if order is not None:
            if len(order) != 3 or any(not isinstance(o, str) for o in order):
                raise ValueError("Order must be a tuple of three strings.")
            if order[0] != "A":
                raise ValueError("Only additive errors are supported.")
            if order[1] not in {"A", "Ad", "N"}:
                raise ValueError(
                    f"Invalid trend specification. Only 'A' (additive), 'Ad' (additive with dampening), "
                    f"or 'N' (no trend) are allowed. Found {order[1]}"
                )
            if order[2] not in {"A", "N"}:
                raise ValueError(
                    f"Invalid seasonal specification. Only 'A' (additive) or 'N' (no seasonal component) "
                    f"are allowed. Found {order[2]}"
                )

            trend = order[1] != "N"
            damped_trend = order[1] == "Ad"
            seasonal = order[2] == "A"

        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.use_transformed_parameterization = use_transformed_parameterization

        if self.seasonal and self.seasonal_periods is None:
            raise ValueError("If seasonal is True, seasonal_periods must be provided.")

        k_states = (
            2
            + int(trend)
            + int(seasonal) * (seasonal_periods if seasonal_periods is not None else 0)
        )
        k_posdef = 1
        k_endog = 1

        super().__init__(
            k_endog,
            k_states,
            k_posdef,
            filter_type,
            verbose=verbose,
            measurement_error=measurement_error,
        )

    @property
    def param_names(self):
        names = [
            "initial_level",
            "initial_trend",
            "initial_seasonal",
            "P0",
            "alpha",
            "beta",
            "gamma",
            "phi",
            "sigma_state",
            "sigma_obs",
        ]
        if not self.trend:
            names.remove("initial_trend")
            names.remove("beta")
        if not self.damped_trend:
            names.remove("phi")
        if not self.seasonal:
            names.remove("initial_seasonal")
            names.remove("gamma")
        if not self.measurement_error:
            names.remove("sigma_obs")

        return names

    @property
    def param_info(self) -> dict[str, dict[str, Any]]:
        info = {
            "P0": {
                "shape": (self.k_states, self.k_states),
                "constraints": "Positive Semi-definite",
            },
            "initial_level": {
                "shape": None if self.k_endog == 1 else (self.k_endog,),
                "constraints": None,
            },
            "initial_trend": {
                "shape": None if self.k_endog == 1 else (self.k_endog,),
                "constraints": None,
            },
            "initial_seasonal": {"shape": (self.seasonal_periods,), "constraints": None},
            "sigma_obs": {
                "shape": None if self.k_endog == 1 else (self.k_endog,),
                "constraints": "Positive",
            },
            "sigma_state": {
                "shape": None if self.k_posdef == 1 else (self.k_posdef,),
                "constraints": "Positive",
            },
            "alpha": {
                "shape": None,
                "constraints": "0 < alpha < 1",
            },
            "beta": {
                "shape": None,
                "constraints": "0 < beta < 1"
                if not self.use_transformed_parameterization
                else "0 < beta < alpha",
            },
            "gamma": {
                "shape": None,
                "constraints": "0 < gamma< 1"
                if not self.use_transformed_parameterization
                else "0 < gamma < (1 - alpha)",
            },
            "phi": {
                "shape": None,
                "constraints": "0 < phi < 1",
            },
        }

        for name in self.param_names:
            info[name]["dims"] = self.param_dims.get(name, None)

        return {name: info[name] for name in self.param_names}

    @property
    def state_names(self):
        states = ["innovation", "level"]
        if self.trend:
            states += ["trend"]
        if self.seasonal:
            states += [f"L{i}.season" for i in range(1, self.seasonal_periods + 1)]

        return states

    @property
    def observed_states(self):
        return ["data"]

    @property
    def shock_names(self):
        return ["innovation"]

    @property
    def param_dims(self):
        coord_map = {
            "P0": (ALL_STATE_DIM, ALL_STATE_AUX_DIM),
            "sigma_obs": (OBS_STATE_DIM,),
            "sigma_state": (OBS_STATE_DIM,),
            "initial_level": (OBS_STATE_DIM,),
            "initial_trend": (OBS_STATE_DIM,),
            "initial_seasonal": (ETS_SEASONAL_DIM,),
            "seasonal_param": (ETS_SEASONAL_DIM,),
        }

        if self.k_endog == 1:
            coord_map["sigma_state"] = None
            coord_map["sigma_obs"] = None
            coord_map["initial_level"] = None
            coord_map["initial_trend"] = None
        if not self.measurement_error:
            del coord_map["sigma_obs"]
        if not self.seasonal:
            del coord_map["seasonal_param"]

        return coord_map

    @property
    def coords(self) -> dict[str, Sequence]:
        coords = make_default_coords(self)
        if self.seasonal:
            coords.update({ETS_SEASONAL_DIM: list(range(1, self.seasonal_periods + 1))})

        return coords

    def make_symbolic_graph(self) -> None:
        P0 = self.make_and_register_variable(
            "P0", shape=(self.k_states, self.k_states), dtype=floatX
        )
        self.ssm["initial_state_cov"] = P0

        initial_level = self.make_and_register_variable(
            "initial_level", shape=(self.k_endog,) if self.k_endog > 1 else (), dtype=floatX
        )
        self.ssm["initial_state", 1] = initial_level

        # The shape of R can be pre-allocated, then filled with the required parameters
        R = pt.zeros((self.k_states, self.k_posdef))

        alpha = self.make_and_register_variable("alpha", shape=(), dtype=floatX)
        R = pt.set_subtensor(R[1, 0], alpha)  # and l_t = ... + alpha * e_t

        # The R[0, 0] entry needs to be adjusted for a shift in the time indices. Consider the (A, N, N) model:
        # y_t = l_{t-1} + e_t
        # l_t = l_{t-1} + alpha * e_t
        # We want the first equation to be in terms of time t on the RHS, because our observation equation is always
        # y_t = Z @ x_t. Re-arranging equation 2, we get l_{t-1} = l_t - alpha * e_t --> y_t = l_t + e_t - alpha * e_t
        # --> y_t = l_t + (1 - alpha) * e_t
        R = pt.set_subtensor(R[0, :], (1 - alpha))

        # Shock and level component always exists, the base case is e_t = e_t and l_t = l_{t-1}
        T_base = pt.as_tensor_variable(np.array([[0.0, 0.0], [0.0, 1.0]]))

        if self.trend:
            initial_trend = self.make_and_register_variable(
                "initial_trend", shape=(self.k_endog,) if self.k_endog > 1 else (), dtype=floatX
            )
            self.ssm["initial_state", 2] = initial_trend

            beta = self.make_and_register_variable("beta", shape=(), dtype=floatX)
            if self.use_transformed_parameterization:
                R = pt.set_subtensor(R[2, 0], beta)
            else:
                R = pt.set_subtensor(R[2, 0], alpha * beta)

            # If a trend is requested, we have the following transition equations (omitting the shocks):
            # l_t = l_{t-1} + b_{t-1}
            # b_t = b_{t-1}
            T_base = pt.as_tensor_variable(([0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]))

        if self.damped_trend:
            phi = self.make_and_register_variable("phi", shape=(), dtype=floatX)
            # We are always in the case where we have a trend, so we can add the dampening parameter to T_base defined
            # in that branch. Transition equations become:
            # l_t = l_{t-1} + phi * b_{t-1}
            # b_t = phi * b_{t-1}
            T_base = pt.set_subtensor(T_base[1:, 2], phi)

        T_components = [T_base]

        if self.seasonal:
            initial_seasonal = self.make_and_register_variable(
                "initial_seasonal", shape=(self.seasonal_periods,), dtype=floatX
            )

            self.ssm["initial_state", 2 + int(self.trend) :] = initial_seasonal

            gamma = self.make_and_register_variable("gamma", shape=(), dtype=floatX)

            if self.use_transformed_parameterization:
                param = gamma
            else:
                param = (1 - alpha) * gamma

            R = pt.set_subtensor(R[2 + int(self.trend), 0], param)

            # Additional adjustment to the R[0, 0] position is required. Start from:
            # y_t = l_{t-1} + s_{t-m} + e_t
            # l_t = l_{t-1} + alpha * e_t
            # s_t = s_{t-m} + gamma * e_t
            # Solve for l_{t-1} and s_{t-m} in terms of l_t and s_t, then substitute into the observation equation:
            # y_t = l_t + s_t - alpha * e_t - gamma * e_t + e_t --> y_t = l_t + s_t + (1 - alpha - gamma) * e_t
            R = pt.set_subtensor(R[0, 0], R[0, 0] - param)

            # The seasonal component is always going to look like a TimeFrequency structural component, see that
            # docstring for more details
            T_seasonal = pt.eye(self.seasonal_periods, k=-1)
            T_seasonal = pt.set_subtensor(T_seasonal[0, -1], 1.0)
            T_components += [T_seasonal]

        self.ssm["selection"] = R

        T = pt.linalg.block_diag(*T_components)
        self.ssm["transition"] = pt.specify_shape(T, (self.k_states, self.k_states))

        Z = np.zeros((self.k_endog, self.k_states))
        Z[0, 0] = 1.0  # innovation
        Z[0, 1] = 1.0  # level
        if self.seasonal:
            Z[0, 2 + int(self.trend)] = 1.0
        self.ssm["design"] = Z

        # Set up the state covariance matrix
        state_cov_idx = ("state_cov", *np.diag_indices(self.k_posdef))
        state_cov = self.make_and_register_variable(
            "sigma_state", shape=() if self.k_posdef == 1 else (self.k_posdef,), dtype=floatX
        )
        self.ssm[state_cov_idx] = state_cov**2

        if self.measurement_error:
            obs_cov_idx = ("obs_cov", *np.diag_indices(self.k_endog))
            obs_cov = self.make_and_register_variable(
                "sigma_obs", shape=() if self.k_endog == 1 else (self.k_endog,), dtype=floatX
            )
            self.ssm[obs_cov_idx] = obs_cov**2
