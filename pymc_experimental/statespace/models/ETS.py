from typing import Any, Sequence

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
            l_t &= l_{t-1} + \alpha \epsilon_t \\
            b_t &= b_{t-1} + \beta \epsilon_t
            \end{align}

    * `ETS(A,N,A)`: Additive seasonal method

        .. math::

            \begin{align}
            y_t &= l_{t-1} + s_{t-m} + \epsilon_t \\
            l_t &= l_{t-1} + \alpha \epsilon_t \\
            s_t &= s_{t-m} + \gamma \epsilon_t
            \end{align}

    * `ETS(A,A,A)`: Additive Holt-Winters method

        .. math::

            \begin{align}
            y_t &= l_{t-1} + b_{t-1} + s_{t-m} + \epsilon_t \\
            l_t &= l_{t-1} + \alpha \epsilon_t \\
            b_t &= b_{t-1} + \beta \epsilon_t \\
            s_t &= s_{t-m} + \gamma \epsilon_t
            \end{align}

    * `ETS(A, Ad, N)`: Dampened trend method

        .. math::

            \begin{align}
            y_t &= l_{t-1} + b_{t-1} + \epsilon_t \\
            l_t &= l_{t-1} + \alpha \epsilon_t \\
            b_t &= \phi b_{t-1} + \beta \epsilon_t
            \end{align}

    * `ETS(A, Ad, A)`: Dampened trend with seasonal method

        .. math::

            \begin{align}
            y_t &= l_{t-1} + b_{t-1} + s_{t-m} + \epsilon_t \\
            l_t &= l_{t-1} + \alpha \epsilon_t \\
            b_t &= \phi b_{t-1} + \beta \epsilon_t \\
            s_t &= s_{t-m} + \gamma \epsilon_t
            \end{align}

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
            "x0",
            "P0",
            "alpha",
            "beta",
            "gamma",
            "phi",
            "sigma_state",
            "sigma_obs",
        ]
        if not self.trend:
            names.remove("beta")
        if not self.damped_trend:
            names.remove("phi")
        if not self.seasonal:
            names.remove("gamma")
        if not self.measurement_error:
            names.remove("sigma_obs")

        return names

    @property
    def param_info(self) -> dict[str, dict[str, Any]]:
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
                "shape": None if self.k_endog == 1 else (self.k_endog,),
                "constraints": "Positive",
            },
            "sigma_state": {
                "shape": None if self.k_posdef == 1 else (self.k_posdef,),
                "constraints": "Positive",
            },
            "alpha": {
                "shape": None,
                "constraints": "0 < Sum(alpha, beta, gamma) < 1",
            },
            "beta": {
                "shape": None,
                "constraints": "0 < Sum(alpha, beta, gamma) < 1",
            },
            "gamma": {
                "shape": None,
                "constraints": "0 < Sum(alpha, beta, gamma) < 1",
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
            "x0": (ALL_STATE_DIM,),
            "P0": (ALL_STATE_DIM, ALL_STATE_AUX_DIM),
            "sigma_obs": (OBS_STATE_DIM,),
            "sigma_state": (OBS_STATE_DIM,),
            "seasonal_param": (ETS_SEASONAL_DIM,),
        }

        if self.k_endog == 1:
            coord_map["sigma_state"] = ()
            coord_map["sigma_obs"] = ()
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
        x0 = self.make_and_register_variable("x0", shape=(self.k_states,), dtype=floatX)
        P0 = self.make_and_register_variable(
            "P0", shape=(self.k_states, self.k_states), dtype=floatX
        )

        # x0, P0, Z, and R do not depend on the user config beyond the shape
        self.ssm["initial_state", :] = x0
        self.ssm["initial_state_cov"] = P0

        # The shape of R can be pre-allocated, then filled with the required parameters
        R = pt.zeros((self.k_states, self.k_posdef))
        R = pt.set_subtensor(R[0, :], 1.0)  # We will always have y_t = ... + e_t

        alpha = self.make_and_register_variable("alpha", shape=(), dtype=floatX)
        R = pt.set_subtensor(R[1, 0], alpha)  # and l_t = ... + alpha * e_t

        # Shock and level component always exists, the base case is e_t = e_t and l_t = l_{t-1}
        T_base = pt.as_tensor_variable(np.array([[0.0, 0.0], [0.0, 1.0]]))

        if self.trend:
            beta = self.make_and_register_variable("beta", shape=(), dtype=floatX)
            R = pt.set_subtensor(R[2, 0], beta)

            # If a trend is requested, we have the following transition equations (omitting the shocks):
            # y_t = l_{t-1} + b_{t-1}
            # l_t = l_{t-1} + b_{t-1}
            # b_t = b_{t-1}
            T_base = pt.as_tensor_variable(([0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]))

        if self.damped_trend:
            phi = self.make_and_register_variable("phi", shape=(), dtype=floatX)
            # We are always in the case where we have a trend, so we can add the dampening parameter to T_base defined
            # in that branch. Transition equations become:
            # y_t = l_{t-1} + phi * b_{t-1}
            # l_t = l_{t-1} + phi * b_{t-1}
            # b_t = phi * b_{t-1}
            T_base = pt.set_subtensor(T_base[1:, 2], phi)

        T_components = [T_base]

        if self.seasonal:
            gamma = self.make_and_register_variable("gamma", shape=(), dtype=floatX)
            R = pt.set_subtensor(R[3, 0], gamma)

            # The seasonal component is always going to look like a TimeFrequency structural component, see that
            # docstring for more details
            T_seasonal = pt.eye(self.seasonal_periods, k=-1)
            T_seasonal = pt.set_subtensor(T_seasonal[0, :], -1)
            T_components += [T_seasonal]

        self.ssm["selection"] = R

        T = pt.linalg.block_diag(*T_components)
        self.ssm["transition"] = pt.specify_shape(T, (self.k_states, self.k_states))

        Z = np.zeros((self.k_endog, self.k_states))
        Z[0, 0] = 1.0  # innovation
        Z[0, 1] = 1.0  # level
        if self.trend:
            Z[0, 2] = 1.0
        if self.seasonal:
            Z[0, 2 + int(self.trend)] = 1.0
        self.ssm["design"] = Z

        # Set up the state covariance matrix
        state_cov_idx = ("state_cov",) + np.diag_indices(self.k_posdef)
        state_cov = self.make_and_register_variable(
            "sigma_state", shape=() if self.k_posdef == 1 else (self.k_posdef,), dtype=floatX
        )
        self.ssm[state_cov_idx] = state_cov**2

        if self.measurement_error:
            obs_cov_idx = ("obs_cov",) + np.diag_indices(self.k_endog)
            obs_cov = self.make_and_register_variable(
                "sigma_obs", shape=() if self.k_endog == 1 else (self.k_endog,), dtype=floatX
            )
            self.ssm[obs_cov_idx] = obs_cov**2
