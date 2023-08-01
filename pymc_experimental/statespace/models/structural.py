import logging
from abc import ABC
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pytensor.tensor as pt
from scipy import linalg

from pymc_experimental.statespace.core.statespace import PyMCStateSpace
from pymc_experimental.statespace.models.utilities import get_slice_and_move_cursor
from pymc_experimental.statespace.utils.constants import (
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    MATRIX_NAMES,
    OBS_STATE_AUX_DIM,
    OBS_STATE_DIM,
    SHOCK_AUX_DIM,
    SHOCK_DIM,
    SHORT_NAME_TO_LONG,
)

_log = logging.getLogger("pymc.experimental.statespace")

MATRIX_NAMES_LONG = [SHORT_NAME_TO_LONG[x] for x in MATRIX_NAMES if x != "P0"]


def order_to_mask(order):
    if isinstance(order, int):
        return np.ones(order).astype(bool)
    else:
        return np.array(order).astype(bool)


def _shift_slice(idx_slice, i):
    return slice(idx_slice.start + i, idx_slice.stop + i)


def _shift_idx(idx, i):
    if isinstance(idx, slice):
        return _shift_slice(idx, i)
    return idx + i


def _shift_indices(idx, k, j=None) -> Tuple[int]:
    if j:
        new_idx = (_shift_idx(idx[0], k), _shift_idx(idx[1], j))
    else:
        new_idx = (_shift_idx(idx[0], k),)

    return new_idx


class StructuralTimeSeries(PyMCStateSpace):
    r"""
    Structural Time Series Model

    The structural time series model, named by [1] and presented in statespace form in [2], is a framework for
    decomposing a univariate time series into level, trend, seasonal, and cycle components. It also admits the
    possibility of exogenous regressors. Unlike the SARIMAX framework, the time series is not assumed to be stationary.

    Notes
    -----

    .. math::
         y_t = \mu_t + \gamma_t + c_t + \varepsilon_t

    """

    def __init__(
        self,
        x0,
        c,
        d,
        T,
        Z,
        R,
        H,
        Q,
        state_names,
        shock_names,
        param_names,
        param_dims,
        coords,
        param_info,
        param_counts,
        param_indices,
        name=None,
        verbose=True,
        filter_type: str = "standard",
    ):
        # Add the initial state covariance to the parameters
        if name is None:
            name = "data"
        self._name = name

        k_states = T.shape[0]
        k_posdef = Q.shape[0]

        outputs = self._add_inital_state_cov_to_properties(
            param_names, param_dims, param_info, param_indices, param_counts, k_states
        )
        param_names, param_dims, param_info, param_indices, param_counts = outputs
        coords = self._add_default_coords(coords, state_names, shock_names)

        self._state_names = state_names
        self._shock_names = shock_names
        self._param_names = param_names
        self._param_dims = param_dims
        self._coords = coords
        self._param_info = param_info

        super().__init__(1, k_states, k_posdef, filter_type=filter_type, verbose=verbose)

        # Initialize the matrices
        self.ssm["initial_state", :] = x0
        self.ssm["initial_state_cov", :, :] = np.eye(k_states)

        self.ssm["state_intercept", :] = c
        self.ssm["obs_intercept", :] = d
        self.ssm["transition", :, :] = T
        self.ssm["design", :, :] = Z
        self.ssm["selection", :, :] = R
        self.ssm["obs_cov", :, :] = H
        self.ssm["state_cov", :, :] = Q

        self.param_indices = param_indices
        self.param_counts = param_counts

    @staticmethod
    def _add_inital_state_cov_to_properties(
        param_names, param_dims, param_info, param_indices, param_counts, k_states
    ):
        param_names += ["P0"]
        param_dims["P0"] = (ALL_STATE_DIM, ALL_STATE_AUX_DIM)
        param_info["P0"] = {
            "shape": (k_states, k_states),
            "constraints": "Positive semi-definite",
            "dims": param_dims["P0"],
        }
        param_indices["P0"] = "initial_state_cov"
        param_counts["P0"] = k_states**2

        return param_names, param_dims, param_info, param_indices, param_counts

    def _add_default_coords(self, coords, state_names, shock_names):
        coords[ALL_STATE_DIM] = state_names
        coords[ALL_STATE_AUX_DIM] = state_names
        coords[SHOCK_DIM] = shock_names
        coords[SHOCK_AUX_DIM] = shock_names
        coords[OBS_STATE_DIM] = [self._name]
        coords[OBS_STATE_AUX_DIM] = [self._name]

        return coords

    @property
    def param_names(self):
        return self._param_names

    @property
    def state_names(self):
        return self._state_names

    @property
    def observed_states(self):
        return [self._name]

    @property
    def shock_names(self):
        return self._shock_names

    @property
    def param_dims(self):
        return self._param_dims

    @property
    def coords(self) -> Dict[str, Sequence]:
        return self._coords

    @property
    def param_info(self) -> Dict[str, Dict[str, Any]]:
        return self._param_info

    def update(self, theta: pt.TensorVariable, mode: Optional[str] = None) -> None:
        """
        Assign parameter values from vector theta into the correct positions in the state space matrices.

        Parameters
        ----------
        theta: TensorVariable
            Vector of all variables in the state space model

        mode: str, optional
            Compile mode used by pytensor
        """

        cursor = 0
        last_param = self.param_names[-1]
        for param in self.param_names:
            ssm_index = self.param_indices[param]

            param_slice, cursor = get_slice_and_move_cursor(
                cursor, self.param_counts[param], last_slice=param == last_param
            )

            if param == "P0":
                self.ssm[ssm_index] = theta[param_slice].reshape((self.k_states, self.k_states))
            else:
                self.ssm[ssm_index] = theta[param_slice]


class Component(ABC):
    r"""
    Base class for a component of a structural timeseries model.

    This base class contains a subset of the class attributes of the PyMCStateSpace class, and none of the class
    methods. The purpose of a component is to allow the partial definition of a structural model. Components are
    assembled into a full model by the StructuralTimeSeries class.
    """

    def __init__(self, k_endog, k_states, k_posdef):
        self.k_endog = k_endog
        self.k_states = k_states
        self.k_posdef = k_posdef

        self.x0 = np.zeros(k_states)
        self.P0 = np.zeros((k_states, k_states))
        self.c = np.zeros(k_states)
        self.d = np.zeros(k_endog)

        self.T = np.zeros((k_states, k_states))
        self.Z = np.zeros((k_endog, k_states))
        self.R = np.zeros((k_states, k_posdef))
        self.H = np.zeros((k_endog, k_endog))
        self.Q = np.zeros((k_posdef, k_posdef))

        self.param_indices = {}

        self.state_names = []
        self.shock_names = []
        self.param_names = []

        self.coords = {}
        self.param_dims = {}
        self.param_info = {}
        self.param_counts = {}

        self._matrix_shift_factors = {
            "initial_state": (self.k_states, None),
            "initial_state_cov": (self.k_states, self.k_states),
            "state_intercept": (self.k_states, None),
            "obs_intercept": (self.k_endog, None),
            "transition": (self.k_states, self.k_states),
            "design": (self.k_states, None),
            "selection": (self.k_states, self.k_posdef),
            "obs_cov": (0, None),
            "state_cov": (self.k_posdef, self.k_posdef),
        }

    def _combine_matrices(self, other):
        x0 = np.concatenate([self.x0, other.x0])
        P0 = np.zeros((x0.shape[0], x0.shape[0]))
        c = np.concatenate([self.c, other.c])
        d = self.d + other.d

        T = linalg.block_diag(self.T, other.T)
        Z = np.concatenate([self.Z, other.Z], axis=-1)
        R = linalg.block_diag(self.R, other.R)
        H = self.H + other.H
        Q = linalg.block_diag(self.Q, other.Q)

        return [x0, P0, c, d, T, Z, R, H, Q]

    @staticmethod
    def _get_new_shapes(matrices):
        _, _, _, _, T, _, _, _, Q = matrices
        k_states = T.shape[0]
        k_posdef = Q.shape[0]

        return k_states, k_posdef

    def _combine_property(self, other, name):
        self_prop = getattr(self, name)
        if isinstance(self_prop, list):
            return self_prop + getattr(other, name)
        elif isinstance(self_prop, dict):
            new_prop = self_prop.copy()
            new_prop.update(getattr(other, name))
            return new_prop

    def _combine_param_indices(self, other):
        new_indices = self.param_indices.copy()
        other_indices = other.param_indices.copy()
        for param_name, index_tuple in other_indices.items():
            matrix_name, *loc = index_tuple
            if matrix_name == "obs_cov":
                new_loc = tuple(loc)
            else:
                i, j = self._matrix_shift_factors[matrix_name]
                new_loc = _shift_indices(loc, i, j)

            new_indices[param_name] = (matrix_name,) + new_loc

        return new_indices

    def __add__(self, other):
        matrices = self._combine_matrices(other)
        k_states, k_posdef = self._get_new_shapes(matrices)

        state_names = self._combine_property(other, "state_names")
        param_names = self._combine_property(other, "param_names")
        shock_names = self._combine_property(other, "shock_names")
        param_info = self._combine_property(other, "param_info")
        param_dims = self._combine_property(other, "param_dims")
        param_counts = self._combine_property(other, "param_counts")
        coords = self._combine_property(other, "coords")

        param_indices = self._combine_param_indices(other)

        new_comp = Component(k_endog=1, k_states=k_states, k_posdef=k_posdef)
        property_names = [
            "state_names",
            "param_names",
            "shock_names",
            "state_dims",
            "coords",
            "param_dims",
            "param_indices",
            "param_info",
            "param_counts",
        ]
        property_values = [
            state_names,
            param_names,
            shock_names,
            param_dims,
            coords,
            param_dims,
            param_indices,
            param_info,
            param_counts,
        ]

        for prop, value in zip(MATRIX_NAMES + property_names, matrices + property_values):
            setattr(new_comp, prop, value)

        return new_comp

    def build(self, name=None, filter_type="standard", verbose=True):
        """
        Build a StructuralTimeSeries statespace model from the current component(s)

        Parameters
        ----------
        name: str, optional
            Name of the exogenous data being modeled. Default is "obs"

        filter_type : str, optional
            The type of Kalman filter to use. Valid options are "standard", "univariate", "single", "cholesky", and
            "steady_state". For more information, see the docs for each filter. Default is "standard".

        verbose : bool, optional
            If True, displays information about the initialized model. Defaults to True.

        Returns
        -------
        PyMCStateSpace
            An initialized instance of a PyMCStateSpace, constructed using the system matrices contained in the
            components.
        """

        x0, c, d, T, Z, R, H, Q = self.x0, self.c, self.d, self.T, self.Z, self.R, self.H, self.Q
        return StructuralTimeSeries(
            x0,
            c,
            d,
            T,
            Z,
            R,
            H,
            Q,
            name=name,
            state_names=self.state_names,
            shock_names=self.shock_names,
            param_names=self.param_names,
            param_dims=self.param_dims,
            coords=self.coords,
            param_info=self.param_info,
            param_counts=self.param_counts,
            param_indices=self.param_indices,
            filter_type=filter_type,
            verbose=verbose,
        )


class LevelTrendComponent(Component):
    r"""
    Level and trend component of a structural time series model

    Parameters
    __________
    order : int

        Number of time derivatives of the trend to include in the model. For example, when order=3, the trend will
        be of the form ``y = a + b * t + c * t ** 2``, where the coefficients ``a, b, c`` come from the initial
        state values.

    innovations_order : int or sequence of int, optional

        The number of stochastic innovations to include in the model. By default, ``innovations_order = order``

    Notes
    -----
    This class implements the level and trend components of the general structural time series model. In the most
    general form, the level and trend is described by a system of two time-varying equations.

    .. math::
        \begin{align}
            \mu_{t+1} &= \mu_t + \nu_t + \zeta_t \\
            \nu_{t+1} &= \nu_t + \xi_t
            \zeta_t &\sim N(0, \sigma_\zeta) \\
            \xi_t &\sim N(0, \sigma_\xi)
        \end{align}

    Where :math:`\mu_{t+1}` is the mean of the timeseries at time t, and :math:`\nu_t` is the drift or the slope of
    the process. When both innovations :math:`\zeta_t` and :math:`\xi_t` are included in the model, it is known as a
    *local linear trend* model. This system of two equations, corresponding to ``order=2``, can be expanded or
    contracted by adding or removing equations. ``order=3`` would add an acceleration term to the sytsem:

    .. math::
        \begin{align}
            \mu_{t+1} &= \mu_t + \nu_t + \zeta_t \\
            \nu_{t+1} &= \nu_t + \eta_t + \xi_t \\
            \eta_{t+1} &= \eta_{t-1} + \omega_t \\
            \zeta_t &\sim N(0, \sigma_\zeta) \\
            \xi_t &\sim N(0, \sigma_\xi) \\
            \omega_t &\sim N(0, \sigma_\omega)
        \end{align}

    After setting all innovation terms to zero and defining initial states :math:`\mu_0, \nu_0, \eta_0`, these equations
    can be collapsed to:

    .. math::
        \mu_t = \mu_0 + \nu_0 \cdot t + \eta_0 \cdot t^2

    Which clarifies how the order and initial states influence the model. In particular, the initial states are the
    coefficients on the intercept, slope, acceleration, and so on.

    In this light, allowing for innovations can be understood as allowing these coefficients to vary over time. Each
    component can be individually selected for time variation by passing a list to the ``innovations_order`` argument.
    For example, a constant intercept with time varying trend and acceleration is specified as ``order=3,
    innovations_order=[0, 1, 1]``.

    By choosing the ``order`` and ``innovations_order``, a large variety of models can be obtained. Notable
    models include:

    * Constant intercept, ``order=1, innovations_order=0``

    .. math::
        \mu_t = \mu

    * Constant linear slope, ``order=2, innovations_order=0``

    .. math::
        \mu_t = \mu_{t-1} + \nu

    * Gaussian Random Walk, ``order=1, innovations_order=1``

    .. math::
        \mu_t = \mu_{t-1} + \zeta_t

    * Gaussian Random Walk with Drift, ``order=2, innovations_order=1``

    .. math::
        \mu_t = \mu_{t-1} + \nu + \zeta_t

    * Smooth Trend, ``order=2, innovations_order=[0, 1]``

    .. math::
        \begin{align}
            \mu_t &= \mu_{t-1} + \nu_{t-1} \\
            \nu_t &= \nu_{t-1} + \xi_t
        \end{align}

    * Local Level, ``order=2, innovations_order=2``

    [1] notes that the smooth trend model produces more gradually changing slopes than the full local linear trend
    model, and is equivalent to an "integrated trend model".

    References
    ----------
    .. [1] Durbin, James, and Siem Jan Koopman. 2012.
        Time Series Analysis by State Space Methods: Second Edition.
        Oxford University Press.

    """

    def __init__(self, order: int = 2, innovations_order: Optional[int] = None):
        if innovations_order is None:
            innovations_order = order

        order = order_to_mask(order)
        k_states = int(sum(order))

        if isinstance(innovations_order, int):
            n = innovations_order
            innovations_order = order_to_mask(k_states)
            if n > 0:
                innovations_order[n:] = False
            else:
                innovations_order[:] = False
        else:
            innovations_order = order_to_mask(innovations_order)

        k_posdef = int(sum(innovations_order))

        super().__init__(1, k_states, k_posdef)

        self.x0 = np.zeros(k_states)

        self.T = np.zeros((k_states, k_states))
        self.T[np.triu_indices_from(self.T)] = 1

        self.R = np.eye(k_states)
        self.R = self.R[:, innovations_order]

        self.Z = np.array([1.0] + [0.0] * (self.k_states - 1))[None]
        self.Q = np.zeros((k_posdef, k_posdef))

        state_names = ["level", "trend", "acceleration", "jerk", "snap", "crackle", "pop"]

        self.param_names = ["initial_trend"]
        self.state_names = state_names[:k_states]
        self.param_indices = {
            "initial_trend": ("initial_state", np.arange(k_states, dtype="int")),
        }
        self.param_dims = {"initial_trend": ("trend_states",)}
        self.param_counts = {"initial_trend": k_states}
        self.coords = {"trend_states": self.state_names}
        self.param_info = {"initial_trend": {"shape": (k_states,), "constraints": "None"}}

        if k_posdef > 0:
            self.param_names += ["trend_sigmas"]
            self.shock_names = list(np.array(self.state_names)[innovations_order])

            self.param_indices["trend_sigmas"] = ("state_cov", *np.diag_indices(k_posdef))
            self.param_dims["trend_sigmas"] = ("trend_shocks",)
            self.param_counts["trend_sigmas"] = k_posdef
            self.coords["trend_shocks"] = self.shock_names
            self.param_info["trend_sigmas"] = {"shape": (k_posdef,), "constraints": "Positive"}

        for name in self.param_names:
            self.param_info[name]["dims"] = self.param_dims[name]


class MeasurementError(Component):
    r"""
    Measurement error term for a structural timeseries model

    Parameters
    ----------

    name: str, optional

        Name of the observed data. Default is "obs".

    Notes
    -----
    This component should only be used in combination with other components, because it has no states. It's only use
    is to add a variance parameter to the model, associated with the observation noise matrix H.

    Examples
    --------
    Create and estimate a deterministic linear trend with measurement error

    .. code:: python

        from pymc_experimental.statespace import structural as st
        import pymc as pm
        import pytensor.tensor as pt

        trend = st.LevelTrendComponent(order=2, innovations_order=0)
        error = st.MeasurementError()
        ss_mod = (trend + error).build()

        with pm.Model(coords=ss_mod.coords) as model:
            P0 = pm.Deterministic('P0', pt.eye(ss_mod.k_states) * 10, dims=ss_mod.param_dims['P0'])
            intitial_trend = pm.Normal('initial_trend', sigma=10, dims=ss_mod.param_dims['initial_trend'])
            sigma_obs = pm.Exponential('sigma_obs', 1, dims=ss_mod.param_dims['sigma_obs'])

            ss_mod.build_statespace_graph(data, mode='JAX')
            idata = pm.sample(nuts_sampler='numpyro')
    """

    def __init__(self, name=None):
        if name is None:
            name = "obs"
        k_endog = 1
        k_states = 0
        k_posdef = 0

        super().__init__(k_endog, k_states, k_posdef)

        self.param_names = [f"sigma_{name}"]
        self.param_indices = {f"sigma_{name}": ("obs_cov", *np.diag_indices(1))}
        self.param_dims = {f"sigma_{name}": (OBS_STATE_DIM,)}
        self.param_info = {
            f"sigma_{name}": {"shape": (1,), "constraints": "Positive", "dims": "None"}
        }
        self.param_counts = {f"sigma_{name}": 1}


class AutoregressiveComponent(Component):
    r"""
    Autoregressive timeseries component

    Parameters
    ----------
    order: int or sequence of int

        If int, the number of lags to include in the model.
        If a sequence, an array-like of zeros and ones indicating which lags to include in the model.

    Notes
    -----
    An autoregressive component can be thought of as a way o introducing serially correlated errors into the model.
    The process is modeled:

    .. math::
        x_t = \sum_{i=1}^p \rho_i x_{t-i}

    Where ``p``, the number of autoregressive terms to model, is the order of the process. By default, all lags up to
    ``p`` are included in the model. To disable lags, pass a list of zeros and ones to the ``order`` argumnet. For
    example, ``order=[1, 1, 0, 1]`` would become:

    .. math::
        x_t = \rho_1 x_{t-1} + \rho_2 x_{t-1} + \rho_4 x_{t-1}

    The coefficient :math:`\rho_3` has been constrained to zero.

    .. warning:: This class is meant to be used as a component in a structural time series model. For modeling of
              stationary processes with ARIMA, use ``statespace.BayesianARIMA``.

    Examples
    --------
    Model a timeseries as an AR(2) process with non-zero mean:

    .. code:: python

        from pymc_experimental.statespace import structural as st
        import pymc as pm
        import pytensor.tensor as pt

        trend = st.LevelTrendComponent(order=1, innovations_order=0)
        ar = st.AutoregressiveComponent(2)
        ss_mod = (trend + ar).build()

        with pm.Model(coords=ss_mod.coords) as model:
            P0 = pm.Deterministic('P0', pt.eye(ss_mod.k_states) * 10, dims=ss_mod.param_dims['P0'])
            intitial_trend = pm.Normal('initial_trend', sigma=10, dims=ss_mod.param_dims['initial_trend'])
            ar_params = pm.Normal('ar_params', dims=ss_mod.param_dims['ar_params'])
            sigma_ar = pm.Exponential('sigma_ar', 1, dims=ss_mod.param_dims['sigma_ar'])

            ss_mod.build_statespace_graph(data, mode='JAX')
            idata = pm.sample(nuts_sampler='numpyro')

    """

    def __init__(self, order=1):
        order = order_to_mask(order)
        ar_lags = np.flatnonzero(order).ravel().astype(int) + 1
        k_states = int(sum(order))
        super().__init__(k_endog=1, k_states=k_states, k_posdef=1)

        self.T = np.eye(k_states, k=-1)
        self.R[0] = 1
        self.Z[0, 0] = 1

        self.state_names = [f"L{i + 1}.data" for i in range(k_states)]
        self.shock_names = ["L1.data"]

        self.param_names = ["ar_params", "sigma_ar"]
        self.param_indices = {
            "ar_params": ("transition", 0, slice(0, k_states)),
            "sigma_ar": ("state_cov", *np.diag_indices(1)),
        }
        self.param_dims = {"ar_params": ("ar_lags",)}
        self.coords = {"ar_lags": ar_lags}
        self.param_info = {
            "ar_params": {"shape": (k_states,), "constraints": "None", "dims": "(ar_lags, )"},
            "sigma_ar": {"shape": (1,), "constraints": "Positive", "dims": None},
        }
        self.param_counts = {"ar_params": k_states, "sigma_ar": 1}


class TimeSeasonality(Component):
    r"""
    Seasonal component, modeled in the time domain

    Parameters
    ----------
    season_length: int
        The number of periods in a single seasonal cycle, e.g. 12 for monthly data with annual seasonal pattern, 7 for
        daily data with weekly seasonal pattern, etc.

    innovations: bool, default True
        Whether to include stochastic innovations in the strength of the seasonal effect

    name: str, default None
        A name for this seasonal component. Used to label dimensions and coordinates. Useful when multiple seasonal
        components are included in the same model. Default is ``f"Seasonal[s={season_length}]"``

    state_names: list of str, default None
        List of strings for seasonal effect labels. If provided, it must be of length ``season_length``. An example
        would be ``state_names = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']`` when data is daily with a weekly
        seasonal pattern (``season_length = 7``).

        If None, states will be numbered ``[State_0, ..., State_s]``
    Notes
    -----
    A seasonal effect is any pattern that repeats every fixed interval. Although there are many possible ways to
    model seasonal effects, the implementation used here is the one described by [1] as the "canonical" time domain
    representation. The seasonal component can be expressed:

    .. math::
        \gamma_t = -\sum_{i=1}^{s-1} \gamma_{t-i} + \omega_t, \quad \omega_t \sim N(0, \sigma_\gamma)

    Where :math:`s` is the ``seasonal_length`` parameter and :math:`\omega_t` is the (optional) stochastic innovation.
    To give interpretation to the :math:`\gamma` terms, it is helpful to work  through the algebra for a simple
    example. Let :math:`s=4`, and omit the shock term. Define initial conditions :math:`\gamma_0, \gamma_{-1},
    \gamma_{-2}`. The value of the seasonal component for the first 5 timesteps will be:

    .. math::
        \begin{align}
            \gamma_1 &= -\gamma_0 - \gamma_{-1} - \gamma_{-2} \\
             \gamma_2 &= -\gamma_1 - \gamma_0 - \gamma_{-1} \\
                       &= -(-\gamma_0 - \gamma_{-1} - \gamma_{-2}) - \gamma_0 - \gamma_{-1}  \\
                       &= (\gamma_0 - \gamma_0 )+ (\gamma_{-1} - \gamma_{-1}) + \gamma_{-2} \\
                       &= \gamma_{-2} \\
              \gamma_3 &= -\gamma_2 - \gamma_1 - \gamma_0  \\
                       &= -\gamma_{-2} - (-\gamma_0 - \gamma_{-1} - \gamma_{-2}) - \gamma_0 \\
                       &=  (\gamma_{-2} - \gamma_{-2}) + \gamma_{-1} + (\gamma_0 - \gamma_0) \\
                       &= \gamma_{-1} \\
              \gamma_4 &= -\gamma_3 - \gamma_2 - \gamma_1 \\
                       &= -\gamma_{-1} - \gamma_{-2} -(-\gamma_0 - \gamma_{-1} - \gamma_{-2}) \\
                       &= (\gamma_{-2} - \gamma_{-2}) + (\gamma_{-1} - \gamma_{-1}) + \gamma_0 \\
                       &= \gamma_0 \\
              \gamma_5 &= -\gamma_4 - \gamma_3 - \gamma_2 \\
                       &= -\gamma_0 - \gamma_{-1} - \gamma_{-2} \\
                       &= \gamma_1
        \end{align}

    This exercise shows that, given a list ``initial_conditions`` of length ``s-1``, the effects of this model will be:

        - Period 1: ``-sum(initial_conditions)``
        - Period 2: ``initial_conditions[-1]``
        - Period 3: ``initial_conditions[-2]``
        - ...
        - Period s: ``initial_conditions[0]``
        - Period s+1: ``-sum(initial_condition)``

    And so on. So for interpretation, the ``season_length - 1`` initial states are, when reversed, the coefficients
    associated with ``state_names[1:]``.

    .. warning:: Although the ``season_names`` argument expects a list of length ``season_length``, only
                ``season_names[1:]`` will be saved as model dimensions, since the 1st coefficient is not estimated (it is the sum
                of the other 11).

    Examples
    --------
    Estimate monthly with a model with a gaussian random walk trend and monthly seasonality:

    .. code:: python

        from pymc_experimental.statespace import structural as st
        import pymc as pm
        import pytensor.tensor as pt
        import pandas as pd

        # Get month names
        state_names = pd.date_range('1900-01-01', '1900-12-31', freq='MS').month_name().tolist()

        # Build the structural model
        grw = st.LevelTrendComponent(order=1, innovations_order=1)
        annual_season = st.TimeSeasonality(season_length=12, name='annual', state_names=state_names, innovations=False)
        ss_mod = (grw + annual_season).build()

        # Estimate with PyMC
        with pm.Model(coords=ss_mod.coords) as model:
            P0 = pm.Deterministic('P0', pt.eye(ss_mod.k_states) * 10, dims=ss_mod.param_dims['P0'])
            intitial_trend = pm.Deterministic('initial_trend', pt.zeros(1), dims=ss_mod.param_dims['initial_trend'])
            annual_coefs = pm.Normal('annual_coefs', sigma=1e-2, dims=ss_mod.param_dims['annual_coefs'])
            trend_sigmas = pm.HalfNormal('trend_sigmas', sigma=1e-6, dims=ss_mod.param_dims['trend_sigmas'])
            ss_mod.build_statespace_graph(data, mode='JAX')
            idata = pm.sample(nuts_sampler='numpyro')

    References
    ----------
    .. [1] Durbin, James, and Siem Jan Koopman. 2012.
        Time Series Analysis by State Space Methods: Second Edition.
        Oxford University Press.
    """

    def __init__(
        self,
        season_length: int,
        innovations: bool = True,
        name: Optional[str] = None,
        state_names: Optional[list] = None,
    ):
        if name is None:
            name = f"Seasonal[s={season_length}]"
        if state_names is None:
            state_names = [f"{name}_{i}" for i in range(season_length)]
        else:
            if len(state_names) != season_length:
                raise ValueError(
                    f"state_names must be a list of length season_length, got {len(state_names)}"
                )
            state_names = state_names.copy()

        # The first state doesn't get a coefficient, it is defined as -sum(state_coefs)
        # TODO: Can I stash that information in the model somewhere so users don't have to know that?
        state_0 = state_names.pop(-1)
        k_states = season_length - 1

        super().__init__(k_endog=1, k_states=k_states, k_posdef=1)

        self.state_names = state_names
        self.T = np.eye(k_states, k=-1)
        self.T[0, :] = -1
        self.Z[0, 0] = 1

        self.param_names = [f"{name}_coefs"]
        self.param_indices = {f"{name}_coefs": ("initial_state", np.arange(k_states, dtype=int))}
        self.param_info = {
            f"{name}_coefs": {
                "shape": (k_states,),
                "constraints": "None",
                "dims": f"({name}_state, )",
            }
        }
        self.param_dims = {f"{name}_coefs": (f"{name}_periods",)}
        self.param_counts[f"{name}_coefs"] = k_states
        self.coords = {f"{name}_periods": self.state_names}

        if innovations:
            self.R[0] = 1

            self.param_names += [f"sigma_{name}"]
            self.param_indices[f"sigma_{name}"] = ("state_cov", *np.diag_indices(1))
            self.param_info[f"sigma_{name}"] = {
                "shape": (1,),
                "constraints": "Positive",
                "dims": "None",
            }
            self.param_counts[f"sigma_{name}"] = 1
            self.shock_names = [f"{name}"]


class FrequencySeasonality(Component):
    r"""
    Seasonal component, modeled in the frequency domain

    Parameters
    ----------
    season_length: float
        The number of periods in a single seasonal cycle, e.g. 12 for monthly data with annual seasonal pattern, 7 for
        daily data with weekly seasonal pattern, etc. Non-integer seasonal_length is also permitted, for example
        365.2422 days in a (solar) year.

    n: int
        Number of fourier features to include in the seasonal component. Default is ``season_length // 2``, which
        is the maximum possible. A smaller number can be used for a more wave-like seasonal pattern.

    name: str, default None
        A name for this seasonal component. Used to label dimensions and coordinates. Useful when multiple seasonal
        components are included in the same model. Default is ``f"Seasonal[s={season_length}, n={n}]"``

    innovations: bool, default True
        Whether to include stochastic innovations in the strength of the seasonal effect

    Notes
    -----
    A seasonal effect is any pattern that repeats every fixed interval. Although there are many possible ways to
    model seasonal effects, the implementation used here is the one described by [1] as the "canonical" frequency domain
    representation. The seasonal component can be expressed:

    .. math::
        \begin{align}
            \gamma_t &= \sum_{j=1}^{2n} \gamma_{j,t} \\
            \gamma_{j, t+1} &= \gamma_{j,t} \cos \lambda_j + \gamma_{j,t}^\star \sin \lambda_j + \omega_{j, t} \\
            \gamma_{j, t}^\star &= -\gamma_{j,t} \sin \lambda_j + \gamma_{j,t}^\star \cos \lambda_j + \omega_{j,t}^\star
            \lambda_j &= \frac{2\pi j}{s}
        \end{align}

    Where :math:`s` is the ``seasonal_length``.

    Unlike a ``TimeSeasonality`` component, a ``FrequencySeasonality`` component does not require integer season
    length. In addition, for long seasonal periods, it is possible to obtain a more compact state space representation
    by choosing ``n << s // 2``. Using ``TimeSeasonality``, an annual seasonal pattern in daily data requires 364
    states, whereas ``FrequencySeasonality`` always requires ``2 * n`` states, regardless of the ``seasonal_length``.
    The price of this compactness is less representational power. At ``n = 1``, the seasonal pattern will be a pure
    sine wave. At ``n = s // 2``, any arbitrary pattern can be represented.

    One cost of the added flexibility of ``FrequencySeasonality`` is reduced interpretability. States of this model are
    coefficients :math:`\gamma_1, \gamma^\star_1, \gamma_2, \gamma_2^\star ..., \gamma_n, \gamma^\star_n` associated
    with different frequencies in the fourier representation of the seasonal pattern. As a result, it is not possible
    to isolate and identify a "Monday" effect, for instance.
    """

    @staticmethod
    def _compute_transition_block(s, j):
        lam = 2 * np.pi * j / s

        return np.array([[np.cos(lam), np.sin(lam)], [-np.sin(lam), np.cos(lam)]])

    def __init__(self, season_length, n=None, name=None, innovations=True):
        if n is None:
            n = int(season_length // 2)
        if name is None:
            name = f"Frequency[s={season_length}, n={n}]"

        k_states = n * 2
        super().__init__(k_endog=1, k_states=k_states, k_posdef=k_states)

        T_mats = [self._compute_transition_block(season_length, j + 1) for j in range(n)]

        self.T = linalg.block_diag(*T_mats)
        self.Z[0, slice(0, self.k_states, 2)] = 1
        self.R = np.eye(self.k_states)

        self.state_names = [f"{name}_{f}_{i}" for i in range(n) for f in ["Cos", "Sin"]]
        self.param_names = [f"{name}"]
        self.param_indices = {f"{name}": ("initial_state", np.arange(k_states, dtype=int))}
        self.param_dims = {name: (f"{name}_state",)}
        self.param_info = {
            f"{name}": {"shape": (k_states,), "constraints": "None", "dims": f"({name}_state, )"}
        }
        self.param_counts[f"{name}"] = k_states
        self.coords = {f"{name}_state": self.state_names}

        if innovations:
            self.param_names += [f"sigma_{name}"]
            self.param_indices[f"sigma_{name}"] = ("state_cov", *np.diag_indices(k_states))
            self.param_info[f"sigma_{name}"] = {
                "shape": (1,),
                "constraints": "Positive",
                "dims": "None",
            }
            self.param_counts[f"sigma_{name}"] = 1
            self.shock_names = [f"{name}_{f}_{i}" for i in range(n) for f in ["Cos", "Sin"]]
