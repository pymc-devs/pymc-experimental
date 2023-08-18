import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
from arviz import InferenceData
from pymc.model import modelcontext
from pymc.util import RandomState
from pytensor import Variable, graph_replace
from pytensor.compile import get_mode

from pymc_experimental.statespace.core.representation import PytensorRepresentation
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
from pymc_experimental.statespace.filters.utilities import stabilize
from pymc_experimental.statespace.utils.constants import (
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    FILTER_OUTPUT_DIMS,
    FILTER_OUTPUT_NAMES,
    FILTER_OUTPUT_TYPES,
    JITTER_DEFAULT,
    MATRIX_DIMS,
    MATRIX_NAMES,
    OBS_STATE_DIM,
    SHOCK_AUX_DIM,
    SHOCK_DIM,
    SHORT_NAME_TO_LONG,
    SMOOTHER_OUTPUT_NAMES,
    TIME_DIM,
    VECTOR_VALUED,
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


def _validate_filter_arg(filter_arg):
    if filter_arg.lower() not in FILTER_OUTPUT_TYPES:
        raise ValueError(
            f'filter_output should be one of {", ".join(FILTER_OUTPUT_TYPES)}, received {filter_arg}'
        )


def _verify_group(group):
    if group not in ["prior", "posterior"]:
        raise ValueError(f'Argument "group" must be one of "prior" or "posterior", found {group}')


class PyMCStateSpace:
    r"""
    Base class for Linear Gaussian Statespace models in PyMC.

    Holds a ``PytensorRepresentation`` and ``KalmanFilter``, and provides a mapping between a PyMC model and the
    statespace model.

    Parameters
    ----------
    k_endog : int
        The number of endogenous variables (observed time series).

    k_states : int
        The number of state variables.

    k_posdef : int
        The number of shocks in the model

    filter_type : str, optional
        The type of Kalman filter to use. Valid options are "standard", "univariate", "single", "cholesky", and
        "steady_state". For more information, see the docs for each filter. Default is "standard".

    verbose : bool, optional
        If True, displays information about the initialized model. Defaults to True.

    measurement_error : bool, optional
        If true, the model contains measurement error. Needed by post-estimation sampling methods to decide how to
        compute the observation errors. If False, these errors are deterministically zero; if True, they are sampled
        from a multivariate normal.

    Notes
    -----
    Based on the statsmodels statespace implementation https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/statespace/representation.py,
    described in [1].

    All statespace models inherit from this base class, which is responsible for providing an interface between a
    PyMC model and a PytensorRepresentation of a linear statespace model. This is done via the ``update`` method,
    which takes as input a vector of PyMC random variables and assigns them to their correct positions inside the
    underlying ``PytensorRepresentation``. Construction of the parameter vector, called ``theta``, is done
    automatically, but depend on the names provided in the ``param_names`` property.

    To implement a new statespace model, one needs to:

    1. Overload the ``param_names`` property to return a list of parameter names.
    2. Overload the ``update`` method to put these parameters into their respective statespace matrices

    In addition, a number of additional properties can be overloaded to provide users with additional resources
    when writing their PyMC models. For details, see the attributes section of the docs for this class.

    Finally, this class holds post-estimation methods common to all statespace models, which do not need to be
    overloaded when writing a custom statespace model.

    Examples
    --------
    The local level model is a simple statespace model. It is a Gaussian random walk with a drift term that itself also
    follows a Gaussian random walk, as described by the following two equations:

    .. math::
        \begin{align}
            y_{t} &= y_{t-1} + x_t + \nu_t \tag{1} \\
            x_{t} &= x_{t-1} + \eta_t \tag{2}
        \end{align}

    Where :math:`y_t` is the observed data, and :math:`x_t` is an unobserved trend term. The model has two unknown
    parameters, the variances on the two innovations, :math:`sigma_\nu` and :math:`sigma_\eta`. Take the hidden state
    vector to be :math:`\begin{bmatrix} y_t & x_t \end{bmatrix}^T` and the shock vector
    :math:`\varepsilon_t = \begin{bmatrix} \nu_t & \eta_t \end{bmatrix}^T`. Then this model can be cast into state-space
    form with the following matrices:

    .. math::
        \begin{align}
            T &= \begin{bmatrix}1 & 1 \\ 0 & 1 \end{bmatrix} &
            R &= \begin{bmatrix}1 & 0 \\ 0 & 1 \end{bmatrix} &
            Q &= \begin{bmatrix} \sigma_\nu & 0 \\ 0 & \sigma_\eta \end{bmatrix} &
            Z &= \begin{bmatrix} 1 & 0 \end{bmatrix}
        \end{align}

    With the remaining statespace matrices as zero matrices of the appropriate sizes. The model has two states,
    two shocks, and one observed state. Knowing all this, a very simple  local level model can be implemented as
    follows:

    .. code:: python

        from pymc_experimental.statespace.core import PyMCStateSpace
        import numpy as np

        class LocalLevel(PyMCStateSpace):
            def __init__():
                # Initialize the superclass. This creates the PytensorRepresentation and the Kalman Filter
                super().__init__(k_endog=1, k_states=2, k_posdef=2)

                # Declare the non-zero, non-parameterized matrices
                self.ssm['transition', :, :] = np.array([[1.0, 1.0], [0.0, 1.0]])
                self.ssm['selection', :, :] = np.eye(2)
                self.ssm['design', :, :] = np.array([[1.0, 0.0]])

            @property
            def param_names(self):
                return ['x0', 'P0', 'sigma_nu', 'sigma_eta']

            def update(self, theta, mode=None):
                # Since the param_names are ['x0', 'P0', 'sigma_nu', 'sigma_eta'], theta will come in as
                # [x0.ravel(), P0.ravel(), sigma_nu, sigma_eta]
                # It will have length 2 + 4 + 1 + 1 = 8

                x0 = theta[:2]
                P0 = theta[2:6].reshape(2,2)
                sigma_nu = theta[6]
                sigma_eta = theta[7]

                # Assign parameters to their correct locations
                self.ssm['initial_state', :] = x0
                self.ssm['initial_state_cov', :, :] = P0
                self.ssm['state_cov', 0, 0] = sigma_nu
                self.ssm['state_cov', 1, 1] = sigma_eta

    After defining priors over the named parameters ``P0``, ``x0``, ``sigma_eta``, and ``sigma_nu``, we can sample
    from this model:

    .. code:: python

        import pymc as pm

        ll = LocalLevel()

        with pm.Model() as mod:
            x0 = pm.Normal('x0', shape=(2,))
            P0_diag = pm.Exponential('P0_diag', 1, shape=(2,))
            P0 = pm.Deterministic('P0', pt.diag(P0_diag))
            sigma_nu = pm.Exponential('sigma_nu', 1)
            sigma_eta = pm.Exponential('sigma_eta', 1)

            ll.build_statespace_graph(data = data)
            idata = pm.sample()


    References
    ----------
    .. [1] Fulton, Chad. "Estimating time series models by state space methods in Python: Statsmodels." (2015).
       http://www.chadfulton.com/files/fulton_statsmodels_2017_v1.pdf

    """

    def __init__(
        self,
        k_endog: int,
        k_states: int,
        k_posdef: int,
        filter_type: str = "standard",
        verbose: bool = True,
        measurement_error: bool = False,
    ):
        self._fit_mode = None
        self._fit_coords = None
        self._needs_exog_data = False
        self._exog_names = []
        self._name_to_variable = {}

        self.k_endog = k_endog
        self.k_states = k_states
        self.k_posdef = k_posdef
        self.measurement_error = measurement_error
        self.data_len = None

        # All models contain a state space representation and a Kalman filter
        self.ssm = PytensorRepresentation(k_endog, k_states, k_posdef)

        # This will be populated with PyMC random matrices after calling _insert_random_variables
        self.subbed_ssm: List[Variable] = None

        if filter_type.lower() not in FILTER_FACTORY.keys():
            raise NotImplementedError(
                "The following are valid filter types: " + ", ".join(list(FILTER_FACTORY.keys()))
            )

        if filter_type == "single" and self.k_endog > 1:
            raise ValueError('Cannot use filter_type = "single" with multiple observed time series')

        self.kalman_filter = FILTER_FACTORY[filter_type.lower()]()
        self.kalman_smoother = KalmanSmoother()
        self.make_symbolic_graph()

        if verbose:
            try:
                self._print_prior_requirements()
            except NotImplementedError:
                pass

    def _print_prior_requirements(self) -> None:
        """
        Prints a short report to the terminal about the priors needed for the model, including their names,
        shapes, named dimensions, and any parameter constraints.
        """
        out = ""
        for param, info in self.param_info.items():
            out += f'\t{param} -- shape: {info["shape"]}, constraints: {info["constraints"]}, dims: {info["dims"]}\n'
        out = out.rstrip()

        _log.info(
            "The following parameters should be assigned priors inside a PyMC "
            f"model block: \n"
            f"{out}"
        )

    def _unpack_statespace_with_placeholders(self) -> Tuple:
        """
        Helper function to quickly obtain all statespace matrices in the standard order. Matrices returned by this
        method will include pytensor placeholders.
        """

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

    def unpack_statespace(self) -> Tuple:
        """
        Helper function to quickly obtain all statespace matrices in the standard order.
        """

        if self.subbed_ssm is None:
            raise ValueError(
                "Cannot unpack the complete statespace system until PyMC model variables have been "
                "inserted. To build the random statespace matrices, call build_statespace_graph() inside"
                "a PyMC model context. "
            )

        return self.subbed_ssm

    @property
    def param_names(self) -> List[str]:
        """
        Names of model parameters

        A list of all parameters expected by the model. Each parameter will be sought inside the active PyMC model
        context when ``build_statespace_graph`` is invoked.
        """
        raise NotImplementedError("The param_names property has not been implemented!")

    @property
    def param_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Information about parameters needed to declare priors

        A dictionary of param_name: dictionary key-value pairs. The return value is used by the
        ``_print_prior_requirements`` method, to print a message telling users how to define the necessary priors for
        the model. Each dictionary should have the following key-value pairs:
            * key: "shape", value: a tuple of integers
            * key: "constraints", value: a string describing the support of the prior (positive,
              positive semi-definite, etc)
            * key: "dims", value: tuple of strings
        """
        raise NotImplementedError("The params_info property has not been implemented!")

    @property
    def state_names(self) -> List[str]:
        """
        A k_states length list of strings, associated with the model's hidden states

        """

        raise NotImplementedError("The state_names property has not been implemented!")

    @property
    def observed_states(self) -> List[str]:
        """
        A k_endog length list of strings, associated with the model's observed states
        """
        raise NotImplementedError("The observed_states property has not been implemented!")

    @property
    def shock_names(self) -> List[str]:
        """
        A k_posdef length list of strings, associated with the model's shock processes

        """
        raise NotImplementedError("The shock_names property has not been implemented!")

    @property
    def default_priors(self) -> Dict[str, Callable]:
        """
        Dictionary of parameter names and callable functions to construct default priors for the model

        Returns a dictionary with param_name: Callable key-value pairs. Used by the ``add_default_priors()`` method
        to automatically add priors to the PyMC model.
        """
        raise NotImplementedError("The default_priors property has not been implemented!")

    @property
    def coords(self) -> Dict[str, Sequence[str]]:
        """
        PyMC model coordinates

        Returns a dictionary of dimension: coordinate key-value pairs, to be provided to ``pm.Model``. Dimensions
        should come from the default names defined in ``statespace.utils.constants`` for them to be detected by
        sampling methods.
        """
        raise NotImplementedError("The coords property has not been implemented!")

    @property
    def param_dims(self) -> Dict[str, Sequence[str]]:
        """
        Dictionary of named dimensions for each model parameter

        Returns a dictionary of param_name: dimension key-value pairs, to be provided to the ``dims`` argument of a
        PyMC random variable. Dimensions should come from the default names defined in ``statespace.utils.constants``
        for them to be detected by sampling methods.

        """
        raise NotImplementedError("The param_dims property has not been implemented!")

    def add_default_priors(self) -> None:
        """
        Add default priors to the active PyMC model context
        """
        raise NotImplementedError("The add_default_priors property has not been implemented!")

    def make_and_register_variable(self, name, shape, dtype=floatX) -> Variable:
        """
        Helper function to create a pytensor symbolic variable and register it in the _name_to_variable dictionary

        Parameters
        ----------
        name : str
            The name of the placeholder variable. Must be the name of a model parameter.
        shape : int or tuple of int
            Shape of the parameter
        dtype : str, default pytensor.config.floatX
            dtype of the parameter

        Notes
        -----
        Symbolic pytensor variables are used in the ``make_symbolic_graph`` method as placeholders for PyMC random
        variables. The change is made in the ``_insert_random_variables`` method via ``pytensor.graph_replace``. To
        make the change, a dictionary mapping pytensor variables to PyMC random variables needs to be constructed.

        The purpose of this method is to:
            1.  Create the placeholder symbolic variables
            2.  Register the placeholder variable in the ``_name_to_variable`` dictionary

        The shape provided here will define the shape of the prior that will need to be provided by the user.

        An error is raised if the provided name has already been registered, or if the name is not present in the
        ``param_names`` property.
        """
        if name not in self.param_names:
            raise ValueError(
                f"{name} is not a model parameter. All placeholder variables should correspond to model "
                f"parameters."
            )

        if name in self._name_to_variable.keys():
            raise ValueError(
                f"{name} is already a registered placeholder variable with shape "
                f"{self._name_to_variable[name].type.shape}"
            )

        placeholder = pt.tensor(name, shape=shape, dtype=dtype)
        self._name_to_variable[name] = placeholder
        return placeholder

    def make_symbolic_graph(self) -> None:
        """
        The purpose of the make_symbolic_graph function is to hide tedious parameter allocations from the user.
        In statespace models, it is extremely rare for an entire matrix to be defined by a single prior distribution.
        Instead, users expect to place priors over single entries of the matrix. The purpose of this function is to
        meet that expectation.

        Every statespace model needs to implement this function.

        Examples
        ----------
        As an example, consider an ARMA(2,2) model, which has five parameters (excluding the initial state distribution):
        2 AR parameters (:math:`\rho_1` and :math:`\rho_2`), 2 MA parameters (:math:`\theta_1` and :math:`theta_2`),
        and a single innovation covariance (:math:`\\sigma`). A common way of writing this statespace is:

        ..math::

            \begin{align}
                T &= \begin{bmatrix} \rho_1 & 1 & 0 \\
                                     \rho_2 & 0 & 1 \\
                                     0      & 0 & 0
                      \\end{bmatrix} \\
                R & = \begin{bmatrix} 1 \\ \theta_1 \\ \theta_2 \\end{bmatrix} \\
                Q &= \begin{bmatrix} \\sigma \\end{bmatrix}
            \\end{align}

        To implement this model, we begin by creating the required matrices, and fill in the "fixed" values -- the ones
        at position (0, 1) and (0, 2) in the T matrix, and at position (0, 0) in the R matrix. These are then saved
        to the class's PytensorRepresentation -- called ``ssm``.

        .. code:: python

            T = np.eye(2, k=1)
            R = np.concatenate([np.ones(1,1), np.zeros((2, 1))], axis=0)

            self.ssm['transition'] = T
            self.ssm['selection'] = R

        Next, placeholders need to be inserted for the random variables rho_1, rho_2, theta_1, theta_2, and sigma.
        This can be done many ways: we could define two vectors, rho and theta, and a scalar for sigma, or five
        scalars. Whatever is chosen, the choice needs to be consistent with the ``param_names`` property.

        Suppose the ``param_names`` are ``[rho, theta, sigma]``, then we make one placeholder for each, and insert it
        into the correct ``ssm`` matrix, at the correct location. To create placeholders, use the
        ``make_and_register_variable`` helper method, which will maintain an internal registry of variables.

        .. code:: python
            rho_parmas = self.make_and_register_variable(name='rho', shape=(2,))
            theta_params = self.make_and_register_variable(name='theta', shape=(2,))
            sigma = self.make_and_register_variable(name='sigma', shape=(1,))

            self.ssm['transition', :, 0] = rho_params
            self.ssm['selection', 1:, 0] = theta_params
            self.ssm['state_cov', 0, 0] = sigma
        """
        raise NotImplementedError("The make_symbolic_statespace method has not been implemented!")

    def _get_matrix_shape_and_dims(self, name: str) -> Tuple[Tuple, Tuple]:
        """
        Get the shape and dimensions of a matrix associated with the specified name.

        This method retrieves the shape and dimensions of a matrix associated with the given name. Importantly,
        it will only find named dimension if they are the "default" dimension names defined in the
        statespace.utils.constant.py file.

        Parameters
        ----------
        name : str
            The name of the matrix whose shape and dimensions need to be retrieved.

        Returns
        -------
        shape: tuple or None
            If no named dimension are found, the shape of the requested matrix, otherwise None.

        dims: tuple or None
            If named dimensions are found, a tuple of strings, otherwise None
        """

        pm_mod = modelcontext(None)
        dims = MATRIX_DIMS.get(name, None)
        dims = dims if all([dim in pm_mod.coords.keys() for dim in dims]) else None

        if name in self.kalman_filter.seq_names:
            shape = (self.data_len,) + self.ssm[SHORT_NAME_TO_LONG[name]].type.shape
            dims = (TIME_DIM,) + dims
        else:
            shape = self.ssm[SHORT_NAME_TO_LONG[name]].type.shape

        shape = shape if dims is None else None

        return shape, dims

    def _get_output_shape_and_dims(self, idata: InferenceData, filter_output: str) -> Tuple:
        """
        Get the shapes and dimensions of the output variables from the provided InferenceData.

        This method extracts the shapes and dimensions of the output variables representing the state estimates
        and covariances from the provided ArviZ InferenceData object. The state estimates are obtained from the
        specified `filter_output` mode, which can be one of "filtered", "predicted", or "smoothed".

        Parameters
        ----------
        idata : arviz.InferenceData
            The ArviZ InferenceData object containing the posterior samples.

        filter_output : str
            The name of the filter output whose shape is being checked. It can be one of "filtered",
            "predicted", or "smoothed".

        Returns
        -------
        mu_shape: tuple(int, int) or None
            Shape of the mean vectors returned by the Kalman filter. Should be (n_data_timesteps, k_states).
            If named dimensions are found, this will be None.

        cov_shape: tuple (int, int, int) or None
            Shape of the hidden state covariance matrices returned by the Kalman filter. Should be
            (n_data_timesteps, k_states, k_states).
            If named dimensions are found, this will be None.

        mu_dims: tuple(str, str) or None
            *Default* named dimensions associated with the mean vectors returned by the Kalman filter, or None if
            the default names are not found.

        cov_dims: tuple (str, str, str) or None
            *Default* named dimensions associated with the covariance matrices returned by the Kalman filter, or None if
            the default names are not found.
        """

        mu_dims = None
        cov_dims = None

        mu_shape = idata[f"{filter_output}_state"].values.shape[2:]
        cov_shape = idata[f"{filter_output}_covariance"].values.shape[2:]

        if all([dim in self._fit_coords for dim in [TIME_DIM, ALL_STATE_DIM, ALL_STATE_AUX_DIM]]):
            time_dim = TIME_DIM
            mu_dims = [time_dim, ALL_STATE_DIM]
            cov_dims = [time_dim, ALL_STATE_DIM, ALL_STATE_AUX_DIM]

            mu_shape = None
            cov_shape = None

        return mu_shape, cov_shape, mu_dims, cov_dims

    def _insert_random_variables(self) -> List[Variable]:
        """
        Replace pytensor symbolic variables with PyMC random variables.

        Examples
        --------
        .. code:: python

            ss_mod = pmss.BayesianSARIMA(order=(2, 0, 2), verbose=False, stationary_initialization=True)
            with pm.Model():
                 x0 = pm.Normal('x0', size=ss_mod.k_states)
                 ar_params = pm.Normal('ar_params', size=ss_mod.p)
                 ma_parama = pm.Normal('ma_params', size=ss_mod.q)
                 sigma_state = pm.Normal('sigma_state')

                 ss_mod._insert_random_variables()
                 matrics = ss_mod.unpack_statespace()

            pm.draw(matrices['transition'], random_seed=RANDOM_SEED)
            >>> array([[-0.90590386,  1.        ,  0.        ],
            >>>        [ 1.25190143,  0.        ,  1.        ],
            >>>        [ 0.        ,  0.        ,  0.        ]])

            pm.draw(matrices['selection'], random_seed=RANDOM_SEED)
            >>> array([[ 1.        ],
            >>>        [-2.46741039],
            >>>        [-0.28947689]])

            pm.draw(matrices['state_cov'], random_seed=RANDOM_SEED)
            >>> array([[-1.69353533]])
        """

        pymc_model = modelcontext(None)
        found_params = []
        with pymc_model:
            for param_name in self.param_names:
                param = getattr(pymc_model, param_name, None)
                if param:
                    found_params.append(param.name)

        missing_params = list(set(self.param_names) - set(found_params))
        if len(missing_params) > 0:
            raise ValueError(
                "The following required model parameters were not found in the PyMC model: "
                + ", ".join(missing_params)
            )

        matrices = list(self._unpack_statespace_with_placeholders())
        replacement_dict = {
            var: pt.atleast_1d(pymc_model[name]) for name, var in self._name_to_variable.items()
        }
        self.subbed_ssm = graph_replace(matrices, replace=replacement_dict, strict=True)

    def add_exogenous(self, exog: pt.TensorVariable) -> None:
        """
        Add an exogenous process to the statespace model

        Parameters
        ----------
        exog: TensorVariable
            An (N, k_endog) tensor representing exogenous processes to be included in the statespace model

        Notes
        -----
        This function can be used to "inject" absolutely any type of dynamics you wish into a statespace model.
        Recall that a statespace model is a system of two matrix equations:

        .. math::
            \begin{align} X_t &= c_t + T_t x_{t-1} + R_t \varepsilon_t & \varepsilon_t &\\sim N(0, Q_t) \\
                          y_t &= d_t + Z_t x_t + \\eta_t & \\eta_t &\\sim N(0, H_t)
            \\end{align}

        Any of the matrices :math:`c, d, T, Z, R, H, Q` can vary across time. When this function is invoked, the
        provided exogenous data is set as the observation intercept, :math:`d_t`. This makes the statespace model
        a model of the residuals :math:`y_t - d_t`. In fact, this is precisely the quantity that is used to compute
        the likelihood of the data during Kalman filtering.
        """
        # User might pass a flat time-varying exog vector, need to make it a column
        if exog.ndim == 1:
            exog = pt.expand_dims(exog, -1)
        elif (exog.ndim == 2) and (exog.type.shape[-1] != 1):
            raise ValueError(
                f"If exogenous data is 2d, it must have a single column, found {exog.type.shape[-1]}"
            )
        elif exog.ndim > 2:
            raise ValueError(f"Exogenous data must be at most 2d, found {exog.ndim} dimensions")

        # Need to specifically ask for the time dim
        d = self.ssm["obs_intercept", :, :]
        self.ssm["obs_intercept"] = d + exog

        self._needs_exog_data = True
        self._exog_names.append(exog.name)

    def build_statespace_graph(
        self,
        data: Union[np.ndarray, pd.DataFrame, pt.TensorVariable],
        register_data: bool = True,
        mode: Optional[str] = None,
        missing_fill_value: Optional[float] = None,
        cov_jitter: Optional[float] = JITTER_DEFAULT,
        return_updates: bool = False,
        include_smoother: bool = True,
    ) -> None:
        """
        Given a parameter vector `theta`, constructs the full computational graph describing the state space model and
        the associated log probability of the data. Hidden states and log probabilities are computed via the Kalman
        Filter. All statespace matrices, as well as Kalman Filter outputs, will be added to the PyMC model on the
        context stack as pm.Deterministic variables.

        Parameters
        ----------
        data : Union[np.ndarray, pd.DataFrame, pt.TensorVariable]
            The observed data used to fit the state space model. It can be a NumPy array, a Pandas DataFrame,
            or a Pytensor tensor variable.

        register_data : bool, optional, default=True
            If True, the observed data will be registered with PyMC as a pm.MutableData variable. In addition,
            a "time" dim will be created an added to the model's coords.

        mode : Optional[str], optional, default=None
            The Pytensor mode used for the computation graph construction. If None, the default mode will be used.
            Other options include "JAX" and "NUMBA".

        missing_fill_value: float, optional, default=-9999
            A value to mask in missing values. NaN values in the data need to be filled with an arbitrary value to
            avoid triggering PyMC's automatic imputation machinery (missing values are instead filled by treating them
            as hidden states during Kalman filtering).

            In general this never needs to be set. But if by a wild coincidence your data includes the value -9999.0,
            you will need to change the missing_fill_value to something else, to avoid incorrectly mark in
            data as missing.

        cov_jitter: float, default 1e-8 or 1e-6 if pytensor.config.floatX is float32
            The Kalman filter is known to be numerically unstable, especially at half precision. This value is added to
            the diagonal of every covariance matrix -- predicted, filtered, and smoothed -- at every step, to ensure
            all matrices are strictly positive semi-definite.

            Obviously, if this can be zero, that's best. In general:
                - Having measurement error makes Kalman Filters more robust. A large source of numerical errors come
                  from the Filtered and Smoothed covariance matrices having a zero in the (0, 0) position, which always
                  occurs when there is no measurement error. You can lower this value in the presence of measurement
                  error.

                - The Univariate Filter is more robust than other filters, and can tolerate a lower jitter value

        return_updates : bool, optional, default=False
            If True, the method will return the update dictionary used by pytensor.Scan to update random variables.
            Only useful for diagnostic purposes.

        include_smoother : bool, optional, default=True
            If True, the Kalman smoother will be included in the computation graph to generate smoothed states
            and covariances. These will be added to the PyMC model on the context stack as pm.Deterministic variables

        Returns
        -------
        None or Dict
            If `return_updates` is True, the method will return a dictionary of updates from the Kalman filter.
            If `return_updates` is False, the method will return None.
        """
        pm_mod = modelcontext(None)

        self._insert_random_variables()
        matrices = self.unpack_statespace()
        obs_coords = pm_mod.coords.get(OBS_STATE_DIM, None)

        self.data_len = data.shape[0]
        data, nan_mask = register_data_with_pymc(
            data,
            n_obs=self.ssm.k_endog,
            obs_coords=obs_coords,
            register_data=register_data,
            missing_fill_value=missing_fill_value,
        )

        registered_matrices = []
        for i, (matrix, name) in enumerate(zip(matrices, MATRIX_NAMES)):
            time_varying_ndim = 2 if name in VECTOR_VALUED else 3
            if not getattr(pm_mod, name, None):
                shape, dims = self._get_matrix_shape_and_dims(name)
                has_dims = dims is not None

                if matrix.ndim == time_varying_ndim and has_dims:
                    dims = (TIME_DIM,) + dims

                x = pm.Deterministic(name, matrix, dims=dims)
                registered_matrices.append(x)
            else:
                registered_matrices.append(matrices[i])

        filter_outputs = self.kalman_filter.build_graph(
            pt.as_tensor_variable(data),
            *registered_matrices,
            mode=mode,
            missing_fill_value=missing_fill_value,
            return_updates=return_updates,
            cov_jitter=cov_jitter,
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
                filtered_states,
                filtered_covariances,
                registered_matrices,
                mode=mode,
                cov_jitter=cov_jitter,
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

    def _build_smoother_graph(
        self,
        filtered_states: pt.TensorVariable,
        filtered_covariances: pt.TensorVariable,
        matrices,
        mode: Optional[str] = None,
        cov_jitter=JITTER_DEFAULT,
    ):
        """
        Build the computation graph for the Kalman smoother.

        This method constructs the computation graph for applying the Kalman smoother to the filtered states
        and covariances obtained from the Kalman filter. The Kalman smoother is used to generate smoothed
        estimates of the latent states and their covariances in a state space model.

        The Kalman smoother provides a more accurate estimate of the latent states by incorporating future
        information in the backward pass, resulting in smoothed state trajectories.

        Parameters
        ----------
        filtered_states : pytensor.tensor.TensorVariable
            The filtered states obtained from the Kalman filter. Returned by the `build_statespace_graph` method.

        filtered_covariances : pytensor.tensor.TensorVariable
            The filtered state covariances obtained from the Kalman filter. Returned by the `build_statespace_graph`
            method.

        mode : Optional[str], default=None
            The mode used by pytensor for the construction of the logp graph. If None, the mode provided to
            `build_statespace_graph` will be used.

        Returns
        -------
        Tuple[pytensor.tensor.TensorVariable, pytensor.tensor.TensorVariable]
            A tuple containing TensorVariables representing the smoothed states and smoothed state covariances
            obtained from the Kalman smoother.
        """

        pymc_model = modelcontext(None)
        with pymc_model:
            *_, T, Z, R, H, Q = matrices

            smooth_states, smooth_covariances = self.kalman_smoother.build_graph(
                T, R, Q, filtered_states, filtered_covariances, mode=mode, cov_jitter=cov_jitter
            )

            return smooth_states, smooth_covariances

    def _build_dummy_graph(self, skip_matrices=None):
        """
        Build a dummy computation graph for the state space model matrices.

        This method creates "dummy" pm.Flat variables representing the matrices used in the state space model.
        The shape and dimensions of each matrix are determined based on the model specifications. These variables
        are used by the `sample_posterior`, `forecast`, and `impulse_response_functions` methods.

        Parameters
        ----------
        skip_matrices : Optional[List[str]], default=None
            A list of matrix names to skip during the dummy graph construction. If provided, the matrices
            specified in this list will not be created in the dummy graph.

        Returns
        -------
        List[pm.Flat]
            A list of pm.Flat variables representing the dummy matrices used in the state space model. The return
            order is always x0, P0, c, d, T, Z, R, H, Q, skipping any matrices indicated in `skip_matrices`.
        """
        modelcontext(None)
        if skip_matrices is None:
            skip_matrices = []

        matrices = []
        for name in MATRIX_NAMES:
            if name in skip_matrices:
                continue

            shape, dims = self._get_matrix_shape_and_dims(name)
            x = pm.Flat(f"{name}", shape=shape, dims=dims)
            matrices.append(x)

        return matrices

    def _sample_conditional(
        self, idata: InferenceData, group: str, random_seed: Optional[RandomState] = None, **kwargs
    ):
        """
        Common functionality shared between `sample_conditional_prior` and `sample_conditional_posterior`. See those
        methods for details.


        Parameters
        ----------
        idata : InferenceData
            An Arviz InferenceData object containing the posterior distribution over model parameters.

        group : str
            InferenceData group from which to draw samples. Should be one of "prior" or "posterior".

        random_seed : int, RandomState or Generator, optional
            Seed for the random number generator.

        kwargs:
            Additional keyword arguments are passed to pymc.sample_posterior_predictive

        Returns
        -------
        InferenceData
            An Arviz InferenceData object containing sampled trajectories from the requested conditional distribution,
            with data variables "filtered_{group}", "predicted_{group}", and "smoothed_{group}".
        """

        _verify_group(group)
        group_idata = getattr(idata, group)
        filters_to_use = ["filtered", "predicted", "smoothed"]

        with pm.Model(coords=self._fit_coords):
            d, Z, H = self._build_dummy_graph(skip_matrices=["x0", "P0", "c", "T", "R", "Q"])

            for filter_output in filters_to_use:
                mu_shape, cov_shape, mu_dims, cov_dims = self._get_output_shape_and_dims(
                    group_idata, filter_output
                )

                mus = pm.Flat(f"{filter_output}_state", shape=mu_shape, dims=mu_dims)
                covs = pm.Flat(f"{filter_output}_covariance", shape=cov_shape, dims=cov_dims)
                latent_states = SequenceMvNormal(
                    f"{filter_output}_{group}",
                    mus=mus,
                    covs=covs,
                    logp=pt.zeros(mus.shape[0]),
                    dims=mu_dims,
                )

                obs_mu, obs_covs = self._observed_mu_cov_from_latent(d, Z, H, latent_states)

                # If there are dims on the latent dim, we can use the defaults to put them on the observed too
                obs_dims = (
                    (TIME_DIM, OBS_STATE_DIM)
                    if not all([x is None for x in mu_dims])
                    else (None, None)
                )

                SequenceMvNormal(
                    f"{filter_output}_{group}_observed",
                    mus=obs_mu,
                    covs=obs_covs,
                    logp=pt.zeros(obs_mu.shape[0]),
                    dims=obs_dims,
                )

            idata_conditional = pm.sample_posterior_predictive(
                group_idata,
                var_names=[
                    f"{name}_{group}{suffix}"
                    for name in filters_to_use
                    for suffix in ["", "_observed"]
                ],
                compile_kwargs={"mode": get_mode(self._fit_mode)},
                random_seed=random_seed,
                **kwargs,
            )

        return idata_conditional.posterior_predictive

    def _sample_unconditional(
        self,
        idata: InferenceData,
        group: str,
        steps: Optional[int] = None,
        use_data_time_dim: bool = False,
        random_seed: Optional[RandomState] = None,
        **kwargs,
    ):
        """
        Draw unconditional sample trajectories according to state space dynamics, using random samples from the
        a provided trace. The state space update equations are:

            X[t+1] = T @ X[t] + R @ eta[t], eta ~ N(0, Q)
            Y[t] = Z @ X[t] + nu[t], nu ~ N(0, H)
            x[0] ~ N(a0, P0)

        Parameters
        ----------
        idata : InferenceData
            An Arviz InferenceData object with a posterior group containing samples from the
            posterior distribution over model parameters.

        steps : Optional[int], default=None
            The number of time steps to sample for the unconditional trajectories. If not provided (None),
            the function will sample trajectories for the entire available time dimension in the posterior.
            Otherwise, it will generate trajectories for the specified number of steps.

        use_data_time_dim : bool, default=False
            If True, the function uses the time dimension present in the provided `idata` object to sample
            unconditional trajectories. If False, a custom time dimension is created based on the number of steps
            specified, or if steps is None, it uses the entire available time dimension in the posterior.

        random_seed : int, RandomState or Generator, optional
            Seed for the random number generator.

        kwargs:
            Additional keyword arguments are passed to pymc.sample_posterior_predictive

        Returns
        -------
        InferenceData
            An Arviz InfereceData with two groups, posterior_latent and posterior_observed

            - posterior_latent represents the latent state trajectories `X[t]`, which follows the dynamics:
              `x[t+1] = T @ x[t] + R @ eta[t]`, where `eta ~ N(0, Q)`.

            - posterior_observed represents the observed state trajectories `Y[t]`, which is obtained from
              the latent state trajectories: `y[t] = Z @ x[t] + nu[t]`, where `nu ~ N(0, H)`.
        """
        _verify_group(group)
        group_idata = getattr(idata, group)
        dims = None
        temp_coords = self._fit_coords.copy()

        if not use_data_time_dim and steps is not None:
            temp_coords.update({TIME_DIM: np.arange(1 + steps, dtype="int")})
            steps = len(temp_coords[TIME_DIM]) - 1
        elif steps is not None:
            n_dimsteps = len(temp_coords[TIME_DIM])
            if n_dimsteps != steps:
                raise ValueError(
                    f"Length of time dimension does not match specified number of steps, expected"
                    f" {n_dimsteps} steps, or steps=None."
                )
        else:
            steps = len(temp_coords[TIME_DIM]) - 1

        if all([dim in self._fit_coords for dim in [TIME_DIM, ALL_STATE_DIM, OBS_STATE_DIM]]):
            dims = [TIME_DIM, ALL_STATE_DIM, OBS_STATE_DIM]

        with pm.Model(coords=temp_coords if dims is not None else None):
            matrices = [x0, P0, c, d, T, Z, R, H, Q] = self._build_dummy_graph()

            if not self.measurement_error:
                H_jittered = pm.Deterministic(
                    "H_jittered", pt.specify_shape(stabilize(H), (self.k_endog, self.k_endog))
                )
                matrices = [x0, P0, c, d, T, Z, R, H_jittered, Q]

            _ = LinearGaussianStateSpace(
                group,
                *matrices,
                steps=steps,
                dims=dims,
                mode=self._fit_mode,
                sequence_names=self.kalman_filter.seq_names,
            )

            idata_unconditional = pm.sample_posterior_predictive(
                group_idata,
                var_names=[f"{group}_latent", f"{group}_observed"],
                compile_kwargs={"mode": self._fit_mode},
                random_seed=random_seed,
                **kwargs,
            )

        return idata_unconditional.posterior_predictive

    def sample_conditional_prior(
        self, idata: InferenceData, random_seed: Optional[RandomState] = None, **kwargs
    ) -> InferenceData:
        """
        Sample from the conditional prior; that is, given parameter draws from the prior distribution,
        compute Kalman filtered trajectories. Trajectories are drawn from a single multivariate normal with mean and
        covariance computed via either the Kalman filter, smoother, or predictions.

        Parameters
        ----------
        idata : InferenceData
            Arviz InferenceData with prior samples for state space matrices x0, P0, c, d, T, Z, R, H, Q.
            Obtained from `pm.sample_prior_predictive` after calling PyMCStateSpace.build_statespace_graph().

        random_seed : int, RandomState or Generator, optional
            Seed for the random number generator.

        kwargs:
            Additional keyword arguments are passed to pymc.sample_posterior_predictive

        Returns
        -------
        InferenceData
            An Arviz InferenceData object containing sampled trajectories from the conditional prior.
            The trajectories are stored in the posterior_predictive group with names "filtered_prior",
             "predicted_prior", and "smoothed_prior".
        """

        return self._sample_conditional(idata, "prior", random_seed, **kwargs)

    def sample_conditional_posterior(
        self, idata: InferenceData, random_seed: Optional[RandomState] = None, **kwargs
    ):
        """
        Sample from the conditional posterior; that is, given parameter draws from the posterior distribution,
        compute Kalman filtered trajectories. Trajectories are drawn from a single multivariate normal with mean and
        covariance computed via either the Kalman filter, smoother, or predictions.

        Parameters
        ----------
        idata : InferenceData
            An Arviz InferenceData object containing the posterior distribution over model parameters.

        random_seed : int, RandomState or Generator, optional
            Seed for the random number generator.

        kwargs:
            Additional keyword arguments are passed to pymc.sample_posterior_predictive

        Returns
        -------
        InferenceData
            An Arviz InferenceData object containing sampled trajectories from the conditional posterior.
            The trajectories are stored in the posterior_predictive group with names "filtered_posterior",
             "predicted_posterior", and "smoothed_posterior".
        """

        return self._sample_conditional(idata, "posterior", random_seed, **kwargs)

    def sample_unconditional_prior(
        self,
        idata: InferenceData,
        steps: Optional[int] = None,
        use_data_time_dim: bool = False,
        random_seed: Optional[RandomState] = None,
        **kwargs,
    ) -> InferenceData:
        """
        Draw unconditional sample trajectories according to state space dynamics, using random samples from the prior
        distribution over model parameters. The state space update equations are:

            X[t+1] = T @ X[t] + R @ eta[t], eta ~ N(0, Q)
            Y[t] = Z @ X[t] + nu[t], nu ~ N(0, H)

        Parameters
        ----------
        idata: InferenceData
            Arviz InferenceData with prior samples for state space matrices x0, P0, c, d, T, Z, R, H, Q.
            Obtained from `pm.sample_prior_predictive` after calling PyMCStateSpace.build_statespace_graph().

        steps : Optional[int], default=None
            The number of time steps to sample for the unconditional trajectories. If not provided (None),
            the function will sample trajectories for the entire available time dimension in the posterior.
            Otherwise, it will generate trajectories for the specified number of steps.

        use_data_time_dim : bool, default=False
            If True, the function uses the time dimension present in the provided `idata` object to sample
            unconditional trajectories. If False, a custom time dimension is created based on the number of steps
            specified, or if steps is None, it uses the entire available time dimension in the posterior.

        random_seed : int, RandomState or Generator, optional
            Seed for the random number generator.

        kwargs:
            Additional keyword arguments are passed to pymc.sample_posterior_predictive

        Returns
        -------
        InferenceData
            An Arviz InfereceData with two data variables, prior_latent and prior_observed

            - prior_latent represents the latent state trajectories `X[t]`, which follows the dynamics:
              `x[t+1] = T @ x[t] + R @ eta[t]`, where `eta ~ N(0, Q)`.

            - prior_observed represents the observed state trajectories `Y[t]`, which is obtained from
              the observation equation: `y[t] = Z @ x[t] + nu[t]`, where `nu ~ N(0, H)`.
        """

        return self._sample_unconditional(
            idata, "prior", steps, use_data_time_dim, random_seed, **kwargs
        )

    def sample_unconditional_posterior(
        self,
        idata,
        steps=None,
        use_data_time_dim=False,
        random_seed: Optional[RandomState] = None,
        **kwargs,
    ) -> InferenceData:
        """
        Draw unconditional sample trajectories according to state space dynamics, using random samples from the
        posterior distribution over model parameters. The state space update equations are:

            X[t+1] = T @ X[t] + R @ eta[t], eta ~ N(0, Q)
            Y[t] = Z @ X[t] + nu[t], nu ~ N(0, H)
            x[0] ~ N(a0, P0)

        Parameters
        ----------
        idata : InferenceData
            An Arviz InferenceData object with a posterior group containing samples from the
            posterior distribution over model parameters.

        steps : Optional[int], default=None
            The number of time steps to sample for the unconditional trajectories. If not provided (None),
            the function will sample trajectories for the entire available time dimension in the posterior.
            Otherwise, it will generate trajectories for the specified number of steps.

        use_data_time_dim : bool, default=False
            If True, the function uses the time dimension present in the provided `idata` object to sample
            unconditional trajectories. If False, a custom time dimension is created based on the number of steps
            specified, or if steps is None, it uses the entire available time dimension in the posterior.

        random_seed : int, RandomState or Generator, optional
            Seed for the random number generator.

        Returns
        -------
        InferenceData
            An Arviz InfereceData with two groups, posterior_latent and posterior_observed

            - posterior_latent represents the latent state trajectories `X[t]`, which follows the dynamics:
              `x[t+1] = T @ x[t] + R @ eta[t]`, where `eta ~ N(0, Q)`.

            - posterior_observed represents the observed state trajectories `Y[t]`, which is obtained from
              the latent state trajectories: `y[t] = Z @ x[t] + nu[t]`, where `nu ~ N(0, H)`.
        """

        return self._sample_unconditional(
            idata, "posterior", steps, use_data_time_dim, random_seed, **kwargs
        )

    def forecast(
        self,
        idata: InferenceData,
        start: Union[int, pd.Timestamp],
        periods: int = None,
        end: Union[int, pd.Timestamp] = None,
        filter_output="smoothed",
        random_seed: Optional[RandomState] = None,
        **kwargs,
    ) -> InferenceData:
        """
        Generate forecasts of state space model trajectories into the future.

        This function combines posterior parameter samples in the provided idata with model dynamics to generate
        forecasts for out-of-sample data. The trajectory is initialized using the filter output specified in
        the filter_output argument.

        Parameters
        ----------
        idata : InferenceData
            An Arviz InferenceData object containing the posterior distribution over model parameters.

        start : Union[int, pd.Timestamp]
            The starting date index or time step from which to generate the forecasts. If the data provided to
            `PyMCStateSpace.build_statespace_graph` had a datetime index, `start` should be a datetime.
            If using integer time series, `start` should be an integer indicating the starting time step. In either
            case, `start` should be in the data index used to build the statespace graph.

        periods : Optional[int], default=None
            The number of time steps to forecast into the future. If `periods` is specified, the `end`
            parameter will be ignored. If `None`, then the `end` parameter must be provided.

        end : Union[int, pd.Timestamp], default=None
            The ending date index or time step up to which to generate the forecasts. If the data provided to
            `PyMCStateSpace.build_statespace_graph` had a datetime index, `start` should be a datetime.
            If using integer time series, `end` should be an integer indicating the ending time step.
            If `end` is provided, the `periods` parameter will be ignored.

        filter_output : str, default="smoothed"
            The type of Kalman Filter output used to initialize the forecasts. The 0th timestep of the forecast will
            be sampled from x[0] ~ N(filter_output_mean[start], filter_output_covariance[start]). Default is "smoothed",
            which uses past and future data to make the best possible hidden state estimate.

        random_seed : int, RandomState or Generator, optional
            Seed for the random number generator.

        kwargs:
            Additional keyword arguments are passed to pymc.sample_posterior_predictive

        Returns
        -------
        InferenceData
            An Arviz InferenceData object containing forecast samples for the latent and observed state
            trajectories of the state space model, named  "forecast_latent" and "forecast_observed".

                - forecast_latent represents the latent state trajectories `X[t]`, which follows the dynamics:
                  `x[t+1] = T @ x[t] + R @ eta[t]`, where `eta ~ N(0, Q)`.

                - forecast_observed represents the observed state trajectories `Y[t]`, which is obtained from
                  the latent state trajectories: `y[t] = Z @ x[t] + nu[t]`, where `nu ~ N(0, H)`.

        """
        _validate_filter_arg(filter_output)
        if periods is None and end is None:
            raise ValueError("Must specify one of either periods or end")
        if periods is not None and end is not None:
            raise ValueError("Must specify exactly one of either periods or end")
        if self._needs_exog_data:
            raise ValueError(
                "Scenario-based forcasting with exogenous variables not currently supported"
            )

        dims = None
        temp_coords = self._fit_coords.copy()

        filter_time_dim = TIME_DIM

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
            if not self.measurement_error:
                H = pm.Deterministic(
                    "H_jittered", pt.specify_shape(stabilize(H), (self.k_endog, self.k_endog))
                )

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
                sequence_names=self.kalman_filter.seq_names,
            )

            idata_forecast = pm.sample_posterior_predictive(
                idata,
                var_names=["forecast_latent", "forecast_observed"],
                compile_kwargs={"mode": self._fit_mode},
                random_seed=random_seed,
                **kwargs,
            )

            return idata_forecast.posterior_predictive

    def impulse_response_function(
        self,
        idata,
        steps: int = None,
        shock_size: Optional[Union[float, np.ndarray]] = None,
        shock_cov: Optional[np.ndarray] = None,
        shock_trajectory: Optional[np.ndarray] = None,
        orthogonalize_shocks: bool = False,
        random_seed: Optional[RandomState] = None,
        **kwargs,
    ):
        """
        Generate impulse response functions (IRF) from state space model dynamics.

        An impulse response function represents the dynamic response of the state space model
        to an instantaneous shock applied to the system. This function calculates the IRF
        based on either provided shock specifications or the posterior state covariance matrix.

        Parameters
        ----------
        idata : az.InferenceData
            An Arviz InferenceData object containing the posterior distribution over model parameters.

        steps : Optional[int], default=None
            The number of time steps to calculate the impulse response. If not provided, it defaults to 40, unless
            a shock trajectory is provided.

        shock_size : Optional[Union[float, np.ndarray]], default=None
            The size of the shock applied to the system. If specified, it will create a covariance
            matrix for the shock with diagonal elements equal to `shock_size`. If float, the diagonal will be filled
            with `shock_size`. If an array, `shock_size` must match the number of shocks in the state space model.

        shock_cov : Optional[np.ndarray], default=None
            A user-specified covariance matrix for the shocks. It should be a 2D numpy array with
            dimensions (n_shocks, n_shocks), where n_shocks is the number of shocks in the state space model.

        shock_trajectory : Optional[np.ndarray], default=None
            A pre-defined trajectory of shocks applied to the system. It should be a 2D numpy array
            with dimensions (n, n_shocks), where n is the number of time steps and k_posdef is the
            number of shocks in the state space model.

        orthogonalize_shocks : bool, default=False
            If True, orthogonalize the shocks using Cholesky decomposition when generating the impulse
            response. This option is only relevant when `shock_trajectory` is not provided.

        random_seed : int, RandomState or Generator, optional
            Seed for the random number generator.

        kwargs:
            Additional keyword arguments are passed to pymc.sample_posterior_predictive

        Returns
        -------
        pm.InferenceData
            An Arviz InferenceData object containing impulse response function in a variable named "irf".
        """
        options = [shock_size, shock_cov, shock_trajectory]
        n_nones = sum(x is not None for x in options)
        sample_posterior_Q = n_nones == 0

        if n_nones > 1:
            raise ValueError("Specify exactly 0 or 1 of shock_size, shock_cov, or shock_trajectory")
        if shock_trajectory is not None:
            n, k = shock_trajectory.shape
            steps = n
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
            x0 = pm.Deterministic("x0_new", pt.zeros(self.k_states), dims=[ALL_STATE_DIM])
            Q = None

            if sample_posterior_Q:
                matrices = self._build_dummy_graph(skip_matrices=["x0"])
                # x0 was not loaded into the model
                P0, c, _, T, _, R, _, Q = matrices

            else:
                matrices = self._build_dummy_graph(skip_matrices=["x0", "Q"])

                # Neither x0 nor Q was loaded into the model
                P0, c, _, T, _, R, _ = matrices

                if shock_cov is not None:
                    Q = pm.Deterministic(
                        "Q_new", pt.as_tensor_variable(shock_cov), dims=[SHOCK_DIM, SHOCK_AUX_DIM]
                    )

            if shock_trajectory is None:
                shock_trajectory = pt.zeros((steps, self.k_posdef))
                if Q is not None:
                    if orthogonalize_shocks:
                        Q = pt.linalg.cholesky(Q)
                    init_shock = pm.MvNormal(
                        "initial_shock", mu=0, cov=stabilize(Q, JITTER_DEFAULT), dims=[SHOCK_DIM]
                    )
                else:
                    init_shock = pm.Deterministic(
                        "initial_shock", pt.as_tensor_variable(shock_size), dims=[SHOCK_DIM]
                    )
                shock_trajectory = pt.set_subtensor(shock_trajectory[0], init_shock)

            else:
                shock_trajectory = pt.as_tensor_variable(shock_trajectory)

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

            compile_kwargs = kwargs.get("compile_kwargs", None)
            if compile_kwargs is None:
                compile_kwargs = {"mode": self._fit_mode}
            else:
                mode = compile_kwargs.get("mode")
                if mode is not None and mode != self._fit_mode:
                    raise ValueError(
                        f"User provided compile mode ({mode}) does not match the compile mode used to "
                        f"construct the model ({self._fit_mode})."
                    )

                compile_kwargs.update({"mode": self._fit_mode})

            irf_idata = pm.sample_posterior_predictive(
                idata,
                var_names=["irf"],
                compile_kwargs=compile_kwargs,
                random_seed=random_seed,
                **kwargs,
            )

            return irf_idata.posterior_predictive

    def _sort_obs_inputs_by_time_varying(self, d, Z):
        seqs = []
        non_seqs = []

        for matrix, name in zip([d, Z], ["d", "Z"]):
            if name in self.kalman_filter.seq_names:
                seqs.append(matrix)
            else:
                non_seqs.append(matrix)

        return seqs, non_seqs

    def _sort_obs_scan_args(self, args):
        args = list(args)

        # If a matrix is time-varying, pytensor will put a [t] on the name
        arg_names = [x.name.replace("[t]", "") for x in args]
        ordered_args = []

        for name in ["d", "Z"]:
            idx = arg_names.index(name)
            ordered_args.append(args[idx])

        return ordered_args

    def _observed_mu_cov_from_latent(self, d, Z, H, latent_states):
        """
        Given a PyMC Random Variable of latent states, construct an RV of for the observed states

        This function constructs an observed mean from the latent mean by applying the observation equation:

        .. math::
            \begin{align} y_t &= Z_t x_t + d_t + \\eta_t & \\eta &\\sim N(0, H_t) \\end{align}

        It also has some logic to handle time-varying matrices Z, d, or H.

        Parameters
        ----------
        latent_states: RandomVariable
            SequenceMvNormal of latent states

        Returns
        -------
        observed_states: RandomVariable
            SequenceMvNormal of observed states
        """
        seqs, non_seqs = self._sort_obs_inputs_by_time_varying(d, Z)
        seqs = [latent_states] + seqs

        def obs_step(*args):
            x, *dZ = args
            d, Z = self._sort_obs_scan_args(dZ)

            return d + Z @ x

        obs_mu, _ = pytensor.scan(
            obs_step, sequences=seqs, non_sequences=non_seqs, strict=False, mode=self._fit_mode
        )

        obs_covs = (
            H
            if "H" in self.kalman_filter.seq_names
            else pt.expand_dims(
                pt.specify_shape(stabilize(H, JITTER_DEFAULT), H.type.shape), (0,)
            ).repeat(obs_mu.shape[0], axis=0)
        )

        return obs_mu, obs_covs
