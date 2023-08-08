from typing import List, Optional, Tuple, Type, Union

import numpy as np
import pandas.core.tools.datetimes
import pytensor
import pytensor.tensor as pt
from pandas import DataFrame

floatX = pytensor.config.floatX
KeyLike = Union[Tuple[Union[str, int]], str]

NEVER_TIME_VARYING = ["initial_state", "initial_state_cov", "a0", "P0"]
VECTOR_VALUED = ["initial_state", "state_intercept", "obs_intercept", "a0", "c", "d"]


def _preprocess_data(data: Union[DataFrame, np.ndarray], expected_dims=3):
    if isinstance(data, pandas.DataFrame):
        data = data.values
    elif not isinstance(data, np.ndarray):
        raise ValueError("Expected pandas Dataframe or numpy array as data")

    if data.ndim < expected_dims:
        n_dims = data.ndim
        n_to_add = expected_dims - n_dims
        expand_idx = tuple(n_dims + np.arange(n_to_add, dtype="int"))
        data = np.expand_dims(data, axis=expand_idx)

    return data


class PytensorRepresentation:
    r"""
    Core class to hold all objects required by linear gaussian statespace models

    Notation for the linear statespace model is taken from [1], while the specific implementation is adapted from
    the statsmodels implementation: https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/statespace/representation.py
    described in [2].

    Parameters
    ----------
    k_endog: int
        Number of observed states (called "endogeous states" in statsmodels)
    k_states: int
        Number of hidden states
    k_posdef: int
        Number of states that have exogenous shocks; also the rank of the selection matrix R.
    design: ArrayLike, optional
        Design matrix, denoted 'Z' in [1].
    obs_intercept: ArrayLike, optional
        Constant vector in the observation equation, denoted 'd' in [1]. Currently
        not used.
    obs_cov: ArrayLike, optional
        Covariance matrix for multivariate-normal errors in the observation equation. Denoted 'H' in
        [1].
    transition: ArrayLike, optional
        Transition equation that updates the hidden state between time-steps. Denoted 'T' in [1].
    state_intercept: ArrayLike, optional
        Constant vector for the observation equation, denoted 'c' in [1]. Currently not used.
    selection: ArrayLike, optional
        Selection matrix that matches shocks to hidden states, denoted 'R' in [1]. This is the identity
        matrix when k_posdef = k_states.
    state_cov: ArrayLike, optional
        Covariance matrix for state equations, denoted 'Q' in [1]. Null matrix when there is no observation
        noise.
    initial_state: ArrayLike, optional
        Experimental setting to allow for Bayesian estimation of the initial state, denoted `alpha_0` in [1]. Default
        It should potentially be removed in favor of the closed-form diffuse initialization.
    initial_state_cov: ArrayLike, optional
        Experimental setting to allow for Bayesian estimation of the initial state, denoted `P_0` in [1]. Default
        It should potentially be removed in favor of the closed-form diffuse initialization.

    Notes
    ----------
    A linear statespace system is defined by two equations:

    .. math::
        \begin{align}
            x_t &= A_t x_{t-1} + c_t + R_t \varepsilon_t \tag{1} \\
            y_t &= Z_t x_t + d_t + \eta_t \tag{2} \\
        \end{align}

    Where :math:`\{x_t\}_{t=0}^T` is a trajectory of hidden states, and :math:`\{y_t\}_{t=0}^T` is a trajectory of
    observable states. Equation 1 is known as the "state transition equation", while describes how the system evolves
    over time. Equation 2 is the "observation equation", and maps the latent state processes to observed data.
    The system is Gaussian when the innovations, :math:`\varepsilon_t`, and the measurement errors, :math:`\eta_t`,
    are normally distributed. The definition is completed by specification of these distributions, as
    well as an initial state distribution:

    .. math::
        \begin{align}
            \varepsilon_t &\sim N(0, Q_t) \tag{3} \\
            \eta_t &\sim N(0, H_t) \tag{4} \\
            x_0 &\sim N(\bar{x}_0, P_0) \tag{5}
        \end{align}

    The 9 matrices that form equations 1 to 5 are summarized in the table below. We call :math:`N` the number of
    observations, :math:`m` the number of hidden states, :math:`p` the number of observed states, and :math:`r` the
    number of innovations.

    +-----------------------------------+-------------------+-----------------------+
    | Name                              | Symbol            | Shape                 |
    +===================================+===================+=======================+
    | Initial hidden state mean         | :math:`x_0`       | :math:`m \times 1`    |
    +-----------------------------------+-------------------+-----------------------+
    | Initial hidden state covariance   | :math:`P_0`       | :math:`m \times m`    |
    +-----------------------------------+-------------------+-----------------------+
    | Hidden state vector intercept     | :math:`c_t`       | :math:`m \times 1`    |
    +-----------------------------------+-------------------+-----------------------+
    | Observed state vector intercept   | :math:`d_t`       | :math:`p \times 1`    |
    +-----------------------------------+-------------------+-----------------------+
    | Transition matrix                 | :math:`T_t`       | :math:`m \times m`    |
    +-----------------------------------+-------------------+-----------------------+
    | Design matrix                     | :math:`Z_t`       | :math:`p \times m`    |
    +-----------------------------------+-------------------+-----------------------+
    | Selection matrix                  | :math:`R_t`       | :math:`m \times r`    |
    +-----------------------------------+-------------------+-----------------------+
    | Observation noise covariance      | :math:`H_t`       | :math:`p \times p`    |
    +-----------------------------------+-------------------+-----------------------+
    | Hidden state innovation covariance| :math:`Q_t`       | :math:`r \times r`    |
    +-----------------------------------+-------------------+-----------------------+

    The shapes listed above are the core shapes, but in the general case all of these matrices (except for :math:`x_0`
    and :math:`P_0`) can be time varying. In this case, a time dimension of shape :math:`n`, equal to the number of
    observations, can be added.

    The purpose of this class is to store these matrices, as well as to allow users to easily index into them. Matrices
    are stored as pytensor ``TensorVariables`` of known shape. Shapes are always accessible via the ``.type.shape``
    method, which should never return ``None``. Matrices can be accessed via normal numpy array slicing after first
    indexing by the name of the desired array. The time dimension is stored on the far right, and is automatically
    sliced away unless specifically requested by the user. See the examples for details.

    Examples
    ----------
    .. code:: python

        from pymc_experimental.statespace.core.representation import PytensorRepresentation
        ssm = PytensorRepresentation(k_endog=1, k_states=3, k_posdef=1)

        # Access matrices by their names
        print(ssm['transition'].type.shape)
        >>> (3, 3)

        # Slice a matrices
        print(ssm['observation_cov', 0, 0].eval())
        >>> 0.0

        # Set elements in a slice of a matrix
        ssm['design', 0, 0] = 1
        print(ssm['design'].eval())
        >>> np.array([[1, 0, 0]])

        # Setting an entire matrix is also permitted. If you set a time dimension, it must be the last dimension, and the
        # core dimensions must agree with those set when the ssm object was instantiated.
        ssm['obs_intercept'] = np.arange(10).reshape(1, 10) # 10 timesteps
        print(ssm['obs_intercept'].eval())
        >>> np.array([[1., 2., 3., 4., 5., 6., 7., 8., 9.]])

    References
    ----------
    .. [1] Durbin, James, and Siem Jan Koopman. 2012.
        Time Series Analysis by State Space Methods: Second Edition.
        Oxford University Press.
    .. [2] Fulton, Chad. "Estimating time series models by state space methods in Python: Statsmodels." (2015).
           http://www.chadfulton.com/files/fulton_statsmodels_2017_v1.pdf
    """

    def __init__(
        self,
        k_endog: int,
        k_states: int,
        k_posdef: int,
        design: Optional[np.ndarray] = None,
        obs_intercept: Optional[np.ndarray] = None,
        obs_cov=None,
        transition=None,
        state_intercept=None,
        selection=None,
        state_cov=None,
        initial_state=None,
        initial_state_cov=None,
    ) -> None:
        self.k_states = k_states
        self.k_endog = k_endog
        self.k_posdef = k_posdef if k_posdef is not None else k_states

        # The last dimension is for time varying matrices; it could be n_obs. Not thinking about that now.
        self.shapes = {
            "design": (self.k_endog, self.k_states, 1),
            "obs_intercept": (self.k_endog, 1),
            "obs_cov": (self.k_endog, self.k_endog, 1),
            "transition": (self.k_states, self.k_states, 1),
            "state_intercept": (self.k_states, 1),
            "selection": (self.k_states, self.k_posdef, 1),
            "state_cov": (self.k_posdef, self.k_posdef, 1),
            "initial_state": (self.k_states,),
            "initial_state_cov": (self.k_states, self.k_states),
        }

        # Initialize the representation matrices
        scope = locals()
        for name, shape in self.shapes.items():
            if scope[name] is not None:
                matrix = self._numpy_to_pytensor(name, scope[name])
                setattr(self, name, matrix)

            else:
                matrix = pt.as_tensor_variable(
                    np.zeros(shape, dtype=floatX), name=name, ndim=len(shape)
                )
                setattr(self, name, matrix)

    def _validate_key(self, key: KeyLike) -> None:
        if key not in self.shapes:
            raise IndexError(f"{key} is an invalid state space matrix name")

    def _update_shape(self, key: KeyLike, value: Union[np.ndarray, pt.TensorType]) -> None:
        if isinstance(value, (pt.TensorConstant, pt.TensorVariable)):
            shape = value.type.shape
        else:
            shape = value.shape

        old_shape = self.shapes[key]
        check_slice = slice(None, 2) if key not in VECTOR_VALUED else slice(None, 1)

        if not all([a == b for a, b in zip(shape[check_slice], old_shape[check_slice])]):
            raise ValueError(
                f"The first two dimensions of {key} must be {old_shape[check_slice]}, found {shape[check_slice]}"
            )

        # Add time dimension dummy if none present
        if len(shape) == 2 and key not in NEVER_TIME_VARYING:
            self.shapes[key] = shape + (1,)

        self.shapes[key] = shape

    def _add_time_dim_to_slice(
        self, name: str, slice_: Union[List[int], Tuple[int]], n_dim: int
    ) -> Tuple[int]:
        # Case 1: There is never a time dim. No changes needed.
        if name in NEVER_TIME_VARYING:
            return slice_

        # Case 2: The matrix has a time dim, and it was requested. No changes needed.
        if len(slice_) == n_dim:
            return slice_

        # Case 3: There's no time dim on the matrix, and none requested. Slice away the dummy dim.
        if len(slice_) < n_dim:
            empty_slice = (slice(None, None, None),)
            n_omitted = n_dim - len(slice_) - 1
            return tuple(slice_) + empty_slice * n_omitted + (0,)

    @staticmethod
    def _validate_key_and_get_type(key: KeyLike) -> Type[str]:
        if isinstance(key, tuple) and not isinstance(key[0], str):
            raise IndexError("First index must the name of a valid state space matrix.")

        return type(key)

    def _validate_matrix_shape(self, name: str, X: np.ndarray) -> None:
        *expected_shape, time_dim = self.shapes[name]
        expected_shape = tuple(expected_shape)

        is_vector = name in VECTOR_VALUED
        not_time_varying = name in NEVER_TIME_VARYING

        if not_time_varying:
            if is_vector:
                if X.ndim != 1:
                    raise ValueError(
                        f"Array provided for {name} has {X.ndim} dimensions, but it must have exactly 1."
                    )

            else:
                if X.ndim != 2:
                    raise ValueError(
                        f"Array provided for {name} has {X.ndim} dimensions, but it must have exactly 2."
                    )

        else:
            if is_vector:
                if X.ndim not in [1, 2]:
                    raise ValueError(
                        f"Array provided for {name} has {X.ndim} dimensions, "
                        f"expecting 1 (static) or 2 (time-varying)"
                    )
                if X.ndim == 2 and X.shape[:-1] != expected_shape:
                    raise ValueError(
                        f"First dimension of array provided for {name} has shape {X.shape[0]}, "
                        f"expected {expected_shape}"
                    )

            else:
                if X.ndim not in [2, 3]:
                    raise ValueError(
                        f"Array provided for {name} has {X.ndim} dimensions, "
                        f"expecting 2 (static) or 3 (time-varying)"
                    )

                if X.ndim == 3 and X.shape[:-1] != expected_shape:
                    raise ValueError(
                        f"First two dimensions of array provided for {name} has shape {X.shape[:-1]}, "
                        f"expected {expected_shape}"
                    )

            # TODO: Think of another way to validate shapes of time-varying matrices if we don't know the data
            #   when the PytensorRepresentation is recreated
            # if X.shape[-1] != self.data.shape[0]:
            #     raise ValueError(
            #         f"Last dimension (time dimension) of array provided for {name} has shape "
            #         f"{X.shape[-1]}, expected {self.data.shape[0]} (equal to the first dimension of the "
            #         f"provided data)"
            #     )

    def _numpy_to_pytensor(self, name: str, X: np.ndarray) -> pt.TensorVariable:
        X = X.copy()
        self._validate_matrix_shape(name, X)

        # Add a time dimension if one isn't provided
        if name not in NEVER_TIME_VARYING:
            if X.ndim == 1 and name in VECTOR_VALUED:
                X = X[..., None]
            elif X.ndim == 2:
                X = X[..., None]

        X_pt = pt.as_tensor(X, name=name, dtype=floatX)
        return X_pt

    def __getitem__(self, key: KeyLike) -> pt.TensorVariable:
        _type = self._validate_key_and_get_type(key)

        # Case 1: user asked for an entire matrix by name
        if _type is str:
            self._validate_key(key)
            matrix = getattr(self, key)

            # Slice away the time dimension if it's a dummy
            if (self.shapes[key][-1] == 1) and (key not in NEVER_TIME_VARYING):
                X = matrix[(slice(None),) * (matrix.ndim - 1) + (0,)]
                X = pt.specify_shape(X, self.shapes[key][:-1])

                X.name = key
                return X

            # If it's never time varying, return everything
            elif key in NEVER_TIME_VARYING:
                X = pt.specify_shape(matrix, self.shapes[key])
                X.name = key

                return X

            # Last possibility is that it's time varying -- also return everything (for now, might need some processing)
            else:
                X = pt.specify_shape(matrix, self.shapes[key])
                X.name = key
                return X

        # Case 2: user asked for a particular matrix and some slices of it
        elif _type is tuple:
            name, *slice_ = key
            self._validate_key(name)

            matrix = getattr(self, name)
            slice_ = self._add_time_dim_to_slice(name, slice_, matrix.ndim)

            return matrix[slice_]

        # Case 3: There is only one slice index, but it's not a string
        else:
            raise IndexError("First index must the name of a valid state space matrix.")

    def __setitem__(self, key: KeyLike, value: Union[float, int, np.ndarray]) -> None:
        _type = type(key)

        # Case 1: key is a string: we are setting an entire matrix.
        if _type is str:
            self._validate_key(key)
            if isinstance(value, np.ndarray):
                value = self._numpy_to_pytensor(key, value)
            else:
                value.name = key

            setattr(self, key, value)
            self._update_shape(key, value)

        # Case 2: key is a string plus a slice: we are setting a subset of a matrix
        elif _type is tuple:
            name, *slice_ = key
            self._validate_key(name)

            matrix = getattr(self, name)

            slice_ = self._add_time_dim_to_slice(name, slice_, matrix.ndim)
            matrix = pt.set_subtensor(matrix[slice_], value)
            matrix = pt.specify_shape(matrix, self.shapes[name])
            matrix.name = name

            setattr(self, name, matrix)
