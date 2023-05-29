from functools import reduce
from typing import List, Optional, Tuple, Type, Union

import numpy as np
import pandas.core.tools.datetimes
import pytensor.tensor as pt
from pandas import DataFrame

KeyLike = Union[Tuple[Union[str, int]], str]


def _preprocess_data(data: Union[DataFrame, np.ndarray], expected_dims=3):
    if isinstance(data, pandas.DataFrame):
        data = data.values
    elif not isinstance(data, np.ndarray):
        raise ValueError("Expected pandas Dataframe or numpy array as data")

    if data.ndim < expected_dims:
        n_dims = data.ndim
        n_to_add = expected_dims - n_dims + 1
        data = reduce(lambda a, b: np.expand_dims(a, -1), [data] * n_to_add)

    return data


class PytensorRepresentation:
    def __init__(
        self,
        data: Union[DataFrame, np.ndarray],
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
        """
        A representation of a State Space model, in Pytensor. Shamelessly copied from the Statsmodels.api implementation
        found here:

        https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/statespace/representation.py

        Parameters
        ----------
        data: ArrayLike
            Array of observed data (called exog in statsmodels)
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

        References
        ----------
        .. [1] Durbin, James, and Siem Jan Koopman. 2012.
            Time Series Analysis by State Space Methods: Second Edition.
            Oxford University Press.
        """

        # self.data = pt.tensor3(name="Data")
        # self.transition = pt.tensor3(name="transition")
        # self.selection = pt.tensor3(name="selection")
        # self.design = pt.tensor3(name="design")
        # self.obs_cov = pt.tensor3(name="obs_cov")
        # self.state_cov = pt.tensor3(name="state_cov")
        # self.state_intercept = pt.tensor3(name="state_cov")
        # self.obs_intercept = pt.tensor3(name="state_cov")
        # self.initial_state = pt.tensor3(name="state_cov")
        # self.initial_state_cov = pt.tensor3(name="state_cov")

        self.data = _preprocess_data(data)
        self.k_states = k_states
        self.k_posdef = k_posdef if k_posdef is not None else k_states

        self.n_obs, self.k_endog, *_ = data.shape

        # The last dimension is for time varying matrices; it could be n_obs. Not thinking about that now.
        self.shapes = {
            "data": (self.k_endog, self.n_obs, 1),
            "design": (self.k_endog, self.k_states, 1),
            "obs_intercept": (self.k_endog, 1, 1),
            "obs_cov": (self.k_endog, self.k_endog, 1),
            "transition": (self.k_states, self.k_states, 1),
            "state_intercept": (self.k_states, 1, 1),
            "selection": (self.k_states, self.k_posdef, 1),
            "state_cov": (self.k_posdef, self.k_posdef, 1),
            "initial_state": (self.k_states, 1, 1),
            "initial_state_cov": (self.k_states, self.k_states, 1),
        }

        # Initialize the representation matrices
        scope = locals()
        for name, shape in self.shapes.items():
            if name == "data":
                continue

            elif scope[name] is not None:
                matrix = self._numpy_to_pytensor(name, scope[name])
                setattr(self, name, matrix)

            else:
                setattr(self, name, pt.zeros(shape))

    def _validate_key(self, key: KeyLike) -> None:
        if key not in self.shapes:
            raise IndexError(f"{key} is an invalid state space matrix name")

    def update_shape(self, key: KeyLike, value: Union[np.ndarray, pt.TensorType]) -> None:
        # TODO: Get rid of these evals

        if isinstance(value, (pt.TensorConstant, pt.TensorVariable)):
            shape = value.shape.eval()
        else:
            shape = value.shape

        old_shape = self.shapes[key]
        if not all([a == b for a, b in zip(shape[:2], old_shape[:2])]):
            raise ValueError(
                f"The first two dimensions of {key} must be {old_shape[:2]}, found {shape[:2]}"
            )

        # Add time dimension dummy if none present
        if len(shape) == 2:
            self.shapes[key] = shape + (1,)

        self.shapes[key] = shape

    def _add_time_dim_to_slice(
        self, name: str, slice_: Union[List[int], Tuple[int]], n_dim: int
    ) -> Tuple[int]:
        no_time_dim = self.shapes[name][-1] == 1

        # Case 1: All dimensions are sliced
        if len(slice_) == n_dim:
            return slice_

        # Case 2a: There is a time dim. Just return.
        if not no_time_dim:
            return slice_

        # Case 2b: There's no time dim. Slice away the dummy dim.
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

        if X.ndim > 3 or X.ndim < 2:
            raise ValueError(
                f"Array provided for {name} has {X.ndim} dimensions, "
                f"expecting 2 (static) or 3 (time-varying)"
            )

        if X.ndim == 2:
            if expected_shape != X.shape:
                raise ValueError(
                    f"Array provided for {name} has shape {X.shape}, expected {expected_shape}"
                )
        if X.ndim == 3:
            if X.shape[:2] != expected_shape:
                raise ValueError(
                    f"First two dimensions of array provided for {name} has shape {X.shape[:2]}, "
                    f"expected {expected_shape}"
                )
            if X.shape[-1] != self.data.shape[0]:
                raise ValueError(
                    f"Last dimension (time dimension) of array provided for {name} has shape "
                    f"{X.shape[-1]}, expected {self.data.shape[0]} (equal to the first dimension of the "
                    f"provided data)"
                )

    def _numpy_to_pytensor(self, name: str, X: np.ndarray) -> pt.TensorVariable:
        X = X.copy()
        self._validate_matrix_shape(name, X)
        # Add a time dimension if one isn't provided
        if X.ndim == 2:
            X = X[..., None]
        return pt.as_tensor(X, name=name)

    def __getitem__(self, key: KeyLike) -> pt.TensorVariable:
        _type = self._validate_key_and_get_type(key)

        # Case 1: user asked for an entire matrix by name
        if _type is str:
            self._validate_key(key)
            matrix = getattr(self, key)

            # Slice away the time dimension if it's a dummy
            if self.shapes[key][-1] == 1:
                return matrix[(slice(None),) * (matrix.ndim - 1) + (0,)]

            # If it's time varying, return everything
            else:
                return matrix

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
            setattr(self, key, value)
            self.update_shape(key, value)

        # Case 2: key is a string plus a slice: we are setting a subset of a matrix
        elif _type is tuple:
            name, *slice_ = key
            self._validate_key(name)

            matrix = getattr(self, name)

            slice_ = self._add_time_dim_to_slice(name, slice_, matrix.ndim)

            matrix = pt.set_subtensor(matrix[slice_], value)
            setattr(self, name, matrix)
