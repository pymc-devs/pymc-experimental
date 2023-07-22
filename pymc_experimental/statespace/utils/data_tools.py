import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pymc import modelcontext
from pytensor.tensor.sharedvar import TensorSharedVariable

from pymc_experimental.statespace.utils.constants import (
    EXTENDED_TIME_DIM,
    OBS_STATE_DIM,
    TIME_DIM,
)

NO_TIME_INDEX_WARNING = (
    "No time index found on the supplied data. A simple range index will be automatically "
    "generated."
)
NO_FREQ_INFO_WARNING = "No frequency was specific on the data's DateTimeIndex."


def get_data_dims(data):
    if not isinstance(data, (pt.TensorVariable, TensorSharedVariable)):
        return

    data_name = getattr(data, "name", None)
    if not data_name:
        return

    pm_mod = modelcontext(None)
    data_dims = pm_mod.named_vars_to_dims.get(data_name, None)
    return data_dims


def _validate_data_shape(data_shape, n_obs, obs_coords=None, check_col_names=False, col_names=None):
    if col_names is None:
        col_names = []

    if len(data_shape) != 2:
        raise ValueError("Data must be a 2d matrix")

    if data_shape[-1] != n_obs:
        raise ValueError(
            f"Shape of data does not match model output. Expected {n_obs} columns, "
            f"found {data_shape[-1]}."
        )

    if check_col_names:
        missing_cols = set(obs_coords) - set(col_names)
        if len(missing_cols) > 0:
            raise ValueError(
                "Columns of DataFrame provided as data do not match state names. The following states were"
                f'not found: {", ".join(missing_cols)}. This may result in unexpected results in complex'
                f"statespace models"
            )


def preprocess_tensor_data(data, n_obs, obs_coords=None):
    data_shape = data.shape.eval()
    _validate_data_shape(data_shape, n_obs, obs_coords)
    warnings.warn(NO_TIME_INDEX_WARNING)
    index = np.arange(data_shape[0], dtype="int")
    extended_index = np.r_[[index[0] - 1], index]

    return data.eval(), index, extended_index


def preprocess_numpy_data(data, n_obs, obs_coords=None):
    _validate_data_shape(data.shape, n_obs, obs_coords)
    warnings.warn(NO_TIME_INDEX_WARNING)
    index = np.arange(data.shape[0], dtype="int")
    extended_index = np.r_[[index[0] - 1], index]

    return data, index, extended_index


def preprocess_pandas_data(data, n_obs, obs_coords=None, check_column_names=False):
    if isinstance(data, pd.Series):
        data = data.to_frame()

    col_names = data.columns
    _validate_data_shape(data.shape, n_obs, obs_coords, check_column_names, col_names)
    if isinstance(data.index, pd.RangeIndex):
        warnings.warn(NO_TIME_INDEX_WARNING)
        return preprocess_numpy_data(data.values, n_obs, obs_coords)
    elif isinstance(data.index, pd.DatetimeIndex):
        if data.index.freq is None:
            warnings.warn(NO_FREQ_INFO_WARNING)
            data.index.freq = data.index.inferred_freq

        index = data.index
        extended_index = index.shift(-1).union(index)

        return data.values, index, extended_index
    else:
        raise IndexError(
            f"Expected pd.DatetimeIndex or pd.RangeIndex on data, found {type(data.index)}"
        )


def register_data_with_pymc(data, n_obs, obs_coords):
    if isinstance(data, (pt.TensorVariable, TensorSharedVariable)):
        values, index, extended_index = preprocess_tensor_data(data, n_obs, obs_coords)
    elif isinstance(data, np.ndarray):
        values, index, extended_index = preprocess_numpy_data(data, n_obs, obs_coords)
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        values, index, extended_index = preprocess_pandas_data(data, n_obs, obs_coords)
    else:
        raise ValueError("Data should be one of pytensor tensor, numpy array, or pandas dataframe")

    pymc_mod = modelcontext(None)
    pymc_mod.add_coord(TIME_DIM, index, mutable=True)
    pymc_mod.add_coord(EXTENDED_TIME_DIM, extended_index, mutable=True)

    pm.MutableData("data", values, dims=[TIME_DIM, OBS_STATE_DIM])
