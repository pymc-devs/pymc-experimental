from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from pymc_experimental.statespace.models.local_level import BayesianLocalLevel
from pymc_experimental.statespace.utils.constants import (
    EXTENDED_TIME_DIM,
    FILTER_OUTPUT_DIMS,
    FILTER_OUTPUT_NAMES,
    MATRIX_DIMS,
    MATRIX_NAMES,
    SMOOTHER_OUTPUT_NAMES,
    TIME_DIM,
)
from pymc_experimental.statespace.utils.data_tools import (
    NO_FREQ_INFO_WARNING,
    NO_TIME_INDEX_WARNING,
)
from pymc_experimental.tests.statespace.utilities.test_helpers import (
    load_nile_test_data,
)

function_names = ["pandas_date_freq", "pandas_date_nofreq", "pandas_nodate", "numpy", "pytensor"]
expected_warning = [
    does_not_raise(),
    pytest.warns(UserWarning, match=NO_FREQ_INFO_WARNING),
    pytest.warns(UserWarning, match=NO_TIME_INDEX_WARNING),
    pytest.warns(UserWarning, match=NO_TIME_INDEX_WARNING),
    pytest.warns(UserWarning, match=NO_TIME_INDEX_WARNING),
]
func_inputs = list(zip(function_names, expected_warning))
floatX = pytensor.config.floatX


@pytest.fixture
def load_dataset():
    data = load_nile_test_data()

    def _load_dataset(f):
        if f == "pandas_date_freq":
            data.index.freq = data.index.inferred_freq
            return data
        if f == "pandas_date_nofreq":
            data.index.freq = None
            return data
        elif f == "pandas_nodate":
            return data.reset_index(drop=True)
        elif f == "numpy":
            return data.values
        elif f == "pytensor":
            return pt.as_tensor_variable(data.values)
        else:
            raise ValueError

    return _load_dataset


@pytest.fixture()
def generate_timeseries():
    def _generate_timeseries(freq):
        index = pd.date_range(start="2000-01-01", freq=freq, periods=100)
        df = pd.DataFrame(np.random.normal(size=100, dtype=floatX), index=index, columns=["level"])
        return df

    return _generate_timeseries


@pytest.fixture()
def create_model(load_dataset):
    ss_mod = BayesianLocalLevel(verbose=False)

    def _create_model(f):
        data = load_dataset(f)
        with pm.Model(coords=ss_mod.coords) as mod:
            ss_mod.add_default_priors()
            ss_mod.build_statespace_graph(data)
        return mod

    return _create_model


@pytest.mark.parametrize("f, warning", func_inputs, ids=function_names)
def test_matrix_coord_assignment(f, warning, create_model):
    with warning:
        pymc_model = create_model(f)

    for matrix in MATRIX_NAMES:
        assert pymc_model.named_vars_to_dims[matrix] == MATRIX_DIMS[matrix]


@pytest.mark.parametrize("f, warning", func_inputs, ids=function_names)
def test_filter_output_coord_assignment(f, warning, create_model):
    with warning:
        pymc_model = create_model(f)

    for output in FILTER_OUTPUT_NAMES + SMOOTHER_OUTPUT_NAMES + ["obs"]:
        assert pymc_model.named_vars_to_dims[output] == FILTER_OUTPUT_DIMS[output]


def test_model_build_without_coords(load_dataset):
    ss_mod = BayesianLocalLevel(verbose=False)
    data = load_dataset("numpy")
    with pm.Model() as mod:
        ss_mod.add_default_priors()
        ss_mod.build_statespace_graph(data, register_data=False)

    assert mod.coords == {}


@pytest.mark.parametrize("f, warning", func_inputs, ids=function_names)
def test_data_index_is_coord(f, warning, create_model):
    with warning:
        pymc_model = create_model(f)
    assert TIME_DIM in pymc_model.coords


@pytest.mark.parametrize("freq", ["Y", "YS", "Q", "QS", "M", "MS", "D", "B"])
def test_extended_date_index(freq, generate_timeseries):
    df = generate_timeseries(freq)
    ss_mod = BayesianLocalLevel(verbose=False)
    with pm.Model() as mod:
        ss_mod.add_default_priors()
        ss_mod.build_statespace_graph(df)

    assert len(mod.coords[TIME_DIM]) == 100
    assert len(mod.coords[EXTENDED_TIME_DIM]) == 101

    index = pd.DatetimeIndex(mod.coords[EXTENDED_TIME_DIM])
    assert index.inferred_freq.startswith(freq.replace("Y", "A"))
