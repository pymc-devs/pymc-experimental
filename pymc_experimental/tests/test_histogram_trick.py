from termios import N_MOUSE
import pymc_experimental as pmx
import numpy as np
import pytest


@pytest.mark.parametrize("use_dask", [True, False])
def test_histogram_init(use_dask):
    data = np.random.randn(10000)
    if use_dask:
        dask = pytest.importorskip("dask")
        dask_df = pytest.importorskip("dask.dataframe")
        data = dask_df.from_array(data)
    histogram = pmx.utils.quantile_histogram(data, n_quantiles=100)
    if use_dask:
        (histogram,) = dask.compute(histogram)
    assert isinstance(histogram, dict)
    assert isinstance(histogram["mid"], np.ndarray)
    assert histogram["mid"].shape == (99,)
    assert histogram["low"].shape == (99,)
    assert histogram["upper"].shape == (99,)
    assert histogram["count"].shape == (99,)
