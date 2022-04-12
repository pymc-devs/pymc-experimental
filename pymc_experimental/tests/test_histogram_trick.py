from termios import N_MOUSE
import pymc_experimental as pmx
import numpy as np
import pytest


@pytest.mark.parametrize("use_dask", [False])
def test_histogram_init(use_dask):
    data = np.random.randn(10000)
    
    histogram = pmx.utils.quantile_histogram(data, n_quantiles=100)
    assert isinstance(histogram, dict)
    assert histogram["mid"].shape == (99, )
    assert histogram["low"].shape == (99, )
    assert histogram["upper"].shape == (99, )
    assert histogram["count"].shape == (99, )