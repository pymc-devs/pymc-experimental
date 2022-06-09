import pymc_experimental as pmx
import pymc as pm
import numpy as np
import pytest


@pytest.mark.parametrize("use_dask", [True, False])
def test_histogram_init(use_dask):
    data = np.random.randn(10000)
    if use_dask:
        dask = pytest.importorskip("dask")
        dask_df = pytest.importorskip("dask.dataframe")
        data = dask_df.from_array(data)
    histogram = pmx.approximations.quantile_histogram(data, n_quantiles=100)
    if use_dask:
        (histogram,) = dask.compute(histogram)
    assert isinstance(histogram, dict)
    assert isinstance(histogram["mid"], np.ndarray)
    assert histogram["mid"].shape == (99,)
    assert histogram["low"].shape == (99,)
    assert histogram["upper"].shape == (99,)
    assert histogram["count"].shape == (99,)


@pytest.mark.parametrize("use_dask", [True, False])
def test_histogram_approx_api(use_dask):
    data = np.random.randn(10000)
    if use_dask:
        dask = pytest.importorskip("dask")
        dask_df = pytest.importorskip("dask.dataframe")
        data = dask_df.from_array(data)
    with pm.Model():
        m = pm.Normal("m")
        s = pm.HalfNormal("s")
        pot = pmx.approximations.histogram_approximation(
            "histogram_potential", pm.Normal.dist(m, s), observed=data, n_quantiles=1000
        )
        trace = pm.sample()  # very fast
