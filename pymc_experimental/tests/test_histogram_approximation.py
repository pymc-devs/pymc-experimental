import pymc_experimental as pmx
import pymc as pm
import numpy as np
import pytest


@pytest.mark.parametrize("use_dask", [True, False])
def test_histogram_init_cont(use_dask):
    data = np.random.randn(10000)
    if use_dask:
        dask = pytest.importorskip("dask")
        dask_df = pytest.importorskip("dask.dataframe")
        data = dask_df.from_array(data)
    histogram = pmx.distributions.histogram_utils.quantile_histogram(data, n_quantiles=100)
    if use_dask:
        (histogram,) = dask.compute(histogram)
    assert isinstance(histogram, dict)
    assert isinstance(histogram["mid"], np.ndarray)
    assert histogram["mid"].shape == (99,)
    assert histogram["low"].shape == (99,)
    assert histogram["upper"].shape == (99,)
    assert histogram["count"].shape == (99,)


@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.parametrize("min_count", [None, 10])
def test_histogram_init_discrete(use_dask, min_count):
    data = np.random.randint(0, 100, size=10000)
    u, c = np.unique(data, return_counts=True)
    if use_dask:
        dask = pytest.importorskip("dask")
        dask_df = pytest.importorskip("dask.dataframe")
        data = dask_df.from_array(data)
    histogram = pmx.distributions.histogram_utils.discrete_histogram(data, min_count=min_count)
    if use_dask:
        (histogram,) = dask.compute(histogram)
    assert isinstance(histogram, dict)
    assert isinstance(histogram["mid"], np.ndarray)
    assert histogram["mid"].dtype == np.int64
    if min_count is not None:
        size = int((c >= min_count).sum())
    else:
        size = len(u)
    assert histogram["mid"].shape == (size,)
    assert histogram["count"].shape == (size,)


@pytest.mark.parametrize("use_dask", [True, False])
def test_histogram_approx_cont(use_dask):
    data = np.random.randn(10000)
    if use_dask:
        dask = pytest.importorskip("dask")
        dask_df = pytest.importorskip("dask.dataframe")
        data = dask_df.from_array(data)
    with pm.Model():
        m = pm.Normal("m")
        s = pm.HalfNormal("s")
        pot = pmx.distributions.histogram_approximation(
            "histogram_potential", pm.Normal.dist(m, s), observed=data, n_quantiles=1000
        )
        trace = pm.sample()  # very fast


@pytest.mark.parametrize("use_dask", [True, False])
def test_histogram_approx_discrete(use_dask):
    data = np.random.randint(0, 100, size=10000)
    if use_dask:
        dask = pytest.importorskip("dask")
        dask_df = pytest.importorskip("dask.dataframe")
        data = dask_df.from_array(data)
    with pm.Model():
        s = pm.Exponential("s", 1.0)
        pot = pmx.distributions.histogram_approximation(
            "histogram_potential", pm.Poisson.dist(s), observed=data, min_count=10
        )
        trace = pm.sample()  # very fast
