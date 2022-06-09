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
@pytest.mark.parametrize("top_k", [None, 90])
def test_histogram_init_discrete(use_dask, top_k):
    data = np.random.randint(0, 100, size=10000)
    n_uniq = len(np.unique(data))
    if use_dask:
        dask = pytest.importorskip("dask")
        dask_df = pytest.importorskip("dask.dataframe")
        data = dask_df.from_array(data)
    histogram = pmx.distributions.histogram_utils.discrete_histogram(data, top_k=top_k)
    if use_dask:
        (histogram,) = dask.compute(histogram)
    assert isinstance(histogram, dict)
    assert isinstance(histogram["mid"], np.ndarray)
    assert histogram["mid"].dtype == np.int64
    if top_k is None:
        top_k = n_uniq
    assert histogram["mid"].shape == (top_k,)
    assert histogram["count"].shape == (top_k,)


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
            "histogram_potential", pm.Poisson.dist(s), observed=data, top_k=90
        )
        trace = pm.sample()  # very fast
