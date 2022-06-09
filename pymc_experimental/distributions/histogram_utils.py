import numpy as np
import functools
from numpy.typing import ArrayLike
from typing import Dict
import pymc as pm

try:
    import dask.dataframe
except ImportError:
    dask = None


__all__ = ["quantile_histogram", "histogram_approximation"]


@functools.singledispatch
def quantile_histogram(data: ArrayLike, n_quantiles=1000) -> Dict[str, ArrayLike]:
    raise NotImplementedError(f"Not implemented for {type(data)}")


@quantile_histogram.register(np.ndarray)
def _(data: ArrayLike, n_quantiles=1000) -> Dict[str, ArrayLike]:
    quantiles = np.quantile(data, np.linspace(0, 1, n_quantiles))
    count, _ = np.histogram(data, quantiles)
    low = quantiles[:-1]
    upper = quantiles[1:]
    result = dict(
        low=low,
        upper=upper,
        mid=(low + upper) / 2,
        count=count,
    )
    return result


if dask is not None:

    @quantile_histogram.register(dask.dataframe.Series)
    def _(data: dask.dataframe.Series, n_quantiles=1000) -> Dict[str, ArrayLike]:
        quantiles = data.quantile(np.linspace(0, 1, n_quantiles)).to_dask_array(lengths=True)
        count, _ = dask.array.histogram(data, quantiles)
        low = quantiles[:-1]
        upper = quantiles[1:]
        result = dict(
            low=low,
            upper=upper,
            mid=(low + upper) / 2,
            count=count,
        )
        return result


@functools.singledispatch
def discrete_histogram(data: ArrayLike, top_k=None):
    raise NotImplementedError(f"Not implemented for {type(data)}")


@discrete_histogram.register(np.ndarray)
def _(data: np.ndarray, top_k=None):
    ...


if dask is not None:

    @discrete_histogram.register(dask.dataframe.Series)
    def _(data: dask.dataframe.Series, top_k=None):
        ...


def histogram_approximation(name, dist, *, observed, n_quantiles=1000):
    histogram = quantile_histogram(observed, n_quantiles=n_quantiles)
    if dask is not None:
        (histogram,) = dask.compute(histogram)
    return pm.Potential(name, pm.logp(dist, histogram["mid"]) * histogram["count"])
