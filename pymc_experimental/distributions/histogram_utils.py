import numpy as np
import functools
from numpy.typing import ArrayLike
from typing import Dict
import pymc as pm

try:
    import dask.dataframe
    import dask.array
except ImportError:
    dask = None


__all__ = ["quantile_histogram", "histogram_approximation"]


@functools.singledispatch
def quantile_histogram(data: ArrayLike, n_quantiles=1000) -> Dict[str, ArrayLike]:
    raise NotImplementedError(f"Not implemented for {type(data)}")


@quantile_histogram.register(np.ndarray)
def _(data: ArrayLike, n_quantiles=1000, zero_inflation=False) -> Dict[str, ArrayLike]:
    if zero_inflation:
        zeros = (data == 0).sum()
        data = data[data > 0]
    quantiles = np.quantile(data, np.linspace(0, 1, n_quantiles))
    count, _ = np.histogram(data, quantiles)
    lower = quantiles[:-1]
    upper = quantiles[1:]
    if zero_inflation:
        count = np.concatenate([[zeros], count])
        lower = np.concatenate([[0], lower])
        upper = np.concatenate([[0], upper])
    result = dict(
        lower=lower,
        upper=upper,
        mid=(lower + upper) / 2,
        count=count,
    )
    return result


if dask is not None:

    @quantile_histogram.register(dask.array.Array)
    def _(
        data: dask.dataframe.Series, n_quantiles=1000, zero_inflation=False
    ) -> Dict[str, ArrayLike]:
        if zero_inflation:
            zeros = (data == 0).sum()
            data = data[data > 0]
        quantiles = dask.array.percentile(data, np.linspace(0, 100, n_quantiles))
        count, _ = dask.array.histogram(data, quantiles)
        lower = quantiles[:-1]
        upper = quantiles[1:]
        if zero_inflation:
            count = dask.array.concatenate([[zeros], count])
            lower = dask.array.concatenate([[0], lower])
            upper = dask.array.concatenate([[0], upper])
        result = dict(
            lower=lower,
            upper=upper,
            mid=(lower + upper) / 2,
            count=count,
        )
        return result

    @quantile_histogram.register(dask.dataframe.Series)
    def _(
        data: dask.dataframe.Series, n_quantiles=1000, zero_inflation=False
    ) -> Dict[str, ArrayLike]:
        return quantile_histogram(
            data.to_dask_array(lengths=True), n_quantiles=n_quantiles, zero_inflation=zero_inflation
        )


@functools.singledispatch
def discrete_histogram(data: ArrayLike, min_count=None):
    raise NotImplementedError(f"Not implemented for {type(data)}")


@discrete_histogram.register(np.ndarray)
def _(data: np.ndarray, min_count=None):
    mid, count = np.unique(data, return_counts=True)
    if min_count is not None:
        mid = mid[count >= min_count]
        count = count[count >= min_count]
    return dict(mid=mid, count=count)


if dask is not None:

    @discrete_histogram.register(dask.array.Array)
    def _(data: dask.array.Array, min_count=None):
        mid, count = dask.array.unique(data, return_counts=True)
        if min_count is not None:
            mid = mid[count >= min_count]
            count = count[count >= min_count]
        return dict(mid=mid, count=count)

    @discrete_histogram.register(dask.dataframe.Series)
    def _(data: dask.dataframe.Series, min_count=None):
        return discrete_histogram(data.to_dask_array(lengths=True), min_count=min_count)


def histogram_approximation(name, dist, *, observed: ArrayLike, **h_kwargs):
    if np.issubdtype(observed.dtype, np.integer):
        histogram = discrete_histogram(observed, **h_kwargs)
    else:
        histogram = quantile_histogram(observed, **h_kwargs)
    if dask is not None:
        (histogram,) = dask.compute(histogram)
    return pm.Potential(name, pm.logp(dist, histogram["mid"]) * histogram["count"])
