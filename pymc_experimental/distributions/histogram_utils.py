import numpy as np
from numpy.typing import ArrayLike
from typing import Dict
import pymc as pm
import xhistogram.core

try:
    import dask.dataframe
    import dask.array
except ImportError:
    dask = None


__all__ = ["quantile_histogram", "histogram_approximation"]


def quantile_histogram(
    data: ArrayLike, n_quantiles=1000, zero_inflation=False
) -> Dict[str, ArrayLike]:
    if dask and isinstance(data, (dask.dataframe.Series, dask.dataframe.DataFrame)):
        data = data.to_dask_array(lengths=True)
    if zero_inflation:
        zeros = (data == 0).sum(0)
        mdata = np.ma.masked_where(data, data > 0)
        qdata = data[data > 0]
    else:
        mdata = data
        qdata = data.flatten()
    quantiles = np.percentile(qdata, np.linspace(0, 100, n_quantiles))
    if dask:
        (quantiles,) = dask.compute(quantiles)
    count, _ = xhistogram.core.histogram(mdata, bins=[quantiles], axis=0)
    count = count.transpose(count.ndim - 1, *range(count.ndim - 1))
    quantiles = quantiles.reshape(quantiles.shape + (1,) * (count.ndim - 1))
    lower = quantiles[:-1]
    upper = quantiles[1:]

    if zero_inflation:
        count = np.concatenate([zeros[None], count])
        lower = np.concatenate([[0], lower])
        upper = np.concatenate([[0], upper])

    result = dict(
        lower=lower,
        upper=upper,
        mid=(lower + upper) / 2,
        count=count,
    )
    return result


def discrete_histogram(data: ArrayLike, min_count=None) -> Dict[str, ArrayLike]:
    if dask and isinstance(data, (dask.dataframe.Series, dask.dataframe.DataFrame)):
        data = data.to_dask_array(lengths=True)
    mid, count_uniq = np.unique(data, return_counts=True)
    if min_count is not None:
        mid = mid[count_uniq >= min_count]
        count_uniq = count_uniq[count_uniq >= min_count]
    bins = np.concatenate([mid, [mid.max() + 1]])
    if dask:
        mid, bins = dask.compute(mid, bins)
    count, _ = xhistogram.core.histogram(data, bins=[bins], axis=0)
    count = count.transpose(count.ndim - 1, *range(count.ndim - 1))
    mid = mid.reshape(mid.shape + (1,) * (count.ndim - 1))
    return dict(mid=mid, count=count)


def histogram_approximation(name, dist, *, observed: ArrayLike, **h_kwargs):
    """Approximate a distribution with a histogram potential.

    Parameters
    ----------
    name : str
        Name for the Potential
    dist : aesara.tensor.var.TensorVariable
        The output of pm.Distribution.dist()
    observed : ArrayLike
        Observed value to construct a histogram. Histogram is computed over 0th axis

    Returns
    -------
    aesara.tensor.var.TensorVariable
        Potential
    """
    if dask and isinstance(observed, (dask.dataframe.Series, dask.dataframe.DataFrame)):
        observed = observed.to_dask_array(lengths=True)
    if np.issubdtype(observed.dtype, np.integer):
        histogram = discrete_histogram(observed, **h_kwargs)
    else:
        histogram = quantile_histogram(observed, **h_kwargs)
    if dask is not None:
        (histogram,) = dask.compute(histogram)
    return pm.Potential(name, pm.logp(dist, histogram["mid"]) * histogram["count"])
