import numpy as np
from numpy.typing import ArrayLike
from typing import Dict
import pymc as pm

try:
    import dask.dataframe
    import dask.array
except ImportError:
    dask = None


__all__ = ["quantile_histogram", "histogram_approximation"]


def quantile_histogram(
    data: ArrayLike, n_quantiles=1000, zero_inflation=False
) -> Dict[str, ArrayLike]:
    if dask and isinstance(data, dask.dataframe.Series):
        data = data.to_dask_array(lengths=True)
    if zero_inflation:
        zeros = (data == 0).sum()
        data = data[data > 0]
    quantiles = np.percentile(data, np.linspace(0, 100, n_quantiles))
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


def discrete_histogram(data: ArrayLike, min_count=None) -> Dict[str, ArrayLike]:
    if dask and isinstance(data, dask.dataframe.Series):
        data = data.to_dask_array(lengths=True)
    mid, count = np.unique(data, return_counts=True)
    if min_count is not None:
        mid = mid[count >= min_count]
        count = count[count >= min_count]
    return dict(mid=mid, count=count)


def histogram_approximation(name, dist, *, observed: ArrayLike, **h_kwargs):
    """Approximate a univariate distribution with a histogram potential.

    Parameters
    ----------
    name : str
        Name for the Potential
    dist : aesara.tensor.var.TensorVariable
        The output of pm.Distribution.dist()
    observed : ArrayLike
        observed value to construct a histogram

    Returns
    -------
    aesara.tensor.var.TensorVariable
        Potential
    """
    if np.issubdtype(observed.dtype, np.integer):
        histogram = discrete_histogram(observed, **h_kwargs)
    else:
        histogram = quantile_histogram(observed, **h_kwargs)
    if dask is not None:
        (histogram,) = dask.compute(histogram)
    return pm.Potential(name, pm.logp(dist, histogram["mid"]) * histogram["count"])
