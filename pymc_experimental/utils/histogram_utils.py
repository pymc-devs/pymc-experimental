import numpy as np
import functools
from numpy.typing import ArrayLike
from typing import Dict

try:
    import dask.dataframe
except ImportError:
    dask = None


__all__ = ["quantile_histogram"]


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
