import numpy as np
import functools
from numpy.typing import ArrayLike
from typing import Dict


__all__ = ["quantile_histogram"]


@functools.singledispatch
def quantile_histogram(data: ArrayLike, n_quantiles=1000) -> Dict[str, ArrayLike]:
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

