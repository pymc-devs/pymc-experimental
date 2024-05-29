from collections.abc import Sequence
from typing import Optional

import arviz as az
import numpy as np
import pymc as pm
import xarray as xr
from arviz import dict_to_dataset
from pymc.backends.arviz import (
    coords_and_dims_for_inferencedata,
    find_constants,
    find_observations,
)
from pymc.util import RandomSeed
from pytensor import Variable


def quadratic(
    vars: Sequence[Variable],
    draws=1_000,
    model=None,
    random_seed: Optional[RandomSeed] = None,
):
    """
    Create a quadratic approximation for a posterior distribution.

    This function generates a quadratic approximation for a given posterior distribution using a specified
    number of draws. This is useful for obtaining a parametric approximation to the posterior distribution
    that can be used for further analysis.

    Parameters
    ----------
    vars : Sequence[Variable]
        A sequence of variables for which the quadratic approximation of the posterior distribution
        is to be created.
    draws : int, optional, default=1_000
        The number of draws to sample from the posterior distribution for creating the approximation.
    model : object, optional, default=None
        The model object that defines the posterior distribution. If None, the default model will be used.
    random_seed : Optional[RandomSeed], optional, default=None
        An optional random seed to ensure reproducibility of the draws. If None, the draws will be
        generated using the current random state.

    Returns
    -------
    arviz.InferenceData
        An `InferenceData` object from the `arviz` library containing the quadratic
        approximation of the posterior distribution. The inferenceData object also
        contains constant and observed data as well as deterministic variables.

    Examples
    --------

    >>> import numpy as np
    >>> import pymc as pm
    >>> import arviz as az
    >>> from pymc_experimental.inference.quadratic import quadratic
    >>> y = np.array([2642, 3503, 4358]*10)
    >>> with pm.Model() as m:
    >>>     logsigma = pm.Uniform("logsigma", 1, 100)
    >>>     mu = pm.Uniform("mu", -10000, 10000)
    >>>     yobs = pm.Normal("y", mu=mu, sigma=pm.math.exp(logsigma), observed=y)
    >>>     idata = quadratic([mu, logsigma])

    Notes
    -----
    This method of approximation may not be suitable for all types of posterior distributions,
    especially those with significant skewness or multimodality.

    See Also
    --------
    fit : Calling the inference function 'fit' like pmx.fit(method="quadratic", vars=[mu, logsigma], model=m)
          will forward the call to 'quadratic'.

    """

    random_seed = np.random.default_rng(seed=random_seed)

    map = pm.find_MAP(vars=vars)

    m = pm.modelcontext(model)

    for var in vars:
        if m.rvs_to_transforms[var] is not None:
            m.rvs_to_transforms[var] = None
            var_value = m.rvs_to_values[var]
            var_value.name = var.name

    H = pm.find_hessian(map, vars=vars)
    cov = np.linalg.inv(H)
    mean = np.concatenate([np.atleast_1d(map[v.name]) for v in vars])
    # posterior = st.multivariate_normal(mean=mean, cov=cov)

    chains = 1

    samples = random_seed.multivariate_normal(mean, cov, size=(chains, draws))

    data_vars = {}
    for i, var in enumerate(vars):
        data_vars[str(var)] = xr.DataArray(samples[:, :, i], dims=("chain", "draw"))

    coords = {"chain": np.arange(chains), "draw": np.arange(draws)}
    ds = xr.Dataset(data_vars, coords=coords)

    idata = az.convert_to_inference_data(ds)

    idata = addDataToInferenceData(model, idata)

    return idata


def addDataToInferenceData(model, trace):
    # Add deterministic variables to inference data
    trace.posterior = pm.compute_deterministics(trace.posterior, model=model, merge_dataset=True)

    coords, dims = coords_and_dims_for_inferencedata(model)

    observed_data = dict_to_dataset(
        find_observations(model),
        library=pm,
        coords=coords,
        dims=dims,
        default_dims=[],
    )

    constant_data = dict_to_dataset(
        find_constants(model),
        library=pm,
        coords=coords,
        dims=dims,
        default_dims=[],
    )

    trace.add_groups(
        {"observed_data": observed_data, "constant_data": constant_data},
        coords=coords,
        dims=dims,
    )

    return trace
