#   Copyright 2024 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

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


def laplace(
    vars: Sequence[Variable],
    draws=1_000,
    chains=1,
    model=None,
    random_seed: Optional[RandomSeed] = None,
    progressbar=True,
):
    """
    Create a Laplace approximation for a posterior distribution.

    This function generates a Laplace approximation for a given posterior distribution using a specified
    number of draws. This is useful for obtaining a parametric approximation to the posterior distribution
    that can be used for further analysis.

    Parameters
    ----------
    vars : Sequence[Variable]
        A sequence of variables for which the Laplace approximation of the posterior distribution
        is to be created.
    draws : int, optional, default=1_000
        The number of draws to sample from the posterior distribution for creating the approximation.
    chains : int, default=1
        The number of chains to sample. For chains=0 only the fit of the Laplace approximation is
        returned.s
    model : object, optional, default=None
        The model object that defines the posterior distribution. If None, the default model will be used.
    random_seed : Optional[RandomSeed], optional, default=None
        An optional random seed to ensure reproducibility of the draws. If None, the draws will be
        generated using the current random state.
    progressbar: bool, optional defaults to True
        Whether to display a progress bar in the command line.

    Returns
    -------
    arviz.InferenceData
        An `InferenceData` object from the `arviz` library containing the Laplace
        approximation of the posterior distribution. The inferenceData object also
        contains constant and observed data as well as deterministic variables.
        InferenceData also contains a group 'fit' with the mean and covariance
        for the Laplace approximation.

    Examples
    --------

    >>> import numpy as np
    >>> import pymc as pm
    >>> import arviz as az
    >>> from pymc_experimental.inference.laplace import laplace
    >>> y = np.array([2642, 3503, 4358]*10)
    >>> with pm.Model() as m:
    >>>     logsigma = pm.Uniform("logsigma", 1, 100)
    >>>     mu = pm.Uniform("mu", -10000, 10000)
    >>>     yobs = pm.Normal("y", mu=mu, sigma=pm.math.exp(logsigma), observed=y)
    >>>     idata = laplace([mu, logsigma])

    Notes
    -----
    This method of approximation may not be suitable for all types of posterior distributions,
    especially those with significant skewness or multimodality.

    See Also
    --------
    fit : Calling the inference function 'fit' like pmx.fit(method="laplace", vars=[mu, logsigma], model=m)
          will forward the call to 'laplace'.

    """

    rng = np.random.default_rng(seed=random_seed)

    map = pm.find_MAP(vars=vars, progressbar=progressbar)

    m = pm.modelcontext(model)

    for var in vars:
        if m.rvs_to_transforms[var] is not None:
            m.rvs_to_transforms[var] = None
            var_value = m.rvs_to_values[var]
            var_value.name = var.name

    H = pm.find_hessian(point=map, vars=vars)
    cov = np.linalg.inv(H)
    mean = np.concatenate([np.atleast_1d(map[v.name]) for v in vars])

    if chains != 0:
        samples = rng.multivariate_normal(mean, cov, size=(chains, draws))

        data_vars = {}
        for i, var in enumerate(vars):
            data_vars[str(var)] = xr.DataArray(samples[:, :, i], dims=("chain", "draw"))

        coords = {"chain": np.arange(chains), "draw": np.arange(draws)}
        ds = xr.Dataset(data_vars, coords=coords)

        idata = az.convert_to_inference_data(ds)
        idata = addDataToInferenceData(model, idata, progressbar)
    else:
        idata = az.InferenceData()

    idata = addFitToInferenceData(vars, idata, mean, cov)

    return idata


def addFitToInferenceData(vars, idata, mean, covariance):
    coord_names = [v.name for v in vars]
    # Convert to xarray DataArray
    mean_dataarray = xr.DataArray(mean, dims=["rows"], coords={"rows": coord_names})
    cov_dataarray = xr.DataArray(
        covariance, dims=["rows", "columns"], coords={"rows": coord_names, "columns": coord_names}
    )

    # Create xarray dataset
    dataset = xr.Dataset({"mean_vector": mean_dataarray, "covariance_matrix": cov_dataarray})

    idata.add_groups(fit=dataset)

    return idata


def addDataToInferenceData(model, trace, progressbar):
    # Add deterministic variables to inference data
    trace.posterior = pm.compute_deterministics(
        trace.posterior, model=model, merge_dataset=True, progressbar=progressbar
    )

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
