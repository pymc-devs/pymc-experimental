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

import warnings
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
from pymc.model.transform.conditioning import remove_value_transforms
from pymc.util import RandomSeed
from pytensor import Variable


def laplace(
    vars: Sequence[Variable],
    draws: Optional[int] = 1000,
    model=None,
    random_seed: Optional[RandomSeed] = None,
    progressbar=True,
):
    """
    Create a Laplace (quadratic) approximation for a posterior distribution.

    This function generates a Laplace approximation for a given posterior distribution using a specified
    number of draws. This is useful for obtaining a parametric approximation to the posterior distribution
    that can be used for further analysis.

    Parameters
    ----------
    vars : Sequence[Variable]
        A sequence of variables for which the Laplace approximation of the posterior distribution
        is to be created.
    draws : Optional[int] with default=1_000
        The number of draws to sample from the posterior distribution for creating the approximation.
        For draws=None only the fit of the Laplace approximation is returned
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
    >>> idata = laplace([mu, logsigma], model=m)

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

    transformed_m = pm.modelcontext(model)

    if len(vars) != len(transformed_m.free_RVs):
        warnings.warn(
            "Number of variables in vars does not eqaul the number of variables in the model.",
            UserWarning,
        )

    map = pm.find_MAP(vars=vars, progressbar=progressbar, model=transformed_m)

    # See https://www.pymc.io/projects/docs/en/stable/api/model/generated/pymc.model.transform.conditioning.remove_value_transforms.html
    untransformed_m = remove_value_transforms(transformed_m)
    untransformed_vars = [untransformed_m[v.name] for v in vars]
    hessian = pm.find_hessian(point=map, vars=untransformed_vars, model=untransformed_m)

    if np.linalg.det(hessian) == 0:
        raise np.linalg.LinAlgError("Hessian is singular.")

    cov = np.linalg.inv(hessian)
    mean = np.concatenate([np.atleast_1d(map[v.name]) for v in vars])

    chains = 1

    if draws is not None:
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
