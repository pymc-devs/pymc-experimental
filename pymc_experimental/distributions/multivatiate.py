#   Copyright 2022 The PyMC Developers
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


from typing import Sequence, Union

import pymc as pm
import pytensor.tensor as pt


def _psivar2musigma(psi: pt.TensorVariable, explained_var: pt.TensorVariable):
    pi = pt.erfinv(2 * psi - 1)
    f = (1 / (2 * pi**2 + 1)) ** 0.5
    sigma = explained_var**0.5 * f
    mu = sigma * pi * 2**0.5
    return mu, sigma


def _R2D2M2CP_beta(
    name: str,
    output_sigma: pt.TensorVariable,
    input_sigma: pt.TensorVariable,
    r2: pt.TensorVariable,
    phi: pt.TensorVariable,
    psi: pt.TensorVariable,
    *,
    dims: Union[str, Sequence[str]],
    centered=False,
):
    """R2D2M2CP_beta prior.
    name: str
        Name for the distribution
    output_sigma: tensor
        standard deviation of the outcome
    input_sigma: tensor
        standard deviation of the explanatory variables
    r2: tensor
        expected R2 for the linear regression
    phi: tensor
        variance weights that sums up to 1
    psi: tensor
        probability of a coefficients to be positive
    """
    tau2 = r2 / (1 - r2)
    explained_variance = phi * pt.expand_dims(tau2 * output_sigma**2, -1)
    mu_param, std_param = _psivar2musigma(psi, explained_variance)
    if not centered:
        with pm.Model(name):
            raw = pm.Normal("raw", dims=dims)
        beta = pm.Deterministic(name, (raw * std_param + mu_param) / input_sigma, dims=dims)
    else:
        beta = pm.Normal(name, mu_param / input_sigma, std_param / input_sigma, dims=dims)
    return beta


def R2D2M2CP(
    name,
    output_sigma,
    input_sigma,
    *,
    dims,
    r2,
    variables_importance=None,
    variance_explained=None,
    r2_std=None,
    positive_probs=0.5,
    positive_probs_std=None,
    centered=False,
):
    """R2D2M2CP Prior.

    Parameters
    ----------
    name : str
        Name for the distribution
    output_sigma : tensor
        Output standard deviation
    input_sigma : tensor
        Input standard deviation
    dims : Union[str, Sequence[str]]
        Dims for the distribution
    r2 : tensor
        :math:`R^2` estimate
    variables_importance : tensor, optional
        Optional estimate for variables importance, positive, by default None
    variance_explained : tensor, optional
        Alternative estimate for variables importance which is point estimate of
        variance explained, should sum up to one, by default None
    r2_std : tensor, optional
        Optional uncertainty over :math:`R^2`, by default None
    positive_probs : tensor, optional
        Optional probability of variables contribution to be positive, by default 0.5
    positive_probs_std : tensor, optional
        Optional uncertainty over effect direction probability, by default None
    centered : bool, optional
        Centered or Non-Centered parametrization of the distribution, by default Non-Centered. Advised to check both

    Returns
    -------
    residual_sigma, coefficients
        Output variance (sigma squared) is split in residual variance and explained variance.

    Raises
    ------
    TypeError
        If parametrization is wrong.

    Notes
    -----
    The R2D2M2CP prior is a modification of R2D2M2 prior.

    - ``(R2D2M2)``CP is taken from https://arxiv.org/abs/2208.07132
    - R2D2M2``(CP)``, (Correlation Probability) is proposed and implemented by Max Kochurov (@ferrine)
    """
    if not isinstance(dims, (list, tuple)):
        dims = (dims,)
    *hierarchy, dim = dims
    input_sigma = pt.as_tensor(input_sigma)
    output_sigma = pt.as_tensor(output_sigma)
    with pm.Model(name) as model:
        if variables_importance is not None and len(model.coords[dim]) > 1:
            if variance_explained is not None:
                raise TypeError("Can't use variable importance with variance explained")
            phi = pm.Dirichlet("phi", pt.as_tensor(variables_importance), dims=hierarchy + [dim])
        elif variance_explained is not None:
            phi = pt.as_tensor(variance_explained)
        else:
            phi = 1 / len(model.coords[dim])
        if r2_std is not None:
            r2 = pm.Beta("r2", mu=r2, sigma=r2_std, dims=hierarchy)
        if positive_probs_std is not None:
            psi = pm.Beta(
                "psi",
                mu=pt.as_tensor(positive_probs),
                sigma=pt.as_tensor(positive_probs_std),
                dims=hierarchy + [dim],
            )
        else:
            psi = pt.as_tensor(positive_probs)
    beta = _R2D2M2CP_beta(
        name, output_sigma, input_sigma, r2, phi, psi, dims=hierarchy + [dim], centered=centered
    )
    resid_sigma = (1 - r2) ** 0.5 * output_sigma
    return resid_sigma, beta
