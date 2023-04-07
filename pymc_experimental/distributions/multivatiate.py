from typing import Sequence, Union

import pymc as pm
import pytensor.tensor as pt


def _psivar2musigma(psi: pt.TensorVariable, var: pt.TensorVariable):
    pi = pt.erfinv(2 * psi - 1)
    f = (1 / (2 * pi**2 + 1)) ** 0.5
    sigma = pt.expand_dims(var, -1) ** 0.5 * f
    mu = sigma * pi * 2**0.5
    return mu, sigma


def _R2D2M2CP_beta(
    name: str,
    variance: pt.TensorVariable,
    param_sigma: pt.TensorVariable,
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
    variance: tensor
        standard deviation of the outcome
    param_sigma: tensor
        standard deviation of the explanatory variables
    r2: tensor
        expected R2 for the linear regression
    phi: tensor
        variance weights that sums up to 1
    psi: tensor
        probability of a coefficients to be positive
    """
    tau2 = r2 / (1 - r2)
    explained_variance = phi * tau2 * pt.expand_dims(variance, -1)
    mu_param, std_param = _psivar2musigma(psi, explained_variance)
    if not centered:
        with pm.Model(name):
            raw = pm.Normal("raw", dims=dims)
        beta = pm.Deterministic(name, (raw * std_param + mu_param) / param_sigma, dims=dims)
    else:
        beta = pm.Normal(name, mu_param / param_sigma, std_param / param_sigma, dims=dims)
    return beta


def R2D2M2CP(
    name,
    variance,
    param_sigma,
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
    variance : tensor
        Output variance
    param_sigma : tensor
        Input standard deviation
    dims : Union[str, Sequence[str]]
        Dims for the distribution
    r2 : tensor
        :math:`R^2` estimate
    variables_importance : tensor, optional
        Optional estimate for variables importance, positive, , by default None
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
    residual_variance, coefficients
        Output variance is split in residual variance and explained variance.

    Raises
    ------
    TypeError
        If parametrization is wrong.

    Notes
    -----
    - ``(R2D2M2)``CP is taken from https://arxiv.org/abs/2208.07132
    - R2D2M2``(CP)``, (Correlation Probability) is proposed and implemented by Max Kochurov (@ferrine)
    """
    if not isinstance(dims, (list, tuple)):
        dims = (dims,)
    *hierarchy, dim = dims
    param_sigma = pt.as_tensor(param_sigma)
    variance = pt.as_tensor(variance)
    with pm.Model(name) as model:
        if variables_importance is not None and len(model.coords[dim]) > 1:
            if variance_explained is not None:
                raise TypeError("Can't use variable importance with variance explained")
            phi = pm.Dirichlet("phi", pt.as_tensor(variables_importance), dims=hierarchy + [dim])
        elif variance_explained:
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
        name, variance, param_sigma, r2, phi, psi, dims=hierarchy + [dim], centered=centered
    )
    variance_resid = (1 - r2) * variance
    return variance_resid, beta
