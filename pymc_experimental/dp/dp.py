#   Copyright 2020 The PyMC Developers
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

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.model import modelcontext

__all__ = ["DirichletProcess"]


def DirichletProcess(name, alpha, base_dist, K, observed=None, sbw_name=None, atoms_name=None):
    r"""
    Truncated Dirichlet Process for Bayesian Nonparametric Density Modelling

    Parameters
    ----------
    alpha: tensor_like of float
        Scale concentration parameter (alpha > 0) specifying the size of "sticks", or generated
        weights, from the stick-breaking process. Ideally, alpha should have a prior and not be
        a fixed constant.
    base_dist: single batched distribution
        The base distribution for a Dirichlet Process. `base_dist` must have shape (K + 1,).
    K: int
        The truncation parameter for the number of components of the Dirichlet Process Mixture.
        The Goldilocks Principle should be used in selecting an appropriate value of K: not too
        low to capture all possible clusters and not too high to induce a heavy computational
        burden for sampling.
    """
    if sbw_name is None:
        sbw_name = "sbw"

    if atoms_name is None:
        atoms_name = "atoms"

    if observed is not None:
        observed = np.asarray(observed)

        if observed.ndim > 1:
            raise ValueError("Multi-dimensional Dirichlet Processes are not " "yet supported.")

        N = observed.shape[0]

    try:
        modelcontext(None)
    except TypeError:
        raise ValueError(
            "PyMC Dirichlet Processes are only available under a pm.Model() context manager."
        )

    sbw = pm.StickBreakingWeights(sbw_name, alpha, K)

    if observed is None:
        return sbw, pm.Deterministic(atoms_name, base_dist)

    """
    idx samples a new atom from `base_dist` with probability alpha/(alpha + N)
    and an existing atom from `observed` with probability N/(alpha + N).

    If a new atom is not sampled, an atom from `observed` is sampled uniformly.
    """
    idx = pm.Bernoulli("idx", p=alpha / (alpha + N), shape=(K + 1,))
    atom_selection = pm.Categorical("atom_selection", p=[1 / N] * N, shape=(K + 1,))

    atoms = pm.Deterministic(
        atoms_name,
        var=pt.stack([pt.constant(observed)[atom_selection], base_dist], axis=-1)[
            pt.arange(K + 1), idx
        ],
    )

    return sbw, atoms
