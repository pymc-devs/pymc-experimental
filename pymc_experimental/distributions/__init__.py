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

# coding: utf-8
"""
Experimental probability distributions for stochastic nodes in PyMC.
"""

from pymc_experimental.distributions.continuous import Chi, GenExtreme
from pymc_experimental.distributions.discrete import GeneralizedPoisson
from pymc_experimental.distributions.histogram_utils import histogram_approximation
from pymc_experimental.distributions.multivariate import R2D2M2CP
from pymc_experimental.distributions.timeseries import DiscreteMarkovChain

__all__ = [
    "DiscreteMarkovChain",
    "GeneralizedPoisson",
    "GenExtreme",
    "R2D2M2CP",
    "histogram_approximation",
    "Chi",
]
