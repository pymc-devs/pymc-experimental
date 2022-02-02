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
import functools
import itertools
import sys

import aesara
import aesara.tensor as at
import numpy as np
import numpy.random as nr

import pytest
import scipy.stats
import scipy.stats.distributions as sp

from aesara.compile.mode import Mode
from aesara.graph.basic import ancestors
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.var import TensorVariable
from numpy import array, inf, log
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
from scipy import integrate
from scipy.special import erf, logit

import pymc as pm

from pymc.aesaraf import floatX, intX
from pymc-experimental.distributions import (
    GenExtreme,
)
from pymc.distributions.shape_utils import to_tuple
from pymc.math import kronecker
from pymc.model import Deterministic, Model, Point, Potential
from pymc.tests.helpers import select_by_precision
from pymc.vartypes import continuous_types

def test_genextreme(self):
    self.check_logp(
        GenExtreme,
        R,
        {"mu": R, "sigma": Rplus, "xi": Domain([-1, -1, -0.5, 0, 0.5, 1, 1])},
        lambda value, mu, sigma, xi: sp.genextreme.logpdf(value, c=-xi, loc=mu, scale=sigma),
    )
    self.check_logcdf(
        GenExtreme,
        R,
        {"mu": R, "sigma": Rplus, "xi": Domain([-1, -1, -0.5, 0, 0.5, 1, 1])},
        lambda value, mu, sigma, xi: sp.genextreme.logcdf(value, c=-xi, loc=mu, scale=sigma),
    )
        
