import numpy as np
import pymc as pm

# general imports
import pytensor.tensor as at
import pytest
from pymc.logprob.utils import ParameterValueError

from pymc_experimental.distributions.timeseries import DiscreteMarkovChain


def test_fail_if_P_not_square():
    P = at.eye(3, 2)
    chain = DiscreteMarkovChain.dist(P=P, x0=0, steps=3)
    with pytest.raises(ParameterValueError):
        pm.logp(chain, np.zeros((3,))).eval()


def test_fail_if_P_not_valid():
    P = at.zeros((3, 3))
    chain = DiscreteMarkovChain.dist(P=P, x0=0, steps=3)
    with pytest.raises(ParameterValueError):
        pm.logp(chain, np.zeros((3,))).eval()
