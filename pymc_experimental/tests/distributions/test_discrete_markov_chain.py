import numpy as np
import pymc as pm

# general imports
import pytensor.tensor as at
import pytest
from pymc.logprob.utils import ParameterValueError

from pymc_experimental.distributions.timeseries import DiscreteMarkovChain


def test_fail_if_P_not_square():
    P = at.eye(3, 2)
    chain = DiscreteMarkovChain.dist(P=P, steps=3)
    with pytest.raises(ParameterValueError):
        pm.logp(chain, np.zeros((3,))).eval()


def test_fail_if_P_not_valid():
    P = at.zeros((3, 3))
    chain = DiscreteMarkovChain.dist(P=P, steps=3)
    with pytest.raises(ParameterValueError):
        pm.logp(chain, np.zeros((3,))).eval()


def test_logp_with_default_init_dist():
    P = at.full((3, 3), 1 / 3)

    chain = DiscreteMarkovChain.dist(P=P, steps=3)

    logp = pm.logp(chain, [0, 0, 0]).eval()
    assert logp == np.log((1 / 3) ** 3)


def test_logp_with_user_defined_init_dist():
    P = at.full((3, 3), 1 / 3)
    x0 = pm.Categorical.dist(p=[0.2, 0.6, 0.2])
    chain = DiscreteMarkovChain.dist(P=P, init_dist=x0, steps=3)

    logp = pm.logp(chain, [0, 0, 0]).eval()
    assert logp == np.log(0.2 * (1 / 3) ** 2)


def test_define_steps_via_shape_arg():
    P = at.full((3, 3), 1 / 3)
    chain = DiscreteMarkovChain.dist(P=P, shape=(3,))

    assert chain.eval().shape[0] == 3
