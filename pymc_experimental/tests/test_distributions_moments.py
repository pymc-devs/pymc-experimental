import pytest

import numpy as np
import scipy.special as special
import pymc as pm
from pymc import Model
from pymc.tests.test_distributions_moments import assert_moment_is_expected
from pymc_experimental.distributions import GeneralizedGamma 


@pytest.mark.parametrize(
    "alpha, p, lambd, size, expected",
    [
        (1, 1, 2, None, 2),
        (1, 1, 2, 5, np.full(5, 2)),
        (1, 1, np.arange(1, 6), None, np.arange(1, 6)),
        (
            np.arange(1, 6),
            2 * np.arange(1, 6),
            10,
            (2, 5),
            np.full(
                (2, 5),
                10
                * special.gamma((np.arange(1, 6) + 1) / (np.arange(1, 6) * 2))
                / special.gamma(np.arange(1, 6) / (np.arange(1, 6) * 2)),
            ),
        ),
    ],
)
def test_generalized_gamma_moment(alpha, p, lambd, size, expected):
    with Model() as model:
        GeneralizedGamma("x", alpha=alpha, p=p, lambd=lambd, size=size)
    assert_moment_is_expected(model, expected)

