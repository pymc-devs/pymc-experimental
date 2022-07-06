# import aesara
import numpy as np
import pytest
import scipy.stats as st

# from aesara import tensor as at
# from scipy import special

from pymc_experimental.distributions import (
    GenExtreme,
)

from pymc.model import Model

from pymc.tests.test_distributions_moments import assert_moment_is_expected


@pytest.mark.parametrize(
    "mu, sigma, xi, size, expected",
    [
        (0, 1, 0, None, 0),
        (1, np.arange(1, 4), 0.1, None, np.arange(1, 4) * (1.1 ** -0.1 - 1) / 0.1),
        (np.arange(5), 1, 0.1, None, np.arange(5) + (1.1 ** -0.1 - 1) / 0.1),
        (
            0,
            1,
            np.linspace(-0.2, 0.2, 6),
            None,
            ((1 + np.linspace(-0.2, 0.2, 6)) ** -np.linspace(-0.2, 0.2, 6) - 1)
            / np.linspace(-0.2, 0.2, 6),
        ),
        (1, 2, 0.1, 5, np.full(5, 1 + 2 * (1.1 ** -0.1 - 1) / 0.1)),
        (
            np.arange(6),
            np.arange(1, 7),
            np.linspace(-0.2, 0.2, 6),
            (3, 6),
            np.full(
                (3, 6),
                np.arange(6)
                + np.arange(1, 7)
                * ((1 + np.linspace(-0.2, 0.2, 6)) ** -np.linspace(-0.2, 0.2, 6) - 1)
                / np.linspace(-0.2, 0.2, 6),
            ),
        ),
    ],
)
def test_genextreme_moment(mu, sigma, xi, size, expected):
    with Model() as model:
        GenExtreme("x", mu=mu, sigma=sigma, xi=xi, size=size)
    assert_moment_is_expected(model, expected)
