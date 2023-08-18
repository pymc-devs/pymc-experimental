import sys
import unittest

import numpy as np
import pytensor
from numpy.testing import assert_allclose
from pytensor.configdefaults import config
from pytensor.gradient import verify_grad as orig_verify_grad

from pymc_experimental.statespace.utils.pytensor_scipy import (
    SolveDiscreteARE,
    solve_discrete_are,
)

floatX = pytensor.config.floatX
solve_discrete_are_enforce = SolveDiscreteARE(enforce_Q_symmetric=True)


def fetch_seed(pseed=None):
    """
    Copied from pytensor.test.unittest_tools
    """

    seed = pseed or config.unittests__rseed
    if seed == "random":
        seed = None

    try:
        if seed:
            seed = int(seed)
        else:
            seed = None
    except ValueError:
        print(
            ("Error: config.unittests__rseed contains " "invalid seed, using None instead"),
            file=sys.stderr,
        )
        seed = None

    return seed


def verify_grad(op, pt, n_tests=2, rng=None, *args, **kwargs):
    """
    Copied from pytensor.test.unittest_tools
    """
    if rng is None:
        rng = np.random.default_rng(fetch_seed())

    # TODO: Needed to increase tolerance for certain tests when migrating to
    # Generators from RandomStates. Caused flaky test failures. Needs further investigation
    if "rel_tol" not in kwargs:
        kwargs["rel_tol"] = 0.05
    if "abs_tol" not in kwargs:
        kwargs["abs_tol"] = 0.05
    orig_verify_grad(op, pt, n_tests, rng, *args, **kwargs)


class TestSolveDiscreteARE(unittest.TestCase):
    def test_forward(self):
        # TEST CASE 4 : darex #1 -- taken from Scipy tests
        a, b, q, r = (
            np.array([[4, 3], [-4.5, -3.5]], dtype=floatX),
            np.array([[1], [-1]], dtype=floatX),
            np.array([[9, 6], [6, 4]], dtype=floatX),
            np.array([[1]], dtype=floatX),
        )
        a, b, q, r = (x.astype(floatX) for x in [a, b, q, r])

        x = solve_discrete_are(a, b, q, r).eval()
        res = a.T.dot(x.dot(a)) - x + q
        res -= (
            a.conj()
            .T.dot(x.dot(b))
            .dot(np.linalg.solve(r + b.conj().T.dot(x.dot(b)), b.T).dot(x.dot(a)))
        )

        atol = 1e-4 if floatX == "float32" else 1e-12
        assert_allclose(res, np.zeros_like(res), atol=atol)

    def test_backward(self):
        a, b, q, r = (
            np.array([[4, 3], [-4.5, -3.5]], dtype=floatX),
            np.array([[1], [-1]], dtype=floatX),
            np.array([[9, 6], [6, 4]], dtype=floatX),
            np.array([[1]], dtype=floatX),
        )
        a, b, q, r = (x.astype(floatX) for x in [a, b, q, r])

        rng = np.random.default_rng(fetch_seed())

        # TODO: Is there a "theoretically motivated" value to use here? I pulled 1e-4 out of a hat
        atol = 1e-4 if floatX == "float32" else 1e-12

        verify_grad(solve_discrete_are_enforce, pt=[a, b, q, r], rng=rng, abs_tol=atol)


if __name__ == "__main__":
    unittest.main()
