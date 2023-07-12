import unittest

from pymc_experimental.statespace import BayesianLocalLevel
from pymc_experimental.tests.statespace.utilities.test_helpers import (
    load_nile_test_data,
)

nile = load_nile_test_data()


def test_local_level_model():
    mod = BayesianLocalLevel(data=nile.values)


if __name__ == "__main__":
    unittest.main()
