import pytest

import numpy as np
import scipy.special as special
import pymc_experimental as pmx
from pymc.tests.test_distributions_random import (
	BaseTestDistributionRandom, 
	seeded_scipy_distribution_builder,
)


class TestGeneralizedGamma(BaseTestDistributionRandom):
    pymc_dist = pmx.GeneralizedGamma
    pymc_dist_params = {"alpha": 2.0, "p": 3.0, "lambd": 5.0}
    expected_rv_op_params = {"alpha": 2.0, "p": 3.0, "lambd": 5.0}
    reference_dist_params = {"a": 2.0 / 3.0, "c": 3.0, "scale": 5.0}
    reference_dist = seeded_scipy_distribution_builder("gengamma")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
    ]

