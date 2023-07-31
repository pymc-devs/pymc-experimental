import warnings
from itertools import product

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from pymc.model_graph import fast_eval

from pymc_experimental.statespace import BayesianARIMA
from pymc_experimental.tests.statespace.utilities.test_helpers import (
    load_nile_test_data,
)

floatX = pytensor.config.floatX


@pytest.fixture
def data():
    return load_nile_test_data()


@pytest.fixture
def rng():
    return np.random.default_rng(1337)


ps = [0, 1, 2, 3]
ds = [0, 1, 2]
qs = [0, 1, 2, 3]
orders = list(product(ps, ds, qs))[1:]
ids = [f"p={x[0]}d={x[1]}q={x[2]}" for x in orders]


@pytest.mark.parametrize("order", orders, ids=ids)
@pytest.mark.parametrize("matrix", ["transition", "selection", "state_cov", "obs_cov", "design"])
def test_SARIMAX_init_matches_statsmodels(data, order, matrix):
    p, d, q = order

    mod = BayesianARIMA(order=(p, d, q), verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm_sarimax = sm.tsa.SARIMAX(data, order=(p, d, q))

    assert_allclose(fast_eval(mod.ssm[matrix]), sm_sarimax.ssm[matrix])


@pytest.mark.parametrize("order", orders, ids=ids)
@pytest.mark.parametrize("matrix", ["transition", "selection", "state_cov", "obs_cov", "design"])
def test_SARIMAX_update_matches_statsmodels(data, order, matrix, rng):
    p, d, q = order

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm_sarimax = sm.tsa.SARIMAX(data, order=(p, d, q))

        param_names = sm_sarimax.param_names
        param_d = {name: rng.normal(scale=0.1) ** 2 for name in param_names}

        res = sm_sarimax.fit_constrained(param_d)
    mod = BayesianARIMA(order=(p, d, q), verbose=False)

    with pm.Model() as pm_mod:
        x0 = pm.Deterministic("x0", pt.zeros(mod.k_states))
        ma_params = pm.Deterministic(
            "ma_params",
            pt.as_tensor_variable(np.array([param_d[k] for k in param_d if k.startswith("ma.")])),
        )
        ar_params = pm.Deterministic(
            "ar_params",
            pt.as_tensor_variable(np.array([param_d[k] for k in param_d if k.startswith("ar.")])),
        )
        state_cov = pm.Deterministic(
            "sigma_state", pt.as_tensor_variable(np.array([[param_d["sigma2"]]]))
        )
        mod.build_statespace_graph(data=data)

    assert_allclose(fast_eval(mod.ssm[matrix]), sm_sarimax.ssm[matrix])
