import warnings
from itertools import product

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose

from pymc_experimental.statespace import BayesianARMA
from pymc_experimental.tests.statespace.utilities.test_helpers import (
    load_nile_test_data,
)

floatX = pytensor.config.floatX


@pytest.fixture
def data():
    return load_nile_test_data()


ps = [0, 1, 2, 3]
qs = [0, 1, 2, 3]
orders = list(product(ps, qs))[1:]
ids = [f"p={x[0]}q={x[1]}" for x in orders]


@pytest.mark.parametrize("order", orders, ids=ids)
@pytest.mark.parametrize("matrix", ["transition", "selection", "state_cov", "obs_cov", "design"])
def test_SARIMAX_init_matches_statsmodels(data, order, matrix):
    p, q = order

    mod = BayesianARMA(order=(p, q), verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm_sarimax = sm.tsa.SARIMAX(data, order=(p, 0, q))

    assert_allclose(mod.ssm[matrix].eval(), sm_sarimax.ssm[matrix])


@pytest.mark.parametrize("order", orders, ids=ids)
@pytest.mark.parametrize("matrix", ["transition", "selection", "state_cov", "obs_cov", "design"])
def test_SARIMAX_update_matches_statsmodels(data, order, matrix):
    p, q = order

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm_sarimax = sm.tsa.SARIMAX(data, order=(p, 0, q))

        param_names = sm_sarimax.param_names
        param_d = {name: np.random.normal(scale=0.1) ** 2 for name in param_names}

        res = sm_sarimax.fit_constrained(param_d)
    mod = BayesianARMA(order=(p, q), verbose=False)

    with pm.Model() as pm_mod:
        x0 = pm.Deterministic("x0", pt.zeros(mod.k_states))
        ma_params = pm.Deterministic(
            "theta",
            pt.as_tensor_variable(np.array([param_d[k] for k in param_d if k.startswith("ma.")])),
        )
        ar_params = pm.Deterministic(
            "rho",
            pt.as_tensor_variable(np.array([param_d[k] for k in param_d if k.startswith("ar.")])),
        )
        state_cov = pm.Deterministic(
            "sigma_state", pt.as_tensor_variable(np.array([[param_d["sigma2"]]]))
        )
        mod.build_statespace_graph(data=data)

    assert_allclose(mod.ssm[matrix].eval(), sm_sarimax.ssm[matrix])
