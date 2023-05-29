import os
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose

from pymc_experimental.statespace import BayesianVARMAX

ROOT = Path(__file__).parent.absolute()
sys.path.append(ROOT)


@pytest.fixture
def data():
    return pd.read_csv(
        os.path.join(ROOT, "test_data/statsmodels_macrodata_processed.csv"), index_col=0
    )


ps = [0, 1, 2, 3]
qs = [0, 1, 2, 3]
orders = list(product(ps, qs))[1:]
ids = [f"p={x[0]}, q={x[1]}" for x in orders]


@pytest.mark.parametrize("order", orders, ids=ids)
@pytest.mark.parametrize("matrix", ["transition", "selection", "state_cov", "obs_cov", "design"])
def test_VARMAX_init_matches_statsmodels(data, order, matrix):
    p, q = order

    mod = BayesianVARMAX(data, order=(p, q), verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm_var = sm.tsa.VARMAX(data, order=(p, q))

    assert_allclose(mod.ssm[matrix].eval(), sm_var.ssm[matrix])


@pytest.mark.parametrize("order", orders, ids=ids)
@pytest.mark.parametrize("var", ["AR", "MA", "state_cov"])
def test_VARMAX_param_counts_match_statsmodels(data, order, var):
    p, q = order

    mod = BayesianVARMAX(data, order=(p, q), verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm_var = sm.tsa.VARMAX(data, order=(p, q))

    count = mod.param_counts[var]
    if var == "state_cov":
        # Statsmodels only counts the lower triangle
        count = mod.k_posdef * (mod.k_posdef - 1)
    assert count == sm_var.parameters[var.lower()]


@pytest.mark.parametrize("order", orders, ids=ids)
@pytest.mark.parametrize("matrix", ["transition", "selection", "state_cov", "obs_cov", "design"])
def test_VARMAX_update_matches_statsmodels(data, order, matrix):
    p, q = order

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm_var = sm.tsa.VARMAX(data, order=(p, q))

    param_counts = [None] + np.cumsum(list(sm_var.parameters.values())).tolist()
    param_slices = [slice(a, b) for a, b in zip(param_counts[:-1], param_counts[1:])]
    param_lists = [trend, ar, ma, reg, state_cov, obs_cov] = [
        sm_var.param_names[idx] for idx in param_slices
    ]
    param_d = {
        k: np.random.normal(scale=0.1) ** 2 for param_list in param_lists for k in param_list
    }

    res = sm_var.fit_constrained(param_d)

    mod = BayesianVARMAX(data, order=(p, q), verbose=False, measurement_error=False)

    with pm.Model() as pm_mod:
        x0 = pm.Deterministic("x0", pt.zeros(mod.k_states))
        ma_params = pm.Deterministic(
            "ma_params", pt.as_tensor_variable(np.array([param_d[var] for var in ma]))
        )
        ar_params = pm.Deterministic(
            "ar_params", pt.as_tensor_variable(np.array([param_d[var] for var in ar]))
        )
        state_chol = np.zeros((mod.k_posdef, mod.k_posdef))
        state_chol[np.tril_indices(mod.k_posdef)] = np.array([param_d[var] for var in state_cov])
        state_cov = pm.Deterministic("state_cov", pt.as_tensor_variable(state_chol @ state_chol.T))
        mod.build_statespace_graph()

    assert_allclose(mod.ssm[matrix].eval(), sm_var.ssm[matrix])
