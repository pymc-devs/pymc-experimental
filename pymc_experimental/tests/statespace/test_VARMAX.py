import warnings
from itertools import product

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from pymc.model_graph import fast_eval

from pymc_experimental.statespace import BayesianVARMAX
from pymc_experimental.tests.statespace.utilities.shared_fixtures import (  # pylint: disable=unused-import
    rng,
)

floatX = pytensor.config.floatX


@pytest.fixture
def data():
    df = pd.read_csv(
        "pymc_experimental/tests/statespace/test_data/statsmodels_macrodata_processed.csv",
        index_col=0,
        parse_dates=True,
    ).astype(floatX)
    df.index.freq = df.index.inferred_freq
    return df


ps = [0, 1, 2, 3]
qs = [0, 1, 2, 3]
orders = list(product(ps, qs))[1:]
ids = [f"p={x[0]}, q={x[1]}" for x in orders]


@pytest.mark.parametrize("order", orders, ids=ids)
@pytest.mark.parametrize("matrix", ["transition", "selection", "state_cov", "obs_cov", "design"])
def test_VARMAX_init_matches_statsmodels(data, order, matrix):
    p, q = order

    mod = BayesianVARMAX(
        k_endog=data.shape[1], order=(p, q), verbose=False, stationary_initialization=True
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm_var = sm.tsa.VARMAX(data, order=(p, q))

    assert_allclose(fast_eval(mod.ssm[matrix]), sm_var.ssm[matrix])


@pytest.mark.parametrize("order", orders, ids=ids)
@pytest.mark.parametrize("var", ["AR", "MA", "state_cov"])
def test_VARMAX_param_counts_match_statsmodels(data, order, var):
    p, q = order

    mod = BayesianVARMAX(k_endog=data.shape[1], order=(p, q), verbose=False)
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
def test_VARMAX_update_matches_statsmodels(data, order, matrix, rng):
    p, q = order

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm_var = sm.tsa.VARMAX(data, order=(p, q))

    param_counts = [None] + np.cumsum(list(sm_var.parameters.values())).tolist()
    param_slices = [slice(a, b) for a, b in zip(param_counts[:-1], param_counts[1:])]
    param_lists = [trend, ar, ma, reg, state_cov, obs_cov] = [
        sm_var.param_names[idx] for idx in param_slices
    ]
    param_d = {k: rng.normal(scale=0.1) ** 2 for param_list in param_lists for k in param_list}

    res = sm_var.fit_constrained(param_d)

    mod = BayesianVARMAX(
        k_endog=data.shape[1],
        order=(p, q),
        verbose=False,
        measurement_error=False,
        stationary_initialization=True,
    )

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
        mod.build_statespace_graph(data=data)

    assert_allclose(fast_eval(mod.ssm[matrix]), sm_var.ssm[matrix])
