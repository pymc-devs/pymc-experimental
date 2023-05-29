import numpy as np
from numpy.testing import assert_allclose

from pymc_experimental.statespace.utils.simulation import (
    numba_mvn_draws,
    simulate_statespace,
)
from pymc_experimental.tests.statespace.utilities.test_helpers import make_test_inputs


def test_numba_mvn_draws():
    cov = np.random.normal(size=(2, 2))
    cov = cov @ cov.T
    draws = numba_mvn_draws(mu=np.zeros((2, 1_000_000)), cov=cov)

    assert_allclose(np.cov(draws), cov, atol=0.01, rtol=0.01)


def test_simulate_statespace():
    data, a0, P0, T, Z, R, H, Q = make_test_inputs(3, 5, 1, 100)
    simulated_states, simulated_obs = simulate_statespace(T, Z, R, H, Q, n_steps=100)

    assert simulated_states.shape == (100, 5)
    assert simulated_obs.shape == (100, 3)


def test_simulate_statespace_with_x0():
    data, a0, P0, T, Z, R, H, Q = make_test_inputs(3, 5, 1, 100)
    simulated_states, simulated_obs = simulate_statespace(
        T, Z, R, H, Q, n_steps=100, x0=a0.squeeze()
    )

    assert simulated_states.shape == (100, 5)
    assert simulated_obs.shape == (100, 3)
    assert np.all(simulated_states[0] == a0)
    assert np.all(simulated_obs[0] == Z @ a0)


def test_simulate_statespace_no_obs_noise():
    data, a0, P0, T, Z, R, H, Q = make_test_inputs(3, 5, 1, 100)
    H = np.zeros_like(H)
    simulated_states, simulated_obs = simulate_statespace(
        T, Z, R, H, Q, n_steps=100, x0=a0.squeeze()
    )

    assert simulated_states.shape == (100, 5)
    assert simulated_obs.shape == (100, 3)


def test_conditional_simulation():
    # TODO: Write me
    pass


def test_unconditional_simulation():
    # TODO: Write me
    pass
