import numpy as np
import pymc as pm
import pytest

import pymc_experimental as pmx


def test_pathfinder():
    # Data of the Eight Schools Model
    J = 8
    y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

    with pm.Model() as model:

        mu = pm.Normal("mu", mu=0.0, sigma=10.0)
        tau = pm.HalfCauchy("tau", 5.0)

        theta = pm.Normal("theta", mu=0, sigma=1, shape=J)
        theta_1 = mu + tau * theta
        obs = pm.Normal("obs", mu=theta, sigma=sigma, shape=J, observed=y)

        idata = pmx.inference.fit_pathfinder()
