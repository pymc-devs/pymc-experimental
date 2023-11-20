import numpy as np
import pymc as pm

import pymc_experimental as pmx


def test_sample():
    """Compare standard PyMC `with pm.Model()` context API against `pmx.model` decorator
    and a functional syntax. Checks whether the kwarge `coords` can be passed.
    """
    coords = {"obs": ["a", "b"]}
    kwargs = {"draws": 50, "tune": 50, "chains": 1, "random_seed": 1}

    with pm.Model(coords=coords) as model:
        pm.Normal("x", 0.0, 1.0, dims="obs")
        idata = pm.sample(**kwargs)

    @pmx.model(coords=coords)
    def model_wrapped():
        pm.Normal("x", 0.0, 1.0, dims="obs")

    mw = model_wrapped()
    idata_wrapped = pm.sample(model=mw, **kwargs)

    np.testing.assert_array_equal(idata.posterior.x, idata_wrapped.posterior.x)
