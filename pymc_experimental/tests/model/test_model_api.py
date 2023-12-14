import numpy as np
import pymc as pm

import pymc_experimental as pmx


def test_logp():
    """Compare standard PyMC `with pm.Model()` context API against `pmx.model` decorator
    and a functional syntax. Checks whether the kwarg `coords` can be passed.
    """
    coords = {"obs": ["a", "b"]}

    with pm.Model(coords=coords) as model:
        pm.Normal("x", 0.0, 1.0, dims="obs")

    @pmx.as_model(coords=coords)
    def model_wrapped():
        pm.Normal("x", 0.0, 1.0, dims="obs")

    mw = model_wrapped()

    @pmx.as_model()
    def model_wrapped2():
        pm.Normal("x", 0.0, 1.0, dims="obs")

    mw2 = model_wrapped2(coords=coords)

    np.testing.assert_equal(model.point_logps(), mw.point_logps())
    np.testing.assert_equal(mw.point_logps(), mw2.point_logps())
