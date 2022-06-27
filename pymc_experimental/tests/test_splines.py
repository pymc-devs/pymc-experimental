import pymc_experimental as pmx
import aesara.tensor as at
import numpy as np
import pytest


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_spline_construction(dtype):
    np_out = pmx.gp.spline.bspline_basis(20, 10, 3, dtype=dtype)
    assert np_out.shape == (20, 10)
    assert np_out.dtype == dtype
    spline_op = pmx.gp.spline.BSplineBasis(dtype)
    out = spline_op(at.constant(20), at.constant(10), at.constant(3))

    assert tuple(out.shape.eval()) == (20, 10)
