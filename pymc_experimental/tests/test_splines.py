import aesara
import pymc_experimental as pmx
import aesara.tensor as at
import numpy as np
import pytest


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sparse", [True, False])
def test_spline_construction(dtype, sparse):
    np_out = pmx.utils.spline.numpy_bspline_basis_regular(20, 10, 3, dtype=dtype)
    assert np_out.shape == (20, 10)
    assert np_out.dtype == dtype
    spline_op = pmx.utils.spline.BSplineBasisRegular(dtype, sparse=sparse)
    out = spline_op(at.constant(20), at.constant(10), at.constant(3))
    if not sparse:
        assert isinstance(out.type, at.TensorType)
    else:
        assert isinstance(out.type, aesara.sparse.SparseTensorType)
    B = out.eval()
    if not sparse:
        np.testing.assert_allclose(B, np_out)
    else:
        np.testing.assert_allclose(B.todense(), np_out)
    assert B.shape == (20, 10)


@pytest.mark.parametrize("shape", [(100,), (100, 5)])
@pytest.mark.parametrize("sparse", [True, False])
def test_interpolation_api(shape, sparse):
    x = np.random.randn(*shape)
    yt = pmx.utils.spline.bspline_regular_interpolation(x, n=1000, sparse=sparse)
    y = yt.eval()
    assert y.shape == (1000, *shape[1:])


@pytest.mark.parametrize(
    "params",
    [
        (dict(dtype=int), (), "bspline basis should be of float type"),
        (dict(sparse="foo", dtype=float), (), "sparse should be True or False"),
        (dict(dtype=float), (100, 30, 0.5), "d should be integer"),
        (dict(dtype=float), (100, 30.5, 1), "k should be integer"),
        (dict(dtype=float), (100.5, 30.5, 1), "n should be integer"),
    ],
)
def test_bad_calls(params):
    kw1, arg2, err = params
    with pytest.raises(TypeError, match=err):
        pmx.utils.spline.BSplineBasisRegular(**kw1)(*arg2)
