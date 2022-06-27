import aesara
import numpy as np
import scipy.interpolate
from aesara.graph.op import Op, Apply
import aesara.tensor as at
import aesara.sparse


def numpy_bspline_basis_regular(n, k, degree=3, dtype=np.float32):
    k_knots = k + degree + 1
    knots = np.linspace(0, 1, k_knots - 2 * degree, dtype=dtype)
    knots = np.r_[[0] * degree, knots, [1] * degree]
    basis_funcs = scipy.interpolate.BSpline(knots, np.eye(k, dtype=dtype), k=degree)
    Bx = basis_funcs(np.linspace(0, 1, n, dtype=dtype))
    return scipy.sparse.csr_matrix(Bx, dtype=dtype)


class BSplineBasisRegular(Op):
    __props__ = ("dtype", "sparse")

    def __init__(self, dtype, sparse=True) -> None:
        super().__init__()
        dtype = np.dtype(dtype)
        assert np.issubdtype(dtype, np.floating)
        self.dtype = dtype
        self.sparse = sparse

    def make_node(self, *inputs) -> Apply:
        n, k, d = map(at.as_tensor, inputs)
        assert k.type in at.int_types, "k should be integer"
        assert n.type in at.int_types, "n should be integer"
        assert d.type in at.int_types, "d should be integer"
        if self.sparse:
            out_type = aesara.sparse.SparseTensorType("csr", self.dtype)()
        else:
            out_type = aesara.tensor.matrix(dtype=self.dtype)
        return Apply(self, [n, k, d], [out_type])

    def perform(self, node, inputs, output_storage, params=None) -> None:
        n, k, d = inputs
        Bx = numpy_bspline_basis_regular(int(n), int(k), int(d), dtype=self.dtype)
        if not self.sparse:
            Bx = Bx.todense()
        output_storage[0][0] = Bx

    def infer_shape(self, fgraph, node, ins_shapes):
        return [(node.inputs[0], node.inputs[1])]


def bspline_basis_regular(n, k, degree=3, dtype=None, sparse=True):
    dtype = dtype or aesara.config.floatX
    return BSplineBasisRegular(dtype=dtype, sparse=sparse)(n, k, degree)


def bspline_regular_interpolation(x, *, n, degree=3, sparse=True):
    """Interpolate sparse grid to dense grid using bsplines.

    Parameters
    ----------
    x : Variable
        Input Variable to interpolate
    n : int
        Resolution of interpolation
    degree : int, optional
        BSpline degree, by default 3
    sparse : bool, optional
        Use sparse operation, by default True

    Returns
    -------
    Variable
        The interpolated variable, interpolation is across 0th axis

    Examples
    --------
    >>> import pymc as pm
    >>> import numpy as np
    >>> half_months = np.linspace(0, 365, 12*2)
    >>> with pm.Model(coords=dict(knots_time=half_months, time=np.arange(365))) as model:
    ...     kernel = pm.gp.cov.ExpQuad(1, ls=365/12)
    ...     # ready to define gp (a latent process over parameters)
    ...     gp = pm.gp.gp.Latent(
    ...         cov_func=kernel
    ...     )
    ...     y_knots = gp.prior("y_knots", half_months[:, None], dims="knots_time")
    ...     y = pm.Deterministic(
    ...         "y",
    ...         bspline_regular_interpolation(y_knots, n=365, degree=3),
    ...         dims="time"
    ...     )
    ...     trace = pm.sample_prior_predictive(1)

    Notes
    -----
    Adopted from `BayesAlpha <https://github.com/quantopian/bayesalpha/blob/676f4f194ad20211fd040d3b0c6e82969aafb87e/bayesalpha/dists.py#L97>`_
    where it was written by @aseyboldt
    """
    x = at.as_tensor(x)
    basis = bspline_basis_regular(n, x.shape[0], degree, sparse=sparse, dtype=x.dtype)
    if sparse:
        return aesara.sparse.dot(basis, x)
    else:
        return aesara.tensor.dot(basis, x)
