import aesara
import numpy as np
import scipy.interpolate
from aesara.graph.op import Op, Apply
import aesara.tensor as at
import aesara.sparse


def numpy_bspline_basis(eval_points: np.ndarray, k: int, degree=3):
    k_knots = k + degree + 1
    knots = np.linspace(0, 1, k_knots - 2 * degree)
    knots = np.r_[[0] * degree, knots, [1] * degree]
    basis_funcs = scipy.interpolate.BSpline(knots, np.eye(k), k=degree)
    Bx = basis_funcs(eval_points).astype(eval_points.dtype)
    return Bx


class BSplineBasis(Op):
    __props__ = ("sparse",)

    def __init__(self, sparse=True) -> None:
        super().__init__()
        if not isinstance(sparse, bool):
            raise TypeError("sparse should be True or False")
        self.sparse = sparse

    def make_node(self, *inputs) -> Apply:
        eval_points, k, d = map(at.as_tensor, inputs)
        if not eval_points.ndim == 1 and np.issubdtype(eval_points.dtype, np.floating):
            raise TypeError("eval_points should be a vector of floats")
        if not k.type in at.int_types:
            raise TypeError("k should be integer")
        if not d.type in at.int_types:
            raise TypeError("degree should be integer")
        if self.sparse:
            out_type = aesara.sparse.SparseTensorType("csr", eval_points.dtype)()
        else:
            out_type = aesara.tensor.matrix(dtype=eval_points.dtype)
        return Apply(self, [eval_points, k, d], [out_type])

    def perform(self, node, inputs, output_storage, params=None) -> None:
        eval_points, k, d = inputs
        Bx = numpy_bspline_basis(eval_points, int(k), int(d))
        if self.sparse:
            Bx = scipy.sparse.csr_matrix(Bx, dtype=eval_points.dtype)
        output_storage[0][0] = Bx

    def infer_shape(self, fgraph, node, ins_shapes):
        return [(node.inputs[0].shape[0], node.inputs[1])]


def bspline_basis(n, k, degree=3, dtype=None, sparse=True):
    dtype = dtype or aesara.config.floatX
    eval_points = np.linspace(0, 1, n, dtype=dtype)
    return BSplineBasis(sparse=sparse)(eval_points, k, degree)


def bspline_interpolation(x, *, n=None, eval_points=None, degree=3, sparse=True):
    """Interpolate sparse grid to dense grid using bsplines.

    Parameters
    ----------
    x : Variable
        Input Variable to interpolate.
        0th coordinate assumed to be mapped regularly on [0, 1] interval
    n : int (optional)
        Resolution of interpolation
    eval_points : vector (optional)
        Custom eval points in [0, 1] interval (or scaled properly using min/max scaling)
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
    if n is not None and eval_points is not None:
        raise ValueError("Please provide one of n or eval_points")
    elif n is not None:
        eval_points = np.linspace(0, 1, n, dtype=x.dtype)
    elif eval_points is None:
        raise ValueError("Please provide one of n or eval_points")
    basis = BSplineBasis(sparse=sparse)(eval_points, x.shape[0], degree)
    if sparse:
        return aesara.sparse.dot(basis, x)
    else:
        return aesara.tensor.dot(basis, x)
