import aesara
import numpy as np
import scipy.interpolate
from aesara.graph.op import Op, Apply
import aesara.tensor as at
import aesara.sparse


def bspline_basis(n, k, degree=3, dtype=np.float32):
    k_knots = k + degree + 1
    knots = np.linspace(0, 1, k_knots - 2 * degree, dtype=dtype)
    knots = np.r_[[0] * degree, knots, [1] * degree]
    basis_funcs = scipy.interpolate.BSpline(knots, np.eye(k, dtype=dtype), k=degree)
    Bx = basis_funcs(np.linspace(0, 1, n, dtype=dtype))
    return scipy.sparse.csr_matrix(Bx, dtype=dtype)


class BSplineBasis(Op):
    __props__ = ("dtype",)

    def __init__(self, dtype) -> None:
        super().__init__()
        dtype = np.dtype(dtype)
        assert np.issubdtype(dtype, np.floating)
        self.dtype = dtype

    def make_node(self, *inputs) -> Apply:
        n, k, d = map(at.as_tensor, inputs)
        assert k.type in at.int_types, "k should be integer"
        assert n.type in at.int_types, "n should be integer"
        assert d.type in at.int_types, "d should be integer"
        return Apply(self, [n, k, d], [aesara.sparse.SparseTensorType("csr", self.dtype)()])

    def perform(self, node, inputs, output_storage, params=None) -> None:
        n, k, d = inputs
        output_storage[0][0] = bspline_basis(n, k, d, dtype=self.dtype)

    def infer_shape(self, fgraph, node, ins_shapes):
        return [(node.inputs[0], node.inputs[1])]
