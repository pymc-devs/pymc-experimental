import pytensor
import pytensor.tensor as pt
import scipy
from pytensor.tensor import TensorVariable, as_tensor_variable
from pytensor.tensor.nlinalg import matrix_dot
from pytensor.tensor.slinalg import solve_discrete_lyapunov

floatX = pytensor.config.floatX


class SolveDiscreteARE(pt.Op):
    __props__ = ("enforce_Q_symmetric",)

    def __init__(self, enforce_Q_symmetric=False):
        self.enforce_Q_symmetric = enforce_Q_symmetric

    def make_node(self, A, B, Q, R):
        A = as_tensor_variable(A)
        B = as_tensor_variable(B)
        Q = as_tensor_variable(Q)
        R = as_tensor_variable(R)

        out_dtype = pytensor.scalar.upcast(A.dtype, B.dtype, Q.dtype, R.dtype)
        X = pytensor.tensor.matrix(dtype=out_dtype)

        return pytensor.graph.basic.Apply(self, [A, B, Q, R], [X])

    def perform(self, node, inputs, output_storage):
        A, B, Q, R = inputs
        X = output_storage[0]

        if self.enforce_Q_symmetric:
            Q = 0.5 * (Q + Q.T)
        X[0] = scipy.linalg.solve_discrete_are(A, B, Q, R).astype(floatX)

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]

    def grad(self, inputs, output_grads):
        # Gradient computations come from Kao and Hennequin (2020), https://arxiv.org/pdf/2011.11430.pdf
        A, B, Q, R = inputs

        (dX,) = output_grads
        X = self(A, B, Q, R)

        K_inner = R + pt.linalg.matrix_dot(B.T, X, B)
        K_inner_inv = pt.linalg.solve(K_inner, pt.eye(R.shape[0]))
        K = matrix_dot(K_inner_inv, B.T, X, A)

        A_tilde = A - B.dot(K)

        dX_symm = 0.5 * (dX + dX.T)
        S = solve_discrete_lyapunov(A_tilde, dX_symm).astype(floatX)

        A_bar = 2 * matrix_dot(X, A_tilde, S)
        B_bar = -2 * matrix_dot(X, A_tilde, S, K.T)
        Q_bar = S
        R_bar = matrix_dot(K, S, K.T)

        return [A_bar, B_bar, Q_bar, R_bar]


def solve_discrete_are(A, B, Q, R, enforce_Q_symmetric=False) -> TensorVariable:
    """
    Solve the discrete Algebraic Riccati equation :math:`A^TXA - X - (A^TXB)(R + B^TXB)^{-1}(B^TXA) + Q = 0`.
    Parameters
    ----------
    A: ArrayLike
        Square matrix of shape M x M
    B: ArrayLike
        Square matrix of shape M x M
    Q: ArrayLike
        Symmetric square matrix of shape M x M
    R: ArrayLike
        Square matrix of shape N x N
    enforce_Q_symmetric: bool
        If True, the provided Q matrix is transformed to 0.5 * (Q + Q.T) to ensure symmetry

    Returns
    -------
    X: pt.matrix
        Square matrix of shape M x M, representing the solution to the DARE
    """

    return SolveDiscreteARE(enforce_Q_symmetric)(A, B, Q, R)
