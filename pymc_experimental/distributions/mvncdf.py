import aesara
from aesara.graph.basic import Apply
from aesara.graph.op import Op
from aesara.tensor.type import TensorType

# Op calculates mvncdf


def conditional_covariance(Sigma, mu, conditioned, conditioned_values):

    """
    Sigma: d x d covariance matrix
    mu: mean, array length d
    conditioned: array of dimensions to condition on
    conditioned_values: array of conditional values equal in length to `conditioned`
    """

    target = (1 - conditioned).astype(bool)

    Sigma_22 = Sigma[conditioned][:, conditioned]
    Sigma_21 = Sigma[conditioned][:, target]
    Sigma_12 = Sigma[target][:, conditioned]
    Sigma_11 = Sigma[target][:, target]

    Sigma_cond = Sigma_11 - np.matmul(np.matmul(Sigma_12, np.linalg.inv(Sigma_22)), Sigma_21)
    mean_cond = np.delete(mu, conditioned) + np.matmul(Sigma_12, np.linalg.inv(Sigma_22)).dot(
        conditioned_values - mu[conditioned]
    )

    return Sigma_cond, mean_cond


class Mvncdf(Op):
    __props__ = ()

    def make_node(self, upper, mu, cov):
        upper = at.as_tensor_variable(upper)
        mu = at.as_tensor_variable(mu)
        cov = at.as_tensor_variable(cov)

        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        # here do the integration
        return scipy.stats.multivariate_normal.logcdf(upper, mu, cov)

    def infer_shape(self, fgraph, node, i0_shapes):
        return

    def grad(self, inputs, output_grads):

        grad_ = []
        for i in range(len(mu)):
            grad_.append(conditional_covariance(cov, mu, i, upper[i]))

        return np.array(grad_)

    def R_op(self, inputs, eval_points):
        # R_op can receive None as eval_points.
        # That mean there is no diferientiable path through that input
        # If this imply that you cannot compute some outputs,
        # return None for those.
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)
