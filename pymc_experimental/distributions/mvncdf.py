import aesara
from aesara.graph.basic import Apply
from aesara.graph.op import Op
from aesara.tensor.type import TensorType
from aesara.tensor.random.basic import MvNormalRV
from aeppl.logprob import _logcdf
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

    inv = pm.math.matrix_inverse(Sigma_22)
    Sigma_cond = Sigma_11 - pm.math.matrix_dot(Sigma_12, inv, Sigma_21)
    mean_cond = pm.math.matrix_dot(Sigma_12, inv, conditioned_values)

    return Sigma_cond, mean_cond


class Mvncdf(Op):
    __props__ = ()

    def make_node(self, value, mu, cov):
        value = at.as_tensor_variable(value)
        mu = at.as_tensor_variable(mu)
        cov = at.as_tensor_variable(cov)

        return Apply(self, [value], [value.type()])

    def perform(self, node, inputs, output_storage):
        # here do the integration
        value, mu, cov = inputs
        output_storage[0] = scipy.stats.multivariate_normal.logcdf(value, mu, cov)

        return output_storage

    def infer_shape(self, fgraph, node, i0_shapes):
        return i0_shapes[0]

    def grad(self, inputs, output_grads):
        value, mu, cov = inputs
        grad_ = []
        for i in range(len(mu)):
            grad_.append(conditional_covariance(cov, mu, i, value[i]))

        return np.array(grad_)*output_grads[0]

    def R_op(self, inputs, eval_points):
        # R_op can receive None as eval_points.
        # That mean there is no diferientiable path through that input
        # If this imply that you cannot compute some outputs,
        # return None for those.
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)

    

    @_logcdf.register(MvNormalRV)
    def mvnormal_logcdf(value, mu, cov):

        return pm.logcdf(MvNormal.dist(mu, cov), value)