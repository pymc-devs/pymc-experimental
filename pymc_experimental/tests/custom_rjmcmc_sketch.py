from step_methods.rjmcmc import *

import functools
import itertools

from scipy.linalg import expm
import scipy
import numpy as np

import pymc as pm
from pymc.aesaraf import compile_pymc

import dill as pickle
import arviz as az


# Since pymc4 seems to be using Aesara (their own version of thano) instead of the old theano
# import aesara_theano_fallback.tensor as tt
import aesara_theano_fallback.tensor as tt


def get_analytical_solution_for_matrix(deltas, ks, initial_values):
    """Solution of the form s[t] = exp(Mt)s[0]"""
    # build matrix M to exponentiate
    M = np.zeros_like(deltas)
    for i, (drow, krow) in enumerate(zip(deltas, ks)):
        for j, (d, k) in enumerate(zip(drow, krow)):
            M[j, i] += d * k
            if i == j:
                # along the diagonal we subtract the column trace
                M[i, j] -= np.sum(deltas[i, ...] * ks[i, ...])

    def f(t):
        return expm(M * t) @ initial_values
    
    return f

def get_vectorized_solution_for_matrix(deltas, ks, initial_values):
    f = get_analytical_solution_for_matrix(deltas, ks, initial_values)

    def function_for_substance(i):
        return lambda t: f(t)[i]

    # Just splitting this into a dictionary for each species cause I'm confused
    return {i:np.vectorize(function_for_substance(i)) for i in range(len(initial_values))}


class LogLike(tt.Op):
    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """

    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        """

        # add inputs as class attributes
        self.loglikelihood = loglike
        self.logpgrad = LogLikeGrad(self.loglikelihood)

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.loglikelihood(theta)

        outputs[0][0] = np.array(logl)  # output the log-likelihood

    def grad(self, inputs, g):
        # TODO test this somehow
        (theta,) = inputs
        return [g[0] * self.logpgrad(theta)]

class LogLikeGrad(tt.Op):
    """
    TODO test this somehow
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """

    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, loglike):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike

    def perform(self, node, inputs, outputs):
        (theta,) = inputs

        outputs[0][0] = scipy.optimize.approx_fprime(theta, self.likelihood, epsilon= np.sqrt(np.finfo(float).eps))



# Generate Data from the true model
timepoints = np.linspace(0, 5, 1000)
true_deltas = np.array([[0, 1],[0, 0]])
true_ks     = np.array([[0, 2],[0, 0]])
initial_values = np.array([5, 0])
true_model = get_vectorized_solution_for_matrix(true_deltas,true_ks, initial_values)
true_data = [true_model[i](timepoints) for i in range(2)]

# Define loglikelihood 
def my_loglike(vec):
    # we'll just hardcode this cause I'm tired

    if (vec[0] != 0 and vec[0] != 1) and (vec[1] != 0 and vec[1] != 1):
        raise(ValueError('Delta parameters can only be one or zero'))

    deltas = np.array([[0, vec[0]],[vec[1], 0]])
    ks = np.array([[0, vec[2]],[vec[3], 0]])
    new_model = get_vectorized_solution_for_matrix(deltas, ks, initial_values)
    simulated_data = [new_model[i](timepoints) for i in range(2)]
    sum = 0
    for i in range(2):
        sum += np.sum(np.log(scipy.stats.norm.pdf(simulated_data[i] - true_data[i])))

    if not np.isfinite(sum):
        raise(ValueError('loglikelihood diverged'))

    return sum

# Generate the likelihood
ndraws = 200
nburn = 50
pymc_model = pm.Model()
theano_loglike = LogLike(lambda x: my_loglike(x))
with pymc_model:
    # Generate priors based on the Supermodel priors
    var_assoc = []
    delta_parameters = []
    k_parameters = []
    for i in range(2):
        for j in range(2):
            if i!=j:
                delta = pm.Bernoulli('delta_{}_{}'.format(i, j), 0.5)
                k = pm.Uniform('k_{}_{}'.format(i, j), lower=0, upper=10)
                var_assoc.append((delta, k))
                delta_parameters.append(delta)
                k_parameters.append(k)
    
    pymc_parameters = [d for d, _ in var_assoc] + [k for _, k in var_assoc]

    theta = tt.as_tensor_variable(pymc_parameters)

    pm.Potential("loglikelihood", theano_loglike(theta))

    

    class Jump(pm.step_methods.arraystep.BlockedStep):
        """ Placeholder class for jumps"""

        # TODO figure out how to generate and gather stats properly
        # generates_stats = True
        # stats_dtypes = [
        #     {
        #         "accept": np.float64
        #         ,"accepted": bool
        #         ,"diverged": bool
        #         , "tune": bool
        #     }
        # ]

        @classmethod
        def get_vars_from_src_dest(self, src, dest):
            return list({k for i, k in enumerate(k_parameters) if src[i] != dest[i]}.union(self.get_deltas_to_flip(src, dest)))

        @classmethod
        def get_deltas_to_flip(self, src, dest):
            return {d for i, d in enumerate(delta_parameters) if src[i] != dest[i]}

        def __new__(cls, src, dest, *args, **kwargs):
            
            # These have to be appended here because BlockedStep does some stuff with it in __new__
            kwargs['vars'] = cls.get_vars_from_src_dest(src, dest)

            return super().__new__(cls, *args, **kwargs)

        def __init__(self, src, dest, p_src_dest, p_dest_src, *args, **kwargs):
            """
            Creates a very simple Jump
            between src and dest configurations with p_src_dest, p_dest_src being the probabilities that this was selected to begin with
            """
            self.src = src
            self.dest = dest

            # uniform scaling (sigma * u) to generate the random var
            self.uniform_scaling = 10

            # Precompute the prior bias in having selected this move type
            self.p_src_dest = p_src_dest
            self.p_dest_src = p_dest_src
            self.diff_logj = np.log(self.p_dest_src) - np.log(self.p_src_dest)

            # Precompute the q term due to variable u generation
            dimension_mismatch = np.sum(dest) - np.sum(src)
            self.q_term = + dimension_mismatch * np.log(self.uniform_scaling)


            self.tune = False
            self.model = pm.modelcontext(None)

            # Generate the dlogp function
            self.rvs = self.get_vars_from_src_dest(src, dest)
            self.vars = pm.inputvars([self.model.rvs_to_values.get(var, var) for var in self.rvs])
            self.logp = self.model.compile_logp()

            # Precompute delta flips
            deltas_to_flip = [str(x) for x in self.get_deltas_to_flip(src, dest)]
            def flip_deltas(point):
                for d in deltas_to_flip:
                    point[d] = 1 - point[d]
            self.flip_deltas = flip_deltas

            def get_forward_transform(rv):
                """Returns a function which performs the forward value_var transform on a scalar"""
                def f(x):
                    return rv.tag.value_var.tag.transform.forward(x, *rv.owner.inputs).eval()
                return f

            # Precompute setting ks
            ks_to_randomize = [(str(k.tag.value_var), get_forward_transform(k)) for i, k in enumerate(k_parameters) if self.dest[i] == 1 and self.src[i] == 0]
            def set_new_ks(point):
                for k, transform in ks_to_randomize:
                    point[k] = transform(np.random.random())
            self.set_new_ks = set_new_ks

        def step(self, point):
            """
            point is a dictionary of {str(value_var): array(x)}
            the transformation on the varlue variables is reversed for some of my computations
            and then applied again
            """
            new_point = {x:y for x, y in point.items()}

            # stats = {}

            # discrete parameter set
            self.flip_deltas(new_point)

            # Continous parameter set 
            self.set_new_ks(new_point)

            # The general RJMCMC kernel will have to provide this object with it
            log_acceptance_fraction = self.logp(new_point) - self.logp(point) + self.diff_logj + self.q_term

            if not np.isfinite(log_acceptance_fraction):
                raise(ValueError())

            # Check for acceptance
            # stats['diverged'] = not np.isfinite(log_acceptance_fraction)
            # stats['accept'] = log_acceptance_fraction

            # If np.isfinite fails then we just consider we're out of bounds (diverged)
            if np.isfinite(log_acceptance_fraction) and np.log(np.random.random()) < min(0, log_acceptance_fraction):
                # stats['accepted'] = True
                # return new_point, [stats]
                return new_point
            else:
                # stats['accepted'] = False
                # return point, [stats]
                return point

    # Delta configurations and subspace sets
    configurations = functools.reduce(lambda x,y: x.union(y), (set(itertools.permutations(x)) for x in itertools.combinations_with_replacement([0, 1], len(delta_parameters))))
    configurations_to_subspaces = {config:{k_parameters[i] for i,d in enumerate(config) if d == 1} for config in configurations}
    # filter out the null subspace if present
    configurations_to_subspaces = {x:y for x,y in configurations_to_subspaces.items() if len(y) > 0}

    # Inter subspace jump steppers
    jump_probas = collections.defaultdict(dict)
    # we filter out self maps and maps to/from the zero dynamics model
    for c1, c2 in filter(lambda x: x[0] != x[1], itertools.product(configurations, repeat = 2)):
         # we'll have everything be equiprobable but no jumping towards zero dynamics
        jump_probas[c1][c2] = 1 if c2 != tuple([0] * len(delta_parameters)) else 0

    # renormalize the jump probabilities
    for c1 in jump_probas:
        sum = np.sum(list(jump_probas[c1].values()))
        jump_probas[c1] = {c2:val/sum for c2, val in jump_probas[c1].items()}

    # Remap jump probabilities into steppers
    jumps = {c1:{c2: Jump(c1, c2, jump_probas[c1][c2], jump_probas[c2][c1]) for c2 in jump_probas[c1] } for c1 in jump_probas}

    step = RJMCMC(delta_parameters, configurations_to_subspaces, jumps, jump_probas)

    trace = pm.sample(ndraws, tune = nburn, step = step, discard_tuned_samples = True, cores=1)

# # Just dump all the data inzto a pickle
# with open('pymc_trace.pkl', 'wb') as buff:
#     pickle.dump({'model':pymc_model, 'trace':trace}, buff)

# with open('pymc_trace.pkl', 'rb') as buff:
#     data = pickle.load(buff)
#     trace = data['trace']

# figure_directory = 'figures/simple_reaction_reversibility/'
# with pymc_model:
#     az.plot_trace(trace.posterior, var_names = ['delta_0_1', 'delta_1_0'])
#     full_screen_save(figure_directory, 'delta_trace.png')
#     az.plot_trace(trace.posterior, var_names = ['k_0_1', 'k_1_0'])
#     full_screen_save(figure_directory, 'k_trace.png')