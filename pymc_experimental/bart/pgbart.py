#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import logging

from copy import copy

import aesara
import numpy as np

from aesara import function as aesara_function

from pymc.aesaraf import inputvars, join_nonshared_inputs, make_shared_replacements
from pymc_experimental.bart.bart import BARTRV
from pymc_experimental.bart.tree import LeafNode, SplitNode, Tree
from pymc.model import modelcontext
from pymc.step_methods.arraystep import ArrayStepShared, Competence

_log = logging.getLogger("pymc")


class PGBART(ArrayStepShared):
    """
    Particle Gibss BART sampling step.

    Parameters
    ----------
    vars: list
        List of value variables for sampler
    num_particles : int
        Number of particles for the conditional SMC sampler. Defaults to 40
    batch : int or tuple
        Number of trees fitted per step. Defaults to  "auto", which is the 10% of the `m` trees
        during tuning and after tuning. If a tuple is passed the first element is the batch size
        during tuning and the second the batch size after tuning.
    model: PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).
    """

    name = "pgbart"
    default_blocked = False
    generates_stats = True
    stats_dtypes = [{"variable_inclusion": object, "bart_trees": object}]

    def __init__(
        self,
        vars=None,
        num_particles=40,
        batch="auto",
        model=None,
    ):
        model = modelcontext(model)
        initial_values = model.initial_point()
        if vars is None:
            vars = model.value_vars
        else:
            vars = [model.rvs_to_values.get(var, var) for var in vars]
            vars = inputvars(vars)
        value_bart = vars[0]
        self.bart = model.values_to_rvs[value_bart].owner.op

        self.X = self.bart.X
        self.Y = self.bart.Y
        self.missing_data = np.any(np.isnan(self.X))
        self.m = self.bart.m
        self.alpha = self.bart.alpha
        self.alpha_vec = self.bart.split_prior
        if self.alpha_vec is None:
            self.alpha_vec = np.ones(self.X.shape[1])

        self.init_mean = self.Y.mean()
        # if data is binary
        Y_unique = np.unique(self.Y)
        if Y_unique.size == 2 and np.all(Y_unique == [0, 1]):
            mu_std = 3 / self.m**0.5
        # maybe we need to check for count data
        else:
            mu_std = self.Y.std() / self.m**0.5

        self.num_observations = self.X.shape[0]
        self.num_variates = self.X.shape[1]
        self.available_predictors = list(range(self.num_variates))

        self.sum_trees = np.full_like(self.Y, self.init_mean).astype(aesara.config.floatX)
        self.a_tree = Tree.init_tree(
            leaf_node_value=self.init_mean / self.m,
            idx_data_points=np.arange(self.num_observations, dtype="int32"),
        )
        self.mean = fast_mean()

        self.normal = NormalSampler(mu_std)
        self.uniform = UniformSampler(0.33, 0.75)
        self.prior_prob_leaf_node = compute_prior_probability(self.alpha)
        self.ssv = SampleSplittingVariable(self.alpha_vec)

        self.tune = True

        if batch == "auto":
            batch = max(1, int(self.m * 0.1))
            self.batch = (batch, batch)
        else:
            if isinstance(batch, (tuple, list)):
                self.batch = batch
            else:
                self.batch = (batch, batch)

        self.log_num_particles = np.log(num_particles)
        self.indices = list(range(2, num_particles))
        self.len_indices = len(self.indices)

        shared = make_shared_replacements(initial_values, vars, model)
        self.likelihood_logp = logp(initial_values, [model.datalogpt], vars, shared)
        self.all_particles = []
        for _ in range(self.m):
            self.a_tree.leaf_node_value = self.init_mean / self.m
            p = ParticleTree(self.a_tree)
            self.all_particles.append(p)
        self.all_trees = np.array([p.tree for p in self.all_particles])
        super().__init__(vars, shared)

    def astep(self, _):
        variable_inclusion = np.zeros(self.num_variates, dtype="int")

        tree_ids = np.random.choice(range(self.m), replace=False, size=self.batch[~self.tune])
        for tree_id in tree_ids:
            # Compute the sum of trees without the old tree that we are attempting to replace
            self.sum_trees_noi = self.sum_trees - self.all_particles[tree_id].tree._predict()
            # Generate an initial set of SMC particles
            # at the end of the algorithm we return one of these particles as the new tree
            particles = self.init_particles(tree_id)

            while True:
                # Sample each particle (try to grow each tree), except for the first two
                stop_growing = True
                for p in particles[2:]:
                    tree_grew = p.sample_tree(
                        self.ssv,
                        self.available_predictors,
                        self.prior_prob_leaf_node,
                        self.X,
                        self.missing_data,
                        self.sum_trees,
                        self.mean,
                        self.m,
                        self.normal,
                    )
                    if tree_grew:
                        self.update_weight(p)
                    if p.expansion_nodes:
                        stop_growing = False
                if stop_growing:
                    break

                # Normalize weights
                w_t, normalized_weights = self.normalize(particles[2:])

                # Resample all but first two particles
                new_indices = np.random.choice(
                    self.indices, size=self.len_indices, p=normalized_weights
                )
                particles[2:] = particles[new_indices]

                # Set the new weight
                for p in particles[2:]:
                    p.log_weight = w_t

            for p in particles[2:]:
                p.log_weight = p.old_likelihood_logp

            _, normalized_weights = self.normalize(particles)
            # Get the new tree and update
            new_particle = np.random.choice(particles, p=normalized_weights)
            new_tree = new_particle.tree

            new_particle.log_weight = new_particle.old_likelihood_logp - self.log_num_particles
            self.all_particles[tree_id] = new_particle
            self.sum_trees = self.sum_trees_noi + new_tree._predict()
            self.all_trees[tree_id] = new_tree.trim()

            if self.tune:
                self.ssv = SampleSplittingVariable(self.alpha_vec)
                for index in new_particle.used_variates:
                    self.alpha_vec[index] += 1
            else:
                for index in new_particle.used_variates:
                    variable_inclusion[index] += 1

        stats = {"variable_inclusion": variable_inclusion, "bart_trees": copy(self.all_trees)}
        return self.sum_trees, [stats]

    def normalize(self, particles):
        """Use logsumexp trick to get w_t and softmax to get normalized_weights.

        w_t is the un-normalized weight per particle, we will assign it to the
        next round of particles, so they all start with the same weight.
        """
        log_w = np.array([p.log_weight for p in particles])
        log_w_max = log_w.max()
        log_w_ = log_w - log_w_max
        w_ = np.exp(log_w_)
        w_sum = w_.sum()
        w_t = log_w_max + np.log(w_sum) - self.log_num_particles
        normalized_weights = w_ / w_sum
        # stabilize weights to avoid assigning exactly zero probability to a particle
        normalized_weights += 1e-12

        return w_t, normalized_weights

    def init_particles(self, tree_id: int) -> np.ndarray:
        """Initialize particles."""
        p0 = self.all_particles[tree_id]
        p1 = copy(p0)
        p1.sample_leafs(
            self.sum_trees,
            self.mean,
            self.m,
            self.normal,
        )

        # The old tree and the one with new leafs do not grow so we update the weights only once
        self.update_weight(p0, old=True)
        self.update_weight(p1, old=True)
        particles = [p0, p1]

        for _ in self.indices:
            pt = ParticleTree(self.a_tree)
            if self.tune:
                pt.kf = self.uniform.random()
            else:
                pt.kf = p0.kf
            particles.append(pt)

        return np.array(particles)

    def update_weight(self, particle, old=False):
        """
        Update the weight of a particle.

        Since the prior is used as the proposal,the weights are updated additively as the ratio of
        the new and old log-likelihoods.
        """
        new_likelihood = self.likelihood_logp(self.sum_trees_noi + particle.tree._predict())
        if old:
            particle.log_weight = new_likelihood
            particle.old_likelihood_logp = new_likelihood
        else:
            particle.log_weight += new_likelihood - particle.old_likelihood_logp
            particle.old_likelihood_logp = new_likelihood

    @staticmethod
    def competence(var, has_grad):
        """PGBART is only suitable for BART distributions."""
        dist = getattr(var.owner, "op", None)
        if isinstance(dist, BARTRV):
            return Competence.IDEAL
        return Competence.INCOMPATIBLE


class ParticleTree:
    """Particle tree."""

    def __init__(self, tree):
        self.tree = tree.copy()  # keeps the tree that we care at the moment
        self.expansion_nodes = [0]
        self.log_weight = 0
        self.old_likelihood_logp = 0
        self.used_variates = []
        self.kf = 0.75

    def sample_tree(
        self,
        ssv,
        available_predictors,
        prior_prob_leaf_node,
        X,
        missing_data,
        sum_trees,
        mean,
        m,
        normal,
    ):
        tree_grew = False
        if self.expansion_nodes:
            index_leaf_node = self.expansion_nodes.pop(0)
            # Probability that this node will remain a leaf node
            prob_leaf = prior_prob_leaf_node[self.tree[index_leaf_node].depth]

            if prob_leaf < np.random.random():
                index_selected_predictor = grow_tree(
                    self.tree,
                    index_leaf_node,
                    ssv,
                    available_predictors,
                    X,
                    missing_data,
                    sum_trees,
                    mean,
                    m,
                    normal,
                    self.kf,
                )
                if index_selected_predictor is not None:
                    new_indexes = self.tree.idx_leaf_nodes[-2:]
                    self.expansion_nodes.extend(new_indexes)
                    self.used_variates.append(index_selected_predictor)
                    tree_grew = True

        return tree_grew

    def sample_leafs(self, sum_trees, mean, m, normal):

        for idx in self.tree.idx_leaf_nodes:
            if idx > 0:
                leaf = self.tree[idx]
                idx_data_points = leaf.idx_data_points
                node_value = draw_leaf_value(
                    sum_trees[idx_data_points],
                    mean,
                    m,
                    normal,
                    self.kf,
                )
                leaf.value = node_value


class SampleSplittingVariable:
    def __init__(self, alpha_vec):
        """
        Sample splitting variables proportional to `alpha_vec`.

        This is equivalent to compute the posterior mean of a Dirichlet-Multinomial model.
        This enforce sparsity.
        """
        self.enu = list(enumerate(np.cumsum(alpha_vec / alpha_vec.sum())))

    def rvs(self):
        r = np.random.random()
        for i, v in self.enu:
            if r <= v:
                return i


def compute_prior_probability(alpha):
    """
    Calculate the probability of the node being a LeafNode (1 - p(being SplitNode)).

    Taken from equation 19 in [Rockova2018].

    Parameters
    ----------
    alpha : float

    Returns
    -------
    list with probabilities for leaf nodes

    References
    ----------
    .. [Rockova2018] Veronika Rockova, Enakshi Saha (2018). On the theory of BART.
    arXiv, `link <https://arxiv.org/abs/1810.00787>`__
    """
    prior_leaf_prob = [0]
    depth = 1
    while prior_leaf_prob[-1] < 1:
        prior_leaf_prob.append(1 - alpha**depth)
        depth += 1
    return prior_leaf_prob


def grow_tree(
    tree,
    index_leaf_node,
    ssv,
    available_predictors,
    X,
    missing_data,
    sum_trees,
    mean,
    m,
    normal,
    kf,
):
    current_node = tree.get_node(index_leaf_node)
    idx_data_points = current_node.idx_data_points

    index_selected_predictor = ssv.rvs()
    selected_predictor = available_predictors[index_selected_predictor]
    available_splitting_values = X[idx_data_points, selected_predictor]
    split_value = get_split_value(available_splitting_values, missing_data)

    if split_value is not None:

        new_idx_data_points = get_new_idx_data_points(
            split_value, idx_data_points, selected_predictor, X
        )
        current_node_children = (
            current_node.get_idx_left_child(),
            current_node.get_idx_right_child(),
        )

        new_nodes = []
        for idx in range(2):
            idx_data_point = new_idx_data_points[idx]
            node_value = draw_leaf_value(
                sum_trees[idx_data_point],
                mean,
                m,
                normal,
                kf,
            )

            new_node = LeafNode(
                index=current_node_children[idx],
                value=node_value,
                idx_data_points=idx_data_point,
            )
            new_nodes.append(new_node)

        new_split_node = SplitNode(
            index=index_leaf_node,
            idx_split_variable=selected_predictor,
            split_value=split_value,
        )

        # update tree nodes and indexes
        tree.delete_leaf_node(index_leaf_node)
        tree.set_node(index_leaf_node, new_split_node)
        tree.set_node(new_nodes[0].index, new_nodes[0])
        tree.set_node(new_nodes[1].index, new_nodes[1])

        return index_selected_predictor


def get_new_idx_data_points(split_value, idx_data_points, selected_predictor, X):

    left_idx = X[idx_data_points, selected_predictor] <= split_value
    left_node_idx_data_points = idx_data_points[left_idx]
    right_node_idx_data_points = idx_data_points[~left_idx]

    return left_node_idx_data_points, right_node_idx_data_points


def get_split_value(available_splitting_values, missing_data):

    if missing_data:
        idx_data_points = idx_data_points[~np.isnan(available_splitting_values)]
        available_splitting_values = available_splitting_values[
            ~np.isnan(available_splitting_values)
        ]

    if available_splitting_values.size > 0:
        idx_selected_splitting_values = discrete_uniform_sampler(len(available_splitting_values))
        split_value = available_splitting_values[idx_selected_splitting_values]

        return split_value


def draw_leaf_value(Y_mu_pred, mean, m, normal, kf):
    """Draw Gaussian distributed leaf values."""
    if Y_mu_pred.size == 0:
        return 0
    else:
        norm = normal.random() * kf
        if Y_mu_pred.size == 1:
            mu_mean = Y_mu_pred.item() / m
        else:
            mu_mean = mean(Y_mu_pred) / m

        draw = norm + mu_mean
        return draw


def fast_mean():
    """If available use Numba to speed up the computation of the mean."""
    try:
        from numba import jit
    except ImportError:
        return np.mean

    @jit
    def mean(a):
        count = a.shape[0]
        suma = 0
        for i in range(count):
            suma += a[i]
        return suma / count

    return mean


def discrete_uniform_sampler(upper_value):
    """Draw from the uniform distribution with bounds [0, upper_value).

    This is the same and np.random.randit(upper_value) but faster.
    """
    return int(np.random.random() * upper_value)


class NormalSampler:
    """Cache samples from a standard normal distribution."""

    def __init__(self, scale):
        self.size = 1000
        self.cache = []
        self.scale = scale

    def random(self):
        if not self.cache:
            self.update()
        return self.cache.pop()

    def update(self):
        self.cache = np.random.normal(loc=0.0, scale=self.scale, size=self.size).tolist()


class UniformSampler:
    """Cache samples from a uniform distribution."""

    def __init__(self, lower_bound, upper_bound):
        self.size = 1000
        self.cache = []
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def random(self):
        if not self.cache:
            self.update()
        return self.cache.pop()

    def update(self):
        self.cache = np.random.uniform(self.lower_bound, self.upper_bound, size=self.size).tolist()


def logp(point, out_vars, vars, shared):
    """Compile Aesara function of the model and the input and output variables.

    Parameters
    ----------
    out_vars: List
        containing :class:`pymc.Distribution` for the output variables
    vars: List
        containing :class:`pymc.Distribution` for the input variables
    shared: List
        containing :class:`aesara.tensor.Tensor` for depended shared data
    """
    out_list, inarray0 = join_nonshared_inputs(point, out_vars, vars, shared)
    f = aesara_function([inarray0], out_list[0])
    f.trust_input = True
    return f
