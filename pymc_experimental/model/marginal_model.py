import warnings
from itertools import zip_longest
from typing import Sequence, Tuple, Union

import numpy as np
import pymc
import pytensor.tensor as pt
from arviz import dict_to_dataset
from pymc import SymbolicRandomVariable
from pymc.backends.arviz import coords_and_dims_for_inferencedata
from pymc.distributions.discrete import Bernoulli, Categorical, DiscreteUniform
from pymc.distributions.transforms import Chain
from pymc.logprob.abstract import _logprob
from pymc.logprob.basic import conditional_logp
from pymc.logprob.transforms import IntervalTransform
from pymc.model import Model
from pymc.pytensorf import compile_pymc, constant_fold, inputvars
from pymc.util import _get_seeds_per_chain, dataset_to_point_list, treedict
from pytensor import Mode
from pytensor.compile import SharedVariable
from pytensor.compile.builders import OpFromGraph
from pytensor.graph import (
    Constant,
    FunctionGraph,
    ancestors,
    clone_replace,
    vectorize_graph,
)
from pytensor.graph.basic import Variable, io_toposort
from pytensor.scan import map as scan_map
from pytensor.tensor import TensorType, TensorVariable
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.rewriting.subtensor import is_full_slice
from pytensor.tensor.shape import Shape
from pytensor.tensor.special import log_softmax
from pytensor.tensor.subtensor import AdvancedSubtensor, Subtensor, get_idx_list
from pytensor.tensor.type_other import NoneTypeT

__all__ = ["MarginalModel"]


class MarginalModel(Model):
    """Subclass of PyMC Model that implements functionality for automatic
    marginalization of variables in the logp transformation

    After defining the full Model, the `marginalize` method can be used to indicate a
    subset of variables that should be marginalized

    Notes
    -----
    Marginalization functionality is still very restricted. Only finite discrete
    variables can be marginalized. Deterministics and Potentials cannot be conditionally
    dependent on the marginalized variables.

    Furthermore, not all instances of such variables can be marginalized. If a variable
    has batched dimensions, it is required that any conditionally dependent variables
    use information from an individual batched dimension. In other words, the graph
    connecting the marginalized variable(s) to the dependent variable(s) must be
    composed strictly of Elemwise Operations. This is necessary to ensure an efficient
    logprob graph can be generated. If you want to bypass this restriction you can
    separate each dimension of the marginalized variable into the scalar components
    and then stack them together. Note that such graphs will grow exponentially in the
    number of  marginalized variables.

    For the same reason, it's not possible to marginalize RVs with multivariate
    dependent RVs.

    Examples
    --------

    Marginalize over a single variable

    .. code-block:: python

        import pymc as pm
        from pymc_experimental import MarginalModel

        with MarginalModel() as m:
            p = pm.Beta("p", 1, 1)
            x = pm.Bernoulli("x", p=p, shape=(3,))
            y = pm.Normal("y", pm.math.switch(x, -10, 10), observed=[10, 10, -10])

            m.marginalize([x])

            idata = pm.sample()

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.marginalized_rvs = []
        self._marginalized_named_vars_to_dims = {}

    def _delete_rv_mappings(self, rv: TensorVariable) -> None:
        """Remove all model mappings referring to rv

        This can be used to "delete" an RV from a model
        """
        assert rv in self.basic_RVs, "rv is not part of the Model"

        name = rv.name
        self.named_vars.pop(name)
        if name in self.named_vars_to_dims:
            self.named_vars_to_dims.pop(name)

        value = self.rvs_to_values.pop(rv)
        self.values_to_rvs.pop(value)

        self.rvs_to_transforms.pop(rv)
        if rv in self.free_RVs:
            self.free_RVs.remove(rv)
            self.rvs_to_initial_values.pop(rv)
        else:
            self.observed_RVs.remove(rv)

    def _transfer_rv_mappings(self, old_rv: TensorVariable, new_rv: TensorVariable) -> None:
        """Transfer model mappings from old_rv to new_rv"""

        assert old_rv in self.basic_RVs, "old_rv is not part of the Model"
        assert new_rv not in self.basic_RVs, "new_rv is already part of the Model"

        self.named_vars.pop(old_rv.name)
        new_rv.name = old_rv.name
        self.named_vars[new_rv.name] = new_rv
        if old_rv in self.named_vars_to_dims:
            self._RV_dims[new_rv] = self._RV_dims.pop(old_rv)

        value = self.rvs_to_values.pop(old_rv)
        self.rvs_to_values[new_rv] = value
        self.values_to_rvs[value] = new_rv

        self.rvs_to_transforms[new_rv] = self.rvs_to_transforms.pop(old_rv)
        if old_rv in self.free_RVs:
            index = self.free_RVs.index(old_rv)
            self.free_RVs.pop(index)
            self.free_RVs.insert(index, new_rv)
            self.rvs_to_initial_values[new_rv] = self.rvs_to_initial_values.pop(old_rv)
        elif old_rv in self.observed_RVs:
            index = self.observed_RVs.index(old_rv)
            self.observed_RVs.pop(index)
            self.observed_RVs.insert(index, new_rv)

    def _marginalize(self, user_warnings=False):
        fg = FunctionGraph(outputs=self.basic_RVs + self.marginalized_rvs, clone=False)

        toposort = fg.toposort()
        rvs_left_to_marginalize = self.marginalized_rvs
        for rv_to_marginalize in sorted(
            self.marginalized_rvs,
            key=lambda rv: toposort.index(rv.owner),
            reverse=True,
        ):
            # Check that no deterministics or potentials dependend on the rv to marginalize
            for det in self.deterministics:
                if is_conditional_dependent(
                    det, rv_to_marginalize, self.basic_RVs + rvs_left_to_marginalize
                ):
                    raise NotImplementedError(
                        f"Cannot marginalize {rv_to_marginalize} due to dependent Deterministic {det}"
                    )
            for pot in self.potentials:
                if is_conditional_dependent(
                    pot, rv_to_marginalize, self.basic_RVs + rvs_left_to_marginalize
                ):
                    raise NotImplementedError(
                        f"Cannot marginalize {rv_to_marginalize} due to dependent Potential {pot}"
                    )

            old_rvs, new_rvs = replace_finite_discrete_marginal_subgraph(
                fg, rv_to_marginalize, self.basic_RVs + rvs_left_to_marginalize
            )

            if user_warnings and len(new_rvs) > 2:
                warnings.warn(
                    "There are multiple dependent variables in a FiniteDiscreteMarginalRV. "
                    f"Their joint logp terms will be assigned to the first RV: {old_rvs[1]}",
                    UserWarning,
                )

            rvs_left_to_marginalize.remove(rv_to_marginalize)

            for old_rv, new_rv in zip(old_rvs, new_rvs):
                new_rv.name = old_rv.name
                if old_rv in self.marginalized_rvs:
                    idx = self.marginalized_rvs.index(old_rv)
                    self.marginalized_rvs.pop(idx)
                    self.marginalized_rvs.insert(idx, new_rv)
                if old_rv in self.basic_RVs:
                    self._transfer_rv_mappings(old_rv, new_rv)
                    if user_warnings:
                        # Interval transforms for dependent variable won't work for non-constant bounds because
                        # the RV inputs are now different and may depend on another RV that also depends on the
                        # same marginalized RV
                        transform = self.rvs_to_transforms[new_rv]
                        if isinstance(transform, IntervalTransform) or (
                            isinstance(transform, Chain)
                            and any(
                                isinstance(tr, IntervalTransform) for tr in transform.transform_list
                            )
                        ):
                            warnings.warn(
                                f"The transform {transform} for the variable {old_rv}, which depends on the "
                                f"marginalized {rv_to_marginalize} may no longer work if bounds depended on other variables.",
                                UserWarning,
                            )
        return self

    def _logp(self, *args, **kwargs):
        return super().logp(*args, **kwargs)

    def logp(self, vars=None, **kwargs):
        m = self.clone()._marginalize()
        if vars is not None:
            if not isinstance(vars, Sequence):
                vars = (vars,)
            vars = [m[var.name] for var in vars]
        return m._logp(vars=vars, **kwargs)

    def clone(self):
        m = MarginalModel(coords=self.coords)
        model_vars = self.basic_RVs + self.potentials + self.deterministics + self.marginalized_rvs
        data_vars = [var for name, var in self.named_vars.items() if var not in model_vars]
        vars = model_vars + data_vars
        cloned_vars = clone_replace(vars)
        vars_to_clone = {var: cloned_var for var, cloned_var in zip(vars, cloned_vars)}
        m.vars_to_clone = vars_to_clone

        m.named_vars = treedict({name: vars_to_clone[var] for name, var in self.named_vars.items()})
        m.named_vars_to_dims = self.named_vars_to_dims
        m.values_to_rvs = {i: vars_to_clone[rv] for i, rv in self.values_to_rvs.items()}
        m.rvs_to_values = {vars_to_clone[rv]: i for rv, i in self.rvs_to_values.items()}
        m.rvs_to_transforms = {vars_to_clone[rv]: i for rv, i in self.rvs_to_transforms.items()}
        m.rvs_to_initial_values = {
            vars_to_clone[rv]: i for rv, i in self.rvs_to_initial_values.items()
        }
        m.free_RVs = [vars_to_clone[rv] for rv in self.free_RVs]
        m.observed_RVs = [vars_to_clone[rv] for rv in self.observed_RVs]
        m.potentials = [vars_to_clone[pot] for pot in self.potentials]
        m.deterministics = [vars_to_clone[det] for det in self.deterministics]

        m.marginalized_rvs = [vars_to_clone[rv] for rv in self.marginalized_rvs]
        m._marginalized_named_vars_to_dims = self._marginalized_named_vars_to_dims
        return m

    def marginalize(
        self,
        rvs_to_marginalize: Union[TensorVariable, str, Sequence[TensorVariable], Sequence[str]],
    ):
        if not isinstance(rvs_to_marginalize, Sequence):
            rvs_to_marginalize = (rvs_to_marginalize,)
        rvs_to_marginalize = [
            self[var] if isinstance(var, str) else var for var in rvs_to_marginalize
        ]

        supported_dists = (Bernoulli, Categorical, DiscreteUniform)
        for rv_to_marginalize in rvs_to_marginalize:
            if rv_to_marginalize not in self.free_RVs:
                raise ValueError(
                    f"Marginalized RV {rv_to_marginalize} is not a free RV in the model"
                )
            if not isinstance(rv_to_marginalize.owner.op, supported_dists):
                raise NotImplementedError(
                    f"RV with distribution {rv_to_marginalize.owner.op} cannot be marginalized. "
                    f"Supported distribution include {supported_dists}"
                )

            if rv_to_marginalize.name in self.named_vars_to_dims:
                dims = self.named_vars_to_dims[rv_to_marginalize.name]
                self._marginalized_named_vars_to_dims[rv_to_marginalize.name] = dims

            self._delete_rv_mappings(rv_to_marginalize)
            self.marginalized_rvs.append(rv_to_marginalize)

        # Raise errors and warnings immediately
        self.clone()._marginalize(user_warnings=True)

    def _to_transformed(self):
        "Create a function from the untransformed space to the transformed space"
        transformed_rvs = []
        transformed_names = []

        for rv in self.free_RVs:
            transform = self.rvs_to_transforms.get(rv)
            if transform is None:
                transformed_rvs.append(rv)
                transformed_names.append(rv.name)
            else:
                transformed_rv = transform.forward(rv, *rv.owner.inputs)
                transformed_rvs.append(transformed_rv)
                transformed_names.append(self.rvs_to_values[rv].name)

        fn = self.compile_fn(inputs=self.free_RVs, outs=transformed_rvs)
        return fn, transformed_names

    def unmarginalize(self, rvs_to_unmarginalize):
        for rv in rvs_to_unmarginalize:
            self.marginalized_rvs.remove(rv)
            if rv.name in self._marginalized_named_vars_to_dims:
                dims = self._marginalized_named_vars_to_dims.pop(rv.name)
            else:
                dims = None
            self.register_rv(rv, name=rv.name, dims=dims)

    def recover_marginals(
        self,
        idata,
        var_names=None,
        return_samples=True,
        extend_inferencedata=True,
        random_seed=None,
    ):
        """Computes posterior log-probabilities and samples of marginalized variables
        conditioned on parameters of the model given InferenceData with posterior group

        When there are multiple marginalized variables, each marginalized variable is
        conditioned on both the parameters and the other variables still marginalized

        All log-probabilities are within the transformed space

        Parameters
        ----------
        idata : InferenceData
            InferenceData with posterior group
        var_names : sequence of str, optional
            List of variable names for which to compute posterior log-probabilities and samples. Defaults to all marginalized variables
        return_samples : bool, default True
            If True, also return samples of the marginalized variables
        extend_inferencedata : bool, default True
            Whether to extend the original InferenceData or return a new one
        random_seed: int, array-like of int or SeedSequence, optional
            Seed used to generating samples

        Returns
        -------
        idata : InferenceData
            InferenceData with where a lp_{varname} and {varname} for each marginalized variable in var_names added to the posterior group

        .. code-block:: python

            import pymc as pm
            from pymc_experimental import MarginalModel

            with MarginalModel() as m:
                p = pm.Beta("p", 1, 1)
                x = pm.Bernoulli("x", p=p, shape=(3,))
                y = pm.Normal("y", pm.math.switch(x, -10, 10), observed=[10, 10, -10])

                m.marginalize([x])

                idata = pm.sample()
                m.recover_marginals(idata, var_names=["x"])


        """
        if var_names is None:
            var_names = [var.name for var in self.marginalized_rvs]

        var_names = [var if isinstance(var, str) else var.name for var in var_names]
        vars_to_recover = [v for v in self.marginalized_rvs if v.name in var_names]
        missing_names = [v.name for v in vars_to_recover if v not in self.marginalized_rvs]
        if missing_names:
            raise ValueError(f"Unrecognized var_names: {missing_names}")

        if return_samples and random_seed is not None:
            seeds = _get_seeds_per_chain(random_seed, len(vars_to_recover))
        else:
            seeds = [None] * len(vars_to_recover)

        posterior = idata.posterior

        # Remove Deterministics
        posterior_values = posterior[
            [rv.name for rv in self.free_RVs if rv not in self.marginalized_rvs]
        ]

        sample_dims = ("chain", "draw")
        posterior_pts, stacked_dims = dataset_to_point_list(posterior_values, sample_dims)

        # Handle Transforms
        transform_fn, transform_names = self._to_transformed()

        def transform_input(inputs):
            return dict(zip(transform_names, transform_fn(inputs)))

        posterior_pts = [transform_input(vs) for vs in posterior_pts]

        rv_dict = {}
        rv_dims = {}
        for seed, rv in zip(seeds, vars_to_recover):
            supported_dists = (Bernoulli, Categorical, DiscreteUniform)
            if not isinstance(rv.owner.op, supported_dists):
                raise NotImplementedError(
                    f"RV with distribution {rv.owner.op} cannot be recovered. "
                    f"Supported distribution include {supported_dists}"
                )

            m = self.clone()
            rv = m.vars_to_clone[rv]
            m.unmarginalize([rv])
            dependent_vars = find_conditional_dependent_rvs(rv, m.basic_RVs)
            joint_logps = m.logp(vars=dependent_vars + [rv], sum=False)

            marginalized_value = m.rvs_to_values[rv]
            other_values = [v for v in m.value_vars if v is not marginalized_value]

            # Handle batch dims for marginalized value and its dependent RVs
            joint_logp = joint_logps[-1]
            for dv in joint_logps[:-1]:
                dbcast = dv.type.broadcastable
                mbcast = marginalized_value.type.broadcastable
                mbcast = (True,) * (len(dbcast) - len(mbcast)) + mbcast
                values_axis_bcast = [
                    i for i, (m, v) in enumerate(zip(mbcast, dbcast)) if m and not v
                ]
                joint_logp += dv.sum(values_axis_bcast)

            rv_shape = constant_fold(tuple(rv.shape))
            rv_domain = get_domain_of_finite_discrete_rv(rv)
            rv_domain_tensor = pt.moveaxis(
                pt.full(
                    (*rv_shape, len(rv_domain)),
                    rv_domain,
                    dtype=rv.dtype,
                ),
                -1,
                0,
            )

            joint_logps = vectorize_graph(
                joint_logp,
                replace={marginalized_value: rv_domain_tensor},
            )
            joint_logps = pt.moveaxis(joint_logps, 0, -1)

            rv_loglike_fn = None
            joint_logps_norm = log_softmax(joint_logps, axis=-1)
            if return_samples:
                sample_rv_outs = pymc.Categorical.dist(logit_p=joint_logps)
                if isinstance(rv.owner.op, DiscreteUniform):
                    sample_rv_outs += rv_domain[0]

                rv_loglike_fn = compile_pymc(
                    inputs=other_values,
                    outputs=[joint_logps_norm, sample_rv_outs],
                    on_unused_input="ignore",
                    random_seed=seed,
                )
            else:
                rv_loglike_fn = compile_pymc(
                    inputs=other_values,
                    outputs=joint_logps_norm,
                    on_unused_input="ignore",
                    random_seed=seed,
                )

            logvs = [rv_loglike_fn(**vs) for vs in posterior_pts]

            logps = None
            samples = None
            if return_samples:
                logps, samples = zip(*logvs)
                logps = np.array(logps)
                samples = np.array(samples)
                rv_dict[rv.name] = samples.reshape(
                    tuple(len(coord) for coord in stacked_dims.values()) + samples.shape[1:],
                )
            else:
                logps = np.array(logvs)

            rv_dict["lp_" + rv.name] = logps.reshape(
                tuple(len(coord) for coord in stacked_dims.values()) + logps.shape[1:],
            )
            if rv.name in m.named_vars_to_dims:
                rv_dims[rv.name] = list(m.named_vars_to_dims[rv.name])
                rv_dims["lp_" + rv.name] = rv_dims[rv.name] + ["lp_" + rv.name + "_dim"]

        coords, dims = coords_and_dims_for_inferencedata(self)
        dims.update(rv_dims)
        rv_dataset = dict_to_dataset(
            rv_dict,
            library=pymc,
            dims=dims,
            coords=coords,
            default_dims=list(sample_dims),
            skip_event_dims=True,
        )

        if extend_inferencedata:
            idata.posterior = idata.posterior.assign(rv_dataset)
            return idata
        else:
            return rv_dataset


class MarginalRV(SymbolicRandomVariable):
    """Base class for Marginalized RVs"""


class FiniteDiscreteMarginalRV(MarginalRV):
    """Base class for Finite Discrete Marginalized RVs"""


def static_shape_ancestors(vars):
    """Identify ancestors Shape Ops of static shapes (therefore constant in a valid graph)."""
    return [
        var
        for var in ancestors(vars)
        if (
            var.owner
            and isinstance(var.owner.op, Shape)
            # All static dims lengths of Shape input are known
            and None not in var.owner.inputs[0].type.shape
        )
    ]


def find_conditional_input_rvs(output_rvs, all_rvs):
    """Find conditionally indepedent input RVs."""
    blockers = [other_rv for other_rv in all_rvs if other_rv not in output_rvs]
    blockers += static_shape_ancestors(tuple(all_rvs) + tuple(output_rvs))
    return [
        var
        for var in ancestors(output_rvs, blockers=blockers)
        if var in blockers
        or (var.owner is None and not isinstance(var, (Constant, SharedVariable)))
    ]


def is_conditional_dependent(
    dependent_rv: TensorVariable, dependable_rv: TensorVariable, all_rvs
) -> bool:
    """Check if dependent_rv is conditionall dependent on dependable_rv,
    given all conditionally independent all_rvs"""

    return dependable_rv in find_conditional_input_rvs((dependent_rv,), all_rvs)


def find_conditional_dependent_rvs(dependable_rv, all_rvs):
    """Find rvs than depend on dependable"""
    return [
        rv
        for rv in all_rvs
        if (rv is not dependable_rv and is_conditional_dependent(rv, dependable_rv, all_rvs))
    ]


def _advanced_indexing_axis_and_ndim(idxs) -> tuple[int, int]:
    """Find the output axis and dimensionality of the advanced indexing group (i.e., array indexing).

    There is a special case: when there are non-consecutive advanced indexing groups, the advanced indexing
    group is always moved to the front.

    See: https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing
    """
    adv_group_axis = None
    for axis, idx in enumerate(idxs):
        if isinstance(idx.type, TensorType):
            adv_group_axis = axis
        elif adv_group_axis is not None:
            # Special non-consecutive case
            adv_group_axis = 0
            break

    adv_group_ndim = max(idx.type.ndim for idx in idxs if isinstance(idx.type, TensorType))
    return adv_group_axis, adv_group_ndim


# TODO: Use typevar for batch_dims
def _broadcast_dims(
    inputs_dims: Sequence[tuple[Union[int, None], ...]]
) -> tuple[Union[int, None], ...]:
    output_ndim = max((len(input_dim) for input_dim in inputs_dims), default=0)
    output_dims = {
        ((None,) * (output_ndim - len(input_dim))) + input_dim for input_dim in inputs_dims
    }
    if len(output_dims) > 1:
        raise ValueError
    return next(iter(output_dims))


def _insert_implicit_expanded_dims(inputs_dims, ndims_params, op_batch_ndim):
    """Insert implicit expanded dims for RandomVariable and SymbolicRandomVariable."""
    new_inputs_dims = []
    for input_dims, core_ndim in zip(inputs_dims, ndims_params):
        if input_dims is None:
            new_inputs_dims.append(None)
        else:
            missing_ndim = (op_batch_ndim + core_ndim) - len(input_dims)
            new_input_dims = ((None,) * missing_ndim) + input_dims
            new_inputs_dims.append(new_input_dims)
    return new_inputs_dims


def subgraph_dims(rv_to_marginalize, other_inputs_rvs, output_rvs) -> list[tuple[int, ...]]:
    """Identify how the batch dims of rv_to_marginalize map to the batch dims of the output_rvs.

    Raises
    ------
    ValueError
        If information from batch_dims is mixed at any point in the subgraph
    NotImplementedError
        If variable related to marginalized batch_dims is used in an operation that is not yet supported

    """

    batch_bcast_axes = rv_to_marginalize.type.broadcastable
    if rv_to_marginalize.owner.op.ndim_supp > 0:
        batch_bcast_axes = batch_bcast_axes[: -rv_to_marginalize.owner.op.ndim_supp]

    if any(batch_bcast_axes):
        # Note: We could support this by distinguishing between original and broadcasted axis
        # But for now this would complicate logic quite a lot for little gain.
        # We could also just allow without a specific type, but we would need to check if they are ever broadcasted
        # by any of the Ops that do implicit broadcasting (Alloc, Elemwise, Blockwise, AdvancedSubtensor, RandomVariable).
        raise NotImplementedError(
            "Marginalization of variables with broadcastable batch axes not supported."
        )

    # Batch axes for core RVs are always on the left
    # We will have info on SymbolicRVs at some point in PyMC
    var_dims: dict[Variable, tuple[Union[int, None], ...]] = {
        rv_to_marginalize: tuple(range(len(batch_bcast_axes)))
    }

    for node in io_toposort([rv_to_marginalize] + other_inputs_rvs, output_rvs):
        inputs_dims = [var_dims.get(inp, None) for inp in node.inputs]

        if not any(inputs_dims):
            # None of the inputs are related to the batch_axes of the marginalized_rv
            # We could pass `None` for everything, but for now that doesn't seem needed
            continue

        elif isinstance(node.op, DimShuffle):
            [key_dims] = inputs_dims
            if any(key_dims[dropped_dim] is not None for dropped_dim in node.op.drop):
                # Note: This is currently not possible as we forbid marginalized variable with broadcasted dims
                raise ValueError(f"{node} drops batch axes of the marginalized variable")

            output_dims = tuple(
                key_dims[i] if isinstance(i, int) else None for i in node.op.new_order
            )
            var_dims[node.outputs[0]] = output_dims

        elif isinstance(node.op, (Elemwise, Blockwise, RandomVariable, SymbolicRandomVariable)):
            # TODO: Add SymbolicRandomVariables to the mix?
            if isinstance(node.op, Elemwise):
                op_batch_ndim = node.outputs[0].type.ndim
            elif isinstance(node.op, Blockwise):
                op_batch_ndim = node.op.batch_ndim(node)
            elif isinstance(node.op, RandomVariable):
                op_batch_ndim = node.default_output().type.ndim
                # The first 3 inputs (rng, size, dtype) don't behave like a regular gufunc
                inputs_dims = [
                    None,
                    None,
                    None,
                    *_insert_implicit_expanded_dims(
                        inputs_dims[3:], node.op.ndims_params, op_batch_ndim
                    ),
                ]
            elif isinstance(node.op, SymbolicRandomVariable):
                ndims_params = getattr(node.op, "ndims_params", None)
                if ndims_params is None:
                    raise NotImplementedError(
                        "Dependent SymbolicRandomVariables without gufunc_signature are not supported"
                    )
                op_batch_ndim = node.op.batch_ndim(node)
                inputs_dims = _insert_implicit_expanded_dims(
                    inputs_dims, ndims_params, op_batch_ndim
                )

            if op_batch_ndim > 0:
                if any(
                    core_dim is not None
                    for input_dims in inputs_dims
                    if input_dims is not None
                    for core_dim in input_dims[op_batch_ndim:]
                ):
                    raise ValueError(
                        f"Node {node} uses batch dimensions of the marginalized variable as a core dimension"
                    )

            # Check batch dims are not broadcasted
            try:
                batch_dims = _broadcast_dims(
                    tuple(
                        input_dims[:op_batch_ndim]
                        for input_dims in inputs_dims
                        if input_dims is not None
                    )
                )
            except ValueError:
                raise NotImplementedError(
                    f"Node {node} mixes batch dimensions of the marginalized variable"
                )
            for out in node.outputs:
                if isinstance(out.type, TensorType):
                    core_dims = out.type.ndim - op_batch_ndim
                    var_dims[out] = batch_dims + (None,) * core_dims

        elif isinstance(node.op, CAReduce):
            # Only non batch_axes dims can be reduced
            [key_dims] = inputs_dims

            axes = node.op.axis
            if isinstance(axes, int):
                axes = (axes,)

            if axes is None or any(key_dims[axis] is not None for axis in axes):
                raise ValueError(
                    f"Reduction node {node} mixes batch dimensions of the marginalized variable"
                )

            output_dims = list(key_dims)
            for axis in sorted(axes, reverse=True):
                output_dims.pop(axis)
            var_dims[node.outputs[0]] = tuple(output_dims)

        elif isinstance(node.op, Subtensor):
            value_dims, *keys_dims = inputs_dims
            # Batch dims in basic indexing must belong to the value variable, since indexing keys are always scalar
            assert all(key_dims is None for key_dims in keys_dims)
            keys = get_idx_list(node.inputs, node.op.idx_list)

            output_dims = []
            for value_dims, idx in zip_longest(value_dims, keys, fillvalue=slice(None)):
                if is_full_slice(idx):
                    output_dims.append(value_dims)
                else:
                    if value_dims is not None:
                        # We are trying to slice or index a batch dim
                        # This is not necessarily problematic, unless this indexed dim is later mixed with other dims
                        # For now, we simply don't try to support it
                        raise NotImplementedError(
                            f"Indexing of batch dimensions of the marginalized variable in node {node} not supported"
                        )
                    if isinstance(idx, slice):
                        # Slice keeps the dim, whereas integer drops it
                        output_dims.append(None)

            var_dims[node.outputs[0]] = tuple(output_dims)

        elif isinstance(node.op, AdvancedSubtensor):
            # AdvancedSubtensor batch axis can show up as both the indexed variable and indexing variable
            value, *keys = node.inputs
            value_dims, *keys_dims = inputs_dims

            # Just to stay sane, we forbid any boolean indexing...
            if any(isinstance(idx.type, TensorType) and idx.type.dtype == "bool" for idx in keys):
                raise NotImplementedError(
                    f"Array indexing with boolean variables in node {node} not supported."
                )

            if value_dims and any(keys_dims):
                # Both indexed variable and indexing variables have batch dimensions
                # I am to lazy to think through these, so we raise for now.
                raise NotImplementedError(
                    f"Simultaneous use of indexed and indexing variables in node {node}, "
                    f"related to batch dimensions of the marginalized variable not supported"
                )

            adv_group_axis, adv_group_ndim = _advanced_indexing_axis_and_ndim(keys)

            if value_dims:
                # Indexed variable has batch dims

                if any(isinstance(idx.type, NoneTypeT) for idx in keys):
                    # TODO: Reason about these as AdvancedIndexing followed by ExpandDims
                    #  (maybe that's how they should be represented in PyTensor?)
                    # Only complication is when the NewAxis force the AdvancedGroup to go to the front
                    # which changes the position of the output batch axes
                    raise NotImplementedError(
                        f"Advanced indexing in node {node} which introduces new axis is not supported"
                    )

                # TODO: refactor this code which is completely shared by the Subtensor Op
                non_adv_dims = []
                for value_dim, idx in zip_longest(value_dims, keys, fillvalue=slice(None)):
                    if is_full_slice(idx):
                        non_adv_dims.append(value_dim)
                    else:
                        if value_dim is not None:
                            # We are trying to slice or index a batch dim
                            raise NotImplementedError(
                                f"Indexing of batch dimensions of the marginalized variable in node {node} not supported"
                            )
                        if isinstance(idx, slice):
                            non_adv_dims.append(None)

                # Insert dims corresponding to advanced indexing among the remaining ones
                output_dims = tuple(
                    non_adv_dims[adv_group_axis:]
                    + [None] * adv_group_ndim
                    + non_adv_dims[adv_group_axis:]
                )

            else:
                # Indexing keys have batch dims.
                # Only array indices can have batch_dims, the rest are just slices or new axis

                # Advanced indexing variables broadcast together, so we apply same rules as in Elemwise
                # However indexing is implicit, so we have to add None dims
                try:
                    adv_dims = _broadcast_dims(
                        tuple(key_dim for key_dim in keys_dims if key_dim is not None)
                    )
                except ValueError:
                    raise ValueError(
                        f"Index node {node} mixes batch dimensions of the marginalized variable"
                    )

                start_non_adv_dims = (None,) * adv_group_axis
                end_non_adv_dims = (None,) * (
                    node.outputs[0].type.ndim - adv_group_axis - adv_group_ndim
                )
                output_dims = start_non_adv_dims + adv_dims + end_non_adv_dims

            var_dims[node.outputs[0]] = output_dims

        else:
            # TODO: Assert, SpecifyShape: easy, batch dims can only be on the input,
            # TODO: Alloc: Easy, core batch dims stay in the same position (because we raised before for bcastable dims)
            raise NotImplementedError(f"Marginalization through operation {node} not supported")

    return [var_dims[output_rv] for output_rv in output_rvs]


def replace_finite_discrete_marginal_subgraph(fgraph, rv_to_marginalize, all_rvs):
    # TODO: This should eventually be integrated in a more general routine that can
    #  identify other types of supported marginalization, of which finite discrete
    #  RVs is just one

    dependent_rvs = find_conditional_dependent_rvs(rv_to_marginalize, all_rvs)
    if not dependent_rvs:
        raise ValueError(f"No RVs depend on marginalized RV {rv_to_marginalize}")

    ndim_supp = {rv.owner.op.ndim_supp for rv in dependent_rvs}
    if max(ndim_supp) > 0:
        raise NotImplementedError(
            "Marginalization of with dependent Multivariate RVs not implemented"
        )

    marginalized_rv_input_rvs = find_conditional_input_rvs([rv_to_marginalize], all_rvs)
    other_direct_rv_ancestors = [
        rv
        for rv in find_conditional_input_rvs(dependent_rvs, all_rvs)
        if rv is not rv_to_marginalize
    ]

    # If the marginalized RV has batched dimensions, check that graph between
    # marginalized RV and dependent RVs is composed strictly of Elemwise Operations.
    # This implies (?) that the dimensions are completely independent and a logp graph
    # can ultimately be generated that is proportional to the support domain and not
    # to the variables dimensions
    # We don't need to worry about this if the  RV is scalar.
    if any(not bcast for bcast in rv_to_marginalize.type.broadcastable):
        # When there are batch dimensions, we call `batch_dims_subgraph` to make sure these are not mixed
        try:
            dependent_rvs_batch_dims = subgraph_dims(
                rv_to_marginalize, other_direct_rv_ancestors, dependent_rvs
            )
        except ValueError as err:
            # This happens when information is mixed. From the user perspective this is a NotImplementedError
            raise NotImplementedError from err

        # We further check that any extra batch dimensions of dependnt RVs beyond those implied by the MarginalizedRV
        # show up on the left, so that collapsing logic in logp can be more straightforward easily.
        # This also ensures the MarginalizedRV still behaves as an RV itself
        marginal_batch_ndim = rv_to_marginalize.type.ndim - rv_to_marginalize.owner.op.ndim_supp
        marginal_batch_dims = tuple(range(marginal_batch_ndim))
        for dependent_rv, dependent_rv_batch_dims in zip(dependent_rvs, dependent_rvs_batch_dims):
            extra_batch_ndim = (
                dependent_rv.type.ndim - marginal_batch_ndim - dependent_rv.owner.op.ndim_supp
            )
            valid_dependent_batch_dims = ((None,) * extra_batch_ndim) + marginal_batch_dims
            if dependent_rv_batch_dims != valid_dependent_batch_dims:
                raise NotImplementedError(
                    "Any extra batch dimensions introduced by dependent RVs must be "
                    "on the left of dimensions introduced by the marginalized RV"
                )

        for dependent_rv, dependent_rv_batch_dims in zip(dependent_rvs, dependent_rvs_batch_dims):
            shared_batch_dims = [
                batch_dim for batch_dim in dependent_rv_batch_dims if batch_dim is not None
            ]
            if shared_batch_dims != sorted(shared_batch_dims):
                raise NotImplementedError(
                    "Shared batch dimensions between marginalized RV and dependent RVs must be aligned positionally"
                )

    input_rvs = [*marginalized_rv_input_rvs, *other_direct_rv_ancestors]
    rvs_to_marginalize = [rv_to_marginalize, *dependent_rvs]

    outputs = rvs_to_marginalize
    # Clone replace inner RV rng inputs so that we can be sure of the update order
    # replace_inputs = {rng: rng.type() for rng in updates_rvs_to_marginalize.keys()}
    # Clone replace outer RV inputs, so that their shared RNGs don't make it into
    # the inner graph of the marginalized RVs
    # FIXME: This shouldn't be needed!
    replace_inputs = {}
    replace_inputs.update({input_rv: input_rv.type() for input_rv in input_rvs})
    cloned_outputs = clone_replace(outputs, replace=replace_inputs)

    marginalization_op = FiniteDiscreteMarginalRV(
        inputs=list(replace_inputs.values()),
        outputs=cloned_outputs,
        ndim_supp=0,
    )
    marginalized_rvs = marginalization_op(*replace_inputs.keys())
    fgraph.replace_all(tuple(zip(rvs_to_marginalize, marginalized_rvs)))
    return rvs_to_marginalize, marginalized_rvs


def get_domain_of_finite_discrete_rv(rv: TensorVariable) -> Tuple[int, ...]:
    op = rv.owner.op
    if isinstance(op, Bernoulli):
        return (0, 1)
    elif isinstance(op, Categorical):
        p_param = rv.owner.inputs[3]
        return tuple(range(pt.get_vector_length(p_param)))
    elif isinstance(op, DiscreteUniform):
        lower, upper = constant_fold(rv.owner.inputs[3:])
        return tuple(range(lower, upper + 1))

    raise NotImplementedError(f"Cannot compute domain for op {op}")


@_logprob.register(FiniteDiscreteMarginalRV)
def finite_discrete_marginal_rv_logp(op, values, *inputs, **kwargs):
    # Clone the inner RV graph of the Marginalized RV
    marginalized_rvs_node = op.make_node(*inputs)
    inner_rvs = clone_replace(
        op.inner_outputs,
        replace={u: v for u, v in zip(op.inner_inputs, marginalized_rvs_node.inputs)},
    )
    marginalized_rv = inner_rvs[0]

    # Obtain the joint_logp graph of the inner RV graph
    inner_rvs_to_values = {rv: rv.clone() for rv in inner_rvs}
    logps_dict = conditional_logp(rv_values=inner_rvs_to_values, **kwargs)

    # Reduce logp dimensions corresponding to broadcasted variables
    joint_logp = logps_dict[inner_rvs_to_values[marginalized_rv]]
    joint_logp_ndim = joint_logp.type.ndim
    for inner_rv, inner_value in inner_rvs_to_values.items():
        if inner_rv is marginalized_rv:
            continue
        dependent_logp = logps_dict[inner_value]
        extra_dims = dependent_logp.type.ndim - joint_logp_ndim
        if extra_dims:
            dependent_logp = dependent_logp.sum(range(0, extra_dims))
        joint_logp += dependent_logp

    # Wrap the joint_logp graph in an OpFromGraph, so that we can evaluate it at different
    # values of the marginalized RV
    # Some inputs are not root inputs (such as transformed projections of value variables)
    # Or cannot be used as inputs to an OpFromGraph (shared variables and constants)
    inputs = list(inputvars(inputs))
    joint_logp_op = OpFromGraph(
        list(inner_rvs_to_values.values()) + inputs, [joint_logp], inline=True
    )

    # Compute the joint_logp for all possible n values of the marginalized RV. We assume
    # each original dimension is independent so that it suffices to evaluate the graph
    # n times, once with each possible value of the marginalized RV replicated across
    # batched dimensions of the marginalized RV

    # PyMC does not allow RVs in the logp graph, even if we are just using the shape
    marginalized_rv_shape = constant_fold(tuple(marginalized_rv.shape), raise_not_constant=False)
    marginalized_rv_domain = get_domain_of_finite_discrete_rv(marginalized_rv)
    marginalized_rv_domain_tensor = pt.moveaxis(
        pt.full(
            (*marginalized_rv_shape, len(marginalized_rv_domain)),
            marginalized_rv_domain,
            dtype=marginalized_rv.dtype,
        ),
        -1,
        0,
    )

    # Arbitrary cutoff to switch to Scan implementation to keep graph size under control
    if len(marginalized_rv_domain) <= 10:
        joint_logps = [
            joint_logp_op(marginalized_rv_domain_tensor[i], *values, *inputs)
            for i in range(len(marginalized_rv_domain))
        ]
    else:

        def logp_fn(marginalized_rv_const, *non_sequences):
            return joint_logp_op(marginalized_rv_const, *non_sequences)

        joint_logps, _ = scan_map(
            fn=logp_fn,
            sequences=marginalized_rv_domain_tensor,
            non_sequences=[*values, *inputs],
            mode=Mode().including("local_remove_check_parameter"),
        )

    joint_logps = pt.logsumexp(joint_logps, axis=0)

    # We have to add dummy logps for the remaining value variables, otherwise PyMC will raise
    return joint_logps, *(pt.constant(0),) * (len(values) - 1)
