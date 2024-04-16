import warnings
from typing import Sequence, Union

import numpy as np
import pymc
import pytensor.tensor as pt
from arviz import InferenceData, dict_to_dataset
from pymc import SymbolicRandomVariable
from pymc.backends.arviz import coords_and_dims_for_inferencedata, dataset_to_point_list
from pymc.distributions.discrete import Bernoulli, Categorical, DiscreteUniform
from pymc.distributions.transforms import Chain
from pymc.logprob.abstract import _logprob
from pymc.logprob.basic import conditional_logp, logp
from pymc.logprob.transforms import IntervalTransform
from pymc.model import Model
from pymc.pytensorf import compile_pymc, constant_fold
from pymc.util import RandomState, _get_seeds_per_chain, treedict
from pytensor import Mode, scan
from pytensor.compile import SharedVariable
from pytensor.graph import Constant, FunctionGraph, ancestors, clone_replace
from pytensor.graph.replace import graph_replace, vectorize_graph
from pytensor.scan import map as scan_map
from pytensor.tensor import TensorType, TensorVariable
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.shape import Shape
from pytensor.tensor.special import log_softmax

__all__ = ["MarginalModel", "marginalize"]

from pymc_experimental.distributions import DiscreteMarkovChain

ModelRVs = TensorVariable | Sequence[TensorVariable] | str | Sequence[str]


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

    @staticmethod
    def from_model(model: Union[Model, "MarginalModel"]) -> "MarginalModel":
        new_model = MarginalModel(coords=model.coords)
        if isinstance(model, MarginalModel):
            marginalized_rvs = model.marginalized_rvs
            marginalized_named_vars_to_dims = model._marginalized_named_vars_to_dims
        else:
            marginalized_rvs = []
            marginalized_named_vars_to_dims = {}

        model_vars = model.basic_RVs + model.potentials + model.deterministics + marginalized_rvs
        data_vars = [var for name, var in model.named_vars.items() if var not in model_vars]
        vars = model_vars + data_vars
        cloned_vars = clone_replace(vars)
        vars_to_clone = {var: cloned_var for var, cloned_var in zip(vars, cloned_vars)}
        new_model.vars_to_clone = vars_to_clone

        new_model.named_vars = treedict(
            {name: vars_to_clone[var] for name, var in model.named_vars.items()}
        )
        new_model.named_vars_to_dims = model.named_vars_to_dims
        new_model.values_to_rvs = {vv: vars_to_clone[rv] for vv, rv in model.values_to_rvs.items()}
        new_model.rvs_to_values = {vars_to_clone[rv]: vv for rv, vv in model.rvs_to_values.items()}
        new_model.rvs_to_transforms = {
            vars_to_clone[rv]: tr for rv, tr in model.rvs_to_transforms.items()
        }
        new_model.rvs_to_initial_values = {
            vars_to_clone[rv]: iv for rv, iv in model.rvs_to_initial_values.items()
        }
        new_model.free_RVs = [vars_to_clone[rv] for rv in model.free_RVs]
        new_model.observed_RVs = [vars_to_clone[rv] for rv in model.observed_RVs]
        new_model.potentials = [vars_to_clone[pot] for pot in model.potentials]
        new_model.deterministics = [vars_to_clone[det] for det in model.deterministics]

        new_model.marginalized_rvs = [vars_to_clone[rv] for rv in marginalized_rvs]
        new_model._marginalized_named_vars_to_dims = marginalized_named_vars_to_dims
        return new_model

    def clone(self):
        return self.from_model(self)

    def marginalize(
        self,
        rvs_to_marginalize: ModelRVs,
    ):
        if not isinstance(rvs_to_marginalize, Sequence):
            rvs_to_marginalize = (rvs_to_marginalize,)
        rvs_to_marginalize = [
            self[var] if isinstance(var, str) else var for var in rvs_to_marginalize
        ]

        for rv_to_marginalize in rvs_to_marginalize:
            if rv_to_marginalize not in self.free_RVs:
                raise ValueError(
                    f"Marginalized RV {rv_to_marginalize} is not a free RV in the model"
                )

            rv_op = rv_to_marginalize.owner.op
            if isinstance(rv_op, DiscreteMarkovChain):
                if rv_op.n_lags > 1:
                    raise NotImplementedError(
                        "Marginalization for DiscreteMarkovChain with n_lags > 1 is not supported"
                    )
                if rv_to_marginalize.owner.inputs[0].type.ndim > 2:
                    raise NotImplementedError(
                        "Marginalization for DiscreteMarkovChain with non-matrix transition probability is not supported"
                    )
            elif not isinstance(rv_op, (Bernoulli, Categorical, DiscreteUniform)):
                raise NotImplementedError(
                    f"Marginalization of RV with distribution {rv_to_marginalize.owner.op} is not supported"
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

    def unmarginalize(self, rvs_to_unmarginalize: Sequence[TensorVariable | str]):
        for rv in rvs_to_unmarginalize:
            if isinstance(rv, str):
                rv = self[rv]
            self.marginalized_rvs.remove(rv)
            if rv.name in self._marginalized_named_vars_to_dims:
                dims = self._marginalized_named_vars_to_dims.pop(rv.name)
            else:
                dims = None
            self.register_rv(rv, name=rv.name, dims=dims)

    def recover_marginals(
        self,
        idata: InferenceData,
        var_names: Sequence[str] | None = None,
        return_samples: bool = True,
        extend_inferencedata: bool = True,
        random_seed: RandomState = None,
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
        for seed, marginalized_rv in zip(seeds, vars_to_recover):
            supported_dists = (Bernoulli, Categorical, DiscreteUniform)
            if not isinstance(marginalized_rv.owner.op, supported_dists):
                raise NotImplementedError(
                    f"RV with distribution {marginalized_rv.owner.op} cannot be recovered. "
                    f"Supported distribution include {supported_dists}"
                )

            m = self.clone()
            marginalized_rv = m.vars_to_clone[marginalized_rv]
            m.unmarginalize([marginalized_rv])
            dependent_vars = find_conditional_dependent_rvs(marginalized_rv, m.basic_RVs)
            joint_logps = m.logp(vars=[marginalized_rv] + dependent_vars, sum=False)

            marginalized_value = m.rvs_to_values[marginalized_rv]
            other_values = [v for v in m.value_vars if v is not marginalized_value]

            # Handle batch dims for marginalized value and its dependent RVs
            marginalized_logp, *dependent_logps = joint_logps
            joint_logp = marginalized_logp + _add_reduce_batch_dependent_logps(
                marginalized_rv.type, dependent_logps
            )

            rv_shape = constant_fold(tuple(marginalized_rv.shape), raise_not_constant=False)
            rv_domain = get_domain_of_finite_discrete_rv(marginalized_rv)
            rv_domain_tensor = pt.moveaxis(
                pt.full(
                    (*rv_shape, len(rv_domain)),
                    rv_domain,
                    dtype=marginalized_rv.dtype,
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
                if isinstance(marginalized_rv.owner.op, DiscreteUniform):
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
                rv_dict[marginalized_rv.name] = samples.reshape(
                    tuple(len(coord) for coord in stacked_dims.values()) + samples.shape[1:],
                )
            else:
                logps = np.array(logvs)

            rv_dict["lp_" + marginalized_rv.name] = logps.reshape(
                tuple(len(coord) for coord in stacked_dims.values()) + logps.shape[1:],
            )
            if marginalized_rv.name in m.named_vars_to_dims:
                rv_dims[marginalized_rv.name] = list(m.named_vars_to_dims[marginalized_rv.name])
                rv_dims["lp_" + marginalized_rv.name] = rv_dims[marginalized_rv.name] + [
                    "lp_" + marginalized_rv.name + "_dim"
                ]

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


def marginalize(model: Model, rvs_to_marginalize: ModelRVs) -> MarginalModel:
    """Marginalize a subset of variables in a PyMC model.

    This creates a class of `MarginalModel` from an existing `Model`, with the specified
    variables marginalized.

    See documentation for `MarginalModel` for more information.

    Parameters
    ----------
    model : Model
        PyMC model to marginalize. Original variables well be cloned.
    rvs_to_marginalize : Sequence[TensorVariable]
        Variables to marginalize in the returned model.

    Returns
    -------
    marginal_model: MarginalModel
        Marginal model with the specified variables marginalized.
    """
    if not isinstance(rvs_to_marginalize, tuple | list):
        rvs_to_marginalize = (rvs_to_marginalize,)
    rvs_to_marginalize = [rv if isinstance(rv, str) else rv.name for rv in rvs_to_marginalize]

    marginal_model = MarginalModel.from_model(model)
    marginal_model.marginalize(rvs_to_marginalize)
    return marginal_model


class MarginalRV(SymbolicRandomVariable):
    """Base class for Marginalized RVs"""


class FiniteDiscreteMarginalRV(MarginalRV):
    """Base class for Finite Discrete Marginalized RVs"""


class DiscreteMarginalMarkovChainRV(MarginalRV):
    """Base class for Discrete Marginal Markov Chain RVs"""


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


def is_elemwise_subgraph(rv_to_marginalize, other_input_rvs, output_rvs):
    # TODO: No need to consider apply nodes outside the subgraph...
    fg = FunctionGraph(outputs=output_rvs, clone=False)

    non_elemwise_blockers = [
        o for node in fg.apply_nodes if not isinstance(node.op, Elemwise) for o in node.outputs
    ]
    blocker_candidates = [rv_to_marginalize] + other_input_rvs + non_elemwise_blockers
    blockers = [var for var in blocker_candidates if var not in output_rvs]

    truncated_inputs = [
        var
        for var in ancestors(output_rvs, blockers=blockers)
        if (
            var in blockers
            or (var.owner is None and not isinstance(var, (Constant, SharedVariable)))
        )
    ]

    # Check that we reach the marginalized rv following a pure elemwise graph
    if rv_to_marginalize not in truncated_inputs:
        return False

    # Check that none of the truncated inputs depends on the marginalized_rv
    other_truncated_inputs = [inp for inp in truncated_inputs if inp is not rv_to_marginalize]
    # TODO: We don't need to go all the way to the root variables
    if rv_to_marginalize in ancestors(
        other_truncated_inputs, blockers=[rv_to_marginalize, *other_input_rvs]
    ):
        return False
    return True


from pytensor.graph.basic import graph_inputs


def collect_shared_vars(outputs, blockers):
    return [
        inp for inp in graph_inputs(outputs, blockers=blockers) if isinstance(inp, SharedVariable)
    ]


def replace_finite_discrete_marginal_subgraph(fgraph, rv_to_marginalize, all_rvs):
    # TODO: This should eventually be integrated in a more general routine that can
    #  identify other types of supported marginalization, of which finite discrete
    #  RVs is just one

    dependent_rvs = find_conditional_dependent_rvs(rv_to_marginalize, all_rvs)
    if not dependent_rvs:
        raise ValueError(f"No RVs depend on marginalized RV {rv_to_marginalize}")

    ndim_supp = {rv.owner.op.ndim_supp for rv in dependent_rvs}
    if len(ndim_supp) != 1:
        raise NotImplementedError(
            "Marginalization with dependent variables of different support dimensionality not implemented"
        )
    [ndim_supp] = ndim_supp
    if ndim_supp > 0:
        raise NotImplementedError("Marginalization with dependent Multivariate RVs not implemented")

    marginalized_rv_input_rvs = find_conditional_input_rvs([rv_to_marginalize], all_rvs)
    dependent_rvs_input_rvs = [
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
    if np.prod(constant_fold(tuple(rv_to_marginalize.shape), raise_not_constant=False)) != 1:
        if not is_elemwise_subgraph(rv_to_marginalize, dependent_rvs_input_rvs, dependent_rvs):
            raise NotImplementedError(
                "The subgraph between a marginalized RV and its dependents includes non Elemwise operations. "
                "This is currently not supported",
            )

    input_rvs = [*marginalized_rv_input_rvs, *dependent_rvs_input_rvs]
    rvs_to_marginalize = [rv_to_marginalize, *dependent_rvs]

    outputs = rvs_to_marginalize
    # We are strict about shared variables in SymbolicRandomVariables
    inputs = input_rvs + collect_shared_vars(rvs_to_marginalize, blockers=input_rvs)

    if isinstance(rv_to_marginalize.owner.op, DiscreteMarkovChain):
        marginalize_constructor = DiscreteMarginalMarkovChainRV
    else:
        marginalize_constructor = FiniteDiscreteMarginalRV

    marginalization_op = marginalize_constructor(
        inputs=inputs,
        outputs=outputs,
        ndim_supp=ndim_supp,
    )

    marginalized_rvs = marginalization_op(*inputs)
    fgraph.replace_all(tuple(zip(rvs_to_marginalize, marginalized_rvs)))
    return rvs_to_marginalize, marginalized_rvs


def get_domain_of_finite_discrete_rv(rv: TensorVariable) -> tuple[int, ...]:
    op = rv.owner.op
    if isinstance(op, Bernoulli):
        return (0, 1)
    elif isinstance(op, Categorical):
        p_param = rv.owner.inputs[3]
        return tuple(range(pt.get_vector_length(p_param)))
    elif isinstance(op, DiscreteUniform):
        lower, upper = constant_fold(rv.owner.inputs[3:])
        return tuple(np.arange(lower, upper + 1))
    elif isinstance(op, DiscreteMarkovChain):
        P = rv.owner.inputs[0]
        return tuple(range(pt.get_vector_length(P[-1])))

    raise NotImplementedError(f"Cannot compute domain for op {op}")


def _add_reduce_batch_dependent_logps(
    marginalized_type: TensorType, dependent_logps: Sequence[TensorVariable]
):
    """Add the logps of dependent RVs while reducing extra batch dims relative to `marginalized_type`."""

    mbcast = marginalized_type.broadcastable
    reduced_logps = []
    for dependent_logp in dependent_logps:
        dbcast = dependent_logp.type.broadcastable
        dim_diff = len(dbcast) - len(mbcast)
        mbcast_aligned = (True,) * dim_diff + mbcast
        vbcast_axis = [i for i, (m, v) in enumerate(zip(mbcast_aligned, dbcast)) if m and not v]
        reduced_logps.append(dependent_logp.sum(vbcast_axis))
    return pt.add(*reduced_logps)


@_logprob.register(FiniteDiscreteMarginalRV)
def finite_discrete_marginal_rv_logp(op, values, *inputs, **kwargs):
    # Clone the inner RV graph of the Marginalized RV
    marginalized_rvs_node = op.make_node(*inputs)
    marginalized_rv, *inner_rvs = clone_replace(
        op.inner_outputs,
        replace={u: v for u, v in zip(op.inner_inputs, marginalized_rvs_node.inputs)},
    )

    # Obtain the joint_logp graph of the inner RV graph
    inner_rv_values = dict(zip(inner_rvs, values))
    marginalized_vv = marginalized_rv.clone()
    rv_values = inner_rv_values | {marginalized_rv: marginalized_vv}
    logps_dict = conditional_logp(rv_values=rv_values, **kwargs)

    # Reduce logp dimensions corresponding to broadcasted variables
    marginalized_logp = logps_dict.pop(marginalized_vv)
    joint_logp = marginalized_logp + _add_reduce_batch_dependent_logps(
        marginalized_rv.type, logps_dict.values()
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

    try:
        joint_logps = vectorize_graph(
            joint_logp, replace={marginalized_vv: marginalized_rv_domain_tensor}
        )
    except Exception:
        # Fallback to Scan
        def logp_fn(marginalized_rv_const, *non_sequences):
            return graph_replace(joint_logp, replace={marginalized_vv: marginalized_rv_const})

        joint_logps, _ = scan_map(
            fn=logp_fn,
            sequences=marginalized_rv_domain_tensor,
            non_sequences=[*values, *inputs],
            mode=Mode().including("local_remove_check_parameter"),
        )

    joint_logps = pt.logsumexp(joint_logps, axis=0)

    # We have to add dummy logps for the remaining value variables, otherwise PyMC will raise
    return joint_logps, *(pt.constant(0),) * (len(values) - 1)


@_logprob.register(DiscreteMarginalMarkovChainRV)
def marginal_hmm_logp(op, values, *inputs, **kwargs):

    marginalized_rvs_node = op.make_node(*inputs)
    inner_rvs = clone_replace(
        op.inner_outputs,
        replace={u: v for u, v in zip(op.inner_inputs, marginalized_rvs_node.inputs)},
    )

    chain_rv, *dependent_rvs = inner_rvs
    P, n_steps_, init_dist_, rng = chain_rv.owner.inputs
    domain = pt.arange(P.shape[-1], dtype="int32")

    # Construct logp in two steps
    # Step 1: Compute the probability of the data ("emissions") under every possible state (vec_logp_emission)

    # First we need to vectorize the conditional logp graph of the data, in case there are batch dimensions floating
    # around. To do this, we need to break the dependency between chain and the init_dist_ random variable. Otherwise,
    # PyMC will detect a random variable in the logp graph (init_dist_), that isn't relevant at this step.
    chain_value = chain_rv.clone()
    dependent_rvs = clone_replace(dependent_rvs, {chain_rv: chain_value})
    logp_emissions_dict = conditional_logp(dict(zip(dependent_rvs, values)))

    # Reduce and add the batch dims beyond the chain dimension
    reduced_logp_emissions = _add_reduce_batch_dependent_logps(
        chain_rv.type, logp_emissions_dict.values()
    )

    # Add a batch dimension for the domain of the chain
    chain_shape = constant_fold(tuple(chain_rv.shape))
    batch_chain_value = pt.moveaxis(pt.full((*chain_shape, domain.size), domain), -1, 0)
    batch_logp_emissions = vectorize_graph(reduced_logp_emissions, {chain_value: batch_chain_value})

    # Step 2: Compute the transition probabilities
    # This is the "forward algorithm", alpha_t = p(y | s_t) * sum_{s_{t-1}}(p(s_t | s_{t-1}) * alpha_{t-1})
    # We do it entirely in logs, though.

    # To compute the prior probabilities of each state, we evaluate the logp of the domain (all possible states) under
    # the initial distribution. This is robust to everything the user can throw at it.
    batch_logp_init_dist = pt.vectorize(lambda x: logp(init_dist_, x), "()->()")(
        batch_chain_value[..., 0]
    )
    log_alpha_init = batch_logp_init_dist + batch_logp_emissions[..., 0]

    def step_alpha(logp_emission, log_alpha, log_P):
        step_log_prob = pt.logsumexp(log_alpha[:, None] + log_P, axis=0)
        return logp_emission + step_log_prob

    P_bcast_dims = (len(chain_shape) - 1) - (P.type.ndim - 2)
    log_P = pt.shape_padright(pt.log(P), P_bcast_dims)
    log_alpha_seq, _ = scan(
        step_alpha,
        non_sequences=[log_P],
        outputs_info=[log_alpha_init],
        # Scan needs the time dimension first, and we already consumed the 1st logp computing the initial value
        sequences=pt.moveaxis(batch_logp_emissions[..., 1:], -1, 0),
    )
    # Final logp is just the sum of the last scan state
    joint_logp = pt.logsumexp(log_alpha_seq[-1], axis=0)

    # If there are multiple emission streams, we have to add dummy logps for the remaining value variables. The first
    # return is the joint probability of everything together, but PyMC still expects one logp for each one.
    dummy_logps = (pt.constant(0),) * (len(values) - 1)
    return joint_logp, *dummy_logps
