import warnings

from collections.abc import Sequence
from typing import Union

import numpy as np
import pymc
import pytensor.tensor as pt

from arviz import InferenceData, dict_to_dataset
from pymc.backends.arviz import coords_and_dims_for_inferencedata, dataset_to_point_list
from pymc.distributions.discrete import Bernoulli, Categorical, DiscreteUniform
from pymc.distributions.transforms import Chain
from pymc.logprob.transforms import IntervalTransform
from pymc.model import Model
from pymc.pytensorf import compile_pymc, constant_fold
from pymc.util import RandomState, _get_seeds_per_chain, treedict
from pytensor.compile import SharedVariable
from pytensor.graph import FunctionGraph, clone_replace, graph_inputs
from pytensor.graph.replace import vectorize_graph
from pytensor.tensor import TensorVariable
from pytensor.tensor.special import log_softmax

__all__ = ["MarginalModel", "marginalize"]

from pymc_experimental.distributions import DiscreteMarkovChain
from pymc_experimental.model.marginal.distributions import (
    MarginalDiscreteMarkovChainRV,
    MarginalFiniteDiscreteRV,
    get_domain_of_finite_discrete_rv,
    reduce_batch_dependent_logps,
)
from pymc_experimental.model.marginal.graph_analysis import (
    find_conditional_dependent_rvs,
    find_conditional_input_rvs,
    is_conditional_dependent,
    subgraph_batch_dim_connection,
)

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
            elif not isinstance(rv_op, Bernoulli | Categorical | DiscreteUniform):
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
        """Create a function from the untransformed space to the transformed space"""
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
            dependent_rvs = find_conditional_dependent_rvs(marginalized_rv, m.basic_RVs)
            logps = m.logp(vars=[marginalized_rv, *dependent_rvs], sum=False)

            # Handle batch dims for marginalized value and its dependent RVs
            dependent_rvs_dim_connections = subgraph_batch_dim_connection(
                marginalized_rv, dependent_rvs
            )
            marginalized_logp, *dependent_logps = logps
            joint_logp = marginalized_logp + reduce_batch_dependent_logps(
                dependent_rvs_dim_connections,
                [dependent_var.owner.op for dependent_var in dependent_rvs],
                dependent_logps,
            )

            marginalized_value = m.rvs_to_values[marginalized_rv]
            other_values = [v for v in m.value_vars if v is not marginalized_value]

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

            batched_joint_logp = vectorize_graph(
                joint_logp,
                replace={marginalized_value: rv_domain_tensor},
            )
            batched_joint_logp = pt.moveaxis(batched_joint_logp, 0, -1)

            joint_logp_norm = log_softmax(batched_joint_logp, axis=-1)
            if return_samples:
                rv_draws = pymc.Categorical.dist(logit_p=batched_joint_logp)
                if isinstance(marginalized_rv.owner.op, DiscreteUniform):
                    rv_draws += rv_domain[0]
                outputs = [joint_logp_norm, rv_draws]
            else:
                outputs = joint_logp_norm

            rv_loglike_fn = compile_pymc(
                inputs=other_values,
                outputs=outputs,
                on_unused_input="ignore",
                random_seed=seed,
            )

            logvs = [rv_loglike_fn(**vs) for vs in posterior_pts]

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


def collect_shared_vars(outputs, blockers):
    return [
        inp for inp in graph_inputs(outputs, blockers=blockers) if isinstance(inp, SharedVariable)
    ]


def replace_finite_discrete_marginal_subgraph(fgraph, rv_to_marginalize, all_rvs):
    dependent_rvs = find_conditional_dependent_rvs(rv_to_marginalize, all_rvs)
    if not dependent_rvs:
        raise ValueError(f"No RVs depend on marginalized RV {rv_to_marginalize}")

    marginalized_rv_input_rvs = find_conditional_input_rvs([rv_to_marginalize], all_rvs)
    other_direct_rv_ancestors = [
        rv
        for rv in find_conditional_input_rvs(dependent_rvs, all_rvs)
        if rv is not rv_to_marginalize
    ]

    # If the marginalized RV has multiple dimensions, check that graph between
    # marginalized RV and dependent RVs does not mix information from batch dimensions
    # (otherwise logp would require enumerating over all combinations of batch dimension values)
    try:
        dependent_rvs_dim_connections = subgraph_batch_dim_connection(
            rv_to_marginalize, dependent_rvs
        )
    except (ValueError, NotImplementedError) as e:
        # For the perspective of the user this is a NotImplementedError
        raise NotImplementedError(
            "The graph between the marginalized and dependent RVs cannot be marginalized efficiently. "
            "You can try splitting the marginalized RV into separate components and marginalizing them separately."
        ) from e

    input_rvs = list(set((*marginalized_rv_input_rvs, *other_direct_rv_ancestors)))
    output_rvs = [rv_to_marginalize, *dependent_rvs]

    # We are strict about shared variables in SymbolicRandomVariables
    inputs = input_rvs + collect_shared_vars(output_rvs, blockers=input_rvs)

    if isinstance(rv_to_marginalize.owner.op, DiscreteMarkovChain):
        marginalize_constructor = MarginalDiscreteMarkovChainRV
    else:
        marginalize_constructor = MarginalFiniteDiscreteRV

    marginalization_op = marginalize_constructor(
        inputs=inputs,
        outputs=output_rvs,  # TODO: Add RNG updates to outputs so this can be used in the generative graph
        dims_connections=dependent_rvs_dim_connections,
    )
    new_output_rvs = marginalization_op(*inputs)
    fgraph.replace_all(tuple(zip(output_rvs, new_output_rvs)))
    return output_rvs, new_output_rvs
