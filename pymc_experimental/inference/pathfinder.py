#   Copyright 2022 The PyMC Developers
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

import collections
import sys
from typing import Optional

import arviz as az
import blackjax
import jax
import numpy as np
import pymc as pm
from packaging import version
from pymc.backends.arviz import coords_and_dims_for_inferencedata
from pymc.blocking import DictToArrayBijection, RaveledVars
from pymc.model import modelcontext
from pymc.sampling.jax import get_jaxified_graph
from pymc.util import RandomSeed, _get_seeds_per_chain, get_default_varnames


def convert_flat_trace_to_idata(
    samples,
    include_transformed=False,
    postprocessing_backend="cpu",
    model=None,
):

    model = modelcontext(model)
    ip = model.initial_point()
    ip_point_map_info = pm.blocking.DictToArrayBijection.map(ip).point_map_info
    trace = collections.defaultdict(list)
    for sample in samples:
        raveld_vars = RaveledVars(sample, ip_point_map_info)
        point = DictToArrayBijection.rmap(raveld_vars, ip)
        for p, v in point.items():
            trace[p].append(v.tolist())

    trace = {k: np.asarray(v)[None, ...] for k, v in trace.items()}

    var_names = model.unobserved_value_vars
    vars_to_sample = list(get_default_varnames(var_names, include_transformed=include_transformed))
    print("Transforming variables...", file=sys.stdout)
    jax_fn = get_jaxified_graph(inputs=model.value_vars, outputs=vars_to_sample)
    result = jax.vmap(jax.vmap(jax_fn))(
        *jax.device_put(list(trace.values()), jax.devices(postprocessing_backend)[0])
    )
    trace = {v.name: r for v, r in zip(vars_to_sample, result)}
    coords, dims = coords_and_dims_for_inferencedata(model)
    idata = az.from_dict(trace, dims=dims, coords=coords)

    return idata


def fit_pathfinder(
    samples=1000,
    random_seed: Optional[RandomSeed] = None,
    postprocessing_backend="cpu",
    model=None,
    **pathfinder_kwargs,
):
    """
    Fit the pathfinder algorithm as implemented in blackjax

    Requires the JAX backend

    Parameters
    ----------
    samples : int
        Number of samples to draw from the fitted approximation.
    random_seed : int
        Random seed to set.
    postprocessing_backend : str
        Where to compute transformations of the trace.
        "cpu" or "gpu".
    pathfinder_kwargs:
        kwargs for blackjax.vi.pathfinder.approximate

    Returns
    -------
    arviz.InferenceData

    Reference
    ---------
    https://arxiv.org/abs/2108.03782
    """
    # Temporarily helper
    if version.parse(blackjax.__version__).major < 1:
        raise ImportError("fit_pathfinder requires blackjax 1.0 or above")

    model = modelcontext(model)

    ip = model.initial_point()
    ip_map = DictToArrayBijection.map(ip)

    new_logprob, new_input = pm.pytensorf.join_nonshared_inputs(
        ip, (model.logp(),), model.value_vars, ()
    )

    logprob_fn_list = get_jaxified_graph([new_input], new_logprob)

    def logprob_fn(x):
        return logprob_fn_list(x)[0]

    [pathfinder_seed, sample_seed] = _get_seeds_per_chain(random_seed, 2)

    print("Running pathfinder...", file=sys.stdout)
    pathfinder_state, _ = blackjax.vi.pathfinder.approximate(
        rng_key=jax.random.key(pathfinder_seed),
        logdensity_fn=logprob_fn,
        initial_position=ip_map.data,
        **pathfinder_kwargs,
    )
    samples, _ = blackjax.vi.pathfinder.sample(
        rng_key=jax.random.key(sample_seed),
        state=pathfinder_state,
        num_samples=samples,
    )

    idata = convert_flat_trace_to_idata(
        samples,
        postprocessing_backend=postprocessing_backend,
        model=model,
    )
    return idata
