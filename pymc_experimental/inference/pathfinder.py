import sys
import collections

import arviz as az
import numpy as np
import pymc as pm
from pymc import modelcontext
from pymc.util import get_default_varnames
from pymc.sampling_jax import get_jaxified_logp, get_jaxified_graph
import jax
import jax.numpy as jnp
import jax.random as random
import blackjax

def convert_flat_trace_to_idata(
    samples, dims=None, coords=None, include_transformed=False, postprocessing_backend="cpu", model=None,
):
    model = modelcontext(model)
    init_position_dict = model.initial_point()
    trace = collections.defaultdict(list)
    astart = pm.blocking.DictToArrayBijection.map(init_position_dict)
    for sample in samples:
        raveld_vars = pm.blocking.RaveledVars(sample, astart.point_map_info)
        point = pm.blocking.DictToArrayBijection.rmap(raveld_vars, init_position_dict)
        for p, v in point.items():
            trace[p].append(v.tolist())

    print("Creating trace...", file=sys.stdout)
    trace = {k: np.asarray(v)[None, ...] for k, v in trace.items()}

    var_names = model.unobserved_value_vars
    vars_to_sample = list(get_default_varnames(var_names, include_transformed=include_transformed))
    print("Transforming variables...", file=sys.stdout)
    jax_fn = get_jaxified_graph(inputs=model.value_vars, outputs=vars_to_sample)
    result = jax.vmap(jax.vmap(jax_fn))(
        *jax.device_put(list(trace.values()), jax.devices(postprocessing_backend)[0])
    )

    trace = {v.name: r for v, r in zip(vars_to_sample, result)}
    idata = az.from_dict(trace, dims=dims, coords=coords)

    return idata


def fit_pathfinder(iterations=5_000, model=None):
    model = modelcontext(model)

    rvs = [rv.name for rv in model.value_vars]
    init_position_dict = model.initial_point()
    init_position = [init_position_dict[rv] for rv in rvs]

    new_logprob, new_input = pm.aesaraf.join_nonshared_inputs(
        init_position_dict, (model.logp(),), model.value_vars, ()
    )

    logprob_fn_list = get_jaxified_graph([new_input], new_logprob)

    def logprob_fn(x):
        return logprob_fn_list(x)[0]

    dim = sum(v.size for v in init_position_dict.values())

    rng_key = random.PRNGKey(314)
    w0 = random.multivariate_normal(rng_key, 2.0 + jnp.zeros(dim), jnp.eye(dim))
    path = blackjax.vi.pathfinder.init(rng_key, logprob_fn, w0, return_path=True, ftol=1e-4)

    pathfinder = blackjax.kernels.pathfinder(rng_key, logprob_fn, ftol=1e-4)
    state = pathfinder.init(w0)

    def inference_loop(rng_key, kernel, initial_state, num_samples):
        @jax.jit
        def one_step(state, rng_key):
            state, info = kernel(rng_key, state)
            return state, (state, info)

        keys = jax.random.split(rng_key, num_samples)
        return jax.lax.scan(one_step, initial_state, keys)

    _, rng_key = random.split(rng_key)
    print("Running pathfinder...", file=sys.stdout)
    _, (_, samples) = inference_loop(rng_key, pathfinder.step, state, iterations)

    dims = {
        var_name: [dim for dim in dims if dim is not None]
        for var_name, dims in model.RV_dims.items()
    }

    idata = convert_flat_trace_to_idata(samples, coords=model.coords, dims=dims)

    return idata
