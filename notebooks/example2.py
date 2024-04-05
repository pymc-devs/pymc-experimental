import pymc as pm

from pymc_experimental.inference.smc.sampling import sample_smc_blackjax
import numpy as np

with pm.Model() as model:
    a = pm.Normal("a", mu=10, sigma=10)
    b = pm.Normal("b", mu=10, sigma=10)
    # either of the following lines produces an error
    d = pm.Dirichlet("d", [1, 1])

    trace = sample_smc_blackjax(
        n_particles=1000,
        kernel="HMC",
        inner_kernel_params={
            "step_size": 0.01,
            "integration_steps": 20,
        },
        iterations_to_diagnose=10,
        target_essn=0.5,
        num_mcmc_steps=10,
    )


real_a = 0.2
real_b = 2
x = np.linspace(1, 100)
y = real_a * x + real_b + np.random.normal(0, 2, len(x))


with pm.Model() as model:
    a = pm.Normal("a", mu=10, sigma=10)
    b = pm.Normal("b", mu=10, sigma=10)
    # either of the following lines produces an error
    c = pm.Normal("c", mu=10, sigma=10, shape=(1,))

    trace = sample_smc_blackjax(
            n_particles=1000,
            kernel="HMC",
            inner_kernel_params={
                "step_size": 0.01,
                "integration_steps": 20,
            },
            iterations_to_diagnose=10,
            target_essn=0.5,
            num_mcmc_steps=10,
    )