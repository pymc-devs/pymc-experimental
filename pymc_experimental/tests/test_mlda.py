import aesara
import aesara.tensor as at
import arviz as az
import numpy as np
import pytest
from aesara.graph.op import Op
from pymc import (
    Binomial,
    CompoundStep,
    ConstantData,
    DEMetropolisZ,
    Metropolis,
    Model,
    MultivariateNormalProposal,
    MutableData,
    MvNormal,
    Normal,
    NormalProposal,
    Potential,
    UniformProposal,
    sample,
    set_data,
)
from pymc.tests.checks import close_to
from pymc.tests.models import (
    mv_simple,
    mv_simple_coarse,
    mv_simple_very_coarse,
    simple_2model_continuous,
)

from pymc_experimental.step_methods.mlda import MLDA, RecursiveDAProposal, extract_Q_estimate


def check_stat(check, idata, name):
    group = idata.posterior
    for (var, stat, value, bound) in check:
        s = stat(group[var].sel(chain=0), axis=0)
        close_to(s, value, bound, name)


def check_stat_dtype(step, idata):
    # TODO: This check does not confirm the announced dtypes are correct as the
    #  sampling machinery will convert them automatically.
    for stats_dtypes in getattr(step, "stats_dtypes", []):
        for stat, dtype in stats_dtypes.items():
            if stat == "tune":
                continue
            assert idata.sample_stats[stat].dtype == np.dtype(dtype)


class TestMLDA:
    steppers = [MLDA]

    def test_proposal_and_base_proposal_choice(self):
        """Test that proposal_dist and base_proposal_dist are set as
        expected by MLDA"""
        _, model, _ = mv_simple()
        _, model_coarse, _ = mv_simple_coarse()
        with model:
            sampler = MLDA(coarse_models=[model_coarse], base_sampler="Metropolis")
            assert isinstance(sampler.proposal_dist, RecursiveDAProposal)
            assert sampler.base_proposal_dist is None
            assert isinstance(sampler.step_method_below.proposal_dist, NormalProposal)

            sampler = MLDA(coarse_models=[model_coarse])
            assert isinstance(sampler.proposal_dist, RecursiveDAProposal)
            assert sampler.base_proposal_dist is None
            assert isinstance(sampler.step_method_below.proposal_dist, UniformProposal)

            initial_point = model.initial_point()
            initial_point_size = sum(initial_point[n.name].size for n in model.value_vars)
            s = np.ones(initial_point_size)
            sampler = MLDA(coarse_models=[model_coarse], base_sampler="Metropolis", base_S=s)
            assert isinstance(sampler.proposal_dist, RecursiveDAProposal)
            assert sampler.base_proposal_dist is None
            assert isinstance(sampler.step_method_below.proposal_dist, NormalProposal)

            sampler = MLDA(coarse_models=[model_coarse], base_S=s)
            assert isinstance(sampler.proposal_dist, RecursiveDAProposal)
            assert sampler.base_proposal_dist is None
            assert isinstance(sampler.step_method_below.proposal_dist, UniformProposal)

            s = np.diag(s)
            sampler = MLDA(coarse_models=[model_coarse], base_sampler="Metropolis", base_S=s)
            assert isinstance(sampler.proposal_dist, RecursiveDAProposal)
            assert sampler.base_proposal_dist is None
            assert isinstance(sampler.step_method_below.proposal_dist, MultivariateNormalProposal)

            sampler = MLDA(coarse_models=[model_coarse], base_S=s)
            assert isinstance(sampler.proposal_dist, RecursiveDAProposal)
            assert sampler.base_proposal_dist is None
            assert isinstance(sampler.step_method_below.proposal_dist, UniformProposal)

            s[0, 0] = -s[0, 0]
            with pytest.raises(np.linalg.LinAlgError):
                MLDA(coarse_models=[model_coarse], base_sampler="Metropolis", base_S=s)

    def test_step_methods_in_each_level(self):
        """Test that MLDA creates the correct hierarchy of step methods when no
        coarse models are passed and when two coarse models are passed."""
        _, model, _ = mv_simple()
        _, model_coarse, _ = mv_simple_coarse()
        _, model_very_coarse, _ = mv_simple_very_coarse()
        with model:
            initial_point = model.initial_point()
            initial_point_size = sum(initial_point[n.name].size for n in model.value_vars)
            s = np.ones(initial_point_size) + 2.0
            sampler = MLDA(
                coarse_models=[model_very_coarse, model_coarse],
                base_S=s,
                base_sampler="Metropolis",
            )
            assert isinstance(sampler.step_method_below, MLDA)
            assert isinstance(sampler.step_method_below.step_method_below, Metropolis)
            assert np.all(sampler.step_method_below.step_method_below.proposal_dist.s == s)

            sampler = MLDA(coarse_models=[model_very_coarse, model_coarse], base_S=s)
            assert isinstance(sampler.step_method_below, MLDA)
            assert isinstance(sampler.step_method_below.step_method_below, DEMetropolisZ)
            assert np.all(sampler.step_method_below.step_method_below.proposal_dist.s == s)

    def test_exceptions_coarse_models(self):
        """Test that MLDA generates the expected exceptions when no coarse_models arg
        is passed, an empty list is passed or when coarse_models is not a list"""
        with pytest.raises(TypeError):
            _, model, _ = mv_simple()
            with model:
                MLDA()

        with pytest.raises(ValueError):
            _, model, _ = mv_simple()
            with model:
                MLDA(coarse_models=[])

        with pytest.raises(ValueError):
            _, model, _ = mv_simple()
            with model:
                MLDA(coarse_models=(model, model))

    def test_nonparallelized_chains_are_random(self):
        """Test that parallel chain are not identical when no parallelisation
        is applied"""
        with Model() as coarse_model:
            Normal("x", 0.3, 1)

        with Model():
            Normal("x", 0, 1)
            for stepper in TestMLDA.steppers:
                step = stepper(coarse_models=[coarse_model])
                idata = sample(chains=2, cores=1, draws=20, tune=0, step=step, random_seed=1)
                samples = idata.posterior["x"].values[:, 5]
                assert len(set(samples)) == 2, f"Non parallelized {stepper} chains are identical."

    def test_parallelized_chains_are_random(self):
        """Test that parallel chain are
        not identical when parallelisation
        is applied"""
        with Model() as coarse_model:
            Normal("x", 0.3, 1)

        with Model():
            Normal("x", 0, 1)
            for stepper in TestMLDA.steppers:
                step = stepper(coarse_models=[coarse_model])
                idata = sample(chains=2, cores=2, draws=20, tune=0, step=step, random_seed=1)
                samples = idata.posterior["x"].values[:, 5]
                assert len(set(samples)) == 2, f"Parallelized {stepper} chains are identical."

    def test_acceptance_rate_against_coarseness(self):
        """Test that the acceptance rate increases
        when the coarse model is closer to
        the fine model."""
        with Model() as coarse_model_0:
            Normal("x", 5.0, 1.0)

        with Model() as coarse_model_1:
            Normal("x", 6.0, 2.0)

        with Model() as coarse_model_2:
            Normal("x", 20.0, 5.0)

        possible_coarse_models = [coarse_model_0, coarse_model_1, coarse_model_2]
        acc = []

        with Model():
            Normal("x", 5.0, 1.0)
            for coarse_model in possible_coarse_models:
                step = MLDA(coarse_models=[coarse_model], subsampling_rates=3)
                idata = sample(chains=1, draws=500, tune=100, step=step, random_seed=1)
                acc.append(idata.sample_stats["accepted"].mean())
            assert acc[0] > acc[1] > acc[2], (
                "Acceptance rate is not "
                "strictly increasing when"
                "coarse model is closer to "
                "fine model. Acceptance rates"
                "were: {}".format(acc)
            )

    def test_mlda_non_blocked(self):
        """Test that MLDA correctly creates non-blocked
        compound steps in level 0 when using a Metropolis
        base sampler."""
        _, model = simple_2model_continuous()
        _, model_coarse = simple_2model_continuous()
        with model:
            for stepper in self.steppers:
                assert isinstance(
                    stepper(
                        coarse_models=[model_coarse],
                        base_sampler="Metropolis",
                        base_blocked=False,
                    ).step_method_below,
                    CompoundStep,
                )

    def test_mlda_blocked(self):
        """Test the type of base sampler instantiated
        when switching base_blocked flag while
        the base sampler is Metropolis and when
        the base sampler is DEMetropolisZ."""
        _, model = simple_2model_continuous()
        _, model_coarse = simple_2model_continuous()
        with model:
            for stepper in self.steppers:
                assert not isinstance(
                    stepper(
                        coarse_models=[model_coarse],
                        base_sampler="Metropolis",
                        base_blocked=True,
                    ).step_method_below,
                    CompoundStep,
                )
                assert isinstance(
                    stepper(
                        coarse_models=[model_coarse],
                        base_sampler="Metropolis",
                        base_blocked=True,
                    ).step_method_below,
                    Metropolis,
                )
                assert isinstance(
                    stepper(coarse_models=[model_coarse]).step_method_below,
                    DEMetropolisZ,
                )

    def test_tuning_and_scaling_on(self):
        """Test that tune and base_scaling change as expected when
        tuning is on."""
        np.random.seed(1234)
        ts = 100
        _, model = simple_2model_continuous()
        _, model_coarse = simple_2model_continuous()
        with model:
            trace_0 = sample(
                tune=ts,
                draws=20,
                step=MLDA(
                    coarse_models=[model_coarse],
                    base_sampler="Metropolis",
                    base_tune_interval=50,
                    base_scaling=100.0,
                ),
                chains=1,
                discard_tuned_samples=False,
                random_seed=1234,
                return_inferencedata=False,
            )

            trace_1 = sample(
                tune=ts,
                draws=20,
                step=MLDA(
                    coarse_models=[model_coarse],
                    base_tune_target="scaling",
                    base_tune_interval=50,
                    base_scaling=100.0,
                ),
                chains=1,
                discard_tuned_samples=False,
                random_seed=1234,
                return_inferencedata=False,
            )

            trace_2 = sample(
                tune=ts,
                draws=20,
                step=MLDA(
                    coarse_models=[model_coarse],
                    base_tune_interval=50,
                    base_scaling=10,
                    base_lamb=100.0,
                ),
                chains=1,
                discard_tuned_samples=False,
                random_seed=1234,
                return_inferencedata=False,
            )

        assert trace_0.get_sampler_stats("tune", chains=0)[0]
        assert trace_0.get_sampler_stats("tune", chains=0)[ts - 1]
        assert not trace_0.get_sampler_stats("tune", chains=0)[ts]
        assert not trace_0.get_sampler_stats("tune", chains=0)[-1]
        assert trace_0.get_sampler_stats("base_scaling", chains=0)[0, 0] == 100.0
        assert trace_0.get_sampler_stats("base_scaling", chains=0)[0, 1] == 100.0
        assert trace_0.get_sampler_stats("base_scaling", chains=0)[-1, 0] < 100.0
        assert trace_0.get_sampler_stats("base_scaling", chains=0)[-1, 1] < 100.0

        assert trace_1.get_sampler_stats("tune", chains=0)[0]
        assert trace_1.get_sampler_stats("tune", chains=0)[ts - 1]
        assert not trace_1.get_sampler_stats("tune", chains=0)[ts]
        assert not trace_1.get_sampler_stats("tune", chains=0)[-1]
        assert trace_1.get_sampler_stats("base_scaling", chains=0)[0] == 100.0
        assert trace_1.get_sampler_stats("base_scaling", chains=0)[-1] < 100.0

        assert trace_2.get_sampler_stats("tune", chains=0)[0]
        assert trace_2.get_sampler_stats("tune", chains=0)[ts - 1]
        assert not trace_2.get_sampler_stats("tune", chains=0)[ts]
        assert not trace_2.get_sampler_stats("tune", chains=0)[-1]
        assert trace_2.get_sampler_stats("base_lambda", chains=0)[0] == 100.0
        assert trace_2.get_sampler_stats("base_lambda", chains=0)[-1] < 100.0

    def test_tuning_and_scaling_off(self):
        """Test that tuning is deactivated when sample()'s tune=0 and that
        MLDA's tune=False is overridden by sample()'s tune."""
        np.random.seed(12345)
        _, model = simple_2model_continuous()
        _, model_coarse = simple_2model_continuous()

        ts_0 = 0
        with model:
            trace_0 = sample(
                tune=ts_0,
                draws=100,
                step=MLDA(
                    coarse_models=[model_coarse],
                    base_sampler="Metropolis",
                    base_tune_interval=50,
                    base_scaling=100.0,
                    tune=False,
                ),
                chains=1,
                discard_tuned_samples=False,
                random_seed=12345,
                return_inferencedata=False,
            )

        ts_1 = 100
        with model:
            trace_1 = sample(
                tune=ts_1,
                draws=20,
                step=MLDA(
                    coarse_models=[model_coarse],
                    base_sampler="Metropolis",
                    base_tune_interval=50,
                    base_scaling=100.0,
                    tune=False,
                ),
                chains=1,
                discard_tuned_samples=False,
                random_seed=12345,
                return_inferencedata=False,
            )

        assert not trace_0.get_sampler_stats("tune", chains=0)[0]
        assert not trace_0.get_sampler_stats("tune", chains=0)[-1]
        assert (
            trace_0.get_sampler_stats("base_scaling", chains=0)[0, 0]
            == trace_0.get_sampler_stats("base_scaling", chains=0)[-1, 0]
            == trace_0.get_sampler_stats("base_scaling", chains=0)[0, 1]
            == trace_0.get_sampler_stats("base_scaling", chains=0)[-1, 1]
            == 100.0
        )

        assert trace_1.get_sampler_stats("tune", chains=0)[0]
        assert trace_1.get_sampler_stats("tune", chains=0)[ts_1 - 1]
        assert not trace_1.get_sampler_stats("tune", chains=0)[ts_1]
        assert not trace_1.get_sampler_stats("tune", chains=0)[-1]
        assert trace_1.get_sampler_stats("base_scaling", chains=0)[0, 0] == 100.0
        assert trace_1.get_sampler_stats("base_scaling", chains=0)[0, 1] == 100.0
        assert trace_1.get_sampler_stats("base_scaling", chains=0)[-1, 0] < 100.0
        assert trace_1.get_sampler_stats("base_scaling", chains=0)[-1, 1] < 100.0

        ts_2 = 0
        with model:
            trace_2 = sample(
                tune=ts_2,
                draws=100,
                step=MLDA(
                    coarse_models=[model_coarse],
                    base_tune_interval=50,
                    base_lamb=100.0,
                    base_tune_target=None,
                ),
                chains=1,
                discard_tuned_samples=False,
                random_seed=12345,
                return_inferencedata=False,
            )

        assert not trace_2.get_sampler_stats("tune", chains=0)[0]
        assert not trace_2.get_sampler_stats("tune", chains=0)[-1]
        assert (
            trace_2.get_sampler_stats("base_lambda", chains=0)[0]
            == trace_2.get_sampler_stats("base_lambda", chains=0)[-1]
            == trace_2.get_sampler_stats("base_lambda", chains=0)[0]
            == trace_2.get_sampler_stats("base_lambda", chains=0)[-1]
            == 100.0
        )

    def test_trace_length(self):
        """Check if trace length is as expected."""
        tune = 100
        draws = 50
        with Model() as coarse_model:
            Normal("n", 0, 2.2, size=(3,))
        with Model():
            Normal("n", 0, 2, size=(3,))
            step = MLDA(coarse_models=[coarse_model])
            idata = sample(tune=tune, draws=draws, step=step, chains=1, discard_tuned_samples=False)
            assert len(idata.warmup_posterior.draw) == tune
            assert len(idata.posterior.draw) == draws

    @pytest.mark.parametrize(
        "variable,has_grad,outcome",
        [("n", True, 1), ("n", False, 1), ("b", True, 0), ("b", False, 0)],
    )
    def test_competence(self, variable, has_grad, outcome):
        """Test if competence function returns expected
        results for different models"""
        with Model() as pmodel:
            Normal("n", 0, 2, size=(3,))
            Binomial("b", n=2, p=0.3)
        assert MLDA.competence(pmodel[variable], has_grad=has_grad) == outcome

    def test_multiple_subsampling_rates(self):
        """Test that when you give a single integer it is applied to all levels and
        when you give a list the list is applied correctly."""
        with Model() as coarse_model_0:
            Normal("n", 0, 2.2, size=(3,))
        with Model() as coarse_model_1:
            Normal("n", 0, 2.1, size=(3,))
        with Model():
            Normal("n", 0, 2.0, size=(3,))

            step_1 = MLDA(coarse_models=[coarse_model_0, coarse_model_1], subsampling_rates=3)
            assert len(step_1.subsampling_rates) == 2
            assert step_1.subsampling_rates[0] == step_1.subsampling_rates[1] == 3

            step_2 = MLDA(coarse_models=[coarse_model_0, coarse_model_1], subsampling_rates=[3, 4])
            assert step_2.subsampling_rates[0] == 3
            assert step_2.subsampling_rates[1] == 4

            with pytest.raises(ValueError):
                step_3 = MLDA(
                    coarse_models=[coarse_model_0, coarse_model_1],
                    subsampling_rates=[3, 4, 10],
                )

    def test_aem_mu_sigma(self):
        """Test that AEM estimates mu_B and Sigma_B in
        the coarse models of a 3-level LR example correctly"""
        # create data for linear regression
        if aesara.config.floatX == "float32":
            p = "float32"
        else:
            p = "float64"
        np.random.seed(123456)
        size = 200
        true_intercept = 1
        true_slope = 2
        sigma = 1
        x = np.linspace(0, 1, size, dtype=p)
        # y = a + b*x
        true_regression_line = true_intercept + true_slope * x
        # add noise
        y = true_regression_line + np.random.normal(0, sigma**2, size)
        s = np.identity(y.shape[0], dtype=p)
        np.fill_diagonal(s, sigma**2)

        # forward model Op - here, just the regression equation
        class ForwardModel(Op):
            if aesara.config.floatX == "float32":
                itypes = [at.fvector]
                otypes = [at.fvector]
            else:
                itypes = [at.dvector]
                otypes = [at.dvector]

            def __init__(self, x, pymc_model):
                self.x = x
                self.pymc_model = pymc_model

            def perform(self, node, inputs, outputs):
                intercept = inputs[0][0]
                x_coeff = inputs[0][1]

                temp = intercept + x_coeff * x + self.pymc_model.bias.data
                with self.pymc_model:
                    set_data({"model_output": temp})
                outputs[0][0] = np.array(temp)

        # create the coarse models with separate biases
        mout = []
        coarse_models = []

        with Model() as coarse_model_0:
            bias = ConstantData("bias", 3.5 * np.ones(y.shape, dtype=p))
            mu_B = MutableData("mu_B", -1.3 * np.ones(y.shape, dtype=p))
            Sigma_B = MutableData("Sigma_B", np.zeros((y.shape[0], y.shape[0]), dtype=p))
            model_output = MutableData("model_output", np.zeros(y.shape, dtype=p))
            Sigma_e = ConstantData("Sigma_e", s)

            # Define priors
            intercept = Normal("Intercept", 0, sigma=20)
            x_coeff = Normal("x", 0, sigma=20)

            theta = at.as_tensor_variable([intercept, x_coeff])

            mout.append(ForwardModel(x, coarse_model_0))

            # Define likelihood
            likelihood = MvNormal("y", mu=mout[0](theta) + mu_B, cov=Sigma_e, observed=y)

            coarse_models.append(coarse_model_0)

        with Model() as coarse_model_1:
            bias = ConstantData("bias", 2.2 * np.ones(y.shape, dtype=p))
            mu_B = MutableData("mu_B", -2.2 * np.ones(y.shape, dtype=p))
            Sigma_B = MutableData("Sigma_B", np.zeros((y.shape[0], y.shape[0]), dtype=p))
            model_output = MutableData("model_output", np.zeros(y.shape, dtype=p))
            Sigma_e = ConstantData("Sigma_e", s)

            # Define priors
            intercept = Normal("Intercept", 0, sigma=20)
            x_coeff = Normal("x", 0, sigma=20)

            theta = at.as_tensor_variable([intercept, x_coeff])

            mout.append(ForwardModel(x, coarse_model_1))

            # Define likelihood
            likelihood = MvNormal("y", mu=mout[1](theta) + mu_B, cov=Sigma_e, observed=y)

            coarse_models.append(coarse_model_1)

        # fine model and inference
        with Model() as model:
            bias = ConstantData("bias", np.zeros(y.shape, dtype=p))
            model_output = MutableData("model_output", np.zeros(y.shape, dtype=p))
            Sigma_e = ConstantData("Sigma_e", s)

            # Define priors
            intercept = Normal("Intercept", 0, sigma=20)
            x_coeff = Normal("x", 0, sigma=20)

            theta = at.as_tensor_variable([intercept, x_coeff])

            mout.append(ForwardModel(x, model))

            # Define likelihood
            likelihood = MvNormal("y", mu=mout[-1](theta), cov=Sigma_e, observed=y)

            step_mlda = MLDA(coarse_models=coarse_models, adaptive_error_model=True)

            trace_mlda = sample(
                draws=100,
                step=step_mlda,
                chains=1,
                tune=200,
                discard_tuned_samples=True,
                random_seed=84759238,
            )

            m0 = step_mlda.step_method_below.model_below.mu_B.get_value()
            s0 = step_mlda.step_method_below.model_below.Sigma_B.get_value()
            m1 = step_mlda.model_below.mu_B.get_value()
            s1 = step_mlda.model_below.Sigma_B.get_value()

            assert np.allclose(m0, -3.5)
            assert np.allclose(m1, -2.2)
            assert np.allclose(s0, 0, atol=1e-3)
            assert np.allclose(s1, 0, atol=1e-3)

    def test_variance_reduction(self):
        """
        Test if the right stats are outputed when variance reduction is used in MLDA,
        if the output estimates are close (VR estimate vs. standard estimate from
        the first chain) and if the variance of VR is lower. Uses a linear regression
        model with multiple levels where approximate levels have fewer data.

        """
        # arithmetic precision
        if aesara.config.floatX == "float32":
            p = "float32"
        else:
            p = "float64"

        # set up the model and data
        seed = 12345
        np.random.seed(seed)
        size = 100
        true_intercept = 1
        true_slope = 2
        sigma = 0.1
        x = np.linspace(0, 1, size, dtype=p)
        # y = a + b*x
        true_regression_line = true_intercept + true_slope * x
        # add noise
        y = true_regression_line + np.random.normal(0, sigma**2, size)
        s = sigma

        x_coarse_0 = x[::3]
        y_coarse_0 = y[::3]
        x_coarse_1 = x[::2]
        y_coarse_1 = y[::2]

        # MCMC parameters
        ndraws = 200
        ntune = 100
        nsub = 3
        nchains = 1

        # define likelihoods with different Q
        class Likelihood(Op):
            if aesara.config.floatX == "float32":
                itypes = [at.fvector]
                otypes = [at.fscalar]
            else:
                itypes = [at.dvector]
                otypes = [at.dscalar]

            def __init__(self, x, y, pymc_model):
                self.x = x
                self.y = y
                self.pymc_model = pymc_model

            def perform(self, node, inputs, outputs):
                intercept = inputs[0][0]
                x_coeff = inputs[0][1]

                temp = np.array(intercept + x_coeff * self.x, dtype=p)
                with self.pymc_model:
                    set_data({"Q": np.array(x_coeff, dtype=p)})
                outputs[0][0] = np.array(
                    -(0.5 / s**2) * np.sum((temp - self.y) ** 2, dtype=p), dtype=p
                )

        # run four MLDA steppers for all combinations of
        # base_sampler and forward model
        for stepper in ["Metropolis", "DEMetropolisZ"]:
            mout = []
            coarse_models = []

            with Model() as coarse_model_0:
                if aesara.config.floatX == "float32":
                    Q = MutableData("Q", np.float32(0.0))
                else:
                    Q = MutableData("Q", np.float64(0.0))

                # Define priors
                intercept = Normal("Intercept", true_intercept, sigma=1)
                x_coeff = Normal("x", true_slope, sigma=1)

                theta = at.as_tensor_variable([intercept, x_coeff])

                mout.append(Likelihood(x_coarse_0, y_coarse_0, coarse_model_0))
                Potential("likelihood", mout[0](theta))

                coarse_models.append(coarse_model_0)

            with Model() as coarse_model_1:
                if aesara.config.floatX == "float32":
                    Q = MutableData("Q", np.float32(0.0))
                else:
                    Q = MutableData("Q", np.float64(0.0))

                # Define priors
                intercept = Normal("Intercept", true_intercept, sigma=1)
                x_coeff = Normal("x", true_slope, sigma=1)

                theta = at.as_tensor_variable([intercept, x_coeff])

                mout.append(Likelihood(x_coarse_1, y_coarse_1, coarse_model_1))
                Potential("likelihood", mout[1](theta))

                coarse_models.append(coarse_model_1)

            with Model() as model:
                if aesara.config.floatX == "float32":
                    Q = MutableData("Q", np.float32(0.0))
                else:
                    Q = MutableData("Q", np.float64(0.0))

                # Define priors
                intercept = Normal("Intercept", true_intercept, sigma=1)
                x_coeff = Normal("x", true_slope, sigma=1)

                theta = at.as_tensor_variable([intercept, x_coeff])

                mout.append(Likelihood(x, y, model))
                Potential("likelihood", mout[-1](theta))

                step = MLDA(
                    coarse_models=coarse_models,
                    base_sampler=stepper,
                    subsampling_rates=nsub,
                    variance_reduction=True,
                    store_Q_fine=True,
                )

                trace = sample(
                    draws=ndraws,
                    step=step,
                    chains=nchains,
                    tune=ntune,
                    cores=1,
                    discard_tuned_samples=True,
                    random_seed=seed,
                    return_inferencedata=False,
                )

                # get fine level stats (standard method)
                Q_2 = trace.get_sampler_stats("Q_2").reshape((nchains, ndraws))
                Q_mean_standard = Q_2.mean(axis=1).mean()
                Q_se_standard = np.sqrt(Q_2.var() / az.ess(np.array(Q_2, np.float64)))

                # get VR stats
                Q_mean_vr, Q_se_vr = extract_Q_estimate(trace, 3)

                # check that returned values are floats and finite.
                assert isinstance(Q_mean_standard, np.floating)
                assert np.isfinite(Q_mean_standard)
                assert isinstance(Q_mean_vr, np.floating)
                assert np.isfinite(Q_mean_vr)
                assert isinstance(Q_se_standard, np.floating)
                assert np.isfinite(Q_se_standard)
                assert isinstance(Q_se_vr, np.floating)
                assert np.isfinite(Q_se_vr)

                # check consistency of QoI across levels.
                Q_1_0 = np.concatenate(trace.get_sampler_stats("Q_1_0")).reshape(
                    (nchains, ndraws * nsub)
                )
                Q_2_1 = np.concatenate(trace.get_sampler_stats("Q_2_1")).reshape((nchains, ndraws))
                # This used to be a scrict zero equality!
                assert np.isclose(Q_1_0.mean(axis=1), 0.0, atol=1e-4)
                assert np.isclose(Q_2_1.mean(axis=1), 0.0, atol=1e-4)

    def test_step_continuous(self):
        def step_fn(C, model_coarse):
            return MLDA(
                coarse_models=[model_coarse],
                base_S=C,
                base_proposal_dist=MultivariateNormalProposal,
            )

        start, model, (mu, C) = mv_simple()
        unc = np.diag(C) ** 0.5
        check = (("x", np.mean, mu, unc / 10), ("x", np.std, unc, unc / 10))
        _, model_coarse, _ = mv_simple_coarse()
        with model:
            step = step_fn(C, model_coarse)
            idata = sample(
                tune=1000,
                draws=1000,
                chains=1,
                step=step,
                start=start,
                model=model,
                random_seed=1,
            )
            check_stat(check, idata, step.__class__.__name__)
            check_stat_dtype(idata, step)

    @aesara.config.change_flags({"floatX": "float64", "warn_float64": "ignore"})
    def test_float64_MLDA(self):
        data = np.random.randn(5)

        with Model() as coarse_model:
            x = Normal("x", initval=np.array(1.0, dtype="float64"))
            obs = Normal("obs", mu=x, sigma=1.0, observed=data + 0.5)

        with Model() as model:
            x = Normal("x", initval=np.array(1.0, dtype="float64"))
            obs = Normal("obs", mu=x, sigma=1.0, observed=data)

        assert x.dtype == "float64"
        assert obs.dtype == "float64"

        with model:
            sample(draws=10, tune=10, chains=1, step=MLDA(coarse_models=[coarse_model]))

    @aesara.config.change_flags({"floatX": "float32", "warn_float64": "warn"})
    def test_float32_MLDA(self):
        data = np.random.randn(5).astype("float32")

        with Model() as coarse_model:
            x = Normal("x", initval=np.array(1.0, dtype="float32"))
            obs = Normal("obs", mu=x, sigma=1.0, observed=data + 0.5)

        with Model() as model:
            x = Normal("x", initval=np.array(1.0, dtype="float32"))
            obs = Normal("obs", mu=x, sigma=1.0, observed=data)

        assert x.dtype == "float32"
        assert obs.dtype == "float32"

        with model:
            sample(draws=10, tune=10, chains=1, step=MLDA(coarse_models=[coarse_model]))
