import numpy as np
import pymc as pm

# general imports
import pytensor.tensor as pt
import pytest
from pymc.logprob.utils import ParameterValueError

from pymc_experimental.distributions.timeseries import DiscreteMarkovChain


class TestDiscreteMarkovRV:
    def test_fail_if_P_not_square(self):
        P = pt.eye(3, 2)
        chain = DiscreteMarkovChain.dist(P=P, steps=3)
        with pytest.raises(ParameterValueError):
            pm.logp(chain, np.zeros((3,))).eval()

    def test_fail_if_P_not_valid(self):
        P = pt.zeros((3, 3))
        chain = DiscreteMarkovChain.dist(P=P, steps=3)
        with pytest.raises(ParameterValueError):
            pm.logp(chain, np.zeros((3,))).eval()

    def test_default_init_dist_warns_user(self):
        P = pt.as_tensor_variable(np.array([[0.1, 0.5, 0.4], [0.3, 0.4, 0.3], [0.9, 0.05, 0.05]]))

        with pytest.warns(UserWarning):
            DiscreteMarkovChain.dist(P=P, steps=3)

    def test_logp_with_default_init_dist(self):
        P = pt.as_tensor_variable(np.array([[0.1, 0.5, 0.4], [0.3, 0.4, 0.3], [0.9, 0.05, 0.05]]))
        chain = DiscreteMarkovChain.dist(P=P, steps=3)

        logp = pm.logp(chain, [0, 1, 2]).eval()
        assert logp == np.log((1 / 3) * 0.5 * 0.3)

    def test_logp_with_user_defined_init_dist(self):
        P = pt.as_tensor_variable(np.array([[0.1, 0.5, 0.4], [0.3, 0.4, 0.3], [0.9, 0.05, 0.05]]))
        x0 = pm.Categorical.dist(p=[0.2, 0.6, 0.2])
        chain = DiscreteMarkovChain.dist(P=P, init_dist=x0, steps=3)

        logp = pm.logp(chain, [0, 1, 2]).eval()
        assert logp == np.log(0.2 * 0.5 * 0.3)

    def test_define_steps_via_shape_arg(self):
        P = pt.full((3, 3), 1 / 3)
        chain = DiscreteMarkovChain.dist(P=P, shape=(3,))

        assert chain.eval().shape[0] == 3

    def test_define_steps_via_dim_arg(self):
        coords = {"steps": [1, 2, 3]}

        with pm.Model(coords=coords):
            P = pt.full((3, 3), 1 / 3)
            chain = DiscreteMarkovChain("chain", P=P, dims=["steps"])

        assert chain.eval().shape[0] == 3

    def test_dims_when_steps_are_defined(self):
        coords = {"steps": [1, 2, 3]}

        with pm.Model(coords=coords):
            P = pt.full((3, 3), 1 / 3)
            chain = DiscreteMarkovChain("chain", P=P, steps=3, dims=["steps"])

        assert chain.eval().shape == (3,)

    def test_multiple_dims_with_steps(self):
        coords = {"steps": [1, 2, 3], "mc_chains": [1, 2, 3]}

        with pm.Model(coords=coords):
            P = pt.full((3, 3), 1 / 3)
            chain = DiscreteMarkovChain("chain", P=P, steps=3, dims=["steps", "mc_chains"])

        assert chain.eval().shape == (3, 3)

    def test_mutiple_dims_with_steps_and_init_dist(self):
        coords = {"steps": [1, 2, 3], "mc_chains": [1, 2, 3]}

        with pm.Model(coords=coords):
            P = pt.full((3, 3), 1 / 3)
            x0 = pm.Categorical.dist(p=[0.1, 0.1, 0.8], size=(3,))
            chain = DiscreteMarkovChain(
                "chain", P=P, init_dist=x0, steps=3, dims=["steps", "mc_chains"]
            )

        assert chain.eval().shape == (3, 3)

    def test_random_draws(self):
        steps = 3
        n_states = 2
        n_draws = 2500
        atol = 0.05

        P = np.full((n_states, n_states), 1 / n_states)
        chain = DiscreteMarkovChain.dist(P=pt.as_tensor_variable(P), steps=steps)

        draws = pm.draw(chain, n_draws)

        # Test x0 is uniform over n_states
        assert np.allclose(
            np.histogram(draws[:, 0], bins=n_states)[0] / n_draws, 1 / n_states, atol=atol
        )

        bigrams = [(chain[i], chain[i + 1]) for chain in draws for i in range(1, steps)]
        freq_table = np.zeros((n_states, n_states))
        for bigram in bigrams:
            i, j = bigram
            freq_table[i, j] += 1
        freq_table /= freq_table.sum(axis=1)[:, None]

        # Test continuation probabilities match P
        assert np.allclose(P, freq_table, atol=atol)
