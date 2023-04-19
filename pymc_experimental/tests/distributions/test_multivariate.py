import numpy as np
import pymc as pm
import pytest

import pymc_experimental as pmx


class TestR2D2M2CP:
    @pytest.fixture(autouse=True)
    def model(self):
        # every method is within a model
        with pm.Model() as model:
            yield model

    @pytest.fixture(params=[True, False])
    def centered(self, request):
        return request.param

    @pytest.fixture(params=[["a"], ["a", "b"], ["one"]])
    def dims(self, model: pm.Model, request):
        for i, c in enumerate(request.param):
            if c == "one":
                model.add_coord(c, range(1))
            else:
                model.add_coord(c, range((i + 2) ** 2))
        return request.param

    @pytest.fixture
    def input_std(self, dims, model):
        input_shape = [int(model.dim_lengths[d].eval()) for d in dims]
        return np.ones(input_shape)

    @pytest.fixture
    def output_std(self, dims, model):
        *hierarchy, _ = dims
        output_shape = [int(model.dim_lengths[d].eval()) for d in hierarchy]
        return np.ones(output_shape)

    @pytest.fixture
    def r2(self):
        return 0.8

    @pytest.fixture(params=[None, 0.1])
    def r2_std(self, request):
        return request.param

    @pytest.fixture(params=[True, False])
    def positive_probs(self, input_std, request):
        if request.param:
            return np.full_like(input_std, 0.5)
        else:
            return 0.5

    @pytest.fixture(params=[True, False])
    def positive_probs_std(self, positive_probs, request):
        if request.param:
            return np.full_like(positive_probs, 0.1)
        else:
            return None

    @pytest.fixture(params=["importance", "explained"])
    def phi_args(self, request, input_std):
        if request.param == "importance":
            return {"variables_importance": np.full_like(input_std, 2)}
        else:
            val = np.full_like(input_std, 2)
            return {"variance_explained": val / val.sum(-1, keepdims=True)}

    def test_init(
        self,
        dims,
        centered,
        input_std,
        output_std,
        r2,
        r2_std,
        positive_probs,
        positive_probs_std,
        phi_args,
    ):
        eps, beta = pmx.distributions.R2D2M2CP(
            "beta",
            output_std,
            input_std,
            dims=dims,
            r2=r2,
            r2_std=r2_std,
            centered=centered,
            positive_probs_std=positive_probs_std,
            positive_probs=positive_probs,
            **phi_args
        )
        assert eps.eval().shape == output_std.shape
        assert beta.eval().shape == input_std.shape
