#   Copyright 2023 The PyMC Developers
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

import hashlib
import sys
import tempfile

import numpy as np
import pytest
import xarray as xr

from pymc_experimental.linearmodel import LinearModel

try:
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    sklearn_available = True
except ImportError:
    sklearn_available = False


@pytest.fixture(scope="module")
def sample_input():
    x, y = LinearModel.generate_model_data()
    return x, y


@pytest.fixture(scope="module")
def fitted_linear_model_instance(sample_input):
    sampler_config = {
        "draws": 500,
        "tune": 300,
        "chains": 2,
        "target_accept": 0.95,
    }
    x, y = sample_input
    model = LinearModel()
    model.fit(x, y)
    return model


def test_save_without_fit_raises_runtime_error():
    x, y = LinearModel.generate_model_data()
    test_model = LinearModel()
    with pytest.raises(RuntimeError):
        test_model.save("saved_model")


def test_fit(fitted_linear_model_instance):
    model = fitted_linear_model_instance

    x_pred = np.random.uniform(low=0, high=1, size=(100, 1))
    pred = model.predict(x_pred)
    assert len(x_pred) == len(pred)
    assert isinstance(pred, np.ndarray)
    post_pred = model.predict_posterior(x_pred)
    assert len(x_pred) == len(post_pred)
    assert isinstance(post_pred, xr.DataArray)


@pytest.mark.skipif(
    sys.platform == "win32", reason="Permissions for temp files not granted on windows CI."
)
def test_save_load(fitted_linear_model_instance):
    model = fitted_linear_model_instance
    temp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
    model.save(temp.name)
    model2 = LinearModel.load(temp.name)
    assert model.idata.groups() == model2.idata.groups()

    x_pred = np.random.uniform(low=0, high=1, size=(100, 1))
    pred1 = model.predict(x_pred)
    pred2 = model2.predict(x_pred)
    # Predictions should have similar statistical characteristics
    assert pred1.mean() == pytest.approx(pred2.mean(), 1e-3)
    assert pred1.var() == pytest.approx(pred2.var(), 1e-2)
    temp.close()


def test_predict(fitted_linear_model_instance):
    model = fitted_linear_model_instance
    x_pred = np.random.uniform(low=0, high=1, size=(100, 1))
    pred = model.predict(x_pred)
    assert len(x_pred) == len(pred)
    assert np.issubdtype(pred.dtype, np.floating)


@pytest.mark.parametrize("combined", [True, False])
def test_predict_posterior(fitted_linear_model_instance, combined):
    model = fitted_linear_model_instance
    n_pred = 150
    x_pred = np.random.uniform(low=0, high=1, size=(n_pred, 1))
    pred = model.predict_posterior(x_pred, combined=combined)
    chains = model.idata.sample_stats.dims["chain"]
    draws = model.idata.sample_stats.dims["draw"]
    expected_shape = (n_pred, chains * draws) if combined else (chains, draws, n_pred)
    assert pred.shape == expected_shape
    assert np.issubdtype(pred.dtype, np.floating)
    # TODO: check that extend_idata has the expected effect


@pytest.mark.parametrize("samples", [None, 300])
@pytest.mark.parametrize("combined", [True, False])
def test_sample_prior_predictive(samples, combined, sample_input):
    x, y = sample_input
    model = LinearModel()
    prior_pred = model.sample_prior_predictive(x, samples, combined=combined)[model.output_var]
    draws = model.sampler_config["draws"] if samples is None else samples
    chains = 1
    expected_shape = (len(x), chains * draws) if combined else (chains, draws, len(x))
    assert prior_pred.shape == expected_shape
    # TODO: check that extend_idata has the expected effect


def test_id(sample_input):
    x, y = sample_input
    model_config = {
        "intercept": {"loc": 0, "scale": 10},
        "slope": {"loc": 0, "scale": 10},
        "obs_error": 2,
    }
    sampler_config = {
        "draws": 1_000,
        "tune": 1_000,
        "chains": 3,
        "target_accept": 0.95,
    }
    model = LinearModel(model_config=model_config, sampler_config=sampler_config)

    expected_id = hashlib.sha256(
        str(model_config.values()).encode() + model.version.encode() + model._model_type.encode()
    ).hexdigest()[:16]

    assert model.id == expected_id


@pytest.mark.skipif(not sklearn_available, reason="scikit-learn package is not available.")
def test_pipeline_integration(sample_input):
    model_config = {
        "intercept": {"loc": 0, "scale": 2},
        "slope": {"loc": 0, "scale": 2},
        "obs_error": 1,
        "default_output_var": "y_hat",
    }
    model = Pipeline(
        [
            ("input_scaling", StandardScaler()),
            (
                "linear_model",
                TransformedTargetRegressor(LinearModel(model_config), transformer=StandardScaler()),
            ),
        ]
    )
    x, y = sample_input
    model.fit(x, y)

    x_pred = np.random.uniform(low=0, high=1, size=(100, 1))
    model.predict(x_pred)
