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
from typing import Dict

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from pymc_experimental.model_builder import ModelBuilder


class test_ModelBuilder(ModelBuilder):
    _model_type = "LinearModel"
    version = "0.1"

    def build_model(self, data=None, model_config=None):

        with pm.Model() as self.model:
            if data is None:
                data = test_ModelBuilder.generate_model_data()
            if model_config is None:
                model_config = self.default_model_config
            x = pm.MutableData("x", data["input"].values)
            y_data = pm.MutableData("y_data", data["output"].values)

            # prior parameters
            a_loc = model_config["a"]["loc"]
            a_scale = model_config["a"]["scale"]
            b_loc = model_config["b"]["loc"]
            b_scale = model_config["b"]["scale"]
            obs_error = model_config["obs_error"]

            # priors
            a = pm.Normal("a", a_loc, sigma=a_scale)
            b = pm.Normal("b", b_loc, sigma=b_scale)
            obs_error = pm.HalfNormal("σ_model_fmc", obs_error)

            # observed data
            y_model = pm.Normal("y_model", a + b * x, obs_error, shape=x.shape, observed=y_data)
            self.output_var = "y_model"

    def _data_setter(self, x: pd.Series, y: pd.Series = None):
        with self.model:
            pm.set_data({"x": x.values})
            if y is not None:
                pm.set_data({"y_data": y.values})

    @property
    def _serializable_model_config(self):
        return self.model_config

    @classmethod
    def generate_model_data(cls, data=None):
        x = np.linspace(start=0, stop=1, num=100)
        y = 5 * x + 3
        y = y + np.random.normal(0, 1, len(x))
        data = pd.DataFrame({"input": x, "output": y})
        return data

    @property
    def default_model_config(self) -> Dict:
        return {
            "a": {"loc": 0, "scale": 10},
            "b": {"loc": 0, "scale": 10},
            "obs_error": 2,
        }

    @property
    def default_sampler_config(self) -> Dict:
        return {
            "draws": 1_000,
            "tune": 1_000,
            "chains": 3,
            "target_accept": 0.95,
        }

    @staticmethod
    def initial_build_and_fit(check_idata=True) -> ModelBuilder:
        data = test_ModelBuilder.generate_model_data()
        model_builder = test_ModelBuilder()
        model_builder.idata = model_builder.fit(X=data["input"], y=data["output"])
        if check_idata:
            assert model_builder.idata is not None
            assert "posterior" in model_builder.idata.groups()
        return model_builder


def test_save_without_fit_raises_runtime_error():
    model_builder = test_ModelBuilder()
    with pytest.raises(RuntimeError):
        model_builder.save("saved_model")


def test_empty_sampler_config_fit():
    sampler_config = {}
    model_builder = test_ModelBuilder(sampler_config=sampler_config)
    data = test_ModelBuilder.generate_model_data()
    model_builder.idata = model_builder.fit(X=data["input"], y=data["output"])
    assert model_builder.idata is not None
    assert "posterior" in model_builder.idata.groups()


def test_fit():
    model = test_ModelBuilder.initial_build_and_fit()
    x_pred = np.random.uniform(low=0, high=1, size=100)
    prediction_data = pd.DataFrame({"input": x_pred})
    pred = model.predict(prediction_data["input"])
    post_pred = model.sample_posterior_predictive(
        prediction_data["input"], extend_idata=True, combined=True
    )
    post_pred.y_model.shape[0] == prediction_data.input.shape


@pytest.mark.skipif(
    sys.platform == "win32", reason="Permissions for temp files not granted on windows CI."
)
def test_save_load():
    test_builder = test_ModelBuilder.initial_build_and_fit()
    temp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
    test_builder.save(temp.name)
    test_builder2 = test_ModelBuilder.load(temp.name)
    assert test_builder.idata.groups() == test_builder2.idata.groups()

    x_pred = np.random.uniform(low=0, high=1, size=100)
    prediction_data = pd.DataFrame({"input": x_pred})
    pred1 = test_builder.predict(prediction_data["input"])
    pred2 = test_builder2.predict(prediction_data["input"])
    assert pred1.shape == pred2.shape
    temp.close()


def test_predict():
    model = test_ModelBuilder.initial_build_and_fit()
    x_pred = np.random.uniform(low=0, high=1, size=100)
    prediction_data = pd.DataFrame({"input": x_pred})
    pred = model.predict(prediction_data["input"])
    # Perform elementwise comparison using numpy
    assert type(pred) == np.ndarray
    assert len(pred) > 0


@pytest.mark.parametrize("combined", [True, False])
def test_sample_posterior_predictive(combined):
    model = test_ModelBuilder.initial_build_and_fit()
    n_pred = 100
    x_pred = np.random.uniform(low=0, high=1, size=n_pred)
    prediction_data = pd.DataFrame({"input": x_pred})
    pred = model.sample_posterior_predictive(
        prediction_data["input"], combined=combined, extend_idata=True
    )
    chains = model.idata.sample_stats.dims["chain"]
    draws = model.idata.sample_stats.dims["draw"]
    expected_shape = (n_pred, chains * draws) if combined else (chains, draws, n_pred)
    assert pred["y_model"].shape == expected_shape
    assert np.issubdtype(pred["y_model"].dtype, np.floating)


def test_id():
    model = test_ModelBuilder()
    expected_id = hashlib.sha256(
        str(model.model_config.values()).encode()
        + model.version.encode()
        + model._model_type.encode()
    ).hexdigest()[:16]

    assert model.id == expected_id
