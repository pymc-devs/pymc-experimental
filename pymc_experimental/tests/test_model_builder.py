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
import pandas as pd
import pymc as pm
import pytest

from pymc_experimental.model_builder import ModelBuilder


class test_ModelBuilder(ModelBuilder):
    _model_type = "LinearModel"
    version = "0.1"

    def build(self):

        with pm.Model() as self.model:
            if self.data is not None:
                x = pm.MutableData("x", self.data["input"].values)
                y_data = pm.MutableData("y_data", self.data["output"].values)

            # prior parameters
            a_loc = self.model_config["a_loc"]
            a_scale = self.model_config["a_scale"]
            b_loc = self.model_config["b_loc"]
            b_scale = self.model_config["b_scale"]
            obs_error = self.model_config["obs_error"]

            # priors
            a = pm.Normal("a", a_loc, sigma=a_scale)
            b = pm.Normal("b", b_loc, sigma=b_scale)
            obs_error = pm.HalfNormal("σ_model_fmc", obs_error)

            # observed data
            if self.data is not None:
                y_model = pm.Normal("y_model", a + b * x, obs_error, shape=x.shape, observed=y_data)

    def _data_setter(self, data: pd.DataFrame):
        with self.model:
            pm.set_data({"x": data["input"].values})
            if "output" in data.columns:
                pm.set_data({"y_data": data["output"].values})

    @classmethod
    def create_sample_input(self):
        x = np.linspace(start=0, stop=1, num=100)
        y = 5 * x + 3
        y = y + np.random.normal(0, 1, len(x))
        data = pd.DataFrame({"input": x, "output": y})

        model_config = {
            "a_loc": 0,
            "a_scale": 10,
            "b_loc": 0,
            "b_scale": 10,
            "obs_error": 2,
        }

        sampler_config = {
            "draws": 1_000,
            "tune": 1_000,
            "chains": 3,
            "target_accept": 0.95,
        }

        return data, model_config, sampler_config

    @staticmethod
    def initial_build_and_fit(check_idata=True) -> ModelBuilder:
        data, model_config, sampler_config = test_ModelBuilder.create_sample_input()
        model_builder = test_ModelBuilder(
            model_config=model_config, sampler_config=sampler_config, data=data
        )
        model_builder.idata = model_builder.fit(data=data)
        if check_idata:
            assert model_builder.idata is not None
            assert "posterior" in model_builder.idata.groups()
        return model_builder


def test_empty_model_config():
    data, model_config, sampler_config = test_ModelBuilder.create_sample_input()
    sampler_config = {}
    model_builder = test_ModelBuilder(
        model_config=model_config, sampler_config=sampler_config, data=data
    )
    model_builder.idata = model_builder.fit(data=data)
    assert model_builder.idata is not None
    assert "posterior" in model_builder.idata.groups()


def test_empty_model_config():
    data, model_config, sampler_config = test_ModelBuilder.create_sample_input()
    model_config = {}
    model_builder = test_ModelBuilder(
        model_config=model_config, sampler_config=sampler_config, data=data
    )
    assert model_builder.model_config == {}


def test_fit():
    model = test_ModelBuilder.initial_build_and_fit()
    x_pred = np.random.uniform(low=0, high=1, size=100)
    prediction_data = pd.DataFrame({"input": x_pred})
    pred = model.predict(prediction_data)
    assert "y_model" in pred.keys()
    post_pred = model.predict_posterior(prediction_data)
    assert "y_model" in post_pred.keys()


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
    pred1 = test_builder.predict(prediction_data)
    pred2 = test_builder2.predict(prediction_data)
    assert pred1["y_model"].shape == pred2["y_model"].shape
    temp.close()


def test_predict():
    model = test_ModelBuilder.initial_build_and_fit()
    x_pred = np.random.uniform(low=0, high=1, size=100)
    prediction_data = pd.DataFrame({"input": x_pred})
    pred = model.predict(prediction_data)
    assert "y_model" in pred
    assert isinstance(pred, dict)
    assert len(prediction_data.input.values) == len(pred["y_model"])
    assert isinstance(pred["y_model"][0], (np.float32, np.float64))


def test_predict_posterior():
    model = test_ModelBuilder.initial_build_and_fit()
    x_pred = np.random.uniform(low=0, high=1, size=100)
    prediction_data = pd.DataFrame({"input": x_pred})
    pred = model.predict_posterior(prediction_data)
    assert "y_model" in pred
    assert isinstance(pred, dict)
    assert len(prediction_data.input.values) == len(pred["y_model"][0])
    assert isinstance(pred["y_model"][0], np.ndarray)


def test_extract_samples():
    # create a fake InferenceData object
    with pm.Model() as model:
        x = pm.Normal("x", mu=0, sigma=1)
        intercept = pm.Normal("intercept", mu=0, sigma=1)
        y_model = pm.Normal("y_model", mu=x * intercept, sigma=1, observed=[0, 1, 2])

        idata = pm.sample(1000, tune=1000)
        post_pred = pm.sample_posterior_predictive(idata)

    # call the function and get the output
    samples_dict = test_ModelBuilder._extract_samples(post_pred)

    # assert that the keys and values are correct
    assert len(samples_dict) == len(post_pred.posterior_predictive)
    for key in post_pred.posterior_predictive:
        expected_value = post_pred.posterior_predictive[key].to_numpy()[0]
        assert np.array_equal(samples_dict[key], expected_value)


def test_id():
    data, model_config, sampler_config = test_ModelBuilder.create_sample_input()
    model = test_ModelBuilder(model_config=model_config, sampler_config=sampler_config, data=data)

    expected_id = hashlib.sha256(
        str(model_config.values()).encode() + model.version.encode() + model._model_type.encode()
    ).hexdigest()[:16]

    assert model.id == expected_id
