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

    def build_model(self, model_config, data=None):
        if data is not None:
            x = pm.MutableData("x", data["input"].values)
            y_data = pm.MutableData("y_data", data["output"].values)

        # prior parameters
        a_loc = model_config["a_loc"]
        a_scale = model_config["a_scale"]
        b_loc = model_config["b_loc"]
        b_scale = model_config["b_scale"]
        obs_error = model_config["obs_error"]

        # priors
        a = pm.Normal("a", a_loc, sigma=a_scale)
        b = pm.Normal("b", b_loc, sigma=b_scale)
        obs_error = pm.HalfNormal("σ_model_fmc", obs_error)

        # observed data
        if data is not None:
            y_model = pm.Normal("y_model", a + b * x, obs_error, shape=x.shape, observed=y_data)

    def _data_setter(self, data: pd.DataFrame):
        with self.model:
            pm.set_data({"x": data["input"].values})
            if "output" in data.columns:
                pm.set_data({"y_data": data["output"].values})

    @classmethod
    def create_sample_input(cls):
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


def test_fit():
    data, model_config, sampler_config = test_ModelBuilder.create_sample_input()
    model = test_ModelBuilder(model_config, sampler_config, data)
    model.fit()
    assert model.idata is not None
    assert "posterior" in model.idata.groups()

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
    data, model_config, sampler_config = test_ModelBuilder.create_sample_input()
    model = test_ModelBuilder(model_config, sampler_config, data)
    model.fit()
    temp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
    model.save(temp.name)
    model2 = test_ModelBuilder.load(temp.name)
    assert model.idata.groups() == model2.idata.groups()

    x_pred = np.random.uniform(low=0, high=1, size=100)
    prediction_data = pd.DataFrame({"input": x_pred})
    pred1 = model.predict(prediction_data)
    pred2 = model2.predict(prediction_data)
    assert pred1["y_model"].shape == pred2["y_model"].shape
    temp.close()

def test_id():
    data, model_config, sampler_config = test_ModelBuilder.create_sample_input()
    model = test_ModelBuilder(model_config, sampler_config, data)
    
    expected_id = hashlib.sha256(
        str(model_config.values()).encode() +
        model.version.encode() +
        model._model_type.encode()
    ).hexdigest()[:16]

    assert model.id == expected_id