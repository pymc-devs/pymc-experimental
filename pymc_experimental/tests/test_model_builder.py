import pymc as pm 
from pymc.model_builder import ModelBuilder
import unittest
import pytest
import re
from nose.tools import *


#relevant libraries
import arviz
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict
import cloudpickle
import arviz as az
import hashlib

class test_ModelBuilder(ModelBuilder):
    _model_type = 'LinearModel'
    version = '0.1'

    def _build(self):
        
        x = pm.MutableData('x', self.data['input'].values)
        y_data = pm.MutableData('y_data', self.data['output'].values)

        a_loc = self.model_config['a_loc']
        a_scale = self.model_config['a_scale']
        b_loc = self.model_config['b_loc']
        b_scale = self.model_config['b_scale']
        obs_error = self.model_config['obs_error']

        a = pm.Normal("a", a_loc, sigma=a_scale)
        b = pm.Normal("b", b_loc, sigma=b_scale)
        obs_error = pm.HalfNormal("σ_model_fmc", obs_error)

        y_model = pm.Normal('y_model', a + b * x, obs_error, observed=y_data)


    def _data_setter(self, data : pd.DataFrame):
        with self.model:
            pm.set_data({'x': data['input'].values})
            try:
                pm.set_data({'y_data': data['output'].values})
            except:
                pm.set_data({'y_data': np.zeros(len(data))})


    @classmethod
    def create_sample_input(cls):
        x = np.linspace(start=0, stop=1, num=100)
        y = 5 * x + 3
        data = pd.DataFrame({'input': x, 'output': y})

        model_config = {
            'a_loc': 7,
            'a_scale': 3,
            'b_loc': 5,
            'b_scale': 3,
            'obs_error': 2,
        }

        sampler_config = {
            'draws': 1_000,
            'tune': 1_000,
            'chains': 1,
            'target_accept': 0.95,
        }

        return data, model_config, sampler_config

def test_model_pickle():
	data, model_config, sampler_config = test_ModelBuilder.create_sample_input() 
	model = test_ModelBuilder(model_config, sampler_config)
	cloned = cloudpickle.loads(cloudpickle.dumps(model))
	assert_equal(model.basic_RVs, cloned.basic_RVs)
	assert_equal(model.id(), cloned.id())


def test_fit():
	with pm.Model() as model:
		x = np.linspace(start=0, stop=1, num=100)
		y = 5 * x + 3
		x = pm.MutableData('x', x)
		y_data = pm.MutableData('y_data', y)

		a_loc = 7
		a_scale = 3
		b_loc = 5
		b_scale = 3
		obs_error = 2

		a = pm.Normal("a", a_loc, sigma=a_scale)
		b = pm.Normal("b", b_loc, sigma=b_scale)
		obs_error = pm.HalfNormal("σ_model_fmc", obs_error)

		y_model = pm.Normal('y_model', a + b * x, obs_error, observed=y_data)

		idata = pm.sample(1000, tune=1000)
	data, model_config, sampler_config = test_ModelBuilder.create_sample_input() 
	model_2 = test_ModelBuilder(model_config, sampler_config)

	assert_equal(model_2.idata,idata)

def test_predict():
	with pm.Model() as model:
		x = np.linspace(start=0, stop=1, num=100)
		y = 5 * x + 3
		x = pm.MutableData('x', x)
		y_data = pm.MutableData('y_data', y)
		a_loc = 7
		a_scale = 3
		b_loc = 5
		b_scale = 3
		obs_error = 2

		a = pm.Normal("a", a_loc, sigma=a_scale)
		b = pm.Normal("b", b_loc, sigma=b_scale)
		obs_error = pm.HalfNormal("σ_model_fmc", obs_error)

		y_model = pm.Normal('y_model', a + b * x, obs_error, observed=y_data)

		idata = pm.sample(1000, tune=1000)
		y_test = pm.sample_posterior_predictive(idata)

		y_test.posterior_predictive.equlas(model.posterior_predictive)

