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
import json
from abc import abstractmethod
from pathlib import Path
from typing import Dict, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


class ModelBuilder:
    """
    ModelBuilder can be used to provide an easy-to-use API (similar to scikit-learn) for models
    and help with deployment.
    """

    _model_type = "BaseClass"
    version = "None"

    def __init__(
        self,
        data: Dict[str, Union[np.ndarray, pd.DataFrame, pd.Series]],
        model_config: Dict = None,
        sampler_config: Dict = None,
    ):
        """
        Initializes model configuration and sampler configuration for the model

        Parameters
        ----------
        model_config : Dictionary, optional
            dictionary of parameters that initialise model configuration. Generated by the user defined create_sample_input method.
        data : Dictionary, required
            It is the data we need to train the model on.
        sampler_config : Dictionary, optional
            dictionary of parameters that initialise sampler configuration. Generated by the user defined create_sample_input method.
        Examples
        --------
        >>> class LinearModel(ModelBuilder):
        >>>     ...
        >>> model = LinearModel(model_config, sampler_config)
        """

        if sampler_config is None:
            sampler_config = {}
        if model_config is None:
            model_config = {}
        self.model_config = model_config  # parameters for priors etc.
        self.sampler_config = sampler_config  # parameters for sampling
        self.data = data
        self.idata = (
            None  # inference data object placeholder, idata is generated during build execution
        )

    @abstractmethod
    def _data_setter(
        self, data: Dict[str, Union[np.ndarray, pd.DataFrame, pd.Series]], x_only: bool = True
    ):
        """
        Sets new data in the model.

        Parameters
        ----------
        data : Dictionary of string and either of numpy array, pandas dataframe or pandas Series
            It is the data we need to set as idata for the model
        x_only : bool
            if data only contains values of x and y is not present in the data

        Examples
        --------
        >>> def _data_setter(self, data : pd.DataFrame):
        >>>     with self.model:
        >>>         pm.set_data({'x': data['input'].values})
        >>>         try: # if y values in new data
        >>>             pm.set_data({'y_data': data['output'].values})
        >>>         except: # dummies otherwise
        >>>             pm.set_data({'y_data': np.zeros(len(data))})
        """

        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def create_sample_input():
        """
        Needs to be implemented by the user in the inherited class.
        Returns examples for data, model_config, sampler_config.
        This is useful for understanding the required
        data structures for the user model.

        Examples
        --------
        >>> @classmethod
        >>> def create_sample_input(cls):
        >>>    x = np.linspace(start=1, stop=50, num=100)
        >>>    y = 5 * x + 3 + np.random.normal(0, 1, len(x)) * np.random.rand(100)*10 +  np.random.rand(100)*6.4
        >>>    data = pd.DataFrame({'input': x, 'output': y})

        >>>    model_config = {
        >>>       'a_loc': 7,
        >>>       'a_scale': 3,
        >>>       'b_loc': 5,
        >>>       'b_scale': 3,
        >>>       'obs_error': 2,
        >>>    }

        >>>    sampler_config = {
        >>>       'draws': 1_000,
        >>>       'tune': 1_000,
        >>>       'chains': 1,
        >>>       'target_accept': 0.95,
        >>>    }
        >>>    return data, model_config, sampler_config
        """

        raise NotImplementedError

    def save(self, fname: str) -> None:
        """
        Saves inference data of the model.

        Parameters
        ----------
        fname : string
            This denotes the name with path from where idata should be saved.

        Examples
        --------
        >>> class LinearModel(ModelBuilder):
        >>>     ...
        >>> data, model_config, sampler_config = LinearModel.create_sample_input()
        >>> model = LinearModel(model_config, sampler_config)
        >>> idata = model.fit(data)
        >>> name = './mymodel.nc'
        >>> model.save(name)
        """

        file = Path(str(fname))
        self.idata.to_netcdf(file)

    @classmethod
    def load(cls, fname: str):
        """
        Creates a ModelBuilder instance from a file,
        Loads inference data for the model.

        Parameters
        ----------
        fname : string
            This denotes the name with path from where idata should be loaded from.

        Returns
        -------
        Returns an instance of ModelBuilder.

        Raises
        ------
        ValueError
            If the inference data that is loaded doesn't match with the model.

        Examples
        --------
        >>> class LinearModel(ModelBuilder):
        >>>     ...
        >>> name = './mymodel.nc'
        >>> imported_model = LinearModel.load(name)
        """

        filepath = Path(str(fname))
        idata = az.from_netcdf(filepath)
        model_builder = cls(
            model_config=json.loads(idata.attrs["model_config"]),
            sampler_config=json.loads(idata.attrs["sampler_config"]),
            data=idata.fit_data.to_dataframe(),
        )
        model_builder.idata = idata
        model_builder.build_model()
        if model_builder.id != idata.attrs["id"]:
            raise ValueError(
                f"The file '{fname}' does not contain an inference data of the same model or configuration as '{cls._model_type}'"
            )

        return model_builder

    def fit(
        self, data: Dict[str, Union[np.ndarray, pd.DataFrame, pd.Series]] = None
    ) -> az.InferenceData:
        """
        Fit a model using the data passed as a parameter.
        Sets attrs to inference data of the model.

        Parameter
        ---------
        data : Dictionary of string and either of numpy array, pandas dataframe or pandas Series
            It is the data we need to train the model on.

        Returns
        -------
        returns inference data of the fitted model.

        Examples
        --------
        >>> data, model_config, sampler_config = LinearModel.create_sample_input()
        >>> model = LinearModel(model_config, sampler_config)
        >>> idata = model.fit(data)
        Auto-assigning NUTS sampler...
        Initializing NUTS using jitter+adapt_diag...
        """

        # If a new data was provided, assign it to the model
        if data is not None:
            self.data = data

        self.build_model()
        self._data_setter(data)

        with self.model:
            self.idata = pm.sample(**self.sampler_config)
            self.idata.extend(pm.sample_prior_predictive())
            self.idata.extend(pm.sample_posterior_predictive(self.idata))

        self.idata.attrs["id"] = self.id
        self.idata.attrs["model_type"] = self._model_type
        self.idata.attrs["version"] = self.version
        self.idata.attrs["sampler_config"] = json.dumps(self.sampler_config)
        self.idata.attrs["model_config"] = json.dumps(self.model_config)
        self.idata.add_groups(fit_data=self.data.to_xarray())
        return self.idata

    def predict(
        self,
        data_prediction: Dict[str, Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
        extend_idata: bool = True,
    ) -> dict:
        """
        Uses model to predict on unseen data and return point prediction of all the samples

        Parameters
        ---------
        data_prediction : Dictionary of string and either of numpy array, pandas dataframe or pandas Series
            It is the data we need to make prediction on using the model.
        extend_idata : Boolean determining whether the predictions should be added to inference data object.
            Defaults to True.

        Returns
        -------
        returns dictionary of sample's mean of posterior predict.

        Examples
        --------
        >>> data, model_config, sampler_config = LinearModel.create_sample_input()
        >>> model = LinearModel(model_config, sampler_config)
        >>> idata = model.fit(data)
        >>> x_pred = []
        >>> prediction_data = pd.DataFrame({'input':x_pred})
        # point predict
        >>> pred_mean = model.predict(prediction_data)
        """

        if data_prediction is not None:  # set new input data
            self._data_setter(data_prediction)

        with self.model:  # sample with new input data
            post_pred = pm.sample_posterior_predictive(self.idata)
            if extend_idata:
                self.idata.extend(post_pred)
        # reshape output
        post_pred = self._extract_samples(post_pred)
        for key in post_pred:
            post_pred[key] = post_pred[key].mean(axis=0)

        return post_pred

    def predict_posterior(
        self,
        data_prediction: Dict[str, Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
        extend_idata: bool = True,
    ) -> Dict[str, np.array]:
        """
        Uses model to predict samples on unseen data.

        Parameters
        ---------
        data_prediction : Dictionary of string and either of numpy array, pandas dataframe or pandas Series
            It is the data we need to make prediction on using the model.
        extend_idata : Boolean determining whether the predictions should be added to inference data object.
            Defaults to True.

        Returns
        -------
        returns dictionary of sample's posterior predict.

        Examples
        --------
        >>> data, model_config, sampler_config = LinearModel.create_sample_input()
        >>> model = LinearModel(model_config, sampler_config)
        >>> idata = model.fit(data)
        >>> x_pred = []
        >>> prediction_data = pd.DataFrame({'input': x_pred})
        # samples
        >>> pred_mean = model.predict_posterior(prediction_data)
        """

        if data_prediction is not None:  # set new input data
            self._data_setter(data_prediction)

        with self.model:  # sample with new input data
            post_pred = pm.sample_posterior_predictive(self.idata)
            if extend_idata:
                self.idata.extend(post_pred)

        # reshape output
        post_pred = self._extract_samples(post_pred)

        return post_pred

    @staticmethod
    def _extract_samples(post_pred: az.data.inference_data.InferenceData) -> Dict[str, np.array]:
        """
        This method can be used to extract samples from posterior predict.

        Parameters
        ----------
        post_pred: arviz InferenceData object

        Returns
        -------
        Dictionary of numpy arrays from InferenceData object
        """

        post_pred_dict = dict()
        for key in post_pred.posterior_predictive:
            post_pred_dict[key] = post_pred.posterior_predictive[key].to_numpy()[0]

        return post_pred_dict

    @property
    def id(self) -> str:
        """
        It creates a hash value to match the model version using last 16 characters of hash encoding.

        Returns
        -------
        Returns string of length 16 characters contains unique hash of the model
        """

        hasher = hashlib.sha256()
        hasher.update(str(self.model_config.values()).encode())
        hasher.update(self.version.encode())
        hasher.update(self._model_type.encode())
        return hasher.hexdigest()[:16]
