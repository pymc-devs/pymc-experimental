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
from typing import Any, Dict, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from pymc.util import RandomState


class ModelBuilder:
    """
    ModelBuilder can be used to provide an easy-to-use API (similar to scikit-learn) for models
    and help with deployment.
    """

    _model_type = "BaseClass"
    version = "None"

    def __init__(
        self,
        data: Union[np.ndarray, pd.DataFrame, pd.Series] = None,
        model_config: Dict = None,
        sampler_config: Dict = None,
    ):
        """
        Initializes model configuration and sampler configuration for the model

        Parameters
        ----------
        data : Dictionary, optional
            It is the data we need to train the model on.
        model_config : Dictionary, optional
            dictionary of parameters that initialise model configuration. Class-default defined by the user default_model_config method.
        sampler_config : Dictionary, optional
            dictionary of parameters that initialise sampler configuration. Class-default defined by the user default_sampler_config method.
        Examples
        --------
        >>> class LinearModel(ModelBuilder):
        >>>     ...
        >>> model = LinearModel(model_config, sampler_config)
        """

        if sampler_config is None:
            sampler_config = self.default_sampler_config
        self.sampler_config = sampler_config
        if model_config is None:
            model_config = self.default_model_config
        self.model_config = model_config  # parameters for priors etc.
        self.data = self.generate_model_data(data=data)
        self.model = None  # Set by build_model
        self.output_var = None  # Set by build_model
        self.idata = None  # idata is generated during fitting
        self.is_fitted_ = False

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

    @property
    @abstractmethod
    def default_model_config(self) -> Dict:
        """
        Returns a class default config dict for model builder if no model_config is provided on class initialization
        Useful for understanding structure of required model_config to allow its customization by users
        Examples
        --------
        >>>     @classmethod
        >>>     def default_model_config(self):
        >>>         Return {
        >>>             'a' : {
        >>>                 'loc': 7,
        >>>                 'scale' : 3
        >>>             },
        >>>             'b' : {
        >>>                 'loc': 3,
        >>>                 'scale': 5
        >>>             }
        >>>              'obs_error': 2
        >>>         }

        Returns
        -------
        model_config : dict
            A set of default parameters for predictor distributions that allow to save and recreate the model.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def default_sampler_config(self) -> Dict:
        """
        Returns a class default sampler dict for model builder if no sampler_config is provided on class initialization
        Useful for understanding structure of required sampler_config to allow its customization by users
        Examples
        --------
        >>>     @classmethod
        >>>     def default_model_config(self):
        >>>         Return {
        >>>             'draws': 1_000,
        >>>             'tune': 1_000,
        >>>             'chains': 1,
        >>>             'target_accept': 0.95,
        >>>         }

        Returns
        -------
        sampler_config : dict
            A set of default settings for used by model in fit process.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def generate_model_data(
        cls, data: Union[np.ndarray, pd.DataFrame, pd.Series] = None
    ) -> pd.DataFrame:
        """
        Returns a default dataset for a class, can be used as a hint to data formatting required for the class

        Examples
        --------
        >>>     @classmethod
        >>>     def generate_model_data(self):
        >>>         x = np.linspace(start=1, stop=50, num=100)
        >>>         y = 5 * x + 3 + np.random.normal(0, 1, len(x)) * np.random.rand(100)*10 +  np.random.rand(100)*6.4
        >>>         data = pd.DataFrame({'input': x, 'output': y})

        Returns
        -------
        data : pd.DataFrame
            The data we want to train the model on.

        """
        raise NotImplementedError

    @abstractmethod
    def build_model(
        model_data: Dict[str, Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
        model_config: Dict[str, Union[int, float, Dict]] = None,
    ) -> None:
        """
        Needs to be implemented by the user in the child class.
        Creates an instance of pm.Model based on provided model_data and model_config, and
        attaches it to self.

        Required Parameters
        ----------
        model_data : dict
            Preformated data that is going to be used in the model. For efficiency reasons it should contain only the necesary data columns,
            not entire available dataset since it's going to be encoded into data used to recreate the model.
            If not provided uses data from self.data
        model_config : dict
            Dictionary where keys are strings representing names of parameters of the model, values are dictionaries of parameters
            needed for creating model parameters. If not provided uses data from self.model_config

        See Also
        --------
        default_model_config : returns default model config

        Returns:
        ----------
        None

        """
        raise NotImplementedError

    def sample_model(self, **kwargs):

        if self.model is None:
            raise RuntimeError(
                "The model hasn't been built yet, call .build_model() first or call .fit() instead."
            )

        with self.model:
            sampler_args = {**self.sampler_config, **kwargs}
            idata = pm.sample(**sampler_args)
            idata.extend(pm.sample_prior_predictive())
            idata.extend(pm.sample_posterior_predictive(idata))

        self.set_idata_attrs(idata)
        return idata

    def set_idata_attrs(self, idata=None):
        if idata is None:
            idata = self.idata
        if idata is None:
            raise RuntimeError("No idata provided to set attrs on.")
        idata.attrs["id"] = self.id
        idata.attrs["model_type"] = self._model_type
        idata.attrs["version"] = self.version
        idata.attrs["sampler_config"] = json.dumps(self.sampler_config)
        idata.attrs["model_config"] = json.dumps(self._serializable_model_config)

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
        if self.idata is not None and "posterior" in self.idata:
            file = Path(str(fname))
            self.idata.to_netcdf(file)
        else:
            raise RuntimeError("The model hasn't been fit yet, call .fit() first")

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
            data=idata.fit_data.to_dataframe(),
            model_config=json.loads(idata.attrs["model_config"]),
            sampler_config=json.loads(idata.attrs["sampler_config"]),
        )
        model_builder.idata = idata
        model_builder.build_model(model_builder.data, model_builder.model_config)
        if model_builder.id != idata.attrs["id"]:
            raise ValueError(
                f"The file '{fname}' does not contain an inference data of the same model or configuration as '{cls._model_type}'"
            )

        return model_builder

    def fit(
        self,
        data: Dict[str, Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
        progressbar: bool = True,
        random_seed: RandomState = None,
        **kwargs: Any,
    ) -> az.InferenceData:
        """
        Fit a model using the data passed as a parameter.
        Sets attrs to inference data of the model.

        Parameter
        ---------
        data : dict
            Dictionary of string and either of numpy array, pandas dataframe or pandas Series. It is the data we need to train the model on.
        progressbar : bool
            Specifies whether the fit progressbar should be displayed
        random_seed : RandomState
            Provides sampler with initial random seed for obtaining reproducible samples
        **kwargs : Any
            Custom sampler settings can be provided in form of keyword arguments. The recommended way is to add custom settings to sampler_config provided by
            create_sample_input, because arguments provided in the form of kwargs will not be saved into the model, therefore will not be available after loading the model

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
        self.model_data = self.generate_model_data(data=self.data)
        if self.model_config is None:
            self.model_config = self.default_model_config
        if self.sampler_config is None:
            self.sampler_config = self.default_sampler_config
        if self.model is None:
            self.build_model(self.model_data, self.model_config)

        self.sampler_config["progressbar"] = progressbar
        self.sampler_config["random_seed"] = random_seed

        self.idata = self.sample_model(**self.sampler_config)
        self.idata.add_groups(fit_data=self.data.to_xarray())
        return self.idata

    def predict(
        self,
        data_prediction: Dict[str, Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
        extend_idata: bool = True,
    ) -> xr.Dataset:
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
        returns posterior mean of predictive samples

        Examples
        --------
        >>> data, model_config, sampler_config = LinearModel.create_sample_input()
        >>> model = LinearModel(model_config, sampler_config)
        >>> idata = model.fit(data)
        >>> x_pred = []
        >>> prediction_data = pd.DataFrame({'input':x_pred})
        >>> pred_mean = model.predict(prediction_data)
        """
        posterior_predictive_samples = self.predict_posterior(data_prediction, extend_idata)
        posterior_means = posterior_predictive_samples.mean(dim=["chain", "draw"], keep_attrs=True)
        return posterior_means

    def predict_posterior(
        self,
        data_prediction: Dict[str, Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
        extend_idata: bool = True,
        combined: bool = False,
    ) -> xr.Dataset:
        """
        Generate posterior predictive samples on unseen data.

        Parameters
        ---------
        data_prediction : Dictionary of string and either of numpy array, pandas dataframe or pandas Series
            It is the data we need to make prediction on using the model.
        extend_idata : Boolean determining whether the predictions should be added to inference data object.
            Defaults to True.
        combined: Combine chain and draw dims into sample. Won’t work if a dim named sample already exists.
            Defaults to False.

        Returns
        -------
        returns posterior predictive samples

        Examples
        --------
        >>> data, model_config, sampler_config = LinearModel.create_sample_input()
        >>> model = LinearModel(model_config, sampler_config)
        >>> idata = model.fit(data)
        >>> x_pred = []
        >>> prediction_data = pd.DataFrame({'input': x_pred})
        >>> pred_samples = model.predict_posterior(prediction_data)
        """

        if data_prediction is not None:  # set new input data
            self._data_setter(data_prediction)

        with self.model:  # sample with new input data
            post_pred = pm.sample_posterior_predictive(self.idata)
            if extend_idata:
                self.idata.extend(post_pred)

        posterior_predictive_samples = az.extract(
            post_pred, "posterior_predictive", combined=combined
        )

        return posterior_predictive_samples

    @property
    @abstractmethod
    def _serializable_model_config(self) -> Dict[str, Union[int, float, Dict]]:
        """
        Converts non-serializable values from model_config to their serializable reversable equivalent.
        Data types like pandas DataFrame, Series or datetime aren't JSON serializable,
        so in order to save the model they need to be formatted.

        Returns
        -------
        model_config: dict
        """

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
