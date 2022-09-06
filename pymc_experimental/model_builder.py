import pymc as pm
import arviz
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict
import cloudpickle
import arviz as az
import hashlib
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)


class ModelBuilder(pm.Model):
    """
    Extention of pm.Model class to improve workflow.

    ModelBuilder class can be used to play around models with ease using direct API calls
    for multiple tasks that one need to deploy a model.  

    Example:
    
    """

    _model_type = "BaseClass"
    version = "None"

    def __init__(self, model_config: Dict, sampler_config: Dict):
        super().__init__()
        self.model_config = model_config  # parameters for priors etc.
        self.sample_config = sampler_config  # parameters for sampling
        self.idata = None  # parameters for

    def _data_setter(
        self, data: Dict[str, Union[np.ndarray, pd.DataFrame, pd.Series]], x_only: bool = True
    ):
        """
        Sets new data in the model.

        Parameter
        --------

        data: Dictionary of string and either of numpy array, pandas dataframe or pandas Series
            It is the data we need to set as idata for the model
        x_only: bool
            if data only contains values of x and y is not present in the data

        Example: 

        def _data_setter(self, data : pd.DataFrame):
        with self.model:
            pm.set_data({'x': data['input'].values})
            try: # if y values in new data
                pm.set_data({'y_data': data['output'].values})
            except: # dummies otherwise
                pm.set_data({'y_data': np.zeros(len(data))})

        """
        raise NotImplementedError

    @classmethod
    def create_sample_input(cls):
        """
        Needs to be implemented by the user in the inherited class.
        Returns examples for data, model_config, samples_config.
        This is useful for understanding the required 
        data structures for the user model.
        """
        raise NotImplementedError

    def save(self, file_prefix, filepath):
        """
        Saves the model as well as inference data of the model.

        Parameters
        ----------
        file_prefix: string
            Passed which denotes the name with which model and idata should be saved.
        filepath: string
            Used as path at which model and idata should be saved
            
        """
        file = Path(filepath + str(file_prefix) + ".nc")
        self.idata.to_netcdf(file)
            

    @classmethod
    def load(cls, file_prefix, filepath):
        """
        Loads model and the idata of used for model.

        Parameters
        ----------
        file_prefix: string
            Passed which denotes the name with which model and idata should be loaded from.
        filepath: string
            Used as path at which model and idata should be loaded from.

        """

        filepath = Path(str(filepath) + str(file_prefix) + ".nc")
        data = az.from_netcdf(filepath)
        self.idata = data
        return self

    # fit and predict methods
    def fit(self, data: Dict[str, Union[np.ndarray, pd.DataFrame, pd.Series]] = None):
        """
        As the name suggests fit can be used to fit a model using the data that is passed as a parameter.
        It returns the inference data.

        Parameter
        ---------
        data: Dictionary of string and either of numpy array, pandas dataframe or pandas Series
            It is the data we need to train the model on.
        """
        if data is not None:
            self.data = data

        if self.basic_RVs == []:
            print("No model found, building model...")
            self.build()

        with self:
            self.idata = pm.sample(**self.sample_config)
            self.idata.extend(pm.sample_prior_predictive())
            self.idata.extend(pm.sample_posterior_predictive(self.idata))

        self.idata.attrs["id"] = self.id()
        self.idata.attrs["model_type"] = self._model_type
        self.idata.attrs["version"] = self.version
        self.idata.attrs["sample_conifg"] = self.sample_config
        self.idata.attrs["model_config"] = self.model_config
        return self.idata

    def predict(
        self,
        data_prediction: Dict[str, Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
        point_estimate: bool = True,
    ):
        """
        Uses model to predict on unseen data and returns posterioir prediction on the data.

        Parameters
        ---------
        data_prediction: Dictionary of string and either of numpy array, pandas dataframe or pandas Series
            It is the data we need to make prediction on using the model.
        point_estimate: bool
            Adds point like estimate used as mean passed as 

        """
        if data_prediction is not None:  # set new input data
            self._data_setter(data_prediction)

        with self.model:  # sample with new input data
            post_pred = pm.sample_posterior_predictive(self.idata.posterior)

        # reshape output
        post_pred = self._extract_samples(post_pred)

        if point_estimate:  # average, if point-like estimate desired
            for key in post_pred:
                post_pred[key] = post_pred[key].mean(axis=0)

        if data_prediction is not None:  # set back original data in model
            self._data_setter(self.data)

        return post_pred

    @staticmethod
    def _extract_samples(post_pred: arviz.data.inference_data.InferenceData) -> Dict[str, np.array]:
        """
        Returns dict of numpy arrays from InferenceData object

        Parameters
        ----------
        post_pred: arviz InferenceData object
        """
        post_pred_dict = dict()
        for key in post_pred.posterior_predictive:
            post_pred_dict[key] = post_pred.posterior_predictive[key].to_numpy()[0]

        return post_pred_dict

    def id(self):
        """
        It creates a hash value to match the model version using last 16 characters of hash encoding.
        """
        hasher = hashlib.sha256()
        hasher.update(str(self.model_config.values()).encode())
        hasher.update(self.version.encode())
        hasher.update(self._model_type.encode())
        hasher.update(str(self.sample_config.values()).encode())
        return hasher.hexdigest()[:16]
