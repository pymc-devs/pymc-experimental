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
    """

    _model_type = "BaseClass"
    version = "None"

    def __init__(self, model_config: Dict, sampler_config: Dict):
        """
        Initialises model configration and sampler configration for the model

        Parameters
        ----------
        model_confid : Dictionary
            dictonary of parameters that initialise model configration. Genrated by the user defiend create_sample_input method.
        sampler_config : Dictionary
            dictonary of parameters that initialise sampler configration. Genrated by the user defiend create_sample_input method.

        Examples
        --------
        >>> class LinearModel(ModelBuilder)
            ...
        >>> model = LinearModel(model_config, sampler_config)
        """

        super().__init__()
        self.model_config = model_config  # parameters for priors etc.
        self.sample_config = sampler_config  # parameters for sampling
        self.idata = None  # parameters for

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

    @classmethod
    def create_sample_input(cls):
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
        >>>    'a_loc': 7,
        >>>    'a_scale': 3,
        >>>    'b_loc': 5,
        >>>    'b_scale': 3,
        >>>    'obs_error': 2,
        >>>    }

        >>>    sampler_config = {
        >>>    'draws': 1_000,
        >>>    'tune': 1_000,
        >>>    'chains': 1,
        >>>    'target_accept': 0.95,
        >>>    }
        >>>    return data, model_config, sampler_config
        """

        raise NotImplementedError

    def save(self, fname):
        """
        Saves inference data of the model.

        Parameters
        ----------
        fname: string
            This denotes the name with path from where idata should be saved.

        Examples
        --------
        >>> name = './mymodel.nc'
        >>> model.save(name)
        """

        file = Path(str(fname))
        self.idata.to_netcdf(file)

    @classmethod
    def load(cls, fname):
        """
        Loads infernce data for the model.

        Parameters
        ----------
        fname: string
            This denotes the name with path from where idata should be loaded from.

        Returns
        -------
        Returns the inference data that is loaded from local system.

        Raises
        ------
        ValueError
            If the inference data that is loaded doesn't match with the model.

        Examples
        --------
        >>> class LinearModel
        ...
        >>> name = './mymodel.nc'
        >>> imported_model = LinearModel.load(name)
        """

        filepath = Path(str(fname))
        data = az.from_netcdf(filepath)
        idata = data
        # Since there is an issue with attrs geting saved in netcdf format which will be fixd in future the following part of code is commented
        # Link of issue -> https://github.com/arviz-devs/arviz/issues/2109
        # if model.idata.attrs is not None:
        #     if model.idata.attrs['id'] == self.idata.attrs['id']:
        #         self = model
        #         self.idata = data
        #         return self
        #     else:
        #         raise ValueError(
        #             f"The route '{file}' does not contain an inference data of the same model '{self.__name__}'"
        #         )
        return idata

    def fit(self, data: Dict[str, Union[np.ndarray, pd.DataFrame, pd.Series]] = None):
        """
        As the name suggests fit can be used to fit a model using the data that is passed as a parameter.
        Sets attrs to inference data of the model.

        Parameter
        ---------
        data : Dictionary of string and either of numpy array, pandas dataframe or pandas Series
            It is the data we need to train the model on.

        Retruns
        --------
        returns infernece data of the fitted model.

        Examples
        --------
        >>> data, model_config, sampler_config = LinearModel.create_sample_input() 
        >>> model = LinearModel(model_config, sampler_config)
        >>> idata = model.fit(data)
        Auto-assigning NUTS sampler...
        Initializing NUTS using jitter+adapt_diag...
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
        Uses model to predict on unseen data.

        Parameters
        ---------
        data_prediction : Dictionary of string and either of numpy array, pandas dataframe or pandas Series
            It is the data we need to make prediction on using the model.
        point_estimate : bool
            Adds point like estimate used as mean passed as 

        Returns
        -------
        returns dictionary of sample's posterior predict. 

        Examples
        --------
        >>> prediction_data = pd.DataFrame({'input':x_pred})
        # only point estimate
        >>> pred_mean = imported_model.predict(prediction_data)
        # samples
        >>> pred_samples = imported_model.predict(prediction_data, point_estimate=False)
        """

        if data_prediction is not None:  # set new input data
            self._data_setter(data_prediction)

        with self.model:  # sample with new input data
            post_pred = pm.sample_posterior_predictive(self.idata)

        # reshape output
        post_pred = self._extract_samples(post_pred)
        if point_estimate:  # average, if point-like estimate desired
            for key in post_pred:
                post_pred[key] = post_pred[key].mean(axis=0)

        return post_pred

    @staticmethod
    def _extract_samples(post_pred: arviz.data.inference_data.InferenceData) -> Dict[str, np.array]:
        """
        This method can be used to extract samples from posteriror predict.
        
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

    def id(self):
        """
        It creates a hash value to match the model version using last 16 characters of hash encoding.

        Return
        ------
        Returns string of length 16 characters containg unique hash of the model
        """

        hasher = hashlib.sha256()
        hasher.update(str(self.model_config.values()).encode())
        hasher.update(self.version.encode())
        hasher.update(self._model_type.encode())
        hasher.update(str(self.sample_config.values()).encode())
        return hasher.hexdigest()[:16]
