from abc import abstractmethod
from typing import Dict, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from pymc_experimental.model_builder import ModelBuilder

# If scikit-learn is available, use its data validator
try:
    from sklearn.utils.validation import check_array, check_X_y
# If scikit-learn is not available, return the data unchanged
except ImportError:

    def check_X_y(X, y, **kwargs):
        return X, y

    def check_array(X, **kwargs):
        return X


class BayesianEstimator(ModelBuilder):
    """
    Base class similar to ModelBuilder but customized for integration with a scikit-learn workflow.
    It is designed to encapsulate parameter inference ("fit") and posterior prediction ("predict")
    for simple models with the following characteristics:
     - One or more input features for each observation
     - One observed output feature for each observation

    Estimators derived from this base class can utilize scikit-learn transformers for input and
    output accessed via `fit` and `predict`. (`TransformedTargetRegressor` would need to be extended
    in order to transform the output of `predict_proba` or `predict_posterior`.)

    Example scikit-learn usage:
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.compose import TransformedTargetRegressor
    >>> model = Pipeline([
    >>>     ('input_scaling', StandardScaler()),
    >>>     ('linear_model', TransformedTargetRegressor(LinearModel(model_config),
    >>>                                                 transformer=StandardScaler()),)
    >>> ])
    >>> model.fit(X_obs, y_obs)
    >>> y_pred = model.predict(X_pred)

    The format for probabilistic output forecasts is currently an xarray `DataSet` of the posterior
    prediction draws for each input prediction. This

    The `sklearn` package is not a dependency for using this class, although if imports from `sklearn`
    are successful, then scikit-learn's data validation functions are used to accept "array-like" input.
    """

    model_type = "BaseClass"
    version = "None"

    def __init__(
        self,
        model_config: Dict = None,
        sampler_config: Dict = None,
    ):
        """
        Initializes model configuration and sampler configuration for the model

        Parameters
        ----------
        model_config : dict
            Parameters that initialise model configuration
        sampler_config : dict
            Parameters that initialise sampler configuration
        """
        if model_config is None:
            model_config = self.default_model_config
        if sampler_config is None:
            sampler_config = self.default_sampler_config
        super().__init__(model_config=model_config, sampler_config=sampler_config)

    @property
    @abstractmethod
    def default_model_config(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def default_sampler_config(self):
        raise NotImplementedError

    def _validate_data(self, X, y=None):
        if y is not None:
            return check_X_y(X, y, accept_sparse=False, y_numeric=True, multi_output=False)
        else:
            return check_array(X, accept_sparse=False)

    @abstractmethod
    def build_model(
        model_data: Dict[str, Union[np.ndarray, pd.DataFrame, pd.Series]] = None,
        model_config: Dict[str, Union[int, float, Dict]] = None,
    ) -> None:

        raise NotImplementedError

    @abstractmethod
    def _data_setter(self, X, y=None):
        """
        Sets new data in the model.

        Parameters
        ----------
        X : array, shape (n_obs, n_features)
            The training input samples.
        y : array, shape (n_obs,)
            The target values (real numbers).

        Returns:
        ----------
        None

        """

        raise NotImplementedError

    def predict_posterior(
        self,
        X_pred: Union[np.ndarray, pd.DataFrame, pd.Series],
        extend_idata: bool = True,
        combined: bool = True,
    ) -> xr.DataArray:
        """
        Generate posterior predictive samples on unseen data.

        Parameters
        ---------
        X_pred : array-like if sklearn is available, otherwise array, shape (n_pred, n_features)
            The input data used for prediction.
        extend_idata : Boolean determining whether the predictions should be added to inference data object.
            Defaults to True.
        combined: Combine chain and draw dims into sample. Won't work if a dim named sample already exists.
            Defaults to True.

        Returns
        -------
        y_pred : DataArray, shape (n_pred, chains * draws) if combined is True, otherwise (chains, draws, n_pred)
            Posterior predictive samples for each input X_pred
        """
        if not hasattr(self, "output_var"):
            raise NotImplementedError(f"Subclasses of {__class__} should set self.output_var")

        X_pred = self._validate_data(X_pred)
        posterior_predictive_samples = self.sample_posterior_predictive(
            X_pred, extend_idata, combined
        )

        if self.output_var not in posterior_predictive_samples:
            raise KeyError(
                f"Output variable {self.output_var} not found in posterior predictive samples."
            )

        return posterior_predictive_samples[self.output_var]

    def predict_proba(
        self,
        X_pred: Union[np.ndarray, pd.DataFrame, pd.Series],
        extend_idata: bool = True,
        combined: bool = False,
    ) -> xr.DataArray:
        """Alias for `predict_posterior`, for consistency with scikit-learn probabilistic estimators."""
        return self.predict_posterior(X_pred, extend_idata, combined)

    def sample_prior_predictive(
        self, X_pred, samples: int = None, extend_idata: bool = False, combined: bool = True
    ):
        """
        Sample from the model's prior predictive distribution.

        Parameters
        ---------
        X_pred : array, shape (n_pred, n_features)
            The input data used for prediction using prior distribution.
        samples : int
            Number of samples from the prior parameter distributions to generate.
            If not set, uses sampler_config['draws'] if that is available, otherwise defaults to 500.
        extend_idata : Boolean determining whether the predictions should be added to inference data object.
            Defaults to False.
        combined: Combine chain and draw dims into sample. Won't work if a dim named sample already exists.
            Defaults to True.

        Returns
        -------
        prior_predictive_samples : DataArray, shape (n_pred, samples)
            Prior predictive samples for each input X_pred
        """
        if samples is None:
            samples = self.sampler_config.get("draws", 500)

        if self.model is None:
            self.build_model()

        self._data_setter(X_pred)

        with self.model:  # sample with new input data
            prior_pred = pm.sample_prior_predictive(samples)
            self.set_idata_attrs(prior_pred)
            if extend_idata:
                if self.idata is not None:
                    self.idata.extend(prior_pred)
                else:
                    self.idata = prior_pred

        prior_predictive_samples = az.extract(prior_pred, "prior_predictive", combined=combined)

        return prior_predictive_samples

    def sample_posterior_predictive(self, X_pred, extend_idata, combined):
        """
        Sample from the model's posterior predictive distribution.

        Parameters
        ---------
        X_pred : array, shape (n_pred, n_features)
            The input data used for prediction using prior distribution..
        extend_idata : Boolean determining whether the predictions should be added to inference data object.
            Defaults to False.
        combined: Combine chain and draw dims into sample. Won't work if a dim named sample already exists.
            Defaults to True.

        Returns
        -------
        posterior_predictive_samples : DataArray, shape (n_pred, samples)
            Posterior predictive samples for each input X_pred
        """
        self._data_setter(X_pred)

        with self.model:  # sample with new input data
            post_pred = pm.sample_posterior_predictive(self.idata)
            if extend_idata:
                self.idata.extend(post_pred)

        posterior_predictive_samples = az.extract(
            post_pred, "posterior_predictive", combined=combined
        )

        return posterior_predictive_samples

    @property
    def _serializable_model_config(self):
        return self.model_config

    def get_params(self, deep=True):
        """
        Get all the model parameters needed to instantiate a copy of the model, not including training data.
        """
        return {"model_config": self.model_config, "sampler_config": self.sampler_config}

    def set_params(self, **params):
        """
        Set all the model parameters needed to instantiate the model, not including training data.
        """
        self.model_config = params["model_config"]
        self.sampler_config = params["sampler_config"]


class LinearModel(BayesianEstimator):
    """
    This class is an implementation of a single-input linear regression model in PYMC using the
    BayesianEstimator base class for interoperability with scikit-learn.
    """

    _model_type = "LinearModel"
    version = "0.1"

    @property
    def default_model_config(self):
        return {
            "intercept": {"loc": 0, "scale": 10},
            "slope": {"loc": 0, "scale": 10},
            "obs_error": 2,
        }

    @property
    def default_sampler_config(self):
        return {
            "draws": 1_000,
            "tune": 1_000,
            "chains": 3,
            "target_accept": 0.95,
        }

    def build_model(self):
        """
        Build the PyMC model.

        Returns
        -------
        None

        Examples
        --------
        >>> self.build_model()
        >>> assert self.model is not None
        >>> assert isinstance(self.model, pm.Model)
        >>> assert "intercept" in self.model.named_vars
        >>> assert "slope" in self.model.named_vars
        >>> assert "σ_model_fmc" in self.model.named_vars
        >>> assert "y_model" in self.model.named_vars
        >>> assert "y_hat" in self.model.named_vars
        >>> assert self.output_var == "y_hat"
        """
        cfg = self.model_config

        # The model is built with placeholder data.
        # Actual data will be set by _data_setter when fitting or evaluating the model.
        # Data array size can change but number of dimensions must stay the same.
        with pm.Model() as self.model:
            x = pm.MutableData("x", np.zeros((1,)), dims="observation")
            y_data = pm.MutableData("y_data", np.zeros((1,)), dims="observation")

            # priors
            intercept = pm.Normal(
                "intercept", cfg["intercept"]["loc"], sigma=cfg["intercept"]["scale"]
            )
            slope = pm.Normal("slope", cfg["slope"]["loc"], sigma=cfg["slope"]["scale"])
            obs_error = pm.HalfNormal("σ_model_fmc", cfg["obs_error"])

            # Model
            y_model = pm.Deterministic("y_model", intercept + slope * x, dims="observation")

            # observed data
            y_hat = pm.Normal(
                "y_hat",
                y_model,
                sigma=obs_error,
                shape=x.shape,
                observed=y_data,
                dims="observation",
            )
            self.output_var = "y_hat"

    def _data_setter(self, X, y=None):
        with self.model:
            pm.set_data({"x": X[:, 0]})
            if y is not None:
                pm.set_data({"y_data": y.squeeze()})

    @classmethod
    def generate_model_data(cls, nsamples=100, data=None):
        """
        Generate model data for linear regression.

        Parameters
        ----------
        nsamples : int, optional
            The number of samples to generate. Default is 100.
        data : np.ndarray, optional
            An optional data array to add noise to.

        Returns
        -------
        tuple
            A tuple of two np.ndarrays representing the feature matrix and target vector, respectively.

        Examples
        --------
        >>> import numpy as np
        >>> x, y = cls.generate_model_data()
        >>> assert isinstance(x, np.ndarray)
        >>> assert isinstance(y, np.ndarray)
        >>> assert x.shape == (100, 1)
        >>> assert y.shape == (100,)
        """
        x = np.linspace(start=0, stop=1, num=nsamples)
        y = 5 * x + 3
        y = y + np.random.normal(0, 1, len(x))

        x = np.expand_dims(x, -1)  # scikit assumes a dimension for features.
        return x, y