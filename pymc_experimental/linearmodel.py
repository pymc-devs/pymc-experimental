import numpy as np
import pymc as pm

from pymc_experimental.model_builder import ModelBuilder


class LinearModel(ModelBuilder):
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

    def build_model(self, data=None):
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
