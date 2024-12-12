from abc import ABC, abstractmethod
from typing import Literal, get_args

import pandas as pd
import pymc as pm

from model.modular.utilities import (
    ColumnType,
    get_X_data,
    hierarchical_prior_to_requested_depth,
    select_data_columns,
)
from patsy import dmatrix

PoolingType = Literal["none", "complete", "partial", None]
valid_pooling = get_args(PoolingType)


def _validate_pooling_params(pooling_columns: ColumnType, pooling: PoolingType):
    """
    Helper function to validate inputs to a GLM component.

    Parameters
    ----------
    pooling_columns: str or list of str
        Data columns used to construct a hierarchical prior
    pooling: str
        Type of pooling to use in the component

    Returns
    -------
    None
    """
    if pooling_columns is not None and pooling == "complete":
        raise ValueError("Index data provided but complete pooling was requested")
    if pooling_columns is None and pooling != "complete":
        raise ValueError(
            "Index data must be provided for partial pooling (pooling = 'partial') or no pooling "
            "(pooling = 'none')"
        )


class GLMModel(ABC):
    """Base class for GLM components. Subclasses should implement the build method to construct the component."""

    def __init__(self, name):
        self.model = None
        self.compiled = False
        self.name = name

    @abstractmethod
    def build(self, model=None):
        pass

    def __add__(self, other):
        return AdditiveGLMComponent(self, other)

    def __mul__(self, other):
        return MultiplicativeGLMComponent(self, other)

    def __str__(self):
        return self.name


class AdditiveGLMComponent(GLMModel):
    """Class to represent an additive combination of GLM components"""

    def __init__(self, left, right):
        self.left = left
        self.right = right
        super().__init__()

    def build(self, *args, **kwargs):
        return self.left.build(*args, **kwargs) + self.right.build(*args, **kwargs)


class MultiplicativeGLMComponent(GLMModel):
    """Class to represent a multiplicative combination of GLM components"""

    def __init__(self, left, right):
        self.left = left
        self.right = right
        super().__init__()

    def build(self, *args, **kwargs):
        return self.left.build(*args, **kwargs) * self.right.build(*args, **kwargs)


class Intercept(GLMModel):
    def __init__(
        self,
        name: str | None = None,
        *,
        pooling_cols: ColumnType = None,
        pooling: PoolingType = "complete",
        hierarchical_params: dict | None = None,
        prior: str = "Normal",
        prior_params: dict | None = None,
    ):
        """
        Class to represent an intercept term in a GLM model.

        By intercept, it is meant any constant term in the model that is not a function of any input data. This can be
        a simple constant term, or a hierarchical prior that creates fixed effects across level of one or more
        categorical variables.

        Parameters
        ----------
        name: str, optional
            Name of the intercept term. If None, a default name is generated based on the index_data.
        pooling_cols: str or list of str, optional
            Columns of the independent data to use as labels for pooling. These columns will be treated as categorical.
            If None, no pooling is applied. If a list is provided, a "telescoping" hierarchy is constructed from left
            to right, with the mean of each subsequent level centered on the mean of the previous level.
        pooling: str, one of ["none", "complete", "partial"], default "complete"
            Type of pooling to use for the intercept term. If "none", no pooling is applied, and each group in the
            index_data is treated as independent. If "complete", complete pooling is applied, and all data are treated
            as coming from the same group. If "partial", a hierarchical prior is constructed that shares information
            across groups in the index_data.
        hierarchical_params: dict, optional
            Additional keyword arguments to configure priors in the hierarchical_prior_to_requested_depth function.
            Options include:
                sigma_dist: str
                    Name of the distribution to use for the standard deviation of the hierarchy. Default is "Gamma"
                sigma_kwargs: dict
                    Additional keyword arguments to pass to the sigma distribution specified by the sigma_dist argument.
                    Default is {"alpha": 2, "beta": 1}
                offset_dist: str, one of ["zerosum", "normal", "laplace"]
                    Name of the distribution to use for the offset distribution. Default is "zerosum"
        prior: str, optional
            Name of the PyMC distribution to use for the intercept term. Default is "Normal".
        prior_params: dict, optional
            Additional keyword arguments to pass to the PyMC distribution specified by the prior argument.

        """
        _validate_pooling_params(pooling_cols, pooling)

        self.pooling_cols = pooling_cols
        self.hierarchical_params = hierarchical_params if hierarchical_params is not None else {}
        self.pooling = pooling if pooling_cols is not None else "complete"

        self.prior = prior
        self.prior_params = prior_params if prior_params is not None else {}

        if pooling_cols is None:
            pooling_cols = []
        elif isinstance(pooling_cols, str):
            pooling_cols = [pooling_cols]

        name = name or f"Intercept(pooling_cols={pooling_cols})"

        super().__init__(name=name)

    def build(self, model: pm.Model | None = None):
        model = pm.modelcontext(model)
        with model:
            if self.pooling == "complete":
                intercept = getattr(pm, self.prior.title())(f"{self.name}", **self.prior_params)
                return intercept

            intercept = hierarchical_prior_to_requested_depth(
                self.name,
                df=get_X_data(model)[self.pooling_cols],
                model=model,
                dims=None,
                no_pooling=self.pooling == "none",
                **self.hierarchical_params,
            )

        return intercept


class Regression(GLMModel):
    def __init__(
        self,
        name: str | None = None,
        *,
        feature_columns: ColumnType | None = None,
        prior: str = "Normal",
        pooling: PoolingType = "complete",
        pooling_columns: ColumnType | None = None,
        hierarchical_params: dict | None = None,
        **prior_params,
    ):
        """
        Class to represent a regression component in a GLM model.

        A regression component is a linear combination of input data and a set of parameters. The input data should be
        a DataFrame with the same index as the observed data. Parameteres can be made hierarchical by providing
        an index_data Series or DataFrame (which should have the same index as the observed data).

        Parameters
        ----------
        name: str, optional
            Name of the intercept term. If None, a default name is generated based on the index_data.
        feature_columns: str or list of str
            Columns of the independent data to use in the regression.
        prior: str, optional
            Name of the PyMC distribution to use for the intercept term. Default is "Normal".
        pooling: str, one of ["none", "complete", "partial"], default "complete"
            Type of pooling to use for the intercept term. If "none", no pooling is applied, and each group in the
            index_data is treated as independent. If "complete", complete pooling is applied, and all data are treated
            as coming from the same group. If "partial", a hierarchical prior is constructed that shares information
            across groups in the index_data.
        pooling_columns: str or list of str, optional
            Columns of the independent data to use as labels for pooling. These columns will be treated as categorical.
            If None, no pooling is applied. If a list is provided, a "telescoping" hierarchy is constructed from left
            to right, with the mean of each subsequent level centered on the mean of the previous level.
        hierarchical_params: dict, optional
            Additional keyword arguments to configure priors in the hierarchical_prior_to_requested_depth function.
            Options include:
                sigma_dist: str
                    Name of the distribution to use for the standard deviation of the hierarchy. Default is "Gamma"
                sigma_kwargs: dict
                    Additional keyword arguments to pass to the sigma distribution specified by the sigma_dist argument.
                    Default is {"alpha": 2, "beta": 1}
                offset_dist: str, one of ["zerosum", "normal", "laplace"]
                    Name of the distribution to use for the offset distribution. Default is "zerosum"
        prior_params:
            Additional keyword arguments to pass to the PyMC distribution specified by the prior argument.
        """
        _validate_pooling_params(pooling_columns, pooling)

        self.feature_columns = feature_columns
        self.pooling = pooling
        self.pooling_columns = pooling_columns

        self.prior = prior
        self.prior_params = prior_params

        name = name if name else f"Regression({feature_columns})"

        super().__init__(name=name)

    def build(self, model=None):
        model = pm.modelcontext(model)
        feature_dim = f"{self.name}_features"

        if feature_dim not in model.coords:
            model.add_coord(feature_dim, self.X.columns)

        with model:
            X = select_data_columns(get_X_data(model), self.feature_columns)

            if self.pooling == "complete":
                beta = getattr(pm, self.prior)(
                    f"{self.name}", **self.prior_params, dims=[feature_dim]
                )
                return X @ beta

            beta = hierarchical_prior_to_requested_depth(
                self.name,
                self.index_data,
                model=model,
                dims=[feature_dim],
                no_pooling=self.pooling == "none",
            )

            regression_effect = (X * beta.T).sum(axis=-1)
            return regression_effect


class Spline(Regression):
    def __init__(
        self,
        name: str,
        *,
        feature_column: str | None = None,
        n_knots: int = 10,
        prior: str = "Normal",
        index_data: pd.Series | None = None,
        pooling: PoolingType = "complete",
        **prior_params,
    ):
        """
        Class to represent a spline component in a GLM model.

        A spline component is a linear combination of basis functions that are piecewise polynomial. The basis functions
        are constructed using the `bs` function from the patsy library. The spline_data should be a Series with the same
        index as the observed data.

        The weights of the spline components can be made hierarchical by providing an index_data Series or DataFrame
        (which should have the same index as the observed data).

        Parameters
        ----------
        name: str, optional
            Name of the intercept term. If None, a default name is generated based on the index_data.
        n_knots: int, default 10
            Number of knots to use in the spline basis.
        feature_column: str
            Column of the independent data to use in the spline.
        index_data: Series or DataFrame, optional
            Index data used to build hierarchical priors. If there are multiple columns, the columns are treated as
            levels of a "telescoping" hierarchy, with the leftmost column representing the top level of the hierarchy,
            and depth increasing to the right.

            The index of the index_data must match the index of the observed data.
        prior: str, optional
            Name of the PyMC distribution to use for the intercept term. Default is "Normal".
        pooling: str, one of ["none", "complete", "partial"], default "complete"
            Type of pooling to use for the intercept term. If "none", no pooling is applied, and each group in the
            index_data is treated as independent. If "complete", complete pooling is applied, and all data are treated
            as coming from the same group. If "partial", a hierarchical prior is constructed that shares information
            across groups in the index_data.
        curve_type: str, one of ["log", "abc", "ns", "nss", "box-cox"]
            Type of curve to build. For details, see the build_curve function.
        prior_params: dict, optional
            Additional keyword arguments to pass to the PyMC distribution specified by the prior argument.
        hierarchical_params: dict, optional
            Additional keyword arguments to configure priors in the hierarchical_prior_to_requested_depth function.
            Options include:
                sigma_dist: str
                    Name of the distribution to use for the standard deviation of the hierarchy. Default is "Gamma"
                sigma_kwargs: dict
                    Additional keyword arguments to pass to the sigma distribution specified by the sigma_dist argument.
                    Default is {"alpha": 2, "beta": 1}
                offset_dist: str, one of ["zerosum", "normal", "laplace"]
                    Name of the distribution to use for the offset distribution. Default is "zerosum"
        """
        _validate_pooling_params(index_data, pooling)
        self.name = name if name else f"Spline({feature_column})"
        self.feature_column = feature_column
        self.n_knots = n_knots
        self.prior = prior
        self.prior_params = prior_params

        super().__init__(name=name)

    def build(self, model: pm.Model | None = None):
        model = pm.modelcontext(model)
        model.add_coord(f"{self.name}_spline", range(self.n_knots))

        with model:
            spline_data = {
                self.feature_column: select_data_columns(
                    get_X_data(model).get_value(), self.feature_column
                )
            }

            X_spline = dmatrix(
                f"bs({self.feature_column}, df={self.n_knots}, degree=3) - 1",
                data=spline_data,
                return_type="dataframe",
            )

            if self.pooling == "complete":
                beta = getattr(pm, self.prior)(
                    f"{self.name}", **self.prior_params, dims=f"{self.feature_column}_spline"
                )
                return X_spline @ beta

            elif self.pooling_columns is not None:
                X = select_data_columns(self.pooling_columns, model)
                beta = hierarchical_prior_to_requested_depth(
                    name=self.name,
                    X=X,
                    model=model,
                    dims=[f"{self.feature_column}_spline"],
                    no_pooling=self.pooling == "none",
                )

            spline_effect = (X_spline * beta.T).sum(axis=-1)
            return spline_effect