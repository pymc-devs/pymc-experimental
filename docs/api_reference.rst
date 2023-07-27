API Reference
***************

This reference provides detailed documentation for all modules, classes, and
methods in the current release of PyMC experimental.

.. currentmodule:: pymc_experimental
.. autosummary::
   :toctree: generated/

   marginal_model.MarginalModel
   model_builder.ModelBuilder

Inference
=========

.. currentmodule:: pymc_experimental.inference
.. autosummary::
   :toctree: generated/

   fit


Distributions
=============

.. currentmodule:: pymc_experimental.distributions
.. autosummary::
   :toctree: generated/

   GenExtreme
   GeneralizedPoisson
   DiscreteMarkovChain
   R2D2M2CP
   histogram_approximation


Model Transformations
=====================

.. currentmodule:: pymc_experimental.model_transform
.. autosummary::
   :toctree: generated/

   conditioning.do
   conditioning.observe


Utils
=====

.. currentmodule:: pymc_experimental.utils
.. autosummary::
   :toctree: generated/

   clone_model
   spline.bspline_interpolation
   prior.prior_from_idata
   model_fgraph.fgraph_from_model
   model_fgraph.model_from_fgraph

Statespace Models
=====
.. currentmodule:: pymc_experimental.statespace
.. autosummary::
   :toctree: generated/

   core.representation.PytensorRepresentation
   core.statespace.PyMCStateSpace
   models.local_level.BayesianLocalLevel
   models.SARIMAX.BayesianARIMA
   models.VARMAX.BayesianVARMAX
   filters.kalman_filter.BaseFilter
   filters.kalman_filter.StandardFilter
   filters.kalman_filter.CholeskyFilter
   filters.kalman_filter.SingleTimeseriesFilter
   filters.kalman_filter.UnivariateFilter
   filters.kalman_filter.SteadyStateFilter
   filters.kalman_smoother.KalmanSmoother
   filters.distributions.LinearGaussianStateSpace
