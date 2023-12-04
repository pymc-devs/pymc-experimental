API Reference
***************

This reference provides detailed documentation for all modules, classes, and
methods in the current release of PyMC experimental.

.. currentmodule:: pymc_experimental
.. autosummary::
   :toctree: generated/

   as_model
   MarginalModel
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

   Chi
   Maxwell
   DiscreteMarkovChain
   GeneralizedPoisson
   BetaNegativeBinomial
   GenExtreme
   R2D2M2CP
   Skellam
   histogram_approximation


Utils
=====

.. currentmodule:: pymc_experimental.utils
.. autosummary::
   :toctree: generated/

   spline.bspline_interpolation
   prior.prior_from_idata
   cache.cache_sampling


Statespace Models
=================
.. automodule:: pymc_experimental.statespace
.. toctree::
   :maxdepth: 1

   statespace/core
   statespace/filters
   statespace/models
