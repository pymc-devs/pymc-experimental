API Reference
***************

Model
=====

This reference provides detailed documentation for all modules, classes, and
methods in the current release of PyMC experimental.

.. currentmodule:: pymc_extras
.. autosummary::
   :toctree: generated/

   as_model
   MarginalModel
   marginalize
   model_builder.ModelBuilder

Inference
=========

.. currentmodule:: pymc_extras.inference
.. autosummary::
   :toctree: generated/

   fit


Distributions
=============

.. currentmodule:: pymc_extras.distributions
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

.. currentmodule:: pymc_extras.utils
.. autosummary::
   :toctree: generated/

   spline.bspline_interpolation
   prior.prior_from_idata
   cache.cache_sampling


Statespace Models
=================
.. automodule:: pymc_extras.statespace
.. toctree::
   :maxdepth: 1

   statespace/core
   statespace/filters
   statespace/models


Model Transforms
================
.. automodule:: pymc_extras.model.transforms
.. autosummary::
   :toctree: generated/

   autoreparam.vip_reparametrize
   autoreparam.VIP


Printing
========
.. currentmodule:: pymc_extras.printing
.. autosummary::
   :toctree: generated/

   model_table
