API Reference
***************

Model
=====

This reference provides detailed documentation for all modules, classes, and
methods in the current release of PyMC experimental.

.. currentmodule:: pymc_experimental
.. autosummary::
   :toctree: generated/

   as_model
   MarginalModel
   marginalize
   model_builder.ModelBuilder
   opt_sample

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


Statespace Models
=================
.. automodule:: pymc_experimental.statespace
.. toctree::
   :maxdepth: 1

   statespace/core
   statespace/filters
   statespace/models


Model Transforms
================
.. automodule:: pymc_experimental.model.transforms
.. autosummary::
   :toctree: generated/

   autoreparam.vip_reparametrize
   autoreparam.VIP


Printing
========
.. currentmodule:: pymc_experimental.printing
.. autosummary::
   :toctree: generated/

   model_table
