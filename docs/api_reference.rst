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
.. automodule:: pymc_experimental.statespace
.. toctree::
   :maxdepth: 1

   statespace/core
   statespace/filters
   statespace/models
