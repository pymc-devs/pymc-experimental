from functools import wraps

from pymc import Model


def as_model(*model_args, **model_kwargs):
    R"""
    Decorator to provide context to PyMC models declared in a function.
    This removes all need to think about context managers and lets you separate creating a generative model from using the model.

    Adapted from `Rob Zinkov's blog post <https://www.zinkov.com/posts/2023-alternative-frontends-pymc/>`_ and inspired by the `sampled <https://github.com/colcarroll/sampled>`_ decorator for PyMC3.

    Examples
    --------
    .. code:: python

        import pymc as pm
        import pymc_experimental as pmx

        # The following are equivalent

        # standard PyMC API with context manager
        with pm.Model(coords={"obs": ["a", "b"]}) as model:
            x = pm.Normal("x", 0., 1., dims="obs")
            pm.sample()

        # functional API using decorator
        @pmx.as_model(coords={"obs": ["a", "b"]})
        def basic_model():
            pm.Normal("x", 0., 1., dims="obs")

        m = basic_model()
        pm.sample(model=m)

    """

    def decorator(f):
        @wraps(f)
        def make_model(*args, **kwargs):
            with Model(*model_args, **model_kwargs) as m:
                f(*args, **kwargs)
            return m

        return make_model

    return decorator
