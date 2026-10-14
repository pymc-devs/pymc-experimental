"""Public namespace for marginalization utilities.

The implementation lives in :mod:`pymc_extras.model.marginal`; this module
re-exports the public API under the shorter ``pymc_extras.marginal`` path.
"""

from pymc_extras.model.marginal.conditional import conditional, recover
from pymc_extras.model.marginal.marginalize import marginalize, unmarginalize

__all__ = ["conditional", "marginalize", "recover", "unmarginalize"]
