#   Copyright 2022 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import logging

from pymc_experimental import gp, statespace, utils
from pymc_experimental.distributions import *
from pymc_experimental.inference.fit import fit
from pymc_experimental.inference.jax_find_map import find_MAP
from pymc_experimental.model.marginal.marginal_model import MarginalModel, marginalize
from pymc_experimental.model.model_api import as_model
from pymc_experimental.version import __version__

_log = logging.getLogger("pmx")

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)
