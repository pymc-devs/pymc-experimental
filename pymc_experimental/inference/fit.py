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


def fit(method, **kwargs):
    """
    Fit a model with an inference algorithm

    Parameters
    ----------
    method : str
        Which inference method to run.
        Supported: pathfinder or laplace

    kwargs are passed on.

    Returns
    -------
    arviz.InferenceData
    """
    if method == "pathfinder":
        from pymc_experimental.inference.pathfinder import fit_pathfinder

        # TODO: edit **kwargs to be more consistent with fit_pathfinder with blackjax and pymc backends.
        return fit_pathfinder(**kwargs)

    if method == "laplace":
        from pymc_experimental.inference.laplace import fit_laplace

        return fit_laplace(**kwargs)
