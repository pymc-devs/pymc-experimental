import pymc
import pytest


def test_imports_from_pymc():
    with pytest.warns(
        UserWarning,
        match="The functionality in this module has been moved to PyMC",
    ):
        from pymc_experimental.utils.model_fgraph import fgraph_from_model as fn

        assert fn is pymc.model.fgraph.fgraph_from_model

        from pymc_experimental.utils.model_fgraph import model_from_fgraph as fn

        assert fn is pymc.model.fgraph.model_from_fgraph

        from pymc_experimental.utils.model_fgraph import clone_model as fn

        assert fn is pymc.model.fgraph.clone_model
