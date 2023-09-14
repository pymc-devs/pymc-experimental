import pymc
import pytest


def test_imports_from_pymc():
    with pytest.warns(
        UserWarning,
        match="The functionality in this module has been moved to PyMC",
    ):
        from pymc_experimental.model_transform.conditioning import do as fn

        assert fn is pymc.do

        from pymc_experimental.model_transform.conditioning import observe as fn

        assert fn is pymc.observe

        from pymc_experimental.model_transform.conditioning import (
            change_value_transforms as fn,
        )

        assert fn is pymc.model.transform.conditioning.change_value_transforms

        from pymc_experimental.model_transform.conditioning import (
            remove_value_transforms as fn,
        )

        assert fn is pymc.model.transform.conditioning.remove_value_transforms
