import pymc_experimental as pmx
from pymc.distributions import transforms
import pytest
import arviz as az
import numpy as np


@pytest.mark.parametrize(
    "case",
    [
        (("a", dict(name="b")), dict(name="b", transform=None, dims=None)),
        (("a", None), dict(name="a", transform=None, dims=None)),
        (("a", transforms.log), dict(name="a", transform=transforms.log, dims=None)),
        (
            ("a", dict(transform=transforms.log)),
            dict(name="a", transform=transforms.log, dims=None),
        ),
        (("a", dict(name="b")), dict(name="b", transform=None, dims=None)),
        (("a", dict(name="b", dims="test")), dict(name="b", transform=None, dims="test")),
        (("a", ("test",)), dict(name="a", transform=None, dims=("test",))),
    ],
)
def test_parsing_arguments(case):
    inp, out = case
    test = pmx.utils.prior.arg_to_param_cfg(*inp)
    assert test == out


@pytest.fixture
def idata():
    a = np.random.randn(4, 1000, 3)
    b = np.exp(np.random.randn(4, 1000, 5))
    return az.convert_to_inference_data(dict(a=a, b=b))


def test_idata_for_tests(idata):
    assert set(idata.posterior.keys()) == {"a", "b"}
    assert len(idata.posterior.coords["chain"]) == 4
    assert len(idata.posterior.coords["draw"]) == 1000
