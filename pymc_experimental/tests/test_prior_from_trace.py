import pymc_experimental as pmx
import pymc as pm
from pymc.distributions import transforms
import pytest


@pytest.mark.parametrize(
    "case",
    [
        (("a", dict(name="b")), dict(name="b", transform=None)),
        (("a", None), dict(name="a", transform=None)),
        (("a", transforms.log), dict(name="a", transform=transforms.log)),
        (("a", dict(transform=transforms.log)), dict(name="a", transform=transforms.log)),
        (("a", dict(name="b")), dict(name="b", transform=None)),
    ],
)
def test_parsing_arguments(case):
    inp, out = case
    test = pmx.utils.prior.arg_to_param_cfg(*inp)
    assert test == out
