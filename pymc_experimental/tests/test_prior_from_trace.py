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
def coords():
    return dict(test=range(3))


@pytest.fixture
def param_cfg():
    return dict(
        a=pmx.utils.prior.arg_to_param_cfg("a"),
        b=pmx.utils.prior.arg_to_param_cfg(
            "b", dict(transform=transforms.sum_to_1, dims=("test",))
        ),
        c=pmx.utils.prior.arg_to_param_cfg("c", dict(transform=transforms.log, dims=("test",))),
    )


@pytest.fixture
def idata(param_cfg, coords):
    vars = dict()
    for k, cfg in param_cfg.items():
        if cfg["dims"] is not None:
            extra_dims = [len(coords[d]) for d in cfg["dims"]]
        else:
            extra_dims = []
        orig = np.random.randn(4, 100, *extra_dims)
        if cfg["transform"] is not None:
            var = cfg["transform"].backward(orig).eval()
        else:
            var = orig
        vars[k] = var
    return az.convert_to_inference_data(vars, coords=coords)


def test_idata_for_tests(idata, param_cfg):
    assert set(idata.posterior.keys()) == set(param_cfg)
    assert len(idata.posterior.coords["chain"]) == 4
    assert len(idata.posterior.coords["draw"]) == 100
