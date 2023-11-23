import pymc as pm
import pytest

from pymc_experimental.utils.hierarchy import zerosum_hierarchy


@pytest.fixture(autouse=True)
def model():
    with pm.Model() as model:
        yield model


def test_hierarchy_init(model):
    model.add_coords(dict(a=range(2), b=range(3)))
    var = zerosum_hierarchy("z0 ~ a", ("a",))
    assert "z0::_a" in model.named_vars
    assert "z0::weight" not in model.named_vars
    var = zerosum_hierarchy(
        "z1 ~ a+b",
        ("a", "b"),
        importance=dict(a=3),
    )
    assert "z1::_a" in model.named_vars
    assert "z1::_b" in model.named_vars
    assert "z1::weight" in model.named_vars
    var = zerosum_hierarchy("z2 ~ a*b", ("a", "b"))
    assert "z2::_a" in model.named_vars
    assert "z2::_b" in model.named_vars
    assert "z2::_a_b" in model.named_vars
    assert "z2::weight" in model.named_vars


def test_zerosum_hierarchy_formula_not_simple():
    with pytest.raises(ValueError, match="Formula should only have Variable terms"):
        zerosum_hierarchy("z ~ a+f(b)", ("a", "b"))


def test_zerosum_hierarchy_formula_group_terms():
    with pytest.raises(ValueError, match="Formula should not have any group terms"):
        zerosum_hierarchy("z ~ a:b + (1|b)", ("a", "b"))


def test_zerosum_hierarchy_formula_no_response():
    with pytest.raises(ValueError, match="Formula should have named Response"):
        zerosum_hierarchy("a", ("a",))


def test_zerosum_hierarchy_response_in_dims():
    with pytest.raises(ValueError, match="Named Response should not be in dims"):
        zerosum_hierarchy("a ~ b", ("a", "b"))


def test_zerosum_non_empty():
    with pytest.raises(ValueError, match="Formula should have at least one Term"):
        zerosum_hierarchy("a ~ 1", ())


def test_zerosum_hierarchy_formula_not_simple():
    with pytest.raises(ValueError, match="There are unexpected importance keys: {'b'}"):
        zerosum_hierarchy("z ~ a", ("a",), importance=dict(b=12.0))
