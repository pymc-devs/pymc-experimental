from pytensor import tensor as pt

from pymc_experimental.model.marginal.graph_analysis import is_conditional_dependent


def test_is_conditional_dependent_static_shape():
    """Test that we don't consider dependencies through "constant" shape Ops"""
    x1 = pt.matrix("x1", shape=(None, 5))
    y1 = pt.random.normal(size=pt.shape(x1))
    assert is_conditional_dependent(y1, x1, [x1, y1])

    x2 = pt.matrix("x2", shape=(9, 5))
    y2 = pt.random.normal(size=pt.shape(x2))
    assert not is_conditional_dependent(y2, x2, [x2, y2])
