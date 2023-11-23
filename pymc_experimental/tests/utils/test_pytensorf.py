import pytensor.tensor as pt
import pytest

from pymc_experimental.utils.pytensorf import (
    named_shuffle_pattern,
    shuffle_named_tensor,
)


@pytest.mark.parametrize(
    "case",
    [
        [("time",), ("time", "group"), (0, "x")],
        [("time", "group"), ("time",), ValueError],
        [("time", "group"), ("group", "time"), (1, 0)],
        [("time",), ("time", None), (0, "x")],
        [(0,), (0, "group"), (0, "x")],
    ],
)
def testnamed_shuffle_pattern(case):
    inp, out, res = case
    if not isinstance(res, tuple):
        with pytest.raises(res):
            named_shuffle_pattern(inp, out)
    else:
        assert named_shuffle_pattern(inp, out) == res


@pytest.mark.parametrize(
    "case",
    [
        [("time",), ("time", "group"), (11, 4, 1)],
        [("time", "group"), ("time",), ValueError],
        [(), ("group",), (11, 1)],
    ],
)
def test_shuffle_named_tensor(case):
    inp, out, res = case
    input_shape = (11, *(len(d) for d in inp))
    tensor = pt.ones(input_shape)
    if not isinstance(res, tuple):
        with pytest.raises(res):
            shuffle_named_tensor(tensor, inp, out)
    else:
        out_tensor = shuffle_named_tensor(tensor, inp, out).eval()
        assert out_tensor.shape == res
