import pytensor.tensor as pt
import pytest
from pytensor.graph import FunctionGraph
from pytensor.graph.basic import equal_computations

from pymc_experimental.utils.pytensorf import toposort_replace


class TestToposortReplace:
    @pytest.mark.parametrize("compatible_type", (True, False))
    @pytest.mark.parametrize("num_replacements", (1, 2))
    @pytest.mark.parametrize("rebuild", (True, False))
    def test_horizontal_dependency(self, compatible_type, num_replacements, rebuild):
        x = pt.vector("x", shape=(5,))
        y = pt.vector("y", shape=(5,))

        out1 = pt.exp(x + y) + pt.log(x + y)
        out2 = pt.cos(out1)

        new_shape = (5,) if compatible_type else (10,)
        new_x = pt.vector("new_x", shape=new_shape)
        new_y = pt.vector("new_y", shape=new_shape)
        if num_replacements == 1:
            replacements = [(y, new_y)]
        else:
            replacements = [(x, new_x), (y, new_y)]

        fg = FunctionGraph([x, y], [out1, out2], clone=False)

        # If types are incompatible, and we don't rebuild or only replace one of the variables,
        # The function should fail
        if not compatible_type and (not rebuild or num_replacements == 1):
            with pytest.raises((TypeError, ValueError)):
                toposort_replace(fg, replacements, rebuild=rebuild)
            return
        toposort_replace(fg, replacements, rebuild=rebuild)

        if num_replacements == 1:
            expected_out1 = pt.exp(x + new_y) + pt.log(x + new_y)
        else:
            expected_out1 = pt.exp(new_x + new_y) + pt.log(new_x + new_y)
        expected_out2 = pt.cos(expected_out1)
        assert equal_computations(fg.outputs, [expected_out1, expected_out2])

    @pytest.mark.parametrize("compatible_type", (True, False))
    @pytest.mark.parametrize("num_replacements", (2, 3))
    @pytest.mark.parametrize("rebuild", (True, False))
    def test_vertical_dependency(self, compatible_type, num_replacements, rebuild):
        x = pt.vector("x", shape=(5,))
        a1 = pt.exp(x)
        a2 = pt.log(a1)
        out = a1 + a2

        new_x = pt.vector("new_x", shape=(5 if compatible_type else 10,))
        if num_replacements == 2:
            replacements = [(x, new_x), (a1, pt.cos(a1)), (a2, pt.sin(a2 + 5))]
        else:
            replacements = [(a1, pt.cos(pt.exp(new_x))), (a2, pt.sin(a2 + 5))]

        fg = FunctionGraph([x], [out], clone=False)

        if not compatible_type and not rebuild:
            with pytest.raises(TypeError):
                toposort_replace(fg, replacements, rebuild=rebuild)
            return
        toposort_replace(fg, replacements, rebuild=rebuild)

        expected_a1 = pt.cos(pt.exp(new_x))
        expected_a2 = pt.sin(pt.log(expected_a1) + 5)
        expected_out = expected_a1 + expected_a2
        assert equal_computations(fg.outputs, [expected_out])
