import pytest

from pymc.distributions import Beta, Binomial, InverseGamma
from pymc.model.core import Model
from pymc.step_methods import Slice

from pymc_experimental import opt_sample


def test_custom_step_raises():
    with Model() as m:
        a = InverseGamma("a", 1, 1)
        b = InverseGamma("b", 1, 1)
        p = Beta("p", a, b)
        y = Binomial("y", n=100, p=p, observed=99)

        with pytest.raises(
            ValueError, match="The `step` argument is not supported in `opt_sample`"
        ):
            opt_sample(step=Slice([a, b]))
