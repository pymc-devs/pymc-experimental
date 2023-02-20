import pytensor
from pytensor.graph import Constant, Type


class StringType(Type[str]):
    def clone(self, **kwargs):
        return type(self)()

    def filter(self, x, strict=False, allow_downcast=None):
        if isinstance(x, str):
            return x
        else:
            raise TypeError("Expected a string!")

    def __str__(self):
        return "string"

    @staticmethod
    def may_share_memory(a, b):
        return isinstance(a, str) and a is b


stringtype = StringType()


class StringConstant(Constant):
    pass


@pytensor._as_symbolic.register(str)
def as_symbolic_string(x, **kwargs):

    return StringConstant(stringtype, x)
