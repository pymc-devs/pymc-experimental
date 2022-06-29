from typing import TypedDict, Optional, Union
import aeppl.transforms


class ParamCfg(TypedDict):
    name: str
    transform: Optional[aeppl.transforms.RVTransform]


def arg_to_param_cfg(key, value: Optional[Union[ParamCfg, aeppl.transforms.RVTransform, str]]):
    if value is None:
        cfg = ParamCfg(name=key, transform=None)
    elif isinstance(value, str):
        cfg = ParamCfg(name=value, transform=None)
    elif isinstance(value, aeppl.transforms.RVTransform):
        cfg = ParamCfg(name=key, transform=value)
    else:
        cfg = value.copy()
        cfg.setdefault("name", key)
        cfg.setdefault("transform", None)
    return cfg
