from typing import TypedDict, Optional, Union, Tuple, Sequence, Dict
import aeppl.transforms


class ParamCfg(TypedDict):
    name: str
    transform: Optional[aeppl.transforms.RVTransform]
    dims: Optional[Union[str, Tuple[str]]]


def _arg_to_param_cfg(
    key, value: Optional[Union[ParamCfg, aeppl.transforms.RVTransform, str, Tuple]] = None
):
    if value is None:
        cfg = ParamCfg(name=key, transform=None, dims=None)
    elif isinstance(value, Tuple):
        cfg = ParamCfg(name=key, transform=None, dims=value)
    elif isinstance(value, str):
        cfg = ParamCfg(name=value, transform=None, dims=None)
    elif isinstance(value, aeppl.transforms.RVTransform):
        cfg = ParamCfg(name=key, transform=value, dims=None)
    else:
        cfg = value.copy()
        cfg.setdefault("name", key)
        cfg.setdefault("transform", None)
        cfg.setdefault("dims", None)
    return cfg


def _parse_args(
    var_names: Sequence[str], **kwargs: Union[ParamCfg, aeppl.transforms.RVTransform, str, Tuple]
) -> Dict[str, ParamCfg]:
    results = dict()
    for var in var_names:
        results[var] = _arg_to_param_cfg(var)
    for key, val in kwargs.items():
        results[key] = _arg_to_param_cfg(key, val)
    return results
