from typing import TypedDict, Optional, Union, Tuple, Sequence, Dict, List
import aeppl.transforms
import arviz
import numpy as np


class ParamCfg(TypedDict):
    name: str
    transform: Optional[aeppl.transforms.RVTransform]
    dims: Optional[Union[str, Tuple[str]]]


class ShapeInfo(TypedDict):
    # shape might not match slice due to a transform
    shape: Tuple[int]
    slice: slice


class VarInfo(TypedDict):
    sinfo: ShapeInfo
    vinfo: ParamCfg


class FlatInfo(TypedDict):
    data: np.ndarray
    info: List[VarInfo]


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


def _flatten(idata: arviz.InferenceData, **kwargs: ParamCfg) -> FlatInfo:
    posterior = idata.posterior
    vars = list()
    info = list()
    begin = 0
    for key, cfg in kwargs.items():
        data = posterior[key].values
        # omitting chain, draw
        shape = data.shape[2:]
        if cfg["transform"] is not None:
            data = cfg["transform"].forward(data).eval()
        data = data.reshape(*data.shape[:2], -1)
        data = data.reshape(-1, data.shape[2])
        end = begin + data.shape[1]
        vars.append(data)
        info.append(dict(shape=shape, slice=slice(begin, end)))
        begin = end
    return dict(data=np.concatenate(vars, axis=-1), infp=info)
