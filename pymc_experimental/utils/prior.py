from typing import TypedDict, Optional, Union, Tuple, Sequence, Dict, List
import aeppl.transforms
import arviz
import pymc as pm
import numpy as np


class ParamCfg(TypedDict):
    name: str
    transform: Optional[aeppl.transforms.RVTransform]
    dims: Optional[Union[str, Tuple[str]]]


class ShapeInfo(TypedDict):
    # shape might not match slice due to a transform
    shape_u: Tuple[int]  # untransformed shape
    shape_t: Tuple[int]  # transformed shape
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
        data = (
            posterior[key]
            # combine all draws from all chains
            .stack(__sample__=["chain", "draw"])
            # move sample dim to the first position
            # no matter where it was before
            .transpose("__sample__", ...)
            # we need numpy data for all the rest functionality
            .values
        )
        # omitting __sample__
        # we need shape in the untransformed space
        shape_u = data.shape[1:]
        if cfg["transform"] is not None:
            # some transforms need original shape
            data = cfg["transform"].forward(data).eval()
            shape_t = data.shape[1:]
        else:
            shape_t = shape_u
        # now we can get rid of shape
        data = data.reshape(data.shape[0], -1)
        end = begin + data.shape[1]
        vars.append(data)
        sinfo = dict(shape_t=shape_t, shape_u=shape_u, slice=slice(begin, end))
        info.append(dict(sinfo=sinfo, vinfo=cfg))
        begin = end
    return dict(data=np.concatenate(vars, axis=-1), info=info)


def _mean_chol(flat_array: np.ndarray):
    mean = flat_array.mean(0)
    cov = np.cov(flat_array, rowvar=False)
    chol = np.linalg.cholesky(cov)
    return mean, chol


def _mvn_prior_from_flat_info(name, flat_info: FlatInfo):
    mean, chol = _mean_chol(flat_info["data"])
    base_dist = pm.Normal(name, np.zeros_like(mean))
    interim = mean + chol @ base_dist
    result = dict()
    for var_info in flat_info["info"]:
        sinfo = var_info["sinfo"]
        vinfo = var_info["vinfo"]
        var = interim[sinfo["slice"]].reshape(sinfo["shape_t"])
        if vinfo["transform"] is not None:
            var = vinfo["transform"].backward(var)
        var = var.reshape(sinfo["shape_u"])
        var = pm.Deterministic(vinfo["name"], var, dims=vinfo["dims"])
        result[vinfo["name"]] = var
    return result


def prior_from_idata(idata, name="trace_prior_", *, var_names: Sequence[str] = (), **kwargs):
    param_cfg = _parse_args(var_names=var_names, **kwargs)
    flat_info = _flatten(idata, **param_cfg)
    return _mvn_prior_from_flat_info(name, flat_info)
