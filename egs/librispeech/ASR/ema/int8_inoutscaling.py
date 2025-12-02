from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torchao.quantization.granularity import PerTensor, PerAxis
from torchao.quantization.observer import (
    AffineQuantizedMinMaxObserver,
    AffineQuantizedObserverBase,
    AffineQuantizedMSEObserver,
)
from torchao.quantization.quant_primitives import MappingType, ZeroPointDomain
from torchao.dtypes import to_affine_quantized_intx_static
from torchao.quantization.quant_api import (
    _replace_with_custom_fn_if_matches_filter,
)
from torchao.core.config import AOBaseConfig
from torchao.quantization import quantize_
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)

from observer import (
    MovingAverageMinMaxObserver,
    MovingAverageMSEObserver,
    HistogramObserver,
)


import matplotlib.pyplot as plt

def plot_hist(data):
    plt.clf()
    plt.hist(data.squeeze().detach().cpu().numpy(), bins=100)
    plt.savefig("delete_it.png")


def q(x: Tensor, dtype: torch.dtype) -> Tensor:
    if dtype == torch.float16:
        mag = x.abs()
        x = torch.where(mag < 2**-14, torch.zeros_like(x), x).to(torch.float16)
    elif dtype == torch.bfloat16:
        mag = x.abs()
        x = torch.where(mag < 2**-126, torch.zeros_like(x), x).to(torch.bfloat16)
    else:
        x = x.to(dtype)
    return x


class QModule(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
        act_obs: torch.nn.Module,
        weight_obs: torch.nn.Module,
        weight_obs_scaled: torch.nn.Module,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.module = module
        self.act_obs = act_obs
        self.weight_obs = weight_obs
        self.weight_obs_scaled = weight_obs_scaled
        self.mode = "observe"
        self.act_scale = 0.0
        self.act_zero_point = None
        self.dtype = module.weight.dtype
        self.dtype_target = act_obs.target_dtype
        self.weight = None
        self.bias = None
        self.act_observed = False
        self.alpha = alpha
        self.scale_out = 1
        self.qmax = get_quant_min_max(self.dtype_target)[1]

    def set_mode(self, mode: str):
        if mode == "quantized":
            self.set_quantized_mode()
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def set_quantized_mode(self):
        assert self.mode == "observe", self.mode
        self.mode = "quantized"
        
        # Get act_scale
        act_scale, _ = self.act_obs.calculate_qparams()
        self.scale_input = 1 / self.reshape_act_scale(act_scale)
        
        # weight = weight * act_scale
        act_scale = self.reshape_weight_scale(act_scale, "cin")
        weight = self.module.weight.data.mul(act_scale)
        
        # get weight_scale
        weight_scale = weight.abs().view(weight.size(0), -1).max(dim=1).values / self.qmax
        weight /= self.reshape_weight_scale(weight_scale, "cout")
        self.weight = weight.round().to(self.dtype)
        self.scale_out = self.reshape_act_scale(weight_scale)

    def forward(self, x: torch.Tensor):
        if self.mode == "quantized":
            x = x * self.scale_input
            x = x.clamp(min=-self.qmax, max=self.qmax).round().double()
            w = self.weight.double()
            x = self.forward_module(x, w, None).to(self.dtype) * self.scale_out
            x = self.addbias(x, self.module.bias)
            return x

        x = self.act_obs(x)
        w = self.module.weight
        self.act_observed = True
        x = self.forward_module(x, w, None)
        x = self.addbias(x, self.module.bias)
        return x

    def forward_module(self, x, w):
        raise NotImplementedError()


class QLinear(QModule):
    def forward_module(self, x, w, b):
        return F.linear(x, w, b)

    def addbias(self, x: Tensor, b: Optional[Tensor]):
        if b is not None:
            x = x + b
        return x

    def reshape_act_scale(self, scale: Tensor) -> Tensor:
        return scale

    def reshape_weight_scale(self, scale: Tensor, channel: str) -> Tensor:
        if channel == "cin":
            return scale
        elif channel == "cout":
            return scale.view(-1, 1)
        else:
            raise ValueError(f"Unsupported channel: {channel}")


class QConv1d(QModule):
    def forward_module(self, x, w, b):
        m = self.module
        return F.conv1d(x, w, b, stride=m.stride, padding=m.padding, groups=m.groups, dilation=m.dilation)

    def addbias(self, x: torch.Tensor, b: Optional[torch.Tensor]):
        if b is not None:
            x = x + b.view(-1, 1)
        return x

    def reshape_act_scale(self, scale: Tensor) -> Tensor:
        return scale.view(-1, 1)

    def reshape_weight_scale(self, scale: Tensor, channel: str) -> Tensor:
        if self.module.groups == self.module.in_channels and self.module.in_channels == self.module.out_channels:
            # DWConv
            return scale.view(-1, 1, 1)
        else:
            if channel == "cin":
                return scale.view(-1, 1)
            elif channel == "cout":
                return scale.view(-1, 1, 1)
            else:
                raise ValueError(f"Unsupported channel: {channel}")


class QEmbedding(nn.Module):
    def __init__(self, module, weight_obs, dtype_target: torch.dtype):
        super().__init__()
        self.dtype = module.weight.dtype
        weight_obs(module.weight)
        weight_scale, weight_zero_point = weight_obs.calculate_qparams()
        weight = to_affine_quantized_intx_static(
            module.weight,
            weight_scale,
            weight_zero_point,
            module.weight.shape,
            dtype_target,
            zero_point_domain=ZeroPointDomain.NONE
        )
        self.weight = weight.tensor_impl.get_plain()[0]
        self.scale = weight_scale
        
    def forward(self, x):
        return F.embedding(x, self.weight).to(self.dtype) * self.scale


def get_quant_min_max(dtype: torch.dtype):
    if dtype == torch.int8:
        return -2**7+1, 2**7-1
    elif dtype == torch.int4:
        return -2**3+1, 2**3-1
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def get_minmax_observer(
    dtype_target: torch.dtype = torch.int8,
    dtype_scale: torch.dtype = torch.float32,
    granularity = PerTensor(),
) -> AffineQuantizedObserverBase:
    qmin, qmax = get_quant_min_max(dtype_target)
    return AffineQuantizedMinMaxObserver(
        MappingType.SYMMETRIC,
        dtype_target,
        granularity=granularity,
        eps=torch.finfo(dtype_scale).eps,
        scale_dtype=dtype_scale,
        zero_point_dtype=dtype_scale,
        zero_point_domain=ZeroPointDomain.NONE,
        quant_min=qmin,
        quant_max=qmax,
    )


def get_mse_observer(
    dtype_target: torch.dtype = torch.int8,
    dtype_scale: torch.dtype = torch.float32,
    granularity = PerTensor(),
) -> AffineQuantizedObserverBase:
    qmin, qmax = get_quant_min_max(dtype_target)
    return AffineQuantizedMSEObserver(
        MappingType.SYMMETRIC,
        dtype_target,
        granularity=granularity,
        eps=torch.finfo(dtype_scale).eps,
        scale_dtype=dtype_scale,
        zero_point_dtype=dtype_scale,
        zero_point_domain=ZeroPointDomain.NONE,
        steps=100,
        quant_min=qmin,
        quant_max=qmax,
    )


def get_moving_avg_minmax_observer(
    dtype_target: torch.dtype = torch.int8,
    dtype_scale: torch.dtype = torch.float32,
    granularity = PerTensor(),
) -> AffineQuantizedObserverBase:
    qmin, qmax = get_quant_min_max(dtype_target)
    return MovingAverageMinMaxObserver(
        MappingType.SYMMETRIC,
        dtype_target,
        granularity=granularity,
        eps=torch.finfo(dtype_scale).eps,
        scale_dtype=dtype_scale,
        zero_point_dtype=dtype_scale,
        zero_point_domain=ZeroPointDomain.NONE,
        gamma=0.99,
        quant_min=qmin,
        quant_max=qmax,
    )


def get_moving_avg_mse_observer(
    dtype_target: torch.dtype = torch.int8,
    dtype_scale: torch.dtype = torch.float32,
    granularity = PerTensor(),
) -> AffineQuantizedObserverBase:
    qmin, qmax = get_quant_min_max(dtype_target)
    return MovingAverageMSEObserver(
        MappingType.SYMMETRIC,
        dtype_target,
        granularity=granularity,
        eps=torch.finfo(dtype_scale).eps,
        scale_dtype=dtype_scale,
        zero_point_dtype=dtype_scale,
        zero_point_domain=ZeroPointDomain.NONE,
        steps=100,
        gamma=0.9,
        quant_min=qmin,
        quant_max=qmax,
    )


def get_histogram_observer(
    dtype_target: torch.dtype = torch.int8,
    dtype_scale: torch.dtype = torch.float32,
    granularity = PerTensor(),
) -> AffineQuantizedObserverBase:
    qmin, qmax = get_quant_min_max(dtype_target)
    return HistogramObserver(
        MappingType.SYMMETRIC,
        dtype_target,
        granularity=granularity,
        eps=torch.finfo(dtype_scale).eps,
        scale_dtype=dtype_scale,
        zero_point_dtype=dtype_scale,
        zero_point_domain=ZeroPointDomain.NONE,
        quant_min=qmin,
        quant_max=qmax,
    )


OBSERVER_DICT = {
    "minmax": get_minmax_observer,
    "mse": get_mse_observer,
    "moving_avg_minmax":  get_moving_avg_minmax_observer,
    "moving_avg_mse": get_moving_avg_mse_observer,
    "histogram": get_histogram_observer,
}


def insert_observers(
    model,
    dtype_target: torch.dtype,
    dtype_scale: torch.dtype,
    act_observer: str = "mse",
    weight_observer: str = "minmax",
    alpha: float = 0.5,
    *args, **kwargs,
):
    _is_linear = lambda m, fqn: isinstance(m, torch.nn.Linear)
    def _is_dwconv(m, fqn):
        return isinstance(m, torch.nn.Conv1d) \
            and m.groups == m.in_channels \
            and m.in_channels == m.out_channels
    def _is_conv(m, fqn):
        return isinstance(m, torch.nn.Conv1d) and m.groups == 1
    _is_embedding = lambda m, fqn: isinstance(m, torch.nn.Embedding)

    def replacement_fn_linear(m):
        act_obs = OBSERVER_DICT[act_observer](dtype_target, dtype_scale, PerAxis(axis=-1)) # 
        weight_obs = OBSERVER_DICT[weight_observer](dtype_target, dtype_scale, PerAxis(axis=1))
        weight_obs_scaled = OBSERVER_DICT[weight_observer](dtype_target, dtype_scale, PerTensor())
        return QLinear(m, act_obs, weight_obs, weight_obs_scaled, alpha=alpha)

    def replacement_fn_dwconv(m):
        act_obs = OBSERVER_DICT[act_observer](dtype_target, dtype_scale, PerAxis(axis=1))  # 
        weight_obs = OBSERVER_DICT[weight_observer](dtype_target, dtype_scale, PerAxis(axis=0))
        weight_obs_scaled = OBSERVER_DICT[weight_observer](dtype_target, dtype_scale, PerTensor())
        return QConv1d(m, act_obs, weight_obs, weight_obs_scaled, alpha=alpha)

    def replacement_fn_conv(m):
        act_obs = OBSERVER_DICT[act_observer](dtype_target, dtype_scale, PerAxis(axis=1))  # 
        weight_obs = OBSERVER_DICT[weight_observer](dtype_target, dtype_scale, PerAxis(axis=1))
        weight_obs_scaled = OBSERVER_DICT[weight_observer](dtype_target, dtype_scale, PerTensor())
        return QConv1d(m, act_obs, weight_obs, weight_obs_scaled, alpha=alpha)

    def replacement_fn_embedding(m):
        weight_obs = OBSERVER_DICT[weight_observer](dtype_target, dtype_scale, PerTensor())
        return QEmbedding(m, weight_obs, dtype_target)

    _replace_with_custom_fn_if_matches_filter(model, replacement_fn_linear, _is_linear)
    _replace_with_custom_fn_if_matches_filter(model, replacement_fn_conv, _is_conv)
    _replace_with_custom_fn_if_matches_filter(model, replacement_fn_dwconv, _is_dwconv)
    _replace_with_custom_fn_if_matches_filter(model, replacement_fn_embedding, _is_embedding)


@dataclass
class StaticQuantConfig(AOBaseConfig):
    mode: str


@register_quantize_module_handler(StaticQuantConfig)
def _apply_static_quant(
    module: torch.nn.Module,
    config: StaticQuantConfig,
):
    """
    Define a transformation associated with `StaticQuantConfig`.
    This is called by `quantize_`, not by the user directly.
    """
    return module.set_mode(config.mode)


def quantize_model(model: nn.Module, mode: str) -> None:
    """
    A helper function to quantize a model with `StaticQuantConfig`.
    This is a wrapper of `quantize_` function.
    """
    is_observed = lambda m, fqn: isinstance(m, QModule)
    quantize_(model, StaticQuantConfig(mode), is_observed)
