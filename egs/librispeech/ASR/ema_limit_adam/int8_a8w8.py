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


def q(
    x: Tensor,
    dtype: torch.dtype,
    check_underflow: bool = False,
    check_overflow: bool = True,
) -> Tensor:
    if dtype == torch.float16:
        mag = x.abs()
        if check_underflow and torch.any(mag < 2**-14):
            print("underflow detected")
            breakpoint()
        if check_overflow and torch.any(mag > 65504):   # max(fp16_normal) = (2 - 2**-10) * 2**15 = 65504
            print("overflow detected")
            breakpoint()
        x = torch.where(mag < 2**-14, torch.zeros_like(x), x).to(torch.float16)
    elif dtype == torch.bfloat16:
        mag = x.abs()
        x = torch.where(mag < 2**-126, torch.zeros_like(x), x).to(torch.bfloat16)
    else:
        x = x.to(dtype)
    return x


class QModule(nn.Module):
    """ Quantize weight -> Quantize act """
    def __init__(
        self,
        module: nn.Module,
        act_obs: AffineQuantizedObserverBase,
        weight_obs: AffineQuantizedObserverBase,
    ):
        super().__init__()
        self.module = module
        self.act_obs = act_obs
        self.mode = "observe"
        self.act_scale = torch.ones(1)
        self.act_zero_point: Optional[Tensor] = None
        self.dtype: torch.dtype = module.weight.dtype
        self.dtype_act: torch.dtype = act_obs.target_dtype
        self.dtype_weight: torch.dtype = weight_obs.target_dtype
        self.dtype_scale: torch.dtype = act_obs.scale_dtype
        self.qmax_act = get_quant_min_max(self.dtype_act)[1]
        self.qmax_weight = get_quant_min_max(self.dtype_weight)[1]
        
        # quantize weight
        weight_obs(module.weight)
        weight_scale, _ = weight_obs.calculate_qparams()
        weight_scale = q(self.reshape_weight_scale(weight_scale, "cout"), self.dtype_scale, True, True)
        self.register_buffer("weight_scale", weight_scale)  
        self.weight_scale: Tensor
        
        weight = module.weight.data / weight_scale
        weight = weight.clamp(min=-self.qmax_weight, max=self.qmax_weight).round() * weight_scale
        self.weight = nn.Parameter(q(weight, self.dtype))
        delattr(module, "weight")
        self.bias: Optional[Tensor] = None
        if module.bias is not None:
            bias = module.bias.data.clone()
            self.bias = nn.Parameter(q(bias, self.dtype))
            delattr(module, "bias")

    def set_mode(self, mode: str):
        if mode == "quantized":
            self.set_quantized_mode()
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def set_quantized_mode(self):
        assert self.mode == "observe", self.mode
        self.mode = "quantized"

        # Get act_scale
        act_scale, act_zero_point = self.act_obs.calculate_qparams()
        act_scale = q(self.reshape_act_scale(act_scale), self.dtype_scale, True, True)
        delattr(self, "act_scale")
        self.register_buffer("act_scale", act_scale)
        self.act_scale: Tensor
        if act_zero_point is not None:
            delattr(self, "act_zero_point")
            self.register_buffer("act_zero_point", act_zero_point)

    def reshape_weight_scale(self, scale: Tensor, channel: str) -> Tensor:
        raise NotImplementedError()

    def reshape_act_scale(self, scale: Tensor) -> Tensor:
        raise NotImplementedError()

    def forward(self, x: Tensor):
        if self.mode == "quantized":
            if self.act_zero_point is not None:
                x = x - self.act_zero_point
            x = x / self.act_scale
            x = x.clamp(min=-self.qmax_act, max=self.qmax_act).round().double()
            w = self.weight / self.weight_scale
            w = w.clamp(min=-self.qmax_weight, max=self.qmax_weight).round().double()
            scale_out = q(self.act_scale * self.weight_scale, self.dtype_scale, True, True)
            scale_out = self.reshape_act_scale(scale_out)
            x = q(self.forward_module(x, w, None), self.dtype) * scale_out
            x = q(x, self.dtype)
            x = self.addbias(x, self.bias)
            return x

        x = self.act_obs(x)
        w = self.weight
        x = self.forward_module(x, w, self.bias)
        return x

    def forward_module(self, x, w):
        raise NotImplementedError()

    def addbias(self, x: Tensor, b: Optional[Tensor]):
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
    def forward_module(self, x: Tensor, w: Tensor, b: Optional[Tensor]):
        m: nn.Conv1d = self.module # type: ignore
        return q(
            F.conv1d(
                x, w, b,
                stride=m.stride,
                padding=m.padding,
                groups=m.groups,
                dilation=m.dilation
            ),
            self.dtype
        )

    def addbias(self, x: Tensor, b: Optional[Tensor]):
        if b is not None:
            x = q(x + b.view(-1, 1), self.dtype)
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
        self.dtype_weight = weight_obs.target_dtype
        self.dtype_scale = weight_obs.scale_dtype
        self.qmax_weight = get_quant_min_max(self.dtype_weight)[1]

        # quantize weight
        weight_obs(module.weight)
        weight_scale, _ = weight_obs.calculate_qparams()
        weight_scale = q(weight_scale, self.dtype_scale, True, True)
        self.register_buffer("weight_scale", weight_scale)
        self.weight_scale: Tensor

        weight = module.weight.data / weight_scale
        weight = weight.clamp(min=-self.qmax_weight, max=self.qmax_weight).round() * weight_scale
        self.weight = nn.Parameter(q(weight, self.dtype))

    def forward(self, x):
        w = self.weight / self.weight_scale
        w = w.clamp(min=-self.qmax_weight, max=self.qmax_weight).round().double()
        x = q(F.embedding(x, w), self.dtype) * self.weight_scale
        return q(x, self.dtype)


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
    dtype_act: torch.dtype,
    dtype_weight: torch.dtype,
    dtype_scale: torch.dtype,
    act_observer: str = "mse",
    weight_observer: str = "minmax",
    per_channel: bool = False,
    *args, **kwargs,
):
    _is_linear = lambda m, fqn: isinstance(m, nn.Linear)
    def _is_dwconv(m, fqn):
        return isinstance(m, nn.Conv1d) \
            and m.groups == m.in_channels \
            and m.in_channels == m.out_channels
    def _is_conv(m, fqn):
        return isinstance(m, nn.Conv1d) and m.groups == 1
    _is_embedding = lambda m, fqn: isinstance(m, nn.Embedding)

    granularity = PerTensor() if not per_channel else PerAxis(axis=0)

    def replacement_fn_linear(m):
        act_obs = OBSERVER_DICT[act_observer](dtype_act, dtype_scale, PerTensor())
        weight_obs = OBSERVER_DICT[weight_observer](dtype_weight, dtype_scale, granularity)
        return QLinear(m, act_obs, weight_obs)

    def replacement_fn_dwconv(m):
        act_obs = OBSERVER_DICT[act_observer](dtype_act, dtype_scale, PerTensor())
        weight_obs = OBSERVER_DICT[weight_observer](dtype_weight, dtype_scale, granularity)
        return QConv1d(m, act_obs, weight_obs)

    def replacement_fn_conv(m):
        act_obs = OBSERVER_DICT[act_observer](dtype_act, dtype_scale, PerTensor())
        weight_obs = OBSERVER_DICT[weight_observer](dtype_weight, dtype_scale, granularity)
        return QConv1d(m, act_obs, weight_obs)

    def replacement_fn_embedding(m):
        weight_obs = OBSERVER_DICT[weight_observer](dtype_weight, dtype_scale, PerTensor())
        return QEmbedding(m, weight_obs, dtype_weight)

    _replace_with_custom_fn_if_matches_filter(model, replacement_fn_linear, _is_linear)
    _replace_with_custom_fn_if_matches_filter(model, replacement_fn_conv, _is_conv)
    _replace_with_custom_fn_if_matches_filter(model, replacement_fn_dwconv, _is_dwconv)
    _replace_with_custom_fn_if_matches_filter(model, replacement_fn_embedding, _is_embedding)


@dataclass
class StaticQuantConfig(AOBaseConfig):
    mode: str


@register_quantize_module_handler(StaticQuantConfig)
def _apply_static_quant(
    module: nn.Module,
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
