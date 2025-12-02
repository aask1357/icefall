from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torchao.quantization.granularity import PerTensor
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
    ):
        super().__init__()
        self.module = module
        self.act_obs = act_obs
        self.weight_obs = weight_obs
        self.mode = "observe"
        self.act_scale = 0.0
        self.act_zero_point = None
        self.dtype = module.weight.dtype
        self.dtype_target = act_obs.target_dtype
        self.weight = None
        self.bias = None
        self.weight_obs(self.module.weight)
        self.act_observed = False

    def set_quantized_mode(self):
        if self.mode == "int":
            self.mode = "quantized"
            return
        elif self.mode == "quantized":
            return
        self.mode = "quantized"
        self.act_scale, self.act_zero_point = self.act_obs.calculate_qparams()
        weight_scale, weight_zero_point = self.weight_obs.calculate_qparams()
        self.weight = to_affine_quantized_intx_static(
            self.module.weight,
            weight_scale,
            weight_zero_point,
            self.module.weight.shape,
            self.dtype_target,
            zero_point_domain=ZeroPointDomain.NONE
        )
        delattr(self.module, "weight")
        if hasattr(self.module, "bias") and self.module.bias is not None:
            self.bias = torch.nn.Parameter(self.module.bias.data.clone())
            delattr(self.module, "bias")

    def set_int_mode(self):
        assert self.mode == "quantized"
        self.mode = "int"

    def forward(self, x: torch.Tensor):
        if self.mode == "int":
            qinput = to_affine_quantized_intx_static(
                x,
                self.act_scale,
                self.act_zero_point,
                x.shape,
                self.dtype_target,
                zero_point_domain=ZeroPointDomain.NONE,
            )
            int8_i, scale_i, _ = qinput.tensor_impl.get_plain()
            int8_w, scale_w, _ = self.weight.tensor_impl.get_plain()
            return self._forward_int8(
                int8_i.to(torch.int32), int8_w.to(torch.int32),
                (scale_i * scale_w).to(self.dtype)
            )
        if self.mode == "observe":
            self.act_observed = True
            x = self.act_obs(x)
            w = self.module.weight
            b = None
            if hasattr(self.module, "bias"):
                b = self.module.bias
            return self.forward_module(x, w, b)
        elif self.mode == "quantized":
            x = to_affine_quantized_intx_static(
                x,
                self.act_scale,
                self.act_zero_point,
                x.shape,
                self.dtype_target,
                zero_point_domain=ZeroPointDomain.NONE,
            )
            int8_i, scale_i, _ = x.tensor_impl.get_plain()
            int8_w, scale_w, _ = self.weight.tensor_impl.get_plain()
            # x = int8_i.to(self.dtype) * scale_i
            # w = int8_w.to(self.dtype) * scale_w
            # b = self.bias
            # return self.forward_module(x, w, b)
            x = int8_i.to(torch.double)
            w = int8_w.to(torch.double)
            # if (scale_i * scale_w <= 2**-14):
            #     breakpoint()
            scale = q(
                q((scale_i * scale_w), scale_i.dtype),
                self.dtype
            )
            # x, w 모두 int로 변환해서 forward_module하는게 칩에서의 동작과 일치하지만,
            # GPU에서 int 연산 지원 안하는 경우가 많아서 그냥 double로 int 연산 simulation함.
            x = q(torch.round(self.forward_module(x, w, None)), self.dtype) * scale
            x = self.addbias(x, self.bias)
            return x

    def forward_module(self, x, w):
        raise NotImplementedError()


class QLinear(QModule):
    def forward_module(self, x, w, b):
        return F.linear(x, w, b)

    def addbias(self, x: torch.Tensor, b: Optional[torch.Tensor]):
        if b is not None:
            x = x + b
        return x

    def _forward_int8(self, x: torch.Tensor, w: torch.Tensor, scale: float):
        x = F.linear(x.to("cpu"), w.to("cpu")).to(device=x.device, dtype=self.dtype) * scale
        if self.bias is not None:
            x = x + self.bias
        return x


class QConv1d(QModule):
    def forward_module(self, x, w, b):
        m = self.module
        return F.conv1d(x, w, b, stride=m.stride, padding=m.padding, groups=m.groups, dilation=m.dilation)

    def addbias(self, x: torch.Tensor, b: Optional[torch.Tensor]):
        if b is not None:
            x = x + b.view(-1, 1)
        return x

    def _forward_int8(self, x: torch.Tensor, w: torch.Tensor, scale: float):
        m = self.module
        try:
            x = F.conv1d(
                x.to("cpu"), w.to("cpu"),
                stride=m.stride, padding=m.padding, groups=m.groups, dilation=m.dilation
            ).to(device=x.device, dtype=self.dtype) * scale
        except:
            print("Something bad happened.")
            breakpoint()
        if self.bias is not None:
            x = x + self.bias.unsqueeze(1)
        return x


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
) -> AffineQuantizedObserverBase:
    qmin, qmax = get_quant_min_max(dtype_target)
    return AffineQuantizedMinMaxObserver(
        MappingType.SYMMETRIC,
        dtype_target,
        granularity=PerTensor(),
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
) -> AffineQuantizedObserverBase:
    qmin, qmax = get_quant_min_max(dtype_target)
    return AffineQuantizedMSEObserver(
        MappingType.SYMMETRIC,
        dtype_target,
        granularity=PerTensor(),
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
) -> AffineQuantizedObserverBase:
    qmin, qmax = get_quant_min_max(dtype_target)
    return MovingAverageMinMaxObserver(
        MappingType.SYMMETRIC,
        dtype_target,
        granularity=PerTensor(),
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
) -> AffineQuantizedObserverBase:
    qmin, qmax = get_quant_min_max(dtype_target)
    return MovingAverageMSEObserver(
        MappingType.SYMMETRIC,
        dtype_target,
        granularity=PerTensor(),
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
) -> AffineQuantizedObserverBase:
    qmin, qmax = get_quant_min_max(dtype_target)
    return HistogramObserver(
        MappingType.SYMMETRIC,
        dtype_target,
        granularity=PerTensor(),
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
    *args, **kwargs
):
    _is_linear = lambda m, fqn: isinstance(m, torch.nn.Linear)
    _is_conv = lambda m, fqn: isinstance(m, torch.nn.Conv1d)
    _is_embedding = lambda m, fqn: isinstance(m, torch.nn.Embedding)

    def replacement_fn_linear(m):
        act_obs = OBSERVER_DICT[act_observer](dtype_target, dtype_scale)
        weight_obs = OBSERVER_DICT[weight_observer](dtype_target, dtype_scale)
        return QLinear(m, act_obs, weight_obs)

    def replacement_fn_conv(m):
        act_obs = OBSERVER_DICT[act_observer](dtype_target, dtype_scale)
        weight_obs = OBSERVER_DICT[weight_observer](dtype_target, dtype_scale)
        return QConv1d(m, act_obs, weight_obs)

    def replacement_fn_embedding(m):
        weight_obs = OBSERVER_DICT[weight_observer](dtype_target, dtype_scale)
        return QEmbedding(m, weight_obs, dtype_target)

    _replace_with_custom_fn_if_matches_filter(model, replacement_fn_linear, _is_linear)
    _replace_with_custom_fn_if_matches_filter(model, replacement_fn_conv, _is_conv)
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
    return module.set_quantized_mode()


def quantize_model(model: nn.Module, mode: str) -> None:
    """
    A helper function to quantize a model with `StaticQuantConfig`.
    This is a wrapper of `quantize_` function.
    """
    is_observed = lambda m, fqn: isinstance(m, QModule)
    quantize_(model, StaticQuantConfig(mode), is_observed)
