from typing import Optional, Union
import torch
from torch import Tensor
import torch.nn as nn


class Quantizer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        q_max: Optional[float],
        eps: float = 0.0,
        x_max: Optional[Union[Tensor, float]] = None,
        act_decay: float = 0.0,
    ) -> torch.Tensor:
        if x_max is None:
            x_max = x.abs().max()
        if q_max is None:
            return torch.clamp(x, -x_max, x_max)

        x_abs = x.abs()
        ctx.act_decay = act_decay
        if act_decay > 0.0:
            ctx.mask = x_abs >= x_max
        scale = q_max / x_abs.max().clamp_min(eps)
        output = torch.clamp((x * scale).round(), -q_max, q_max) / scale
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        if ctx.act_decay > 0.0:
            scale = grad_output.norm() * ctx.act_decay
            mask = ctx.mask.to(grad_output.dtype)
            grad_output = grad_output + scale * mask
        return grad_output, None, None, None, None


def quantize(
    x: torch.Tensor,
    q_max: float,
    eps: float = 0.0,
    x_max: Optional[Union[Tensor, float]] = None,
    act_decay: float = 0.0,
) -> torch.Tensor:
    return Quantizer.apply(x, q_max, eps, x_max, act_decay)


def quantile(x: Tensor, q: float) -> Tensor:
    # Too Slow
    k = round(q * x.numel()) - 1
    return x.flatten().kthvalue(k).values


def quantile_from_sorted(x_sorted: Tensor, q: float) -> Tensor:
    k = round(q * x_sorted.numel()) - 1
    return x_sorted.flatten()[k]


class PercentileObserver(nn.Module):
    def __init__(self, gamma=0.99):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("max", torch.tensor(0.0))
        self.register_buffer("p99_99", torch.tensor(0.0))
        self.register_buffer("p99_9", torch.tensor(0.0))
        self.register_buffer("p99", torch.tensor(0.0))

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            with torch.no_grad():
                x_abs = x.abs().flatten().sort().values
                # max_val = x_abs.max()
                # p99_99 = quantile(x_abs, 0.9999)
                # p99_9 = quantile(x_abs, 0.999)
                # p99 = quantile(x_abs, 0.99)
                max_val = x_abs[-1]
                p99_99 = quantile_from_sorted(x_abs, 0.9999)
                p99_9 = quantile_from_sorted(x_abs, 0.999)
                p99 = quantile_from_sorted(x_abs, 0.99)

                self.max.mul_(self.gamma).add_(max_val * (1 - self.gamma))
                self.p99_99.mul_(self.gamma).add_(p99_99 * (1 - self.gamma))
                self.p99_9.mul_(self.gamma).add_(p99_9 * (1 - self.gamma))
                self.p99.mul_(self.gamma).add_(p99 * (1 - self.gamma))
        return x


class ConstantQuantizer(nn.Module):
    def __init__(self, clamp_value: float, n_bits: int, eps: float, act_decay: float):
        super().__init__()
        self.clamp_value = clamp_value
        self.q_max = float(2 ** (n_bits - 1) - 1)
        self.eps = eps
        self.act_decay = act_decay

    def forward(self, x: Tensor) -> Tensor:
        return quantize(x, self.q_max, self.eps, self.clamp_value, self.act_decay)


class HistogramQuantizer(nn.Module):
    def __init__(
        self,
        percentile: float,
        gamma: float = 0.99,
        n_bits: int = 8,
        eps: float = 0.0,
        act_decay: float = 0.0
    ):
        super().__init__()
        self.percentile = percentile / 100.0
        self.gamma = gamma
        self.register_buffer("clamp_value", torch.tensor(100.0))
        self.q_max = float(2 ** (n_bits - 1) - 1)
        self.eps = eps
        self.act_decay = act_decay

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            with torch.no_grad():
                x_abs = x.abs().flatten().sort().values
                clamp_value = quantile_from_sorted(x_abs, self.percentile)
                self.clamp_value.mul_(self.gamma).add_(clamp_value * (1 - self.gamma))
        else:
            clamp_value = self.clamp_value
        return quantize(x, self.q_max, self.eps, clamp_value, self.act_decay)


def get_activation_quantizer(method: str, n_bits: int, eps: float, act_decay: float) -> nn.Module:
    if method == "None":
        return nn.Identity()
    elif method.startswith("Constant-"):
        clamp_value = float(method.split("-")[1])
        return ConstantQuantizer(clamp_value, n_bits, eps, act_decay)
    elif method == "Observer":
        return PercentileObserver()
    elif method.startswith("Histogram-"):
        clamp_value = float(method.split("-")[1])
        return HistogramQuantizer(clamp_value, n_bits, eps, act_decay)
    else:
        raise ValueError(f"invalid clamp_method '{method}'")
