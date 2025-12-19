from typing import Optional, Union
import torch
from torch import Tensor
import torch.nn as nn


class Quantizer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        q_max: Union[Tensor, float],
        x_min: Optional[Union[Tensor, float]],
        x_max: Union[Tensor, float],
        eps: float = 1.0e-5,
    ) -> torch.Tensor:
        if x_min is None:
            y = x * (q_max / x_max)
            y = y.round()
            y = y * (x_max / q_max)
        else:
             y = (x - x_min) * (q_max / (x_max - x_min))
             y = y.round()
             y = y * ((x_max - x_min) / q_max) + x_min
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output, None, None, None, None


def quantize(
    x: torch.Tensor,
    q_max: Union[Tensor, float],
    x_min: Optional[Union[Tensor, float]],
    x_max: Union[Tensor, float],
    eps: float = 1.0e-5,
    inplace: bool = False,
    clamp: bool = True,
) -> torch.Tensor:
    round_x: Tensor = Quantizer.apply(x, q_max, x_min, x_max, eps)
    if not clamp:
        return round_x

    if x_min is None:
        clamp_max = x_max * (1 + 0.5 / q_max)
        clamp_min = -clamp_max
    else:
        clamp_min = x_min - (x_max - x_min) * (0.5 / q_max)
        clamp_max = x_max + (x_max - x_min) * (0.5 / q_max)
    if inplace:
        x = round_x.clamp_(min=clamp_min, max=clamp_max)
    else:
        x = round_x.clamp(min=clamp_min, max=clamp_max)
    return x


def quantile(x: Tensor, q: float) -> Tensor:
    # Too Slow
    k = round(q * x.numel()) - 1
    return x.flatten().kthvalue(k).values


def quantile_from_sorted(x_sorted: Tensor, q: float) -> Tensor:
    k = round(q * x_sorted.numel()) - 1
    return x_sorted.flatten()[k]


class HistogramQuantizer(nn.Module):
    def __init__(
        self,
        percentile: float = 99.0,    # 
        gamma: float = 0.95,
        n_bits: int = 8,
        eps: float = 1.0e-5,
        inplace: bool = False,
    ):
        super().__init__()
        self.percentile = percentile / 100.0
        self.gamma = gamma
        self.register_buffer("act_min", torch.tensor(-100.0))
        self.register_buffer("act_max", torch.tensor(100.0))
        self.q_max = float(2 ** (n_bits - 1) - 1) * 2
        self.eps = eps
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            with torch.no_grad():
                x_sorted = x.flatten().sort().values
                act_min = quantile_from_sorted(x_sorted, 1 - self.percentile)
                act_max = quantile_from_sorted(x_sorted, self.percentile)
                self.act_min.mul_(self.gamma).add_(act_min, alpha=1 - self.gamma)
                self.act_max.mul_(self.gamma).add_(act_max, alpha=1 - self.gamma)
        return quantize(x, self.q_max, self.act_min, self.act_max, self.eps, self.inplace)


class WeightQuantizer(nn.Module):
    def __init__(self, n_bits: int, eps: float = 1.0e-5):
        super().__init__()
        self.q_max = float(2 ** (n_bits - 1) - 1)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return quantize(x, self.q_max, None, x.abs().max(), self.eps, clamp=False)
