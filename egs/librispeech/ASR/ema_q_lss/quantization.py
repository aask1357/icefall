from typing import Optional, Union
import torch
from torch import Tensor
import torch.nn as nn

from icefall.quantization import omniquant


class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.round()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class ActQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int,
        eps: float = 1.0e-5,
        inplace: bool = False,
        act_gamma: bool = True,
    ):
        super().__init__()
        self.q_max = float(2 ** (n_bits - 1) - 1)
        self.eps = eps
        self.inplace = inplace
        self._is_initialized_cpu = True
        self.register_buffer("x_abs_max", torch.tensor(1.0))
        if act_gamma:
            self._is_initialized_cpu = True
            self.gamma = nn.Parameter(torch.ones((1,)))
        else:
            self.gamma = None

    @torch.no_grad()
    def get_q_quantile(self, x: Tensor) -> Tensor:
        if self.q == 1.0:
            return x.abs().amax()
        x_sorted = x.flatten().abs().sort().values
        k = round(self.q * x_sorted.numel())
        q_quantile = x_sorted[k]
        return q_quantile

    @torch.autocast("cuda", enabled=False)
    def forward(self, x: Tensor) -> Tensor:
        if not self._is_initialized_cpu:
            x_abs_max = x.abs().amax()
            self.gamma.data.fill_(x_abs_max)
            self.x_abs_max.data.fill_(x_abs_max)
            self._is_initialized_cpu = True

        x_abs_max = x.abs().amax()
        self.x_abs_max.data.mul_(0.95).add_(x_abs_max, alpha=0.05)
        if self.gamma is not None:
            self.gamma.data.clamp_(min=self.eps, max=self.x_abs_max)
            scale = self.gamma / self.q_max
        else:
            scale = self.x_abs_max / self.q_max
        return omniquant(x, scale, self.q_max, self.inplace)


class WeightQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int,
        eps: float = 1.0e-5,
        mode: str = "max",
    ):
        super().__init__()
        assert mode in ["scale", "max", "max_gamma"]
        self.q_max = float(2 ** (n_bits - 1) - 1)
        self.eps = eps
        self.mode = mode
        self.gamma = nn.Parameter(torch.ones((1,)))

        self._is_initialized_cpu = True
        if self.mode == "scale":
            self._is_initialized_cpu = True
        elif self.mode == "max":
            self.gamma = None

    @torch.autocast("cuda", enabled=False)
    def forward(self, x: Tensor) -> Tensor:
        """
        return round(x_clamped / scale) * scale
        """
        if not self._is_initialized_cpu:
            assert self.mode == "scale", self.mode
            scale = x.detach().abs().amax()
            self.gamma.data.fill_(scale)
            self._is_initialized_cpu = True

        if self.mode == "scale":
            scale_max = x.detach().abs().amax()
            self.gamma.data.clamp_(min=self.eps, max=scale_max)
            scale = self.gamma / self.q_max
        elif self.mode == "max":
            x_abs_max = x.abs().amax().float()
            scale = x_abs_max / self.q_max
        elif self.mode == "max_gamma":
            self.gamma.data.clamp_(min=0.01, max=1.0)
            x_abs_max = x.abs().amax().float()
            scale = x_abs_max * (self.gamma / self.q_max)

        return omniquant(x, scale, self.q_max)
