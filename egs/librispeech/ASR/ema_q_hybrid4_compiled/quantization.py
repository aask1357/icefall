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


@torch.compile()
def amax_abs(x: Tensor) -> Tensor:
    return x.abs().amax()


@torch.compile()
def quantile_fp32(x: Tensor, k: int) -> Tensor:
    x_sorted = x.flatten().abs().sort().values
    q_quantile = x_sorted[k]
    return q_quantile


@torch.compile()
def quantile_fp16(x: Tensor, k: int) -> Tensor:
    x_sorted = x.flatten().abs().sort().values
    q_quantile = x_sorted[k]
    return q_quantile

# @torch.compile()
# def quantile(x: Tensor, k: int) -> Tensor:
#     x_sorted = x.flatten().abs().sort().values
#     q_quantile = x_sorted[k]
#     return q_quantile


@torch.compile()
def inplace_ema(x: Tensor, new: Tensor, alpha: float) -> Tensor:
    x.mul_(alpha).add_(new, alpha=1 - alpha)
    return x


@torch.compile()
def mul(a: Tensor, b: Tensor, alpha: float) -> Tensor:
    return a * b / alpha


class ActQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int,
        decay: float = 0.95,
        inplace: bool = False,
        quantile: float = 0.99,
        learnable_gamma: bool = True,
        gamma_min: Optional[float] = 0.1,
        gamma_max: Optional[float] = 10.0,
    ):
        super().__init__()
        self.q_max = float(2 ** (n_bits - 1) - 1)
        self.decay = decay
        self.inplace = inplace
        self.q = quantile
        self.register_buffer("q_quantile", torch.ones((1,)))
        self.q_quantile: Tensor
        self._is_initialized_cpu = False
        self.register_buffer("is_initialized", torch.tensor(False, dtype=torch.bool))
        self.is_initialized: Tensor
        self.learnable_gamma = learnable_gamma
        if self.learnable_gamma:
            self.gamma = nn.Parameter(torch.ones((1,)))
            self.gamma_min = gamma_min
            self.gamma_max = gamma_max

    @torch.no_grad()
    def get_q_quantile(self, x: Tensor) -> Tensor:
        if self.q == 1.0:
            return amax_abs(x)

        k = round(self.q * x.numel())
        if x.dtype == torch.float32:
            return quantile_fp32(x, k)
        elif x.dtype == torch.float16:
            return quantile_fp16(x, k)
        else:
            x_sorted = x.flatten().abs().sort().values
            return x_sorted[k]

    @torch.autocast("cuda", enabled=False)
    def forward(self, x: Tensor) -> Tensor:
        if not self._is_initialized_cpu:
            self._is_initialized_cpu = self.is_initialized.item()
            if not self._is_initialized_cpu:
                q_quantile = self.get_q_quantile(x)
                self.q_quantile.data.copy_(q_quantile)
                self.is_initialized.data.fill_(True)
                self._is_initialized_cpu = True

        if self.training:
            q_quantile = self.get_q_quantile(x)
            inplace_ema(self.q_quantile, q_quantile, self.decay)

        if self.learnable_gamma:
            self.gamma.data.clamp_(min=self.gamma_min, max=self.gamma_max)
            scale = mul(self.q_quantile, self.gamma, self.q_max)
        else:
            scale = self.q_quantile / self.q_max
        return omniquant(x, scale, self.q_max, self.inplace)


@torch.compile()
def mul_amax_abs_detach(x: Tensor, alpha: float) -> Tensor:
    return x.detach().abs().amax().float() * alpha


class WeightQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int,
        eps: float = 1.0e-5,
        decay: float = 0.95,
        mode: str = "max",
        scale: float = 0.3,
    ):
        super().__init__()
        assert mode in ["scale", "max", "max_gamma"]
        self.q_max = float(2 ** (n_bits - 1) - 1)
        self.eps = eps
        self.decay = decay
        self.mode = mode
        if self.mode == "max":
            self.gamma = None
        else:
            self.gamma = nn.Parameter(torch.ones((1,)))

        self._is_initialized_cpu = True
        if self.mode == "scale":
            self.register_buffer("is_initialized", torch.tensor(False, dtype=torch.bool))
            self._is_initialized_cpu = False
            self.is_initialized: Tensor
            self.scale = scale

    @torch.autocast("cuda", enabled=False)
    def forward(self, x: Tensor) -> Tensor:
        """
        return round(x_clamped / scale) * scale
        """
        if not self._is_initialized_cpu:
            self._is_initialized_cpu = self.is_initialized.item()
            if not self._is_initialized_cpu:
                if self.mode == "scale":
                    scale = mul_amax_abs_detach(x.flatten(), 1 / self.scale)
                    self.gamma.data.fill_(scale)
                self.is_initialized.data.fill_(True)
                self._is_initialized_cpu = True

        with torch.autocast("cuda", enabled=False):
            if self.mode == "scale":
                scale_max = mul_amax_abs_detach(x.flatten(), 1 / self.scale)
                self.gamma.data.clamp_(min=self.eps, max=scale_max)
                scale = self.gamma * (self.scale / self.q_max)
            elif self.mode == "max":
                scale = mul_amax_abs_detach(x.flatten(), 1 / self.q_max)
            else:
                self.gamma.data.clamp_(min=0.01, max=1.0)
                scale = mul_amax_abs_detach(x.flatten(), self.gamma / self.q_max)

        return omniquant(x, scale, self.q_max)
