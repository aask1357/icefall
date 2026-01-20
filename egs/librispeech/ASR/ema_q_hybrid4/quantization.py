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
            return x.abs().amax()
        x_sorted = x.flatten().abs().sort().values
        k = round(self.q * x_sorted.numel())
        q_quantile = x_sorted[k]
        return q_quantile

    @torch.autocast("cuda", enabled=False)
    def forward(self, x: Tensor) -> Tensor:
        q_quantile = self.get_q_quantile(x)
        if not self._is_initialized_cpu:
            self._is_initialized_cpu = self.is_initialized.item()
            if not self._is_initialized_cpu:
                self.q_quantile.data.copy_(q_quantile)
                self.is_initialized.data.fill_(True)
                self._is_initialized_cpu = True

        if self.training:
            self.q_quantile.mul_(self.decay).add_(q_quantile, alpha=1 - self.decay)

        with torch.autocast("cuda", enabled=False):
            scale = self.q_quantile / self.q_max
            if self.learnable_gamma:
                self.gamma.data.clamp_(min=self.gamma_min, max=self.gamma_max)
                scale = scale * self.gamma
        return omniquant(x, scale, self.q_max, self.inplace)


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
        self.scale = scale
        self.gamma = nn.Parameter(torch.ones((1,)))

        self._is_initialized_cpu = True
        if self.mode == "scale":
            self.register_buffer("is_initialized", torch.tensor(False, dtype=torch.bool))
            self._is_initialized_cpu = False
            self.is_initialized: Tensor
        elif self.mode == "max":
            self.gamma = None

    @torch.autocast("cuda", enabled=False)
    def forward(self, x: Tensor) -> Tensor:
        """
        return round(x_clamped / scale) * scale
        """
        if not self._is_initialized_cpu:
            self._is_initialized_cpu = self.is_initialized.item()
            if not self._is_initialized_cpu:
                assert self.mode == "scale", self.mode
                scale = x.detach().abs().amax() / self.scale
                self.gamma.data.fill_(scale)
                self.is_initialized.data.fill_(True)
                self._is_initialized_cpu = True

        with torch.autocast("cuda", enabled=False):
            if self.mode == "scale":
                scale_max = x.detach().abs().amax() / self.scale
                self.gamma.data.clamp_(min=self.eps, max=scale_max)
                scale = self.gamma * (self.scale / self.q_max)
            elif self.mode == "max":
                x_abs_max = x.abs().amax().float()
                scale = x_abs_max / self.q_max
            elif self.mode == "max_gamma":
                self.gamma.data.clamp_(min=0.01, max=1.0)
                x_abs_max = x.abs().amax().float()
                scale = x_abs_max * (self.gamma / self.q_max)

        return omniquant(x, scale, self.q_max)
