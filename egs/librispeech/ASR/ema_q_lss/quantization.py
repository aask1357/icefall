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
        self.is_initialized = True
        self.register_buffer("x_abs_max", torch.tensor(1.0))
        if act_gamma:
            self.is_initialized = False
            self.gamma = nn.Parameter(torch.ones((1,)))
        else:
            self.gamma = None
        self.finetuning = False

    def set_quantizer_finetuning_mode(self):
        if self.gamma is not None:
            self.is_initialized = True
            gamma = self.gamma.data
            delattr(self, "gamma")
            self.register_buffer("gamma", gamma)
            delattr(self, "x_abs_max")
        self.finetuning = True

    @torch.autocast("cuda", enabled=False)
    def forward(self, x: Tensor) -> Tensor:
        if not self.is_initialized:
            x_abs_max = x.abs().amax()
            self.gamma.data.fill_(x_abs_max)
            self.x_abs_max.data.fill_(x_abs_max)
            self.is_initialized = True

        if self.training and not self.finetuning:
            x_abs_max = x.abs().amax()
            self.x_abs_max.data.mul_(0.95).add_(x_abs_max, alpha=0.05)
        if self.gamma is not None:
            if self.training and not self.finetuning:
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

        self.is_initialized = True
        if self.mode == "scale":
            self.is_initialized = False
        elif self.mode == "max":
            self.gamma = None
        self.finetuning = False

    def set_quantizer_finetuning_mode(self, x=None):
        if self.mode == "scale":
            self.is_initialized = True
            gamma = self.gamma.data
            delattr(self, "gamma")
            self.register_buffer("gamma", gamma)
        elif x is None:
            self.is_initialized = False
        elif self.mode == "max":
            self.register_buffer("gamma", x.detach().abs().amax())
            self.mode = "scale"
        elif self.mode == "max_gamma":
            gamma = self.gamma.data
            delattr(self, "gamma")
            self.register_buffer("gamma", x.detach().abs().amax() * gamma)
            self.mode = "scale"
        self.finetuning = True

    @torch.autocast("cuda", enabled=False)
    def forward(self, x: Tensor) -> Tensor:
        """
        return round(x_clamped / scale) * scale
        """
        if not self.is_initialized:
            if self.mode == "scale":
                scale = x.detach().abs().amax()
                self.gamma.data.fill_(scale)
            else:
                self.set_quantizer_finetuning_mode(x)
            self.is_initialized = True

        if self.mode == "scale":
            if self.training and not self.finetuning:
                scale_max = x.detach().abs().amax()
                self.gamma.data.clamp_(min=self.eps, max=scale_max)
            scale = self.gamma / self.q_max
        elif self.mode == "max":
            x_abs_max = x.abs().amax().float()
            scale = x_abs_max / self.q_max
        elif self.mode == "max_gamma":
            if self.training:
                self.gamma.data.clamp_(min=0.01, max=1.0)
            x_abs_max = x.abs().amax().float()
            scale = x_abs_max * (self.gamma / self.q_max)

        return omniquant(x, scale, self.q_max)
