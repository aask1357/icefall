from typing import Optional, Union
import torch
from torch import Tensor
import torch.nn as nn

from icefall.quantization import omniquant, customquant


class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.round()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class Quantizer(nn.Module):
    def __init__(
        self,
        n_bits: int,
        eps: float = 1.0e-5,
        gamma: float = 0.95,
        inplace: bool = False,
        mode: str = "omni_max",
        exp_q_factor: bool = False,
    ):
        super().__init__()
        self.q_max = float(2 ** (n_bits - 1) - 1)
        self.eps = eps
        self.gamma = gamma
        quantize_method, mode = mode.split("_")
        self.quantize_method = quantize_method
        self.mode = mode
        self.exp_q_factor = exp_q_factor and self.mode != "scale"
        self.q_factor = nn.Parameter(torch.ones((1,)))
        if self.exp_q_factor:
            self.q_factor.data.copy_(torch.log(self.q_factor.data))
        self.inplace = inplace

        self._is_initialized_cpu = True
        if self.mode in ["scale", "emamax"]:
            self.register_buffer("is_initialized", torch.tensor(False, dtype=torch.bool))
            self._is_initialized_cpu = False
            self.is_initialized: Tensor
        
        if self.mode in ["emamax", "max"]:
            self.register_buffer("x_abs_max", torch.tensor(0.0))
            self.x_abs_max: Tensor

    def forward(self, x: Tensor) -> Tensor:
        """
        return round(x_clamped / scale) * scale
        """
        if not self._is_initialized_cpu:
            self._is_initialized_cpu = self.is_initialized.item()
            if not self._is_initialized_cpu:
                if self.mode == "scale":
                    scale = x.abs().amax() / self.q_max
                    self.q_factor.data.fill_(scale)
                elif self.mode == "emamax":
                    self.x_abs_max.data.fill_(x.abs().amax())
                self.is_initialized.data.fill_(True)
                self._is_initialized_cpu = True

        if self.mode == "scale":
            scale = self.q_factor
        else:
            with torch.amp.autocast("cuda", enabled=False):
                if self.exp_q_factor:
                    self.q_factor.data.clamp_(min=-4.6, max=0.0)
                    q_factor = torch.exp(self.q_factor)
                else:
                    self.q_factor.data.clamp_(min=0.01, max=1.0)
                    q_factor = self.q_factor

                if self.training:
                    x_abs_max = x.detach().abs().amax().float()
                    self.x_abs_max.data.mul_(self.gamma).add_(x_abs_max * (1 - self.gamma))
                    if self.mode == "emamax":
                        x_abs_max = self.x_abs_max
                    scale = x_abs_max * (q_factor / self.q_max)
                else:
                    x_abs_max = self.x_abs_max
                    scale = x_abs_max * (q_factor / self.q_max)

        if self.quantize_method == "omni":
            x = omniquant(x, scale, self.q_max, self.inplace)
        elif self.quantize_method == "custom":
            x = customquant(x, scale, self.q_max-self.eps, self.inplace)
        return x
