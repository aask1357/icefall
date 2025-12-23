from typing import Optional, Union
import torch
from torch import Tensor
import torch.nn as nn


class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.round()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class Quantizer(nn.Module):
    def __init__(self, n_bits: int, eps: float = 1.0e-5, inplace: bool = False):
        super().__init__()
        self.q_max = float(2 ** (n_bits - 1) - 1)
        self.eps = eps
        self.q_factor = nn.Parameter(torch.ones((1,)))
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        self.q_factor.data.clamp_(min=self.eps, max=1)
        with torch.amp.autocast("cuda", enabled=False):
            x = x.float()
            scale = (x.abs().amax() * self.q_factor).clamp_min(self.eps) / self.q_max
            x = x.div(scale)
            x = Round.apply(x)
            x = x.clamp(min=-self.q_max-0.5, max=self.q_max+0.5)
            x = x.mul(scale)
        return x
