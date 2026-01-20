from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        return x.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class QuantizedLinear(nn.Linear):
    def __init__(self, *args, n_bits_act: Optional[int] = 8, n_bits_weight: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "scale_input",
            torch.ones((self.in_features,)) if n_bits_act is not None else None
        )
        self.register_buffer("scale_output", torch.ones((self.out_features,)))
        self.scale_input: Optional[Tensor]
        self.scale_output: Tensor
        self.q_act = 2**(n_bits_act-1) - 1 + 0.5 if n_bits_act is not None else float("inf")
        self.q_weight = 2**(n_bits_weight-1) - 1 + 0.5

    def rescale_weight(self, target_norm: float = 0.1) -> None:
        norm = self.weight.data.norm() / self.weight.data.numel()**0.5
        scale = target_norm / norm
        self.weight.data *= scale
        self.register_buffer("scale_weight", 1 / scale)
        self.scale_weight: Tensor

    def forward(self, x: Tensor) -> Tensor:
        if self.scale_input is not None:
            x = x * self.scale_input
            x = RoundFunction.apply(x)          # type: ignore
            x = x.clamp(min=-self.q_act, max=self.q_act)

        weight = self.weight * self.scale_weight
        weight = RoundFunction.apply(weight)    # type: ignore
        weight = weight.clamp(min=-self.q_weight, max=self.q_weight)
        weight = weight * self.scale_output.view(-1, 1)

        return F.linear(x, weight, self.bias)


class QuantizedConv1d(nn.Conv1d):
    def __init__(
        self,
        *args,
        causal: bool = False,
        n_bits_act: Optional[int] = 8,
        n_bits_weight: int = 8,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        d, k, s = self.dilation[0], self.kernel_size[0], self.stride[0]
        self.causal_padding = d * (k - 1) - (s - 1) if causal else 0
        self.register_buffer(
            "scale_input",
            torch.ones((self.in_channels, 1)) if n_bits_act is not None else None
        )
        self.register_buffer("scale_output", torch.ones((self.out_channels, 1)))
        self.scale_input: Optional[Tensor]
        self.scale_output: Tensor
        self.q_act = 2**(n_bits_act-1) - 1 + 0.5 if n_bits_act is not None else float("inf")
        self.q_weight = 2**(n_bits_weight-1) - 1 + 0.5

    def rescale_weight(self, target_norm: float = 0.1) -> None:
        norm = self.weight.data.norm() / self.weight.data.numel()**0.5
        scale = target_norm / norm
        self.weight.data *= scale
        self.register_buffer("scale_weight", 1 / scale)
        self.scale_weight: Tensor

    def forward(self, x: Tensor) -> Tensor:
        if self.scale_input is not None:
            x = x * self.scale_input
            x = RoundFunction.apply(x)          # type: ignore
            x = x.clamp(min=-self.q_act, max=self.q_act)

        weight = self.weight * self.scale_weight
        weight = RoundFunction.apply(weight)    # type: ignore
        weight = weight.clamp(min=-self.q_weight, max=self.q_weight)
        weight = weight * self.scale_output.view(-1, 1, 1)

        if self.causal_padding > 0:
            x = F.pad(x, (self.causal_padding, 0))
        return F.conv1d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QuantizedEmbedding(nn.Embedding):
    def __init__(self, *args, n_bits_weight: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("scale_output", torch.tensor(1.0))
        self.scale_output: Tensor
        self.q_weight = 2**(n_bits_weight-1) - 1 + 0.5

    def rescale_weight(self, target_norm: float = 0.1) -> None:
        norm = self.weight.data.norm() / self.weight.data.numel()**0.5
        scale = target_norm / norm
        self.weight.data *= scale
        self.register_buffer("scale_weight", 1 / scale)
        self.scale_weight: Tensor

    def forward(self, x: Tensor) -> Tensor:
        if x.numel() < self.num_embeddings:
            # Scale output.
            x = (
                F.embedding(
                    x,
                    self.weight,
                    self.padding_idx,
                    None,
                    2.0,  # None, 2.0 relate to normalization
                    self.scale_grad_by_freq,
                    self.sparse,
                )
            )
            x = x * self.scale_weight
            x = RoundFunction.apply(x)    # type: ignore
            x = x.clamp(min=-self.q_weight, max=self.q_weight)
            return x * self.scale_output
        else:
            # Scale weight.
            weight = self.weight * self.scale_weight
            weight = RoundFunction.apply(weight)    # type: ignore
            weight = weight.clamp(min=-self.q_weight, max=self.q_weight)
            weight = weight * self.scale_output
            return (
                F.embedding(
                    x,
                    weight,
                    self.padding_idx,
                    None,
                    2.0,  # None, 2.0 relate to normalization
                    self.scale_grad_by_freq,
                    self.sparse,
                )
            )
