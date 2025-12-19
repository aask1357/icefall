from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from scaling import ScaledLinear, ScaledConv1d, ScaledEmbedding
from quantization import WeightQuantizer, HistogramQuantizer


class QuantizedLinear(ScaledLinear):
    def __init__(
        self,
        *args,
        n_bits_act: Optional[int] = 8,
        n_bits_weight: int = 8,
        percentile: float = 99.0,
        gamma: float = 0.95,
        eps: float = 1.0e-5,
        inplace: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        ActQuantizer = nn.Identity if n_bits_act is None else HistogramQuantizer
        self.quantizer_act = ActQuantizer(percentile, gamma, n_bits_act, eps, inplace)
        self.quantizer_weight = WeightQuantizer(n_bits_weight, eps)

    def get_weight(self) -> Tensor:
        return self.quantizer_weight(self.weight) * self.weight_scale.exp()

    def forward(self, x: Tensor) -> Tensor:
        x = self.quantizer_act(x)
        return F.linear(x, self.get_weight(), self.get_bias())


class QuantizedConv1d(ScaledConv1d):
    def __init__(
        self,
        *args,
        causal: bool = False,
        n_bits_act: Optional[int] = 8,
        n_bits_weight: int = 8,
        percentile: float = 99.0,
        gamma: float = 0.95,
        eps: float = 1.0e-5,
        inplace: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        ActQuantizer = nn.Identity if n_bits_act is None else HistogramQuantizer
        self.quantizer_act = ActQuantizer(percentile, gamma, n_bits_act, eps, inplace)
        self.quantizer_weight = WeightQuantizer(n_bits_weight, eps)
        d, k, s = self.dilation[0], self.kernel_size[0], self.stride[0]
        self.causal_padding = d * (k - 1) - (s - 1) if causal else 0

    def get_weight(self) -> Tensor:
        return self.quantizer_weight(self.weight) * self.weight_scale.exp()

    def forward(self, x: Tensor) -> Tensor:
        if self.causal_padding > 0:
            x = F.pad(x, (self.causal_padding, 0))
        x = self.quantizer_act(x)
        return F.conv1d(
            x, self.get_weight(), self.get_bias(),
            self.stride, self.padding, self.dilation, self.groups
        )


class QuantizedEmbedding(ScaledEmbedding):
    def __init__(self, *args, n_bits_weight: int = 8, eps: float = 1.0e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantizer = WeightQuantizer(n_bits_weight, eps)

    def get_weight(self) -> Tensor:
        return self.quantizer(self.weight) * self.scale.exp()

    def forward(self, x: Tensor) -> Tensor:
        if x.numel() < self.num_embeddings:
            # Scale output.
            x = (
                F.embedding(
                    x,
                    self.quantizer(self.weight),
                    self.padding_idx,
                    None,
                    2.0,  # None, 2.0 relate to normalization
                    self.scale_grad_by_freq,
                    self.sparse,
                )
            )
            return x * self.scale.exp()
        else:
            # Scale weight.
            return F.embedding(
                x,
                self.get_weight(),
                self.padding_idx,
                None,
                2.0,  # None, 2.0 relate to normalization
                self.scale_grad_by_freq,
                self.sparse,
            )
