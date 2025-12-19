# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

import torch
import torch.nn as nn
from scaling import ScaledLinear
from quantized_layers import QuantizedLinear

from icefall.utils import is_jit_tracing


class Joiner(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
        n_bits_act: Optional[int] = None,
        n_bits_weight: int = 8,
        quantizer_gamma: float = 0.95,
        eps: float = 1.0e-5,
    ):
        super().__init__()

        self.encoder_proj = QuantizedLinear(
            encoder_dim, joiner_dim,
            n_bits_act=n_bits_act,
            n_bits_weight=n_bits_weight,
            gamma=quantizer_gamma,
            eps=eps,
        )
        self.decoder_proj = QuantizedLinear(
            decoder_dim, joiner_dim,
            n_bits_act=n_bits_act,
            n_bits_weight=n_bits_weight,
            gamma=quantizer_gamma,
            eps=eps,
        )
        self.output_linear = QuantizedLinear(
            joiner_dim, vocab_size,
            n_bits_act=n_bits_act,
            n_bits_weight=n_bits_weight,
            gamma=quantizer_gamma,
            eps=eps,
        )

    def remove_weight_reparameterizations(self):
        encoder_proj = nn.Linear(
            self.encoder_proj.in_features,
            self.encoder_proj.out_features,
            bias=self.encoder_proj.bias is not None,
        )
        decoder_proj = nn.Linear(
            self.decoder_proj.in_features,
            self.decoder_proj.out_features,
            bias=self.decoder_proj.bias is not None,
        )
        output_linear = nn.Linear(
            self.output_linear.in_features,
            self.output_linear.out_features,
            bias=self.output_linear.bias is not None,
        )
        encoder_proj.weight.data.copy_(self.encoder_proj.get_weight())
        encoder_proj.bias.data.copy_(self.encoder_proj.get_bias())
        decoder_proj.weight.data.copy_(self.decoder_proj.get_weight())
        decoder_proj.bias.data.copy_(self.decoder_proj.get_bias())
        output_linear.weight.data.copy_(self.output_linear.get_weight())
        output_linear.bias.data.copy_(self.output_linear.get_bias())
        self.encoder_proj = encoder_proj
        self.decoder_proj = decoder_proj
        self.output_linear = output_linear

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
        project_input: bool = True,
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            Output from the encoder. Its shape is (N, T, s_range, C).
          decoder_out:
            Output from the decoder. Its shape is (N, T, s_range, C).
           project_input:
            If true, apply input projections encoder_proj and decoder_proj.
            If this is false, it is the user's responsibility to do this
            manually.
        Returns:
          Return a tensor of shape (N, T, s_range, C).
        """
        if not is_jit_tracing():
            assert encoder_out.ndim == decoder_out.ndim

        if project_input:
            logit = self.encoder_proj(encoder_out) + self.decoder_proj(decoder_out)
        else:
            logit = encoder_out + decoder_out

        logit = self.output_linear(torch.tanh(logit))

        return logit
