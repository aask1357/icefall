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

import torch
import torch.nn as nn
from scaling import ScaledLinear

from icefall.utils import is_jit_tracing


class DecoderProj(nn.Module):
    def __init__(self, in_channels, out_channels, weight_norm):
        super().__init__()
        self.relu = nn.ReLU()
        if weight_norm:
            def _Linear(in_channels, out_channels):
                return nn.utils.parametrizations.weight_norm(
                    nn.Linear(in_channels, out_channels))
        else:
            _Linear = ScaledLinear
        self.linear = _Linear(in_channels, out_channels)
    
    def forward(self, x):
        y = self.relu(x)
        y = self.linear(y)
        return x + y


class Joiner(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
        dec_residual: bool = False,
        weight_norm: bool = False,
    ):
        super().__init__()

        self.weight_norm = weight_norm
        if weight_norm:
            def _Linear(in_channels, out_channels):
                return nn.utils.parametrizations.weight_norm(nn.Linear(in_channels, out_channels))
        else:
            _Linear = ScaledLinear
        self.encoder_proj = _Linear(encoder_dim, joiner_dim)
        if dec_residual:
            self.decoder_proj = DecoderProj(decoder_dim, joiner_dim, weight_norm)
        else:
            self.decoder_proj = _Linear(decoder_dim, joiner_dim)
        self.output_linear = _Linear(joiner_dim, vocab_size)

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
