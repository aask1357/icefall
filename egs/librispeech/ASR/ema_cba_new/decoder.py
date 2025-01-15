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
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm as weight_norm_fn
from scaling import ScaledConv1d, ScaledEmbedding

from icefall.utils import is_jit_tracing
from chip_fp16 import Q, q


@torch.jit.script
def normalize(x: Tensor, scale: float) -> Tensor:
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    return x * (scale / norm.clamp(min=1e-8))


class NormalizedEmbedding(nn.Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
        self.sqrt_dim: float = self.embedding_dim ** 0.5

    def forward(self, x: Tensor) -> Tensor:
        weight = normalize(self.weight, self.sqrt_dim)
        return F.embedding(
            x,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight, mode='fan_out')
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        Q(self)


class Decoder(nn.Module):
    """This class modifies the stateless decoder from the following paper:

        RNN-transducer with stateless prediction network
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9054419

    It removes the recurrent connection from the decoder, i.e., the prediction
    network. Different from the above paper, it adds an extra Conv1d
    right after the embedding layer.

    TODO: Implement https://arxiv.org/pdf/2109.07513.pdf
    """

    def __init__(
        self,
        vocab_size: int,
        decoder_dim: int,
        blank_id: int,
        context_size: int,
        weight_norm: bool = False,
    ):
        """
        Args:
          vocab_size:
            Number of tokens of the modeling unit including blank.
          decoder_dim:
            Dimension of the input embedding, and of the decoder output.
          blank_id:
            The ID of the blank symbol.
          context_size:
            Number of previous words to use to predict the next word.
            1 means bigram; 2 means trigram. n means (n+1)-gram.
        """
        super().__init__()

        self.embedding = ScaledEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=decoder_dim,
        )
        self.blank_id = blank_id

        assert context_size >= 1, context_size
        self.context_size = context_size
        self.vocab_size = vocab_size
        if context_size > 1:
            self.conv = ScaledConv1d(
                in_channels=decoder_dim,
                out_channels=decoder_dim,
                kernel_size=context_size,
                padding=0,
                groups=decoder_dim,
                bias=False,
            )
        else:
            # It is to support torch script
            self.conv = nn.Identity()

    def forward(
        self,
        y: torch.Tensor,
        need_pad: bool = True  # Annotation should be Union[bool, torch.Tensor]
        # but, torch.jit.script does not support Union.
    ) -> torch.Tensor:
        """
        Args:
          y:
            A 2-D tensor of shape (N, U).
          need_pad:
            True to left pad the input. Should be True during training.
            False to not pad the input. Should be False during inference.
        Returns:
          Return a tensor of shape (N, U, decoder_dim).
        """
        if isinstance(need_pad, torch.Tensor):
            # This is for torch.jit.trace(), which cannot handle the case
            # when the input argument is not a tensor.
            need_pad = bool(need_pad)

        y = y.to(torch.int64)
        # this stuff about clamp() is a temporary fix for a mismatch
        # at utterance start, we use negative ids in beam_search.py
        if torch.jit.is_tracing():
            # This is for exporting to PNNX via ONNX
            embedding_out = self.embedding(y)
        else:
            embedding_out = self.embedding(y.clamp(min=0)) * (y >= 0).unsqueeze(-1)
        if self.context_size > 1:
            embedding_out = embedding_out.permute(0, 2, 1)
            if need_pad:
                embedding_out = F.pad(embedding_out, pad=(self.context_size - 1, 0))
            else:
                # During inference time, there is no need to do extra padding
                # as we only need one output
                if not is_jit_tracing():
                    assert embedding_out.size(-1) == self.context_size
            embedding_out = self.conv(embedding_out)
            embedding_out = embedding_out.permute(0, 2, 1)
        embedding_out = F.relu(embedding_out)
        return embedding_out
