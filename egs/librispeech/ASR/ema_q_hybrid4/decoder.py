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
import torch.nn.functional as F
from scaling import ScaledConv1d, ScaledEmbedding
from quantized_layers import QuantizedEmbedding, QuantizedConv1d

from icefall.utils import is_jit_tracing


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
        n_bits_act: Optional[int] = None,
        n_bits_weight: int = 8,
        eps: float = 1.0e-5,
        decay: float = 0.95,
        weight_quantizer_mode: str = "max",
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

        self.embedding = QuantizedEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=decoder_dim,
            n_bits_weight=n_bits_weight,
            eps=eps,
            decay=decay,
            weight_quantizer_mode=weight_quantizer_mode,
        )
        self.blank_id = blank_id

        assert context_size >= 1, context_size
        self.context_size = context_size
        self.vocab_size = vocab_size
        if context_size > 1:
            self.conv = QuantizedConv1d(
                in_channels=decoder_dim,
                out_channels=decoder_dim,
                kernel_size=context_size,
                padding=0,
                groups=decoder_dim,
                bias=False,
                n_bits_act=n_bits_act,
                n_bits_weight=n_bits_weight,
                eps=eps,
                decay=decay,
                quantile=1.0,
                learnable_gamma=False,
                weight_quantizer_mode=weight_quantizer_mode,
                inplace=True,
            )
        else:
            # It is to support torch script
            self.conv = nn.Identity()

    def remove_weight_reparameterizations(self):
        embedding = nn.Embedding(
            num_embeddings=self.embedding.num_embeddings,
            embedding_dim=self.embedding.embedding_dim,
            _weight=self.embedding.get_weight(),
        )
        conv = nn.Conv1d(
            in_channels=self.conv.in_channels,
            out_channels=self.conv.out_channels,
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
            bias=self.conv.bias is not None,
        )
        embedding.weight.data.copy_(self.embedding.get_weight())
        conv.weight.data.copy_(self.conv.get_weight())
        if self.conv.bias is not None:
            conv.bias.data.copy_(self.conv.bias)
        self.embedding = embedding
        self.conv = conv

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
