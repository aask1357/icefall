import typing as tp
import random

import torch
from torch.nn import functional as F
from torch import Tensor, nn
from lhotse.utils import LOG_EPSILON
from icefall.iir.iir import EMA

from encoder_interface import EncoderInterface
from scaling import (
    ScaledConv1dAct,
    MovingAverageNorm,
)

from quantization import get_activation_quantizer


class CausalScaledConv1dAct(ScaledConv1dAct):
    def __init__(self, *args, use_cache=False, **kwargs):
        super().__init__(*args, **kwargs, padding_mode='zeros')
        d, k, s = self.dilation[0], self.kernel_size[0], self.stride[0]
        self.causal_padding = d * (k - 1) - (s - 1)
        assert self.padding[0] == 0, self.padding
        self.cache = torch.empty(0)
        self.use_cache = use_cache

    def empty_cache(self):
        self.cache = torch.empty(0)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_cache:
            padding = x.new_zeros(x.size(0), x.size(1), self.causal_padding)
            B = min(self.cache.size(0), x.size(0))
            if B > 0:
                padding[:B] = self.cache[:B]
            x = torch.cat((padding, x), dim=2)
            self.cache = x.detach()[:, :, -self.causal_padding:]
        else:
            x = F.pad(x, (self.causal_padding, 0))
        return super().forward(x)


class CausalSE(nn.Module):
    r"""
    Causal Squeeze-and-excitation module using Exponential Moving Average Layer.

    Args:
        dim (int): Dimension to be used for two fully connected (FC) layers

    Inputs: inputs, input_lengths
        - **inputs**: The output of the last convolution layer. `FloatTensor` of size
            ``(batch, dimension, seq_length)``
        - **input_lengths**: The length of input tensor. ``(batch)``

    Returns: output
        - **output**: Output of SELayer `FloatTensor` of size
            ``(batch, dimension, seq_length)``
    """
    def __init__(
        self,
        dim: int,
        activation: str = 'ReLU',
        chunksize: int = 16,
        se_gate: str = "sigmoid",
        gamma: float = 0.9,
        use_cache: bool = False,
        clamp_method: str = "None",
        n_bits_weight: int = 8,
        n_bits_act: int = 8,
        eps: float = 0.0,
        act_decay: float = 0.0,
        conv_scale_gamma: float = 0.95,
        ema_normalize: bool = False,
        manorm_after_ema: bool = False,
    ) -> None:
        super().__init__()
        assert dim % 8 == 0, 'Dimension should be divisible by 8.'
        self.dim = dim
        self.chunksize = chunksize
        self.activation = activation

        self.ema = EMA(dim, r_max=gamma, init_method="uniform", use_cache=use_cache, normalize=ema_normalize)
        self.norm = MovingAverageNorm(dim, gamma=conv_scale_gamma) if manorm_after_ema else nn.Identity()

        self.sequential = nn.Sequential(
            get_activation_quantizer(clamp_method, n_bits_act, eps, act_decay=act_decay),
            ScaledConv1dAct(dim, dim // 8, 1, n_bits=n_bits_weight, eps=eps, learn_weight_scale=False,
                gamma=conv_scale_gamma, activation=activation,
                activation_kwargs=dict(), target="std"),
            get_activation_quantizer(clamp_method, n_bits_act, eps, act_decay=act_decay),
            ScaledConv1dAct(dim // 8, dim, 1, n_bits=n_bits_weight, eps=eps, learn_weight_scale=False,
                gamma=conv_scale_gamma, activation="Identity", target="std"),
        )
        self.scale = nn.Parameter(torch.ones(dim))
        if se_gate == "sigmoid":
            self.gate = nn.Sigmoid()
        elif se_gate == "tanh":
            self.gate = nn.Tanh()
        else:
            raise ValueError(f"invalid se_gate '{se_gate}'")
    
    def remove_weight_reparameterizations(self):
        weight = self.sequential[0].weight
        kwargs = dict(device=weight.device, dtype=weight.dtype)
        self.ema.remove_weight_reparameterizations()

        dim = self.sequential[0].in_channels
        seq = self.sequential
        self.sequential = nn.Sequential(
            nn.Conv1d(dim, dim//8, 1, **kwargs),
            getattr(nn, self.activation)(),
            nn.Conv1d(dim//8, dim, 1, **kwargs)
        )
        self.sequential[0].weight.data.copy_(seq[0].get_weight())
        self.sequential[0].bias.data.copy_(seq[0].get_bias())

        idx = 2 if isinstance(seq[-1], nn.Conv1d) else 3
        self.sequential[2].weight.data.copy_(seq[idx].get_weight())
        self.sequential[2].bias.data.copy_(seq[idx].get_bias())

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        
        # get one vector per chunksize.
        # [B, C, L] -> [B, C, floor(L/cs)] (cs=chunksize)
        seq_lengths = x.size(2)
        chunksize = self.chunksize
        padding = (seq_lengths + chunksize - 1) // chunksize * chunksize - seq_lengths
        x = F.pad(x, (0, padding))      # pad right so that (L+P) % cs = 0
        B, C, L_P = x.shape
        x = x.view(B, C, L_P//chunksize, chunksize).mean(dim=3)  # [B, C, (L+P)//cs]

        # main network
        x = self.ema(x)
        x = self.norm(x)
        output = self.sequential(x)
        output = output.mul_(self.scale.view(-1, 1))
        output = self.gate(output)
        
        # expand length from (L+P)//cs to L
        # [B, C, (L+P)//cs] -> [B, C, L+P]
        output = output.repeat_interleave(chunksize, dim=2)
        output = output[:, :, :seq_lengths]

        return output.mul_(residual)


class ConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        channels_hidden: int,
        kernel_size: int = 5,
        dilation: int = 1,
        activation: str = 'ReLU',
        activation_kwargs: dict = {'inplace': True},
        dropout: float = 0.0,
        se_activation: str = 'ReLU',
        se_gate: str = "sigmoid",
        gamma: float = 0.9,
        use_cache: bool = False,
        chunksize: int = 16,
        clamp_method: str = "None",
        skip: str = "residual",
        n_bits_weight: int = 8,
        n_bits_act: int = 8,
        eps: float = 0.0,
        act_decay: float = 0.0,
        conv_scale_gamma: float = 0.95,
        channelwise_scale: bool = False,
        ema_normalize: bool = False,
        manorm_after_ema: bool = False,
    ) -> None:
        super(ConvBlock, self).__init__()

        self.clamp0 = get_activation_quantizer(clamp_method, n_bits_act, eps, act_decay=act_decay)
        self.pointwise1 = ScaledConv1dAct(
            channels, channels_hidden, 1,
            learn_weight_scale=False, n_bits=n_bits_weight, eps=eps,
            gamma=conv_scale_gamma, activation=activation, activation_kwargs=activation_kwargs,
        )
        self.clamp1 = get_activation_quantizer(clamp_method, n_bits_act, eps, act_decay=act_decay)
        self.depthwise = CausalScaledConv1dAct(
            channels_hidden, channels_hidden, kernel_size, groups=channels_hidden,
            dilation=dilation, use_cache=use_cache,
            learn_weight_scale=False, n_bits=n_bits_weight, eps=eps,
            gamma=conv_scale_gamma, activation=activation, activation_kwargs=activation_kwargs
        )
        self.clamp2 = get_activation_quantizer(clamp_method, n_bits_act, eps, act_decay=act_decay)
        self.pointwise2 = ScaledConv1dAct(
            channels_hidden, channels, 1, bias=False,
            learn_weight_scale=False, n_bits=n_bits_weight, eps=eps,
            gamma=conv_scale_gamma,
        )
        self.se = CausalSE(
            channels, se_activation, se_gate=se_gate,
            gamma=gamma, use_cache=use_cache, chunksize=chunksize, clamp_method=clamp_method,
            n_bits_weight=n_bits_weight, n_bits_act=n_bits_act, eps=eps, act_decay=act_decay,
            conv_scale_gamma=conv_scale_gamma, ema_normalize=ema_normalize,
            manorm_after_ema=manorm_after_ema,
        )
        self.dropout = nn.Dropout(dropout, inplace=True)

        if skip != "residual":
            num_features = channels if channelwise_scale else 1
            self.scale = nn.Parameter(torch.ones(num_features, 2))
            self.skip = skip.split("-")[0]
            if skip in ["bypass-zeroinit"]:
                self.scale.data[:, 0].zero_()
            else:
                self.scale.data.div_(2**0.5)
            self.norm = MovingAverageNorm(channels, gamma=conv_scale_gamma, inplace=True)

        self.initial_cache = None

    def remove_weight_reparameterizations(self):
        kwargs = dict(device=self.pointwise1.weight.device, dtype=self.pointwise1.weight.dtype)
        bias = False if self.pointwise1.bias is None else True

        conv = self.pointwise1
        self.pointwise1 = nn.Conv1d(conv.in_channels, conv.out_channels, 1,
                                    bias=bias, **kwargs)
        self.pointwise1.weight.data.copy_(conv.get_weight())
        if bias:
            self.pointwise1.bias.data.copy_(conv.get_bias())
        conv = self.depthwise
        self.depthwise = CausalConv1d(
            conv.in_channels, conv.out_channels, conv.kernel_size,
            groups=conv.groups, bias=bias, **kwargs)
        self.depthwise.weight.data.copy_(conv.get_weight())
        if bias:
            self.depthwise.bias.data.copy_(conv.get_bias())
        conv = self.pointwise2
        self.pointwise2 = nn.Conv1d(conv.in_channels, conv.out_channels, 1, **kwargs)
        self.pointwise2.weight.data.copy_(conv.get_weight())
        self.pointwise2.bias.data.copy_(conv.get_bias())
        self.se.remove_weight_reparameterizations()

    def forward(
        self,
        inputs: Tensor,
        input_lengths: Tensor,
        warmup: float,
    ) -> tp.Tuple[Tensor, Tensor]:
        r"""
        Forward propagate a `inputs` for convolution block.

        Args:
            **inputs** (torch.FloatTensor): Input of convolution block `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            **output** (torch.FloatTensor): Output of convolution block `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **output_lengths** (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        x = inputs
        x_len = input_lengths

        x = self.clamp0(x)
        x = self.pointwise1(x)
        if self.initial_cache is not None:
            x = torch.cat([self.initial_cache, x], dim=2)
        x = self.clamp1(x)
        x = self.depthwise(x)
        x = self.clamp2(x)
        x = self.pointwise2(x)
        x = self.se(x)
        x = self.dropout(x)

        if self.skip == "residual":
            x = inputs + x
        elif self.skip == "bypass":
            x = x * self.scale[:, 0:1] + inputs * self.scale[:, 1:2]
            x = self.norm(x)

        return x, x_len


class Conv1dSubsampling(nn.Module):
    def __init__(self, in_ch, out_ch, subsampling_factor,
                 clamp_method, n_bits_weight, n_bits_act, eps, act_decay,
                 conv_scale_gamma: float = 0.9):
        super().__init__()

        self.subsampling_factor = subsampling_factor
        sf = subsampling_factor

        self.conv = nn.Sequential(
            get_activation_quantizer(clamp_method, n_bits_act, eps, act_decay=act_decay),
            ScaledConv1dAct(
                in_ch, out_ch, 1, bias=False,
                learn_weight_scale=False, n_bits=n_bits_weight, eps=eps, gamma=conv_scale_gamma,
            ),
            get_activation_quantizer(clamp_method, n_bits_act, eps, act_decay=act_decay),
            ScaledConv1dAct(
                out_ch, out_ch, 2*sf, stride=sf, padding=0, groups=out_ch, bias=False,
                learn_weight_scale=False, n_bits=n_bits_weight, eps=eps, gamma=conv_scale_gamma,
            ),
        )

    def remove_weight_reparameterizations(self):
        sf = self.subsampling_factor
        conv = self.conv
        kwargs = dict(device=conv[0].weight.device, dtype=conv[0].weight.dtype)
        self.conv = nn.Sequential(
            nn.Conv1d(conv[0].in_channels, conv[0].out_channels, 1, bias=False, **kwargs),
            nn.Conv1d(conv[1].in_channels, conv[1].out_channels, 2*sf, stride=sf, padding=0,
                      groups=conv[1].groups, **kwargs),
        )
        self.conv[0].weight.data.copy_(conv[0].get_weight())
        self.conv[1].weight.data.copy_(conv[1].get_weight())
        self.conv[1].bias.data.copy_(conv[1].get_bias())

    def forward(self, x, x_len):
        # x = F.pad(x, (2, 2), value=LOG_EPSILON)
        x = self.conv(x)
        x_len = torch.floor(x_len / self.subsampling_factor)
        return x, x_len


class Encoder(EncoderInterface):
    def __init__(
        self,
        num_features: int,
        subsampling_factor: int = 4,
        channels: int = 256,
        channels_expansion: int = 1024,
        kernel_size: int = 8,
        dilations: tp.List[int] = [8 for _ in range(11)],
        output_channels: int = 512,
        dropout: float = 0.075,
        activation: str = 'ReLU',
        activation_kwargs: dict = {},
        se_activation: str = 'ReLU',
        se_gate: str = "tanh",
        gamma: float = 0.93,
        use_cache: bool = False,
        mean: float = -6.375938187300722,
        std: float = 4.354657227491409,
        chunksize: int = 16,
        skip: str = "residual",
        clamp_method: str = "None",
        n_bits_weight: int = 8,
        n_bits_act: int = 8,
        eps: float = 0.0,
        act_decay: float = 0.0,
        conv_scale_gamma: float = 0.95,
        ema_normalize: bool = False,
        manorm_after_ema: bool = False,
    ) -> None:
        assert skip in [
            "residual", "residual-zeroinit",
            "bypass", "bypass-zeroinit", "bypass-oneinit", "bypass-lin"
        ], f"invalid skip type '{skip}'"
        super().__init__()

        self.mean = mean
        self.rstd = 1 / std
        self.num_features = num_features
        self.subsampling_factor = subsampling_factor

        self.conv_pre = Conv1dSubsampling(
            num_features,
            channels,
            subsampling_factor=subsampling_factor,
            clamp_method=clamp_method,
            n_bits_weight=n_bits_weight,
            n_bits_act=n_bits_act,
            eps=eps,
            act_decay=act_decay,
            conv_scale_gamma=conv_scale_gamma,
        )

        self.cnn = nn.ModuleList()
        for idx_block, dilation in enumerate(dilations, start=1):
            layer = ConvBlock(
                channels, channels_expansion, kernel_size, dilation,
                activation, activation_kwargs, dropout, se_activation,
                se_gate, gamma=gamma,
                use_cache=use_cache, chunksize=chunksize,
                clamp_method=clamp_method, skip=skip,
                n_bits_weight=n_bits_weight, n_bits_act=n_bits_act, eps=eps, act_decay=act_decay,
                conv_scale_gamma=conv_scale_gamma, ema_normalize=ema_normalize,
                manorm_after_ema=manorm_after_ema,
            )
            self.cnn.append(layer)
        self.proj = nn.Sequential(
            get_activation_quantizer(clamp_method, n_bits_act, eps, act_decay=act_decay),
            ScaledConv1dAct(channels, output_channels, 1, bias=False, learn_weight_scale=False,
                n_bits=n_bits_weight, gamma=conv_scale_gamma,),
        )

    @torch.no_grad()
    def remove_weight_reparameterizations(self):
        proj: ScaledConv1dAct = self.proj   # type: ignore
        kwargs = dict(device=proj.weight.device, dtype=proj.weight.dtype)
        self.proj = nn.Conv1d(proj.in_channels, proj.out_channels, 1, bias=False, **kwargs)
        self.proj.weight.data.copy_(proj.get_weight())
        self.conv_pre.remove_weight_reparameterizations()

        for layer in self.cnn:
            layer.remove_weight_reparameterizations()

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        warmup: float = 1.0,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, None]:
        """
        Args:
          x: [batch, time, channels]
          x_lens: [batch], containing the number of frames in `x`
            before padding.
          warmup:
            A floating point value that gradually increases from 0 throughout
            training; when it is >= 1.0 we are "fully warmed up".  It is used
            to turn modules on sequentially.

        Returns:
          A tuple of 3 tensors:
            - embeddings: its shape is (N, T', d_model), where T' is the output
              sequence lengths.
            - lengths: a tensor of shape (batch_size,) containing the number of
              frames in `embeddings` before padding.
            - None (for compatibility)
        """
        # lengths = ((x_lens - 3) // 2 - 1) // 2 # issue an warning
        #
        # Note: rounding_mode in torch.div() is available only in torch >= 1.8.0
        sf = self.subsampling_factor
        x = F.pad(x, (0, 0, sf, 0), value=LOG_EPSILON)
        x = (x - self.mean) * self.rstd
        x = x.transpose(1, 2)
        lengths = x_lens

        x, lengths = self.conv_pre(x, lengths)     # [batch_size, channels, time]

        if not torch.jit.is_tracing():
            assert x.size(2) == lengths.max().item()

        for block in self.cnn:
            x, lengths = block(x, lengths, warmup)   # [batch_size, channels, time]
        x = self.proj(x)  # [batch_size, channels_out, time]

        x = x.transpose(1, 2)   # [B, T, C]
        return x, lengths, None


def test_model(check_unused_params: bool = False):
    import re
    device = "cpu"
    model = Encoder(
        80,
        dilations=[1 for _ in range(11)],
        kernel_size=8,
        channels=256,
        channels_expansion=1024,
        output_channels=512,
        se_gate="tanh",
    ).to(device)

    x, lengths, _ = model(
        torch.randn(96, 1656, 80, device=device),
        torch.tensor([16000 for _ in range(96)], dtype=torch.int64, device=device)
    )
    print(x.shape, lengths)
    if check_unused_params:
        (x*0).mean().backward()
        for n, p in model.named_parameters():
            if p.grad is None:
                print(n, p.shape)


if __name__ == "__main__":
    test_model(False)
    # model = CausalSE(8)
    # x = torch.randn(1, 8, 3)
    # print(x)
    # print(model(x))
