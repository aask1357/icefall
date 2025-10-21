import typing as tp
import random

import torch
from torch.nn import functional as F
from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm as weight_norm_fn
from lhotse.utils import LOG_EPSILON
from icefall.iir.iir import EMA

from encoder_interface import EncoderInterface
from scaling import (
    ActivationBalancer,
    BasicNorm,
    Whiten
)
from chip_fp16 import Q, q


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        Q(self)


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, use_cache=False, **kwargs):
        super().__init__(*args, **kwargs)
        d, k, s = self.dilation[0], self.kernel_size[0], self.stride[0]
        self.causal_padding = d * (k - 1) - (s - 1)
        self.cache = torch.empty(0)
        self.use_cache = use_cache
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='linear')
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        Q(self)
    
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
        y = F.conv1d(x, self.weight, self.bias, self.stride, self.padding,
                     self.dilation, self.groups)
        return y


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
        act_bal: bool = False,
        chunksize: int = 16,
        se_gate: str = "sigmoid",
        gamma: float = 0.9,
        use_cache: bool = False,
        weight_norm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % 8 == 0, 'Dimension should be divisible by 8.'
        self.dim = dim
        self.chunksize = chunksize
        self.activation = activation
        self.weight_norm = weight_norm

        self.ema = EMA(dim, r_max=gamma, init_method="uniform", use_cache=use_cache)
        if weight_norm:
            def Conv(*args, **kwargs):
                return weight_norm_fn(Conv1d(*args, **kwargs))
        else:
            Conv = Conv1d
        Act = getattr(nn, activation)
        if activation == "PReLU":
            activation_kwargs = {"num_parameters": 1}
        elif act_bal:
            activation_kwargs = {"inplace": False}
        else:
            activation_kwargs = {"inplace": True}
        ActBal = ActivationBalancer if act_bal else nn.Identity
        self.sequential = nn.Sequential(
            Conv(dim, dim // 8, 1),
            ActBal(dim // 8, channel_dim=1, max_abs=10.0,
                                min_positive=0.05, max_positive=1.0),
            Act(**activation_kwargs),
            Conv(dim // 8, dim, 1),
            ActBal(dim, channel_dim=1, max_abs=10.0,
                                min_positive=0.05, max_positive=1.0),
        )
        if se_gate == "sigmoid":
            self.gate = Q(nn.Sigmoid())
        elif se_gate == "tanh":
            self.gate = Q(nn.Tanh())
        else:
            raise ValueError(f"invalid se_gate '{se_gate}'")
    
    def remove_weight_reparameterizations(self):
        weight = self.sequential[0].weight
        kwargs = dict(device=weight.device, dtype=weight.dtype)
        self.ema.remove_weight_reparameterizations()
        if not self.weight_norm:
            return
        dim = self.sequential[0].in_channels
        seq = self.sequential
        self.sequential = nn.Sequential(
            Q(nn.Conv1d(dim, dim//8, 1, **kwargs)),
            getattr(nn, self.activation)(),
            Q(nn.Conv1d(dim//8, dim, 1, **kwargs))
        )
        self.sequential[0].weight.data.copy_(q(seq[0].get_weight()))
        self.sequential[0].bias.data.copy_(q(seq[0].get_bias()))
        
        idx = 2 if isinstance(seq[-1], nn.Conv1d) else 3
        self.sequential[2].weight.data.copy_(q(seq[idx].get_weight()))
        self.sequential[2].bias.data.copy_(q(seq[idx].get_bias()))

    def forward(self, x: Tensor) -> tp.Tuple[Tensor, Tensor]:
        """
        Args:
            inputs: [batch, dimension, seq_length]
            input_lengths: [batch]

        Returns:
            output: (batch, dimension, seq_length)
        """
        skip = x
        
        # get one vector per chunksize.
        # [B, C, L] -> [B, C, floor(L/cs)] (cs=chunksize)
        seq_lengths = x.size(2)
        chunksize = self.chunksize
        padding = (seq_lengths + chunksize - 1) // chunksize * chunksize - seq_lengths
        x = F.pad(x, (0, padding))      # pad right so that (L+P) % cs = 0
        B, C, L_P = x.shape
        x = x.view(B, C, L_P//chunksize, chunksize).mean(dim=3)  # [B, C, (L+P)//cs]
        x = q(x)
        
        # main network
        x = q(self.ema(x))
        output = self.sequential(x)
        output = self.gate(output)
        
        # expand length from (L+P)//cs to L
        # [B, C, (L+P)//cs] -> [B, C, L+P]
        output = output.repeat_interleave(chunksize, dim=2)
        output = output[:, :, :seq_lengths]

        return q(output.mul_(skip))


class ConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        channels_hidden: int,
        kernel_size: int = 5,
        dilation: int = 1,
        activation: str = 'ReLU',
        norm: str = 'BatchNorm',
        dropout: float = 0.0,
        se_activation: str = 'ReLU',
        act_bal: bool = False,
        whitener: bool = False,
        zero_init_residual: bool = False,
        se_gate: str = "sigmoid",
        gamma: float = 0.9,
        use_cache: bool = False,
        chunksize: int = 16,
        scale_limit: float = 2.0,
        weight_norm: bool = False,
    ) -> None:
        super(ConvBlock, self).__init__()
        self.norm = norm
        self.scale_limit = scale_limit
        
        bias = True
        if norm == "BatchNorm":
            Norm = nn.BatchNorm1d
            bias = False
        elif norm == "SyncBatchNorm":
            Norm = nn.SyncBatchNorm
            bias = False
        elif norm == "BasicNorm":
            Norm = BasicNorm
        else:
            raise RuntimeError(f"invalid norm {norm}")
        
        if activation == "PReLU":
            activation_kwargs = {"num_parameters": 1}
        else:
            activation_kwargs = {"inplace": True}

        if whitener:
            Wht = Whiten
        else:
            Wht = nn.Identity

        self.activation = getattr(nn, activation)(**activation_kwargs)
        self.whiten = Wht(
            num_groups=1, whitening_limit=5.0,
            prob=(0.025, 0.25), grad_scale=0.01)
        self.pointwise1 = Conv1d(channels, channels_hidden, 1, bias=bias)
        self.norm1 = Norm(channels_hidden)
        self.depthwise = CausalConv1d(
            channels_hidden, channels_hidden, kernel_size, groups=channels_hidden,
            dilation=dilation, use_cache=use_cache, bias=bias)
        self.norm2 = Norm(channels_hidden)
        self.pointwise2 = Conv1d(channels_hidden, channels, 1, bias=bias)
        self.norm3 = Norm(channels, affine=False)
        self.se = CausalSE(channels, se_activation, act_bal, se_gate=se_gate,
                           gamma=gamma, use_cache=use_cache, chunksize=chunksize,
                           weight_norm=weight_norm,)
        self.dropout = nn.Dropout(dropout, inplace=True)
        
        if zero_init_residual:
            self.scale = nn.Parameter(torch.zeros(channels, 1))
        else:
            self.scale = nn.Parameter(torch.ones(channels, 1))
        self.initial_cache = None

    def remove_weight_reparameterizations(self):
        kwargs = dict(device=self.pointwise1.weight.device, dtype=self.pointwise1.weight.dtype)
        bias = False if self.pointwise1.bias is None else True
        self.se.remove_weight_reparameterizations()
        
        if self.norm in ["BatchNorm", "SyncBatchNorm"]:
            # y = ((x * weight - mean) / std * gamma + beta)
            # <=> y = x * (gamma/std*weight) + (-mean*gamma/std + beta)
            # mean, std, gamma, beta: [Ci], weight: [Co, Ci, K]
            channels = self.pointwise1.in_channels
            channels_hidden = self.pointwise1.out_channels
            kernel_size = self.depthwise.kernel_size
            
            mean = self.norm1.running_mean
            std = self.norm1.running_var.data.add(self.norm1.eps).sqrt()
            weight = (self.norm1.weight/std) * self.pointwise1.weight.data
            bias = -(mean*self.norm1.weight/std) + self.norm1.bias
            self.pointwise1 = Conv1d(channels, channels_hidden, 1, bias=True, **kwargs)
            self.pointwise1.weight.data.copy_(q(weight.view(-1, 1, 1)*conv_weight))
            self.pointwise1.bias.data.copy_(q(bias))
            self.norm1 = nn.Identity()
            self.initial_cache = torch.zeros(1, channels_hidden, kernel_size-1, **kwargs)
            
            mean = self.norm2.running_mean
            std = self.norm2.running_var.data.add(self.norm2.eps).sqrt()
            weight = (self.norm2.weight/std) * self.depthwise.weight.data
            bias = -(mean*self.norm2.weight/std) + self.norm1.bias
            self.depthwise = Q(Conv1d(
                channels_hidden, channels_hidden, kernel_size, 
                groups=channels_hidden, bias=True, **kwargs))
            self.depthwise.weight.data.copy_(q(weight.view(-1, 1, 1)*conv_weight))
            self.depthwise.bias.data.copy_(q(bias))
            self.norm2 = nn.Identity()

    def forward(
        self,
        inputs: Tensor,
        input_lengths: Tensor,
    ) -> tp.Tuple[Tensor, Tensor]:
        x = inputs
        x_len = input_lengths

        x = self.pointwise1(x)
        x = self.norm1(x)
        x = self.activation(x)
        if self.initial_cache is not None:
            x = torch.cat([self.initial_cache, x], dim=2)
        x = self.depthwise(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.pointwise2(x)
        x = self.norm3(x)
        x = self.se(x)
        x = self.dropout(x)
        
        self.scale.data.clamp_(min=-self.scale_limit, max=self.scale_limit)
        x = torch.addcmul(inputs, x, self.scale)    # x = inputs + x * self.scale

        return self.whiten(q(x)), x_len


class Conv1dSubsampling(nn.Module):
    def __init__(self, in_ch, out_ch, subsampling_factor,
                 whitener, weight_norm):
        super().__init__()
        if whitener:
            Wht = Whiten
        else:
            Wht = nn.Identity
        self.subsampling_factor = subsampling_factor
        sf = subsampling_factor
        def norm(module):
            if weight_norm:
                return weight_norm_fn(module)
            return module
        
        self.pointwise = norm(Conv1d(in_ch, out_ch, 1, bias=False))
        self.depthwise = norm(Conv1d(out_ch, out_ch, 2*sf, stride=sf, padding=0,
                              groups=out_ch, bias=False))
        self.whiten = Wht(num_groups=1, whitening_limit=2.0, prob=(0.025, 0.25), grad_scale=0.025)
    
    def remove_weight_reparameterizations(self):
        conv = self.pointwise
        kwargs = dict(device=conv.weight.device, dtype=conv.weight.dtype)
        self.pointwise = Q(nn.Conv1d(
            conv.in_channels, conv.out_channels, 1, bias=False, **kwargs))
        self.pointwise.weight.data.copy_(conv.get_weight())
        
        sf = self.subsampling_factor
        conv = self.depthwise
        self.depthwise = Q(nn.Conv1d(
            conv.in_channels, conv.out_channels, 2*sf, stride=sf, padding=0,
            groups=conv.groups, **kwargs))
        self.depthwise.weight.data.copy_(q(conv.get_weight()))
        self.depthwise.bias.data.copy_(q(conv.get_bias()))
        
        self.act_bal = nn.Identity()
        self.whiten = nn.Identity()
    
    def forward(self, x, x_len):
        x = self.pointwise(x)
        x = F.pad(x, (self.subsampling_factor, 0))
        x = self.depthwise(x)
        x = self.whiten(x)
        x_len = torch.floor(x_len / self.subsampling_factor)
        return x, x_len


class Encoder(EncoderInterface):
    def __init__(
        self,
        num_features: int,
        subsampling_factor: int = 4,
        channels: int = 384,
        channels_expansion: int = 512,
        kernel_size: int = 4,
        dilations: tp.List[int] = [1, 2, 4, 1, 2, 4],
        output_channels: int = 640,
        dropout: float = 0.075,
        activation: str = 'ReLU',
        norm: str = 'BatchNorm',
        se_activation: str = 'ReLU',
        is_pnnx: bool = False,
        act_bal: bool = False,
        whitener: bool = False,
        zero_init_residual: bool = False,
        se_gate: str = "sigmoid",
        gamma: float = 0.9,
        use_cache: bool = False,
        mean: float = -6.375938187300722,
        std: float = 4.354657227491409,
        chunksize: int = 16,
        scale_limit: float = 1.0,
        weight_norm: bool = False,
        scaled_conv: bool = True, # not used.
    ) -> None:
        super().__init__()

        self.mean = mean
        self.rstd = 1 / std
        self.norm = norm
        self.num_features = num_features
        self.subsampling_factor = subsampling_factor
        
        self.conv_pre = Conv1dSubsampling(
            num_features,
            channels,
            subsampling_factor=subsampling_factor,
            whitener=whitener,
            weight_norm=weight_norm,
        )
        self.is_pnnx = is_pnnx

        self.cnn = nn.ModuleList()
        for dilation in dilations:
            layer = ConvBlock(
                channels, channels_expansion, kernel_size, dilation,
                activation, norm, dropout, se_activation,
                act_bal, whitener, zero_init_residual, se_gate,
                gamma=gamma, use_cache=use_cache, chunksize=chunksize,
                scale_limit=scale_limit, weight_norm=weight_norm,
            )
            self.cnn.append(layer)

        self.proj = Conv1d(channels, output_channels, 1, bias=False)
        if weight_norm:
            self.proj = weight_norm_fn(self.proj)
    
    @torch.no_grad()
    def remove_weight_reparameterizations(self):
        self.conv_pre.remove_weight_reparameterizations()
        # ToDo: merge self.proj and self.norm into a single nn.Conv1d
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
        x = (x - self.mean) * self.rstd
        x = q(x).transpose(1, 2)
        lengths = x_lens

        x, lengths = self.conv_pre(x, lengths)     # [batch_size, channels, time]

        if not torch.jit.is_tracing():
            assert x.size(2) == lengths.max().item()

        for block in self.cnn:
            x, lengths = block(x, lengths)   # [batch_size, channels, time]
        x = self.proj(x)        # [batch_size, channels_out, time]
        x = x.transpose(1, 2)   # [B, T, C]
        return x, lengths, None


def test_model(check_unused_params: bool = False):
    import re
    device = "cpu"
    model = Encoder(
        80,
        dilations=[1 for _ in range(22)],
        kernel_size=8,
        channels=384,
        channels_expansion=1536,
        output_channels=512,
        norm='BatchNorm',
        num_rnn_layers=0,
        zero_init_residual=True,
        se_gate="tanh",
    ).to(device)
    conv2d_params, conv1d_params, lstm_params = 0, 0, 0
    conv1d_wo_se, conv1d_se = 0, 0
    for p in model.conv_pre.parameters():
        conv2d_params += p.numel()
    for n, p in model.cnn.named_parameters():
        conv1d_params += p.numel()
        if re.search(r"\.se\.", n):
            conv1d_se += p.numel()
        else:
            conv1d_wo_se += p.numel()
    for n, p in model.proj.named_parameters():
        conv1d_params += p.numel()
        conv1d_wo_se += p.numel()
    if model.lstm is not None:
        for p in model.lstm.parameters():
            lstm_params += p.numel()
    total_params = conv2d_params + conv1d_params + lstm_params
    x, lengths, _ = model(
        torch.randn(2, 500, 80, device=device),
        torch.tensor([100, 500], dtype=torch.int64, device=device)
    )
    print(x.shape, lengths)
    print(
        f"conv2d: {conv2d_params/1000_000:.2f}M\n"
        f"conv1d: {conv1d_params/1000_000:.2f}M\n"
        f"    s&e: {conv1d_se/1000_000:.2f}M\n"
        f"    else: {conv1d_wo_se/1000_000:.2f}M\n"
        f"lstm: {lstm_params/1000_000:.2f}M\n"
        f"total: {total_params/1000_000:.2f}M"
    )
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
