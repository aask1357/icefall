import typing as tp

import torch
from torch.nn import functional as F
from torch import Tensor, nn
from lhotse.utils import LOG_EPSILON
from icefall.iir.iir import EMA

from encoder_interface import EncoderInterface
from scaling import (
    ActivationBalancer,
    BasicNorm,
    ScaledConv1d,
    ScaledSyncBatchNorm,
    ScaledBatchNorm1d,
)
from quantized_layers import QuantizedConv1d
from chip_fp16 import Q, q


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, use_cache=False, **kwargs):
        super().__init__(*args, **kwargs)
        d, k, s = self.dilation[0], self.kernel_size[0], self.stride[0]
        self.causal_padding = d * (k - 1) - (s - 1)
        self.cache = torch.empty(0)
        self.use_cache = use_cache
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


class CausalScaledConv1d(ScaledConv1d):
    def __init__(self, *args, use_cache=False, **kwargs):
        super().__init__(*args, **kwargs, padding_mode='zeros')
        d, k, s = self.dilation[0], self.kernel_size[0], self.stride[0]
        self.causal_padding = d * (k - 1) - (s - 1)
        assert self.padding[0] == 0, self.padding
        self.cache = torch.empty(0)
        self.use_cache = use_cache
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
        act_bal: bool = False,
        chunksize: int = 16,
        se_gate: str = "sigmoid",
        gamma: float = 0.9,
        use_cache: bool = False,
        n_bits_act: tp.Optional[int] = None,
        n_bits_weight: int = 4,
        eps: float = 1.0e-5,
        weight_quantizer_mode: str = "max",
    ) -> None:
        super().__init__()
        assert dim % 8 == 0, 'Dimension should be divisible by 8.'
        self.dim = dim
        self.chunksize = chunksize
        self.activation = activation

        self.ema = EMA(dim, r_max=gamma, init_method="uniform", use_cache=use_cache)
        def Conv(*args, **kwargs) -> nn.Module:
            return QuantizedConv1d(
                *args,
                n_bits_act=n_bits_act,
                n_bits_weight=n_bits_weight,
                eps=eps,
                weight_quantizer_mode=weight_quantizer_mode,
                **kwargs
            )
        ActBal = ActivationBalancer if act_bal else nn.Identity

        Act = getattr(nn, activation)
        self.sequential = nn.Sequential(
            Conv(dim, dim // 8, 1, inplace=False),
            ActBal(1),
            Act(inplace=not act_bal),
            Conv(dim // 8, dim, 1, inplace=act_bal),
            ActBal(1),
        )
        if se_gate == "sigmoid":
            self.gate = Q(nn.Sigmoid())
        elif se_gate == "tanh":
            self.gate = Q(nn.Tanh())
        else:
            raise ValueError(f"invalid se_gate '{se_gate}'")
    
    def remove_weight_reparameterizations(self, ema: bool = True):
        weight = self.sequential[0].weight
        kwargs = dict(device=weight.device, dtype=weight.dtype)
        if ema:
            self.ema.remove_weight_reparameterizations()

        dim = self.sequential[0].in_channels
        seq = self.sequential
        self.sequential = nn.Sequential(
            Q(nn.Conv1d(dim, dim//8, 1, **kwargs)),
            getattr(nn, self.activation)(),
            Q(nn.Conv1d(dim//8, dim, 1, **kwargs))
        )
        self.sequential[0].weight.data.copy_(seq[0].get_weight())
        self.sequential[0].bias.data.copy_(seq[0].get_bias())
        
        idx = 2 if isinstance(seq[-1], nn.Conv1d) else 3
        self.sequential[2].weight.data.copy_(seq[idx].get_weight())
        self.sequential[2].bias.data.copy_(seq[idx].get_bias())

    def forward(self, x: Tensor) -> tp.Tuple[Tensor, Tensor]:
        """
        Args:
            inputs: [batch, dimension, seq_length]
            input_lengths: [batch]

        Returns:
            output: (batch, dimension, seq_length)
        """
        residual = x
        
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

        return q(output.mul_(residual))


class ConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        channels_hidden: int,
        kernel_size: int = 5,
        dilation: int = 1,
        activation: str = 'ReLU',
        activation_kwargs: dict = {'inplace': True},
        norm: str = 'BatchNorm',
        dropout: float = 0.0,
        se_activation: str = 'ReLU',
        act_bal: bool = False,
        zero_init_residual: bool = False,
        se_gate: str = "sigmoid",
        gamma: float = 0.9,
        use_cache: bool = False,
        chunksize: int = 16,
        scale_limit: float = 2.0,
        skip: str = "residual",
        n_bits_act: tp.Optional[int] = None,
        n_bits_weight: int = 8,
        eps: float = 1.0e-5,
        weight_quantizer_mode: str = "max",
    ) -> None:
        super().__init__()

        self.scale_limit = scale_limit
        def Conv(*args, **kwargs) -> nn.Module:
            return QuantizedConv1d(
                *args,
                n_bits_act=n_bits_act,
                n_bits_weight=n_bits_weight,
                eps=eps,
                weight_quantizer_mode=weight_quantizer_mode,
                **kwargs
            )
        def CausalConv(*args, use_cache: bool = False, **kwargs) -> nn.Module:
            return QuantizedConv1d(
                *args,
                causal=True,
                n_bits_act=n_bits_act,
                n_bits_weight=n_bits_weight,
                eps=eps,
                inplace=True,
                weight_quantizer_mode=weight_quantizer_mode,
                **kwargs
            )

        self.norm = norm
        if norm == "BatchNorm":
            Norm = nn.BatchNorm1d
        elif norm == "SyncBatchNorm":
            Norm = nn.SyncBatchNorm
        elif norm == "BasicNorm":
            Norm = BasicNorm
        else:
            raise RuntimeError(f"invalid norm {norm}")

        if act_bal:
            ActBal = ActivationBalancer
            activation_kwargs = {'inplace': False}
        else:
            ActBal = nn.Identity

        self.activation = getattr(nn, activation)(**activation_kwargs)
        self.act_bal = ActBal(1)
        self.pointwise1 = Conv(channels, channels_hidden, 1, inplace=False)
        self.norm1 = Norm(channels_hidden, affine=False)
        self.depthwise = CausalConv(
            channels_hidden, channels_hidden, kernel_size, groups=channels_hidden,
            dilation=dilation, use_cache=use_cache)
        self.norm2 = Norm(channels_hidden, affine=False)
        self.pointwise2 = Conv(channels_hidden, channels, 1, bias=False, inplace=True)
        self.se = CausalSE(
            channels, se_activation, act_bal, se_gate=se_gate,
            gamma=gamma, use_cache=use_cache, chunksize=chunksize,
            n_bits_act=n_bits_act,
            n_bits_weight=n_bits_weight,
            eps=eps,
            weight_quantizer_mode=weight_quantizer_mode,
        )
        self.dropout = nn.Dropout(dropout, inplace=True)

        skip = skip.split("-")
        self.skip = skip[0]
        if len(skip) > 1:
            zero_init_residual = skip[1]
        else:
            zero_init_residual = False
        self.scale = nn.Parameter(torch.ones(1))
        if zero_init_residual == "zeroinit":
            self.scale.data.zero_()
        elif self.skip == "bypass":
            self.scale.data.fill_(0.5)
        self.register_buffer("initial_cache", None)

    def remove_weight_reparameterizations(self, fuse_bn: bool = True, ema: bool = True):
        kwargs = dict(device=self.pointwise1.weight.device, dtype=self.pointwise1.weight.dtype)
        bias = False if self.pointwise1.bias is None else True
        conv = self.pointwise1
        self.pointwise1 = Q(nn.Conv1d(conv.in_channels, conv.out_channels, 1,
                                    bias=bias, **kwargs))
        self.pointwise1.weight.data.copy_(conv.get_weight())
        if bias:
            self.pointwise1.bias.data.copy_(conv.get_bias())
        conv = self.depthwise
        self.depthwise = nn.Conv1d(
            conv.in_channels, conv.out_channels, conv.kernel_size,
            groups=conv.groups, bias=bias, **kwargs)
        self.depthwise.weight.data.copy_(conv.get_weight())
        if bias:
            self.depthwise.bias.data.copy_(conv.get_bias())
        conv = self.pointwise2
        self.pointwise2 = Q(nn.Conv1d(conv.in_channels, conv.out_channels, 1, **kwargs))
        self.pointwise2.weight.data.copy_(conv.get_weight())
        self.pointwise2.bias.data.copy_(conv.get_bias())
        self.se.remove_weight_reparameterizations(ema=ema)
        if self.initial_cache is None:
            delattr(self, "initial_cache")
        self.register_buffer(
            "initial_cache",
            torch.zeros(
                1,
                self.depthwise.in_channels,
                self.depthwise.kernel_size[0]-1,
                **kwargs
            )   # [B=1, C, K-1]
        )

        if not fuse_bn:
            return

        if self.norm in ["BatchNorm", "SyncBatchNorm"]:
            # y = ((x - mean) / std * gamma + beta) * weight
            # <=> y = x * (gamma/std*weight) + (-mean*gamma/std*weight + beta*weight)
            # mean, std, gamma, beta: [Ci], weight: [Co, Ci, K]
            mean = self.norm1.running_mean
            std = self.norm1.running_var.data.add(self.norm1.eps).sqrt()
            conv_weight = self.depthwise.weight
            conv_bias = self.depthwise.bias
            weight = 1 / std
            bias = -mean / std
            initial_cache = torch.stack(
                [-bias.view(1, -1) for _ in range(self.depthwise.kernel_size[0]-1)],
                dim=2,
            )   # [B=1, C, K-1]
            tmp = torch.cat([bias.view(1, -1, 1), -initial_cache], dim=2)  # [B=1, C, K]
            bias = F.conv1d(tmp, conv_weight, groups=self.depthwise.groups)
            
            channels = conv_weight.size(0)
            self.depthwise = Q(nn.Conv1d(
                channels, channels, conv_weight.size(2), groups=channels, bias=True,
                **kwargs))
            self.depthwise.weight.data.copy_(weight.view(-1, 1, 1)*conv_weight)
            bias = bias.squeeze(2).squeeze(0)
            if conv_bias is not None:
                bias = bias + conv_bias
            self.depthwise.bias.data.copy_(bias)
            weight = torch.where(
                weight.abs() > 1e-12,
                weight,
                torch.tensor([1e-12], device=weight.device, dtype=weight.dtype)
            )
            initial_cache /= weight.view(1, -1, 1)
            initial_cache.clamp_(
                min=torch.finfo(torch.float16).min,
                max=torch.finfo(torch.float16).max
            )
            if self.initial_cache is None:
                delattr(self, "initial_cache")
            self.register_buffer("initial_cache", initial_cache)
            self.norm1 = nn.Identity()
            
            mean = self.norm2.running_mean
            std = self.norm2.running_var.data.add(self.norm2.eps).sqrt()
            conv_weight = self.pointwise2.weight
            conv_bias = self.pointwise2.bias
            weight = 1 / std
            bias = -mean / std
            bias = F.conv1d(bias.view(1, -1, 1), conv_weight)
            
            self.pointwise2 = Q(nn.Conv1d(
                conv_weight.size(1), conv_weight.size(0), 1, bias=True, **kwargs))
            self.pointwise2.weight.data.copy_(weight.view(1, -1, 1)*conv_weight)
            self.pointwise2.bias.data.copy_(bias.squeeze(2).squeeze(0) + conv_bias)
            self.norm2 = nn.Identity()

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

        x = self.pointwise1(x)
        x = self.act_bal(x)
        x = self.activation(x)
        x = self.norm1(x)
        if self.initial_cache is not None:
            cache = self.initial_cache.repeat(x.size(0), 1, 1)
            x = torch.cat([cache, x], dim=2)
        x = self.depthwise(x)
        x = self.act_bal(x)
        x = self.activation(x)
        x = self.norm2(x)
        x = self.pointwise2(x)
        x = self.act_bal(x)
        x = self.se(x)
        x = self.dropout(x)

        if self.skip == "residual":
            # x = inputs + x * self.scale
            self.scale.data.clamp_(min=-self.scale_limit, max=self.scale_limit)
            x = torch.addcmul(inputs, x, self.scale)
        elif self.skip == "bypass":
            # x = inputs * (1- self.scale) + x * self.scale
            #   = inputs + self.scale * (x - inputs)
            self.scale.data.clamp_(min=0.0, max=1.0)
            x = torch.addcmul(inputs, x - inputs, self.scale)

        return q(x), x_len


class Conv1dSubsampling(nn.Module):
    def __init__(
        self, in_ch, out_ch, subsampling_factor,
        n_bits_act: tp.Optional[int] = None,
        n_bits_weight: int = 8,
        eps: float = 1.0e-5,
        weight_quantizer_mode: str = "max",
    ):
        super().__init__()
        def Conv(*args, **kwargs) -> nn.Module:
            return QuantizedConv1d(
                *args,
                n_bits_act=n_bits_act,
                n_bits_weight=n_bits_weight,
                eps=eps,
                inplace=True,
                weight_quantizer_mode=weight_quantizer_mode,
                **kwargs
            )

        self.subsampling_factor = subsampling_factor
        sf = subsampling_factor

        self.conv = nn.Sequential(
            Conv(in_ch, out_ch, 1, bias=False),
            Conv(out_ch, out_ch, 2*sf, stride=sf, padding=0, groups=out_ch, bias=False),
        )

    def remove_weight_reparameterizations(self):
        sf = self.subsampling_factor
        conv = self.conv
        kwargs = dict(device=conv[0].weight.device, dtype=conv[0].weight.dtype)
        self.conv = nn.Sequential(
            Q(nn.Conv1d(conv[0].in_channels, conv[0].out_channels, 1, bias=False, **kwargs)),
            Q(nn.Conv1d(conv[1].in_channels, conv[1].out_channels, 2*sf, stride=sf, padding=0,
                      groups=conv[1].groups, **kwargs)),
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
        num_features: int = 80,
        subsampling_factor: int = 4,
        channels: int = 256,
        channels_expansion: int = 1024,
        kernel_size: int = 8,
        dilations: tp.List[int] = [1 for _ in range(11)],
        dropout: float = 0.075,
        activation: str = 'ReLU',
        activation_kwargs: dict = {'inplace': True},
        norm: str = 'BatchNorm',
        se_activation: str = 'ReLU',
        act_bal: bool = False,
        zero_init_residual: bool = True,
        se_gate: str = "tanh",
        gamma: float = 0.97,
        chunksize: int = 8,
        use_cache: bool = False,
        mean: float = -6.375938187300722,
        std: float = 4.354657227491409,
        scale_limit: float = 2.0,
        skip: str = "residual",
        n_bits_act: tp.Optional[int] = None,
        n_bits_weight: int = 8,
        eps: float = 1.0e-5,
        weight_quantizer_mode: str = "max",
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
            n_bits_act=n_bits_act,
            n_bits_weight=n_bits_weight,
            eps=eps,
            weight_quantizer_mode=weight_quantizer_mode,
        )

        self.cnn = nn.ModuleList()
        for dilation in dilations:
            layer = ConvBlock(
                channels, channels_expansion, kernel_size, dilation,
                activation, activation_kwargs, norm, dropout, se_activation,
                act_bal, zero_init_residual, se_gate, gamma=gamma,
                use_cache=use_cache, chunksize=chunksize, scale_limit=scale_limit,
                skip=skip, 
                n_bits_act=n_bits_act,
                n_bits_weight=n_bits_weight,
                eps=eps,
                weight_quantizer_mode=weight_quantizer_mode,
            )
            self.cnn.append(layer)
    
    @torch.no_grad()
    def remove_weight_reparameterizations(self, fuse_bn: bool = True, ema: bool = True):
        self.conv_pre.remove_weight_reparameterizations()
        for layer in self.cnn:
            layer.remove_weight_reparameterizations(fuse_bn=fuse_bn, ema=ema)

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
        x = q(x).transpose(1, 2)
        lengths = x_lens

        x, lengths = self.conv_pre(x, lengths)     # [batch_size, channels, time]

        if not torch.jit.is_tracing():
            assert x.size(2) == lengths.max().item()

        x_in = x
        for block in self.cnn:
            x, lengths = block(x, lengths, warmup)   # [batch_size, channels, time]

        x = x.transpose(1, 2)   # [B, T, C]
        return x, lengths, None


def test_model(check_unused_params: bool = True):
    device = "cpu"
    model = Encoder(
        n_bits_act=8,
        n_bits_weight=4,
    ).to(device)

    x, lengths, _ = model(
        torch.randn(2, 500, 80, device=device),
        torch.tensor([100, 500], dtype=torch.int64, device=device)
    )
    print(x.shape, lengths)
    (x*0).mean().backward()

    if check_unused_params:
        for n, p in model.named_parameters():
            if p.grad is None:
                print(n, p.shape)
        print("Total parameters:", sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    test_model(True)
