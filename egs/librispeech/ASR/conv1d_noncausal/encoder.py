import typing as tp
import random

import torch
from torch.nn import functional as F
from torch import Tensor, nn
from lhotse.utils import LOG_EPSILON


from encoder_interface import EncoderInterface
from scaling import (
    ActivationBalancer,
    BasicNorm,
    ScaledConv1d,
)
from chip_fp16 import Q, q


class SE(nn.Module):
    r"""
    Squeeze-and-excitation module using Exponential Moving Average Layer.

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
        scaled_conv: bool = False,
        act_bal: bool = False,
        se_gate: str = "sigmoid",
    ) -> None:
        super().__init__()
        assert dim % 8 == 0, 'Dimension should be divisible by 8.'
        self.dim = dim
        self.scaled_conv = scaled_conv
        self.activation = activation

        Conv = ScaledConv1d if scaled_conv else nn.Conv1d
        Act = getattr(nn, activation)
        if act_bal:
            self.sequential = nn.Sequential(
                Conv(dim, dim // 8, 1),
                ActivationBalancer(1),
                Act(inplace=False),
                Conv(dim // 8, dim, 1),
                ActivationBalancer(1),
            )
        else:
            self.sequential = nn.Sequential(
                Conv(dim, dim // 8, 1),
                Act(inplace=True),
                Conv(dim // 8, dim, 1),
            )
        if se_gate == "sigmoid":
            self.gate = Q(nn.Sigmoid())
        elif se_gate == "tanh":
            self.gate = Q(nn.Tanh())
        else:
            raise ValueError(f"invalid se_gate '{se_gate}'")
    
    def remove_weight_reparameterizations(self):
        if not self.scaled_conv:
            return
        weight = self.sequential[0].weight
        kwargs = dict(device=weight.device, dtype=weight.dtype)
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
        self.scaled_conv = False

    def forward(self, x: Tensor) -> tp.Tuple[Tensor, Tensor]:
        """
        Args:
            inputs: [batch, dimension, seq_length]
            input_lengths: [batch]

        Returns:
            output: (batch, dimension, seq_length)
        """
        residual = x
        
        x = q(x.mean(dim=2, keepdim=True))  # [B, C, L] -> [B, C, 1]
        output = self.sequential(x)
        output = self.gate(output)

        return q(residual * output)


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
        scaled_conv: bool = False,
        act_bal: bool = False,
        zero_init_residual: bool = False,
        se_gate: str = "sigmoid",
    ) -> None:
        super(ConvBlock, self).__init__()
        self.scaled_conv = scaled_conv
        self.norm = norm
        
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

        if scaled_conv:
            Conv = ScaledConv1d
        else:
            Conv = nn.Conv1d
        
        if act_bal:
            ActBal = ActivationBalancer
            activation_kwargs = {'inplace': False}
        else:
            ActBal = nn.Identity
        
        self.activation = getattr(nn, activation)(**activation_kwargs)
        self.act_bal = ActBal(1)
        self.pointwise1 = Conv(channels, channels_hidden, 1)
        self.norm1 = Norm(channels_hidden)
        self.depthwise = Conv(
            channels_hidden, channels_hidden, kernel_size, padding=(kernel_size-1)//2,
            groups=channels_hidden, dilation=dilation)
        self.norm2 = Norm(channels_hidden)
        self.pointwise2 = Conv(channels_hidden, channels, 1)
        self.se = SE(channels, se_activation, scaled_conv, act_bal, se_gate=se_gate)
        self.dropout = nn.Dropout(dropout, inplace=True)
        
        self.scale = None
        if zero_init_residual:
            self.scale = nn.Parameter(torch.zeros(1))
        self.initial_cache = None

    def remove_weight_reparameterizations(self):
        kwargs = dict(device=self.pointwise1.weight.device, dtype=self.pointwise1.weight.dtype)
        bias = False if self.pointwise1.bias is None else True
        if self.scaled_conv:
            conv = self.pointwise1
            self.pointwise1 = Q(nn.Conv1d(conv.in_channels, conv.out_channels, 1,
                                        bias=bias, **kwargs))
            self.pointwise1.weight.data.copy_(conv.get_weight())
            if bias:
                self.pointwise1.bias.data.copy_(conv.get_bias())
            conv = self.depthwise
            self.depthwise = Conv1d(
                conv.in_channels, conv.out_channels, conv.kernel_size,
                groups=conv.groups, bias=bias, **kwargs)
            self.depthwise.weight.data.copy_(conv.get_weight())
            if bias:
                self.depthwise.bias.data.copy_(conv.get_bias())
            conv = self.pointwise2
            self.pointwise2 = Q(nn.Conv1d(conv.in_channels, conv.out_channels, 1, **kwargs))
            self.pointwise2.weight.data.copy_(conv.get_weight())
            self.pointwise2.bias.data.copy_(conv.get_bias())
            self.se.remove_weight_reparameterizations()
        
        if self.norm in ["BatchNorm", "SyncBatchNorm"]:
            # y = ((x - mean) / std * gamma + beta) * weight
            # <=> y = x * (gamma/std*weight) + (-mean*gamma/std*weight + beta*weight)
            # mean, std, gamma, beta: [Ci], weight: [Co, Ci, K]
            mean = self.norm1.running_mean
            std = self.norm1.running_var.data.add(self.norm1.eps).sqrt()
            conv_weight = self.depthwise.weight
            conv_bias = self.depthwise.bias
            weight = (self.norm1.weight/std)
            bias = -(mean*self.norm1.weight/std) + self.norm1.bias
            self.initial_cache = torch.stack(
                [-bias.view(1, -1) for _ in range(self.depthwise.kernel_size[0]-1)],
                dim=2
            )   # [B=1, C, K-1]
            tmp = torch.cat([bias.view(1, -1, 1), -self.initial_cache], dim=2)  # [B=1, C, K]
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
            self.initial_cache /= weight.view(1, -1, 1)
            self.norm1 = nn.Identity()
            
            mean = self.norm2.running_mean
            std = self.norm2.running_var.data.add(self.norm2.eps).sqrt()
            conv_weight = self.pointwise2.weight
            conv_bias = self.pointwise2.bias
            weight = (self.norm2.weight/std)
            bias = -(mean*self.norm2.weight/std) + self.norm2.bias
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
        x = self.depthwise(x)
        x = self.act_bal(x)
        x = self.activation(x)
        x = self.norm2(x)
        x = self.pointwise2(x)
        x = self.act_bal(x)
        x = self.se(x)
        x = self.dropout(x)
        
        if self.scale is not None:
            x = torch.addcmul(inputs, x, self.scale)    # x = inputs + x * self.scale
        elif warmup < 1:
            x = torch.add(inputs, x, alpha=warmup)      # x = inputs + x * warmup
        else:
            x = x + inputs

        return q(x), x_len


class Conv1dSubsamplingLinear(nn.Module):
    def __init__(self, in_ch, out_ch, act_bal, scaled_conv):
        super().__init__()
        self.scaled_conv = scaled_conv
        if scaled_conv:
            Conv = ScaledConv1d
        else:
            Conv = nn.Conv1d
        
        if act_bal:
            ActBal = ActivationBalancer
        else:
            ActBal = nn.Identity
        
        self.conv = nn.Sequential(
            Conv(in_ch, out_ch, 1, bias=False),
            Conv(out_ch, out_ch, 8, stride=4, padding=0, groups=out_ch),
            ActBal(1)
        )
    
    def remove_weight_reparameterizations(self):
        if not self.scaled_conv:
            return
        conv = self.conv
        kwargs = dict(device=conv[0].weight.device, dtype=conv[0].weight.dtype)
        self.conv = nn.Sequential(
            Q(nn.Conv1d(conv[0].in_channels, conv[0].out_channels, 1, bias=False, **kwargs)),
            Q(nn.Conv1d(conv[1].in_channels, conv[1].out_channels, 8, stride=4, padding=0,
                      groups=conv[1].groups, **kwargs)),
        )
        self.conv[0].weight.data.copy_(conv[0].get_weight())
        self.conv[1].weight.data.copy_(conv[1].get_weight())
        self.conv[1].bias.data.copy_(conv[1].get_bias())
    
    def forward(self, x, x_len):
        x = F.pad(x, (2, 2), value=LOG_EPSILON)
        x = self.conv(x)
        x_len = torch.floor(x_len / 4)
        return x, x_len


class Encoder(EncoderInterface):
    def __init__(
        self,
        num_features: int,
        subsampling_factor: int = 4,
        channels: int = 384,
        channels_expansion: int = 512,
        kernel_size: int = 9,
        dilations: tp.List[int] = [1, 2, 4, 1, 2, 4],
        output_channels: int = 640,
        dropout: float = 0.075,
        activation: str = 'ReLU',
        activation_kwargs: dict = {'inplace': True},
        norm: str = 'BatchNorm',
        se_activation: str = 'ReLU',
        is_pnnx: bool = False,
        scaled_conv: bool = False,
        act_bal: bool = False,
        conv1d_subsampling_version: int = 1,
        zero_init_residual: bool = False,
        se_gate: str = "sigmoid",
        mean: float = -6.375938187300722,
        std: float = 4.354657227491409,
    ) -> None:
        assert kernel_size % 2 == 1
        super().__init__()

        self.mean = mean
        self.rstd = 1 / std
        self.norm = norm
        self.num_features = num_features
        self.subsampling_factor = subsampling_factor
        self.scaled_conv = scaled_conv
        if subsampling_factor != 4:
            raise NotImplementedError("Support only 'subsampling_factor=4'.")
        
        Conv = ScaledConv1d if scaled_conv else nn.Conv1d
        
        self.conv_pre = Conv1dSubsamplingLinear(
            num_features,
            channels,
            scaled_conv=scaled_conv,
            act_bal=act_bal
        )
        self.is_pnnx = is_pnnx

        self.cnn = nn.ModuleList()
        for dilation in dilations:
            layer = ConvBlock(
                channels, channels_expansion, kernel_size, dilation,
                activation, activation_kwargs, norm, dropout, se_activation,
                scaled_conv, act_bal, zero_init_residual, se_gate,
            )
            self.cnn.append(layer)
        self.proj = Conv(channels * 2, output_channels, 1, bias=False)
    
    @torch.no_grad()
    def remove_weight_reparameterizations(self):
        if self.scaled_conv:
            proj: ScaledConv1d = self.proj   # type: ignore
            kwargs = dict(device=proj.weight.device, dtype=proj.weight.dtype)
            self.proj = Q(nn.Conv1d(proj.in_channels, proj.out_channels, 1, bias=False, **kwargs))
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
        x = (x - self.mean) * self.rstd
        x = q(x).transpose(1, 2)
        lengths = x_lens

        x, lengths = self.conv_pre(x, lengths)     # [batch_size, channels, time]

        if not torch.jit.is_tracing():
            assert x.size(2) == lengths.max().item()

        x_in = x
        for block in self.cnn:
            x, lengths = block(x, lengths, warmup)   # [batch_size, channels, time]
        x = self.proj(torch.cat((x, x_in), dim=1))  # [batch_size, channels_out, time]

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
        scaled_conv=True,
        conv1d_subsampling_version=2,
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
