import typing as tp
import random

import torch
from torch.nn import functional as F
from torch import Tensor, nn

from encoder_interface import EncoderInterface
from scaling import (
    ActivationBalancer,
    BasicNorm,
    DoubleSwish,
    ScaledConv2d,
    ScaledConv1d,
)
from lstm import RNNEncoder, RNNEncoderLayer


class IdentityLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def forward(self, x: Tensor, x_len: Tensor) -> tp.Tuple[Tensor, None]:
        return x, None


class WrappedSyncBatchNorm(nn.SyncBatchNorm):
    def forward(self, x: Tensor, x_length: Tensor) -> Tensor:
        return super().forward(x)


class WrappedBatchNorm1d(nn.BatchNorm1d):
    def forward(self, x: Tensor, x_length: Tensor) -> Tensor:
        return super().forward(x)


class WrappedBatchNorm2d(nn.BatchNorm2d):
    def forward(self, x: Tensor, x_length: Tensor) -> Tensor:
        return super().forward(x)


class WrappedBasicNorm(BasicNorm):
    def __init__(
        self,
        num_channels: int,
        channel_dim: int = 1,
        eps: float = 0.25,
        learn_eps: bool = True,
    ):
        super().__init__(num_channels, channel_dim, eps, learn_eps)

    def forward(self, x: Tensor, x_length: Tensor) -> Tensor:
        return super().forward(x)


class WrappedScaledConv1d(ScaledConv1d):
    def forward(self, x: Tensor, x_len: Tensor) -> Tensor:
        return super().forward(x)


class Conv2dSubsampling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layer1_channels: int = 8,
        layer2_channels: int = 32,
        layer3_channels: int = 128,
        activation: str = 'ReLU',
        norm: str = 'BatchNorm',
        is_pnnx: bool = False,
    ) -> None:
        """
        Args:
          in_channels:
            Number of channels in. The input shape is (N, T, in_channels).
            Caution: It requires: T >= 9, in_channels >= 9.
          out_channels
            Output dim. The output shape is (N, ((T-3)//2-1)//2, out_channels)
          layer1_channels:
            Number of channels in layer1
          layer1_channels:
            Number of channels in layer2
          is_pnnx:
            True if we are converting the model to PNNX format.
            False otherwise.
        """
        assert in_channels >= 9
        super().__init__()

        if activation == "DoubleSwish":
            act = DoubleSwish
        else:
            act = getattr(nn, activation)
        self.conv = nn.Sequential(
            ScaledConv2d(
                in_channels=1,
                out_channels=layer1_channels,
                kernel_size=3,
                padding=0,
            ),
            ActivationBalancer(channel_dim=1),
            act(),
            ScaledConv2d(
                in_channels=layer1_channels,
                out_channels=layer2_channels,
                kernel_size=3,
                stride=2,
            ),
            ActivationBalancer(channel_dim=1),
            act(),
            ScaledConv2d(
                in_channels=layer2_channels,
                out_channels=layer3_channels,
                kernel_size=3,
                stride=2,
            ),
            ActivationBalancer(channel_dim=1),
            act(),
        )
        self.out = ScaledConv1d(
            layer3_channels * (((in_channels - 3) // 2 - 1) // 2), out_channels, 1
        )
        # set learn_eps=False because out_norm is preceded by `out`, and `out`
        # itself has learned scale, so the extra degree of freedom is not
        # needed.
        if norm == "BasicNorm":
            self.out_norm = WrappedBasicNorm(out_channels, learn_eps=False)
        elif norm == "BatchNorm":
            self.out_norm = WrappedBatchNorm1d(out_channels)
        elif norm == "SyncBatchNorm":
            self.out_norm = WrappedSyncBatchNorm(out_channels)
        elif norm == "ExactBatchNorm":
            raise NotImplementedError()
        # constrain median of output to be close to zero.
        self.out_balancer = ActivationBalancer(
            channel_dim=-1, min_positive=0.45, max_positive=0.55
        )

        # ncnn supports only batch size == 1
        self.is_pnnx = is_pnnx
        self.conv_out_dim = self.out.weight.shape[1]

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            Its shape is (N, idim, T).

        Returns:
          Return a tensor of shape (N, odim, ((T-3)//2-1)//2)
        """
        if not self.is_pnnx:
            lengths = (((lengths - 3) >> 1) - 1) >> 1
        else:
            lengths1 = torch.floor((lengths - 3) / 2)
            lengths = torch.floor((lengths1 - 1) / 2)
            lengths = lengths.to(lengths)
        
        # On entry, x is (N, idim, T)
        x = x.unsqueeze(1)  # (N, idim, T) -> (N, 1, idim, T) i.e., (N, C, H, W)
        x = self.conv(x)

        if torch.jit.is_tracing() and self.is_pnnx:
            x = x.reshape(1, self.conv_out_dim, -1)
            x = self.out(x)
        else:
            # Now x is of shape (N, odim, ((T-3)//2-1)//2, ((idim-3)//2-1)//2)
            b, c, f, t = x.size()
            x = self.out(x.view(b, c * f, t))

        # Now x is of shape (N, odim, ((T-3)//2-1))//2)
        x = self.out_norm(x, lengths)
        x = self.out_balancer(x)
        
        return x, lengths


class CausalConv1dOnnx(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d, k, s = self.dilation[0], self.kernel_size[0], self.stride[0]
        self.causal_padding = d * (k - 1) - (s - 1)

    def initialize_cache(self, x: Tensor) -> Tensor:
        return torch.zeros(
            x.size(0), self.in_channels, self.causal_padding, device=x.device)

    def forward(self, x: Tensor, cache: Tensor) -> tp.Tuple[Tensor, Tensor]:
        x = torch.cat((cache, x), dim=2)
        cache = x[:, :, -self.causal_padding:]
        y = F.conv1d(x, self.weight, self.bias, self.stride, self.padding,
                     self.dilation, self.groups)
        return y, cache


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d, k, s = self.dilation[0], self.kernel_size[0], self.stride[0]
        self.causal_padding = d * (k - 1) - (s - 1)
        # nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        # if self.bias is not None:
        #     self.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, [self.causal_padding, 0])
        y = F.conv1d(x, self.weight, self.bias, self.stride, self.padding,
                     self.dilation, self.groups)
        return y


class CausalScaledConv1d(ScaledConv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, padding_mode='zeros')
        d, k, s = self.dilation[0], self.kernel_size[0], self.stride[0]
        self.causal_padding = d * (k - 1) - (s - 1)
        assert self.padding[0] == 0, self.padding
        # nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        # if self.bias is not None:
        #     self.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, [self.causal_padding, 0])
        return super().forward(x)


class CausalSE(nn.Module):
    r"""
    Causal Squeeze-and-excitation module.

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
        chunksize: tp.List[int] = [16],
        se_gate: str = "sigmoid",
    ) -> None:
        super().__init__()
        assert dim % 8 == 0, 'Dimension should be divisible by 8.'
        self.dim = dim
        self.chunksize = chunksize

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
            self.gate = nn.Sigmoid()
        elif se_gate == "tanh":
            self.gate = nn.Tanh()
        else:
            raise ValueError(f"invalid se_gate '{se_gate}'")

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
        chunksize = random.choice(self.chunksize)
        padding = (seq_lengths + chunksize - 1) // chunksize * chunksize - seq_lengths
        x = F.pad(x, (0, padding))      # pad right so that (L+P) % cs = 0
        lengths_cumsum = torch.arange(
            1, seq_lengths + padding + 1, device=x.device, dtype=x.dtype
        )

        x = x.cumsum(dim=2) / lengths_cumsum       # output[:, :, -1] = output[:, :, L]
        x = x[:, :, chunksize-1::chunksize]   # output[:, :, cs-1, 2*cs-1, 3*cs-1, ...]
        
        # main network
        output = self.sequential(x)
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
        norm: str = 'BatchNorm',
        dropout: float = 0.0,
        se_activation: str = 'ReLU',
        scaled_conv: bool = False,
        act_bal: bool = False,
        zero_init_residual: bool = False,
        se_gate: str = "sigmoid",
    ) -> None:
        super(ConvBlock, self).__init__()
        
        bias = True
        if norm == "BatchNorm":
            Norm = WrappedBatchNorm1d
            bias = False
        elif norm == "SyncBatchNorm":
            Norm = WrappedSyncBatchNorm
            bias = False
        elif norm == "BasicNorm":
            Norm = WrappedBasicNorm
        else:
            raise RuntimeError(f"invalid norm {norm}")

        if scaled_conv:
            Conv = ScaledConv1d
            CausalConv = CausalScaledConv1d
        else:
            Conv = nn.Conv1d
            CausalConv = CausalConv1d
        
        if act_bal:
            ActBal = ActivationBalancer
            activation_kwargs = {'inplace': False}
        else:
            ActBal = nn.Identity
        
        self.activation = getattr(nn, activation)(**activation_kwargs)
        self.act_bal = ActBal(1)
        self.pointwise1 = Conv(channels, channels_hidden, 1, bias=bias)
        self.norm1 = Norm(channels_hidden)
        self.depthwise = CausalConv(
            channels_hidden, channels_hidden, kernel_size, groups=channels_hidden,
            dilation=dilation, bias=bias)
        self.norm2 = Norm(channels_hidden)
        self.pointwise2 = Conv(channels_hidden, channels, 1)
        self.se = CausalSE(channels, se_activation, scaled_conv, act_bal, se_gate=se_gate)
        self.dropout = nn.Dropout(dropout, inplace=True)
        
        self.scale = None
        if zero_init_residual:
            self.scale = nn.Parameter(torch.zeros(1))

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
        x = self.norm1(x, x_len)
        x = self.depthwise(x)
        x = self.act_bal(x)
        x = self.activation(x)
        x = self.norm2(x, x_len)
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

        return x, x_len


class Conv1dSubsampling(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation,
        norm,
        channels_hidden,
        scaled_conv: bool = False,
        act_bal: bool = False,
    ) -> None:
        super().__init__()
        
        bias = True
        if norm == "BatchNorm":
            Norm = WrappedBatchNorm1d
            bias = False
        elif norm == "SyncBatchNorm":
            Norm = WrappedSyncBatchNorm
            bias = False
        elif norm == "BasicNorm":
            Norm = WrappedBasicNorm
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
            activation_kwargs = {'inplace': True}
        
        self.activation = getattr(nn, activation)(**activation_kwargs)
        self.act_bal = ActBal(1)
        self.pointwise1 = Conv(in_channels, channels_hidden, 1, bias=bias)
        self.norm1 = Norm(channels_hidden)
        self.depthwise = Conv(
            channels_hidden, channels_hidden, 8, stride=4, padding=2,
            groups=channels_hidden, bias=bias
        )
        self.norm2 = Norm(channels_hidden)
        self.pointwise2 = Conv(channels_hidden, out_channels, 1)
    
    def forward(self, x, x_len):
        x = self.pointwise1(x)
        x = self.act_bal(x)
        x = self.activation(x)
        x = self.norm1(x, x_len)
        
        x_len = torch.floor(x_len / 4)
        
        x = self.depthwise(x)
        x = self.act_bal(x)
        x = self.activation(x)
        x = self.norm2(x, x_len)
        x = self.pointwise2(x)
        x = self.act_bal(x)

        return x, x_len


class Conv1dSubsamplingLinear(nn.Module):
    def __init__(self, in_ch, out_ch, act_bal, scaled_conv):
        super().__init__()
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
            Conv(out_ch, out_ch, 8, stride=4, padding=2, groups=out_ch),
            ActBal(1)
        )
    
    def forward(self, x, x_len):
        x = self.conv(x)
        x_len = torch.floor(x_len / 4)
        return x, x_len


class Encoder(EncoderInterface):
    def __init__(
        self,
        num_features: int,
        subsampling_factor: int = 4,
        use_conv2d_subsampling: bool = True,
        channels: int = 384,
        channels_expansion: int = 512,
        kernel_size: int = 4,
        dilations: tp.List[int] = [1, 2, 4, 1, 2, 4],
        output_channels: int = 640,
        dropout: float = 0.075,
        activation: str = 'ReLU',
        activation_kwargs: dict = {'inplace': True},
        norm: str = 'BatchNorm',
        se_activation: str = 'ReLU',
        is_pnnx: bool = False,
        num_rnn_layers: int = 3,
        dim_feedforward: int = 2048,
        rnn_hidden_size: int = 640,
        grad_norm_threshold: float = 25.0,
        layer_dropout: float = 0.075,
        aux_layer_period: int = 0,
        scaled_conv: bool = False,
        act_bal: bool = False,
        conv1d_subsampling_version: int = 1,
        zero_init_residual: bool = False,
        se_gate: str = "sigmoid",
    ) -> None:
        super().__init__()

        self.norm = norm
        self.num_features = num_features
        self.subsampling_factor = subsampling_factor
        if subsampling_factor != 4:
            raise NotImplementedError("Support only 'subsampling_factor=4'.")
        
        Conv = ScaledConv1d if scaled_conv else nn.Conv1d
        
        if use_conv2d_subsampling:
            self.conv_pre = Conv2dSubsampling(
                num_features,
                channels,
                activation=activation,
                norm=norm,
                is_pnnx=is_pnnx,
            )
        elif conv1d_subsampling_version == 1:
            self.conv_pre = Conv1dSubsampling(
                num_features,
                channels,
                activation=activation,
                norm=norm,
                channels_hidden=channels_expansion,
                scaled_conv=scaled_conv,
                act_bal=act_bal
            )
        elif conv1d_subsampling_version == 2:
            self.conv_pre = Conv1dSubsamplingLinear(
                num_features,
                channels,
                scaled_conv=scaled_conv,
                act_bal=act_bal
            )
        else:
            raise ValueError(f"Invalid conv1d-subsampling-version {conv1d_subsampling_version}")

        self.is_pnnx = is_pnnx

        self.cnn = nn.ModuleList()
        for dilation in dilations:
            layer = ConvBlock(
                channels, channels_expansion, kernel_size, dilation,
                activation, activation_kwargs, norm, dropout, se_activation,
                scaled_conv, act_bal, zero_init_residual, se_gate
            )
            self.cnn.append(layer)
        self.proj = Conv(channels * 2, output_channels, 1, bias=False)
        
        self.lstm = None
        if num_rnn_layers > 0:
            encoder_layer = RNNEncoderLayer(
                d_model=output_channels,
                dim_feedforward=dim_feedforward,
                rnn_hidden_size=rnn_hidden_size,
                grad_norm_threshold=grad_norm_threshold,
                dropout=dropout,
                layer_dropout=layer_dropout,
            )
            self.lstm = RNNEncoder(
                encoder_layer,
                num_rnn_layers,
                aux_layers=list(
                    range(
                        num_rnn_layers // 3,
                        num_rnn_layers - 1,
                        aux_layer_period,
                    )
                )
                if aux_layer_period > 0
                else None,
            )

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
        x = x.transpose(1, 2)
        lengths = x_lens

        x, lengths = self.conv_pre(x, lengths)     # [batch_size, channels, time]

        if not torch.jit.is_tracing():
            assert x.size(2) == lengths.max().item()

        x_in = x
        for block in self.cnn:
            x, lengths = block(x, lengths, warmup)   # [batch_size, channels, time]
        x = self.proj(torch.cat((x, x_in), dim=1))  # [batch_size, channels_out, time]

        if self.lstm is not None:
            x = x.permute(2, 0, 1)  # [T, B, C]
            x = self.lstm(x, warmup=warmup)[0]
            x = x.permute(1, 0, 2)  # [B, T, C]
        else:
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
        scaled_conv=True,
        use_conv2d_subsampling=True,
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
