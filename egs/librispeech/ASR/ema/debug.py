import typing as tp
import random
import math
import pathlib

import torch
from torch import jit
from torch.nn import functional as F, Identity as ActivationBalancer
from torch import Tensor, nn
from lhotse.utils import LOG_EPSILON
from icefall.iir.iir import EMA


FP16 = False


def q(x: torch.Tensor) -> torch.Tensor:
    if not FP16:
        return x
    x = x.to(torch.float16)
    mag = x.abs()
    return torch.where(mag < 2**-14, torch.zeros_like(x), x).float()


def hook(module: torch.nn.Module, args, output: torch.Tensor) -> torch.Tensor:
    output = q(output)
    return output


def Q(m: torch.nn.Module) -> torch.nn.Module:
    if not FP16:
        return m
    m.register_forward_hook(hook)
    return m


class ScaledLinear(nn.Linear):
    def __init__(
        self,
        *args,
        initial_scale: float = 1.0,
        initial_speed: float = 1.0,
        **kwargs,
    ):
        super(ScaledLinear, self).__init__(*args, **kwargs)
        initial_scale = torch.tensor(initial_scale).log()
        self.weight_scale = nn.Parameter(initial_scale.clone().detach())
        if self.bias is not None:
            self.bias_scale = nn.Parameter(initial_scale.clone().detach())
        else:
            self.register_parameter("bias_scale", None)

        self._reset_parameters(
            initial_speed
        )  # Overrides the reset_parameters in nn.Linear
        Q(self)

    def _reset_parameters(self, initial_speed: float):
        std = 0.1 / initial_speed
        a = (3**0.5) * std
        nn.init.uniform_(self.weight, -a, a)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)
        fan_in = self.weight.shape[1] * self.weight[0][0].numel()
        scale = fan_in**-0.5  # 1/sqrt(fan_in)
        with torch.no_grad():
            self.weight_scale += torch.tensor(scale / std).log()

    def get_weight(self):
        return self.weight * self.weight_scale.exp()

    def get_bias(self):
        if self.bias is None or self.bias_scale is None:
            return None
        else:
            return self.bias * self.bias_scale.exp()

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.get_weight(), self.get_bias())


@jit.script
def iir_torch(x: Tensor, denom_flipped: Tensor, out: Tensor) -> None:
    '''input:
        x: [T, C, B], denom_flipped: [C, 2]
    output: [1+T, C, B]
    In-place calculation. Faster, but backward is not supported.'''
    T = x.size(0)
    
    # reshape
    denom_flipped = denom_flipped.unsqueeze(1)     # [C, 1, N]
    x = x.transpose(0, 1)       # [C, T, B]
    out = out.transpose(0, 1)   # [C, 1+T, B]
    
    # iir filtering
    out[:, 1:2, :] = x[:, 0:1, :] + denom_flipped[:, :, 1:2] @ out[:, 0:1, :]   # [C, 1, B]
    for t in range(1, T):
        torch.baddbmm(  # [C, 1, B] + [C, 1, N] @ [C, N, B] = [C, 1, B]
            x[:, t:t+1, :],             # [C, 1, B]
            denom_flipped,              # [C, 1, N]
            out[:, t-1:t+1, :],         # [C, N, B]
            out=out[:, t+1:t+2, :]      # [C, 1, B]
        )


def iir_backward_grad_weight(out: Tensor, grad_out: Tensor, weight: Tensor) -> Tensor:
    '''input:
        out: [B, C, 1+T] where out[:, ;, 0] = 0
        grad_out: [B, C, T+1] where grad_out[:, :, -1] = 0
        weight: [C, 1, 2]
    out[t:t+2] * weight + in = out[t+2]
    <=> {out[1:], 0} = conv({0, out[:-1]}, weight, padding=1) + in
    Therefore, grad_weight = conv_backward(grad_out[1:], out[:-1], weight, padding=1))
    ex)    o1        o4  X : output (X is a value that must not participate in backward)
        w0 w1        w0 w1 : weight convolution
        .   0  o1 o2 o3  . : input (. is a 0-padding)
    <=> o1~o4 X = conv(0 o1~o3, w, p=1)
    <=> grad_weight = conv_backward({0, o1~o3}, {g1~g4, 0}, p=1)'''
    C = out.size(1)
    return torch.ops.aten.convolution_backward(grad_out, out[:, :, :-1], weight, None,
        (1,), (1,), (1,), False, (0,), C, (False, True, False))[1].squeeze(1)


class IIRTorchFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, denom_flipped: Tensor) -> Tensor:
        # input:
        #   x: [B, C, T]
        #   denom_flipped: [C, N, 2]
        # output: [B, C*N, 1+T] where output[:, :, 0] = 0
        B, C, T = x.shape
        N = denom_flipped.size(1)
        xt = x.transpose(0, 2)              # [T, C, B]
        xt = xt.repeat_interleave(N, dim=1) # [T, C*N, B]
        out = torch.zeros(1+T, C*N, B, device=x.device, dtype=x.dtype)
        weight = denom_flipped.view(C*N, 2)
        iir_torch(xt, weight, out)          # [1+T, C*N, B]
        out = out.transpose(0, 2)           # [B, C*N, 1+T]
        ctx.save_for_backward(weight, out)
        ctx._N = N

        return out      # [B, C*N, 1+T]

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tp.Tuple[Tensor, tp.Optional[Tensor], None]:
        # grad_out: [B, C*N, 1+T]
        weight, out = ctx.saved_tensors
        B, CN, T_1 = out.shape
        N = ctx._N
        C = CN // N
        T = T_1 - 1
        grad_out = grad_out[:, :, 1:].flip(2).transpose(0, 2)   # [T, C*N, B]

        grad_in = torch.zeros(T+1, CN, B, device=out.device, dtype=out.dtype)
        iir_torch(grad_out, weight, grad_in)                    # [1+T, C*N, B]
        grad_in = grad_in.flip(0).transpose(0, 2)               # [B, C*N, T+1]
        if ctx.needs_input_grad[1]:
            grad_weight = iir_backward_grad_weight(out, grad_in, weight.unsqueeze(1))
            grad_weight = grad_weight.view(C, N, 2)
        else:
            grad_weight = None
        
        grad_in = grad_in[:, :, :T].reshape(B, C, N, T).sum(dim=2)  # [B, C, T]

        return grad_in, grad_weight, None


def iir(x: Tensor, weight: Tensor) -> Tensor:
    ''' inputs:
        x: [B, C, T]
        weight: [C, N//2, 2]
    output:
        y: [B, C*N//2, T]
    '''
    x = IIRTorchFunction.apply(x, weight)
    return x[:, :, 1:]


def inv_sigmoid(x: float) -> float:
    assert 0. < x < 1.
    return math.log(x / (1 - x))


class EMA_(nn.Module):
    '''First-order IIR filter layer.
    weight: [C, 2] where weight[:, 0] = 0, weight[:, 1] = some_value
    y[:, :, i+1] = y[:, :, i] * gamma + (1 - gamma) * x[:, :, i+1]
    where gamma = r_max * sigmoid(weight), 0 < r_max <= 1
    '''
    
    def __init__(
        self,
        channels: int,
        r_max: float = 1.0,
        init_method: str = "constant",
        init_value: float = 0.9,
        reversed: bool = False,
        device = None,
        dtype = torch.float32,
    ):
        assert 0 < r_max <= 1.0, r_max
        assert abs(init_value) <= r_max, f"{init_value}, {r_max}"
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.channels = channels
        self.reversed = reversed
        self.r_max = r_max
        
        if init_method == "constant":
            weight_value = inv_sigmoid(init_value / r_max)   # value = r_max * sigmoid(weight)
            self.weight = nn.Parameter(
                torch.empty(channels, **factory_kwargs).fill_(weight_value)
            )
        elif init_method == "uniform":
            weight_value = inv_sigmoid(init_value / r_max)
            self.weight = nn.Parameter(
                torch.empty(channels, **factory_kwargs).uniform_(-weight_value, weight_value)
            )
        else:
            raise ValueError(f"Invalid init_method {init_method}")
        self.weight_reparameterized = False
    
    @torch.no_grad()
    def remove_weight_reparameterizations(self) -> None:
        gamma = self.get_weight()
        self.weight = nn.Parameter(gamma)
        self.weight_reparameterized = True
    
    def get_weight(self) -> Tensor:
        if self.weight_reparameterized:
            return self.weight
        return self.r_max * torch.sigmoid(self.weight).view(self.channels, 1, 1)
    
    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, T]
        gamma = self.get_weight()
        denom_flipped = F.pad(gamma, (1, 0))    # [C, 1, 2] (zero-pad left)
        
        x = x * (1.0 - gamma.view(1, self.channels, 1))     # [B, C, T]
        if self.reversed:
            x = x.flip(2)
            
        x = iir(x, denom_flipped)       # [B, C, T]
        if self.reversed:
            x = x.flip(2)
        return x


class ScaledConv1d(nn.Conv1d):
    # See docs for ScaledLinear
    def __init__(
        self,
        *args,
        initial_scale: float = 1.0,
        initial_speed: float = 1.0,
        **kwargs,
    ):
        super(ScaledConv1d, self).__init__(*args, **kwargs)
        initial_scale = torch.tensor(initial_scale).log()

        self.bias_scale: tp.Optional[nn.Parameter]  # for torchscript

        self.weight_scale = nn.Parameter(initial_scale.clone().detach())
        if self.bias is not None:
            self.bias_scale = nn.Parameter(initial_scale.clone().detach())
        else:
            self.register_parameter("bias_scale", None)
        self._reset_parameters(
            initial_speed
        )  # Overrides the reset_parameters in base class
        Q(self)

    def _reset_parameters(self, initial_speed: float):
        std = 0.1 / initial_speed
        a = (3**0.5) * std
        nn.init.uniform_(self.weight, -a, a)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)
        fan_in = self.weight.shape[1] * self.weight[0][0].numel()
        scale = fan_in**-0.5  # 1/sqrt(fan_in)
        with torch.no_grad():
            self.weight_scale += torch.tensor(scale / std).log()

    def get_weight(self):
        return self.weight * self.weight_scale.exp()

    def get_bias(self):
        bias = self.bias
        bias_scale = self.bias_scale
        if bias is None or bias_scale is None:
            return None
        else:
            return bias * bias_scale.exp()

    def forward(self, input: Tensor) -> Tensor:
        F = torch.nn.functional
        if self.padding_mode != "zeros":
            return F.conv1d(
                F.pad(
                    input,
                    self._reversed_padding_repeated_twice,
                    mode=self.padding_mode,
                ),
                self.get_weight(),
                self.get_bias(),
                self.stride,
                (0,),
                self.dilation,
                self.groups,
            )
        return F.conv1d(
            input,
            self.get_weight(),
            self.get_bias(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d, k, s = self.dilation[0], self.kernel_size[0], self.stride[0]
        self.causal_padding = d * (k - 1) - (s - 1)
        # nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        # if self.bias is not None:
        #     self.bias.data.zero_()
        Q(self)

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
        assert self.causal_padding > 0
        Q(self)

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, [self.causal_padding, 0])
        y = super().forward(x)
        self.cache = y[:, :, -self.causal_padding:]
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
        scaled_conv: bool = False,
        act_bal: bool = False,
        chunksize: tp.List[int] = [16],
        se_gate: str = "sigmoid",
    ) -> None:
        super().__init__()
        assert dim % 8 == 0, 'Dimension should be divisible by 8.'
        self.dim = dim
        self.chunksize = chunksize
        self.scaled_conv = scaled_conv
        self.activation = activation

        self.ema = EMA(dim, r_max=0.93, init_method="uniform")
        
        self.scaled_conv = scaled_conv
        Conv = ScaledConv1d if scaled_conv else nn.Conv1d
        Act = getattr(nn, activation)
        self.activation = activation
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
        weight = self.sequential[0].weight
        kwargs = dict(device=weight.device, dtype=weight.dtype)
        self.ema.remove_weight_reparameterizations()
        if not self.scaled_conv:
            return
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
        chunksize = random.choice(self.chunksize)
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
        else:
            raise RuntimeError(f"invalid norm {norm}")
        
        self.scaled_conv = scaled_conv
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
        self.pointwise1 = Conv(channels, channels_hidden, 1)
        self.norm1 = Norm(channels_hidden)
        self.depthwise = CausalConv(
            channels_hidden, channels_hidden, kernel_size, groups=channels_hidden,
            dilation=dilation)
        self.norm2 = Norm(channels_hidden)
        self.pointwise2 = Conv(channels_hidden, channels, 1)
        self.se = CausalSE(channels, se_activation, scaled_conv, act_bal, se_gate=se_gate)
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
            self.depthwise = CausalConv1d(
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
            self.depthwise.bias.data.copy_(bias.squeeze(2).squeeze(0))
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
        if self.initial_cache is not None:
            x = torch.cat([self.initial_cache, x], dim=2)
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


class Encoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        subsampling_factor: int = 4,
        use_conv2d_subsampling: bool = False,
        channels: int = 384,
        channels_expansion: int = 512,
        kernel_size: int = 4,
        dilations: tp.List[int] = [1 for _ in range(22)],
        output_channels: int = 512,
        dropout: float = 0.075,
        activation: str = 'ReLU',
        activation_kwargs: dict = {'inplace': True},
        norm: str = 'BatchNorm',
        se_activation: str = 'ReLU',
        is_pnnx: bool = False,
        scaled_conv: bool = True,
        act_bal: bool = True,
        conv1d_subsampling_version: int = 2,
        zero_init_residual: bool = False,
        se_gate: str = "sigmoid",
    ) -> None:
        super().__init__()

        self.norm = norm
        self.num_features = num_features
        self.subsampling_factor = subsampling_factor
        if subsampling_factor != 4:
            raise NotImplementedError("Support only 'subsampling_factor=4'.")
        self.scaled_conv = scaled_conv
        Conv = ScaledConv1d if scaled_conv else nn.Conv1d
        
        if use_conv2d_subsampling:
            raise RuntimeError(use_conv2d_subsampling)
        elif conv1d_subsampling_version == 1:
            raise RuntimeError(conv1d_subsampling_version)
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
        x = x.transpose(1, 2)
        lengths = x_lens

        x, lengths = self.conv_pre(x, lengths)     # [batch_size, channels, time]

        x_in = x
        for block in self.cnn:
            x, lengths = block(x, lengths, warmup)   # [batch_size, channels, time]
        x = self.proj(torch.cat((x, x_in), dim=1))  # [batch_size, channels_out, time]
        x = x.transpose(1, 2)   # [B, T, C]
        
        return x, lengths, None


class Joiner(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
    ):
        super().__init__()

        self.encoder_proj = ScaledLinear(encoder_dim, joiner_dim)
        self.decoder_proj = ScaledLinear(decoder_dim, joiner_dim)
        self.output_linear = ScaledLinear(joiner_dim, vocab_size)

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
        project_input: bool = True,
    ) -> torch.Tensor:
        assert encoder_out.ndim == decoder_out.ndim

        if project_input:
            logit = self.encoder_proj(encoder_out) + self.decoder_proj(decoder_out)
        else:
            logit = encoder_out + decoder_out

        logit = self.output_linear(torch.tanh(logit))

        return logit


def plot(figures):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.figure(figsize=(10, 10))
    for idx, (name, image) in enumerate(figures.items(), start=1):
        plt.subplot(len(figures), 1, idx)
        im = plt.imshow(image, origin="lower", interpolation="nearest", aspect='auto')
        plt.colorbar(im)
        plt.title(name)
    plt.savefig("delete_it.png")


def test_model():
    device = "cpu"
    dtype = torch.float64
    model = Encoder(
        80,
        dilations=[1 for _ in range(22)],
        kernel_size=8,
        channels=384,
        channels_expansion=1536,
        output_channels=512,
        norm='BatchNorm',
        scaled_conv=True,
        use_conv2d_subsampling=False,
        conv1d_subsampling_version=2,
        zero_init_residual=True,
        se_gate="tanh",
    ).to(device=device, dtype=dtype).eval()
    joiner = Joiner(512, 512, 512, 500).to(device=device, dtype=dtype).eval()

    checkpoint = torch.load("ema/kws_abs/epoch-120.pt", map_location='cpu')
    encoder_state_dict = {}
    joiner_state_dict = {}
    for n, p in checkpoint["model"].items():
        if n.startswith("encoder."):
            encoder_state_dict[n[len("encoder."):]] = p
        elif n.startswith("joiner"):
            joiner_state_dict[n[len("joiner."):]] = p
    model.load_state_dict(encoder_state_dict)
    joiner.load_state_dict(joiner_state_dict)
    # model.remove_weight_reparameterizations()
    
    from pathlib import Path
    from decode import get_parser
    from train import get_params, get_transducer_model
    from keyword_spotting import get_model
    import sentencepiece as spm
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    params = get_params()
    params.update(vars(args))
    
    sp = spm.SentencePieceProcessor()
    sp.LoadFromFile(params.bpe_model)
    params.blank_id = sp.PieceToId("<blk>")
    params.unk_id = sp.PieceToId("<unk>")
    params.vocab_size = sp.GetPieceSize()
    model_orig = get_transducer_model(params)
    get_model(params, model_orig, device, args, sp)
    model_orig = model_orig.to(dtype)
    model_orig, joiner = model_orig.encoder, model_orig.joiner
    
    hop, window = 160, 400
    with open("input.buf", "rb") as f:
        x = torch.frombuffer(f.read(), dtype=torch.float32)
    with open("param.buf", "rb") as f:
        ft_weight = torch.frombuffer(f.read(512*400*4), dtype=torch.float32).view(512, 400)
        mel_fbank = torch.frombuffer(f.read(80*256*4), dtype=torch.float32).view(80, 256)
    with open("cache.buf", "rb") as f:
        cache_conv_pre = torch.frombuffer(f.read(384*4*4), dtype=torch.float32).view(1, 384, 4)
    x = F.pad(x, (window - hop, 0))

    spec_len = (x.size(0) - window + hop) // hop
    y = x.as_strided(size=(spec_len, window), stride=(hop, 1))

    # remove DC
    row_means = torch.mean(y, dim=1).unsqueeze(1)
    y = y - row_means

    # preemphasis
    y_prev = F.pad(y.unsqueeze(0), (1, 0), mode="replicate").squeeze(0)
    y = y - 0.97 * y_prev[:, :-1]

    # window + stft
    y = F.linear(y.float(), ft_weight, None)

    # magnitude
    y = y.view(spec_len, 2, 256).abs().sum(dim=1)

    # log mel
    y = F.linear(y.float(), mel_fbank)
    mel_orig = y.log().mul(2).clamp_min(-15.9453).unsqueeze(0).to(dtype)        # [B, T, C]
    mel = (mel_orig + 6.375938187300722).transpose(1, 2) * 0.22963919954179052  # [B, C, T]
    mel_len = torch.tensor([mel.size(2)], dtype=torch.int64, device=y.device)
    with torch.no_grad():
        # eps_normalized = (-15.9453 + 6.375938187300722) * 0.22963919954179052
        # mel_padded = torch.cat([torch.ones(1, mel.size(1), 4, device=mel.device, dtype=mel.dtype)*eps_normalized, mel_normalized], dim=2)
        # x, x_len = model_orig.conv_pre(mel, mel_len)
            
        # for block in model_orig.cnn[:]:
        #     skip = x
        #     x = block.pointwise1(x)
        #     x = block.activation(x)
        #     x = block.norm1(x)
        #     x = block.depthwise(x)
        #     x = block.activation(x)
        #     x = block.norm2(x)
        #     x = block.pointwise2(x)
        #     x = skip + x * block.scale
        x, *_ = model_orig(mel_orig, mel_len)
        x = x.transpose(1, 2)
        
        y, y_len = model.conv_pre(mel, mel_len)
        y_in = y
        for block in model.cnn[:]:
            # y, y_len = block(y, y_len, 1.0)
            skip = y
            y = block.pointwise1(y)
            y = block.activation(y)
            y = block.norm1(y)
            y = block.depthwise(y)
            y = block.activation(y)
            y = block.norm2(y)
            y = block.pointwise2(y)
            y = block.se(y)
            y = skip + y * block.scale
        y = model.proj(torch.cat((y, y_in), dim=1))
    
    x, y = x.squeeze(0), y.squeeze(0)
    figures = {
        "original ver.": x.cpu().numpy(),
        "porting ver.": y.cpu().numpy(),
        "diff": (x - y).cpu().numpy()
    }
    plot(figures)


if __name__ == "__main__":
    test_model()
