"""
IIR layer, second-order.
"""

import typing as tp
import time
import math
import random
from collections import defaultdict

from scipy import signal

import torch
from torch import jit
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from tqdm import tqdm



def verbose() -> bool:
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return False
    return True


try:
    from .kernels import iir_2o_cuda as iir_cuda
    _IIR_KERNEL_AVAILABLE = True
    if verbose():
        print("*** IIR v7 loaded.")
except ImportError:
    raise ImportError("IIR cuda kernel not found. Compile with 'python setup_iir_2o.py build_ext --inplace'")
    _IIR_KERNEL_AVAILABLE = False
    if verbose():
        print("*** IIR cuda kernel not found. Using Torch Native operation.")
        print("*** For speedup, compile with '$python setup_iir.py build_ext --inplace'")

try:
    from modules.kernels import delay_cuda
    _DELAY_KERNEL_AVAILABLE = True
    if verbose():
        print("*** Delay cuda kernel loaded.")
except ImportError:
    _DELAY_KERNEL_AVAILABLE = False
    if verbose():
        print("*** Delay cuda kernel not found. Using Torch Native operation.")
        # print("*** For speedup, compile with '$python setup_delay.py build_ext --inplace'")


def unpad_delay_torch(x: Tensor, delay: Tensor, max_delay: int) -> None:
    '''
    Args:
        x: [B, C, T+max_delay]
        delay: [C] (torch.int32 Tensor)
    Return:
        out: [B, C, T]'''
    B, C, T_padded = x.shape
    T = T_padded - max_delay
    out = []
    for c in range(C):
        delay_c = delay[c].item()
        out.append(x[:, c:c+1, delay_c:T+delay_c])
    return torch.cat(out, dim=1)


class DelayCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, delay: Tensor, max_delay: int) -> Tensor:
        '''
        Args:
            x: [B, C, T+max_delay]
            delay: [C] (torch.int32 Tensor)
        Return:
            out: [B, C, T]
        '''
        B, C, T_padded = x.shape
        T = T_padded - max_delay
        out = torch.empty(B, C, T, device=x.device, dtype=x.dtype)
        delay_cuda.unpad(x, out, delay)
        ctx.save_for_backward(delay)
        ctx.max_delay = max_delay

        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tp.Tuple[Tensor, None, None]:
        '''
        Args:
            grad_out: [B, C, T]
        Return:
            grad_x: [B, C, T+max(delay)]
            grad_delay: None
            grad_max_delay: None
        '''
        delay, = ctx.saved_tensors
        B, C, T = grad_out.shape
        grad_x = torch.empty(B, C, T + ctx.max_delay,
                             device=grad_out.device, dtype=grad_out.dtype)
        delay_cuda.pad(grad_out, grad_x, delay)
        return grad_x, None, None


def unpad_delay(x: Tensor, delay: Tensor, max_delay: int) -> Tensor:
    '''
    Args:
        x: [B, C, T+max_delay]
        delay: [C] (torch.int32 Tensor)
    Return:
        out: [B, C, T]'''
    if x.is_cuda and _DELAY_KERNEL_AVAILABLE:
        return DelayCudaFunction.apply(x, delay, max_delay)
    return unpad_delay_torch(x, delay, max_delay)


@jit.script
def iir_torch(x: Tensor, denom_flipped: Tensor, scale: Tensor, out: Tensor) -> None:
    '''
    Args:
        x: [T, C, B]
        denom_flipped: [C, 2]
        scale: [C]
        out: [2+T, C, B]
    In-place calculation. Faster, but backward is not supported.
    '''
    T = x.size(0)
    
    # reshape
    denom_flipped = denom_flipped.unsqueeze(1)     # [C, 1, 2]
    x = x.transpose(0, 1) * scale.view(-1, 1, 1)   # [C, T, B]
    out = out.transpose(0, 1)   # [C, 2+T, B]
    
    # iir filtering
    for t in range(T):
        torch.baddbmm(  # [C, 1, B] + [C, 1, 2] @ [C, 2, B] = [C, 1, B]
            x[:, t:t+1, :],         # [C, 1, B]
            denom_flipped,          # [C, 1, 2]
            out[:, t:t+2, :],       # [C, 2, B]
            out=out[:, t+2:t+3, :]  # [C, 1, B]
        )


def iir_default(
    x: Tensor,
    denom_flipped: Tensor,
    scale: Tensor,
    cache: tp.Optional[None] = None
) -> Tensor:
    '''
    Args:
        x: [B, C, T]
        denom_flipped: [C, 2]
        scale: [C]
        cache: None or [B, C, 2]
    Return: [B, C, 2+T] where output[:, :, 0:2] = cache or 0 (if cache is None)
    Slower, but backward is supported.
    '''
    B, C, T = x.shape
    denom_flipped = denom_flipped.transpose(0, 1).unsqueeze(2)  # [2, C, 1]
    x = x.transpose(0, 2) * scale.view(1, -1, 1)                # [T, C, B]
    
    if cache is None:
        cache = torch.zeros(2, C, B, device=x.device, dtype=x.dtype)    # [2, C, B]
        out = []
    else:
        cache = cache.permute(2, 1, 0)   # [2, C, B]
        out = [cache]
    for t in range(T):
        _x = (cache * denom_flipped).sum(0, keepdim=True)   # [2, C, B], [2, C, 1] -> [1, C, B]
        _x = _x + x[t:t+1, :, :]                            # [1, C, B]
        out.append(_x)
        cache = torch.cat([cache[1:, :, :], _x], dim=0)     # [2, C, B]
    out = torch.cat(out, dim=0)                 # [2+T, C, B]
    out = out.transpose(0, 2)                   # [B, C, 2+T]
    return out


def iir_gpu(
    x: Tensor,
    denom_flipped: Tensor,
    scale: Tensor,
    cache: tp.Optional[None] = None
):
    if cache is None:
        cache = torch.zeros(B, C, 2, device=x.device, dtype=x.dtype)    # [B, C, 2]
    out = torch.cat([cache, x], dim=2)  # [B, C, 2+T]
    B, C, T = x.shape
    T_padded = (T+31) // 32 * 32
    x = F.pad(x, (0, T_padded - T))
    scale = scale.view(C, 1, 1)
    
    # a: [5, C, 32] / b: [4, C, 32]
    a = torch.zeros(5, C, 32, device=x.device, dtype=x.dtype)
    b = torch.zeros(4, C, 32, device=x.device, dtype=x.dtype)
    def fill_a_b(n, a_n, b_n):
        for i in range(32):
            if i > n and i % 2**(n-1):
                pass
                
    
    a_list, b_list = [], []
    a0 = denom_flipped[:, 1]
    b0 = denom_flipped[:, 0]
    a_list.append(a0)
    b_list.append(b0)
    a_m2 = a0
    b_m2 = b0
    a[0, c, 1::2] = a0
    
    a_m1 = a0 * a0 + b0
    b_m1 = a0 * b0
    a_list.append(a_m1)
    b_list.append(b_m1)
    
    for n in range(2, 17):
        a = a0 * a_m1 + b0 * a_m2
        b = a0 * b_m1 + b0 * b_m2
        a_list.append(a)
        b_list.append(b)
    
    for i in range(32):
        if i % 4 in [2, 3]:
            a[1, c, i] = a0
            b[0, c, i] = b0
        if i % 8 in [4, 5, 6, 7]:
            a[2, c, i] = a0
            b[1, c, i] = b0
    for t in range(0, T_padded, 32):
        for stage in range(5):
            y = scale * x[:, :, t:t+32]
            y[:, :, 0] += denom_flipped[:, :, 0] * cache[:, :, 0]
            y[:, :, 0] += denom_flipped[:, :, 1] * cache[:, :, 1]
            y[:, :, 1] += denom_flipped[:, :, 0] * cache[:, :, 1]
            
            y += a


def iir_backward_grad_weight(
    out: Tensor,
    grad_out: Tensor,
    weight: Tensor,
    inv_scale: Tensor
) -> Tensor:
    '''
    Args:
        out: [B, C, 2+T]
        grad_out: [B, C, T]
        weight: [C, 2]
        inv_scale: [C]
    Return:
        grad_weight: [C, 2]
    
    out[t:t+2] * weight + in * scale = out[t+2]
    <=> out[2:] = conv(out[:-1], weight) + in * scale
    Therefore, grad_weight = conv_backward(grad_out[2:], out[:-1], weight))
    ex)     o0 ........ o4 o5 : output
        w0  w1 -> -> -> w0 w1 : weight convolution
        o-2 o-1 o0 .... o3 o4 : input
    <=> o0~o5 = conv(o-2~o4, w) + DontCare
    <=> grad_weight = conv_backward(o-2~o4, g0~g5)
    '''
    # B, C, T_2 = out.shape
    # return F.conv1d(
    #     out[:, :, :-1].transpose(0, 1).reshape(1, C*B, T_2-1),
    #     grad_out.transpose(0, 1),
    #     groups=C,
    # ).squeeze(0)
    
    C = out.size(1)
    return torch.ops.aten.convolution_backward(
        grad_out, out[:, :, :-1], weight.unsqueeze(1),
        None,   # Optional[bias_size]
        (1,),   # stride
        (0,),   # padding
        (1,),   # dilation
        False,  # transposed
        (0,),   # output_padding
        C,      # groups
        (False, True, False)    # output_mask: [grad_input, grad_weight, grad_bias]
    )[1].squeeze(1).mul_(inv_scale.unsqueeze(1))


def iir_backward_grad_scale(
    out: Tensor,
    grad_in: Tensor,
    inv_scale: Tensor
) -> Tensor:
    '''
    Args:
        out: [B, C, T]
        grad_in: [B, C, T]
        inv_scale: [C]
    Return:
        grad_scale: [C]
    '''
    return (out * grad_in).sum(dim=(0, 2)).mul_(inv_scale)


class IIRTorchFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, scale: Tensor) -> Tensor:
        # Args:
        #   x: [B, C, T]
        #   weight: [C, 2]
        #   scale: [C]
        # Return: [B, C, 2+T] where output[:, :, 0:2] = 0
        B, C, T = x.shape
        xt = x.transpose(0, 2)                      # [T, C, B]
        out = torch.zeros(2+T, C, B, device=x.device, dtype=x.dtype)
        iir_torch(xt, weight, scale, out)           # [2+T, C, B]
        out = out.transpose(0, 2)                   # [B, C, 2+T]
        ctx.save_for_backward(x, weight, scale, out)

        return out      # [B, C, 2+T]

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tp.Tuple[Tensor, tp.Optional[Tensor], tp.Optional[Tensor]]:
        # grad_out: [B, C, 2+T]
        x, weight, scale, out = ctx.saved_tensors
        B, C, T_2 = out.shape
        T = T_2 - 2
        grad_out = grad_out[:, :, 2:].flip(2).transpose(0, 2)   # [T, C, B]

        grad_in = torch.zeros(T_2, C, B, device=out.device, dtype=out.dtype)
        iir_torch(grad_out, weight, scale, grad_in)             # [2+T, C, B]
        grad_in = grad_in.flip(0).transpose(0, 2)               # [B, C, T+2]
        grad_in = grad_in[:, :, :T]  # [B, C, T]
        
        grad_weight = None
        inv_scale = scale.clamp(min=1e-12).reciprocal()
        if ctx.needs_input_grad[1]:
            grad_weight = iir_backward_grad_weight(out, grad_in, weight, inv_scale)
        
        grad_scale = None
        if ctx.needs_input_grad[2]:
            grad_scale = iir_backward_grad_scale(x, grad_in, inv_scale)     # [C]

        return grad_in, grad_weight, grad_scale


class IIRCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, scale: Tensor) -> Tensor:
        # Args:
        #   x: [B, C, T]
        #   weight: [C, 2]
        #   scale: [C]
        # Return: [B, C, 2+T] where output[:, :, 0:2] = 0
        B, C, T = x.shape
        out = torch.zeros(B, C, 2+T, device=x.device, dtype=x.dtype)
        iir_cuda.forward(x, weight, scale, out)    # [B, C, 2+T]
        ctx.save_for_backward(x, weight, scale, out)

        return out      # [B, C, 2+T]

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tp.Tuple[Tensor, tp.Optional[Tensor], tp.Optional[Tensor]]:
        '''
        Args:
            grad_out: [B, C, 2+T]
                -> grad_out[:, :, 0:2] should not participate in the gradient calculation.
        Return:
            grad_in: [B, C, T]
            grad_weight: [C, 2]
            grad_scale: [C]
        
        Default Impl)
            forward)
                y = x * scale                   # Requires more calculation & memory
                out = iir(y, weight)
            backward)
                grad_y = iir_backward(out, grad_out)
                grad_weight = iir_backward_grad_weight(out, grad_y)
                grad_x = grad_y * weight
                grad_scale = grad_y * x
        Efficient Impl)
            forward)
                out = iir(x, weight, scale)     # Efficient
            backward)
                grad_x = iir_backward(out, grad_out, weight)
                grad_weight = iir_backward_grad_weight(out, grad_y)
                            = iir_backward_grad_weight(out, grad_x / scale)
                            = iir_backward_grad_weight(out, grad_x) / scale
                grad_scale = grad_y * x
                           = (grad_x / scale) * x
                           = (grad_x * x) / scale
        '''
        x, weight, scale, out = ctx.saved_tensors
        B, C, T_2 = out.shape
        T = T_2 - 2
        grad_out = grad_out[:, :, 2:].flip(2)   # [B, C, T]

        grad_in = torch.zeros(B, C, T_2, device=out.device, dtype=out.dtype)
        iir_cuda.forward(grad_out, weight, scale, grad_in)  # [B, C, 2+T]
        grad_in = grad_in.flip(2)               # [B, C, T+2]
        grad_in = grad_in[:, :, :T]             # [B, C, T]
        
        grad_weight = None
        inv_scale = scale.clamp(min=1e-12).reciprocal()
        if ctx.needs_input_grad[1]:
            grad_weight = iir_backward_grad_weight(out, grad_in, weight, inv_scale)

        grad_scale = None
        if ctx.needs_input_grad[2]:
            grad_scale = iir_backward_grad_scale(x, grad_in, inv_scale) # [C]
            
        return grad_in, grad_weight, grad_scale


def iir(x: Tensor, weight: Tensor, scale: Tensor) -> Tensor:
    '''
    Args:
        x: [B, C, T]
        weight: [C, 2]
        scale: [C]
    Return:
        y: [B, C, 2+T]
    '''
    if x.is_cuda and _IIR_KERNEL_AVAILABLE:
        return IIRCudaFunction.apply(x, weight, scale)

    return IIRTorchFunction.apply(x, weight, scale)


class Radius(jit.ScriptModule):
    def __init__(
        self, channels: int, r_max: float, init_kwargs: tp.Dict[str, tp.Any],
        device=None, dtype=None
    ) -> None:
        super().__init__()
        self.r_max = r_max
        r_min, r_max = self.calculate_min_max(**init_kwargs)
        self.weight = nn.Parameter(
            torch.empty(channels, device=device, dtype=dtype).uniform_(r_min, r_max)
        )
    
    def get(self) -> Tensor:
        return self.forward(self.weight)


class LinearRadius(Radius):
    def calculate_min_max(self, r_min: float, r_max: float, **kwargs) -> tp.Tuple[float, float]:
        assert 0.0 <= r_min <= r_max <= self.r_max
        return r_min, r_max

    def forward(self, x: Tensor) -> Tensor:
        return x.clamp(min=-self.r_max, max=self.r_max)


class TanhRadius(Radius):
    def calculate_min_max(self, r_min: float, r_max: float, **kwargs) -> tp.Tuple[float, float]:
        assert 0.0 <= r_min <= r_max <= self.r_max
        r_min = math.atanh(r_min / self.r_max)
        r_max = math.atanh(r_max / self.r_max)
        return r_min, r_max

    def forward(self, x: Tensor) -> Tensor:
        return self.r_max * torch.tanh(x)


def inv_sigmoid(x: float) -> float:
    return math.log(x / (1 - x))


class SigmoidRadius(Radius):
    def calculate_min_max(self, r_min: float, r_max: float, **kwargs) -> tp.Tuple[float, float]:
        assert 0.0 <= r_min <= r_max <= self.r_max
        r_min = inv_sigmoid(r_min / self.r_max)
        r_max = inv_sigmoid(r_max / self.r_max)
        return r_min, r_max

    def forward(self, x: Tensor) -> Tensor:
        return self.r_max * torch.sigmoid(x)


def inv_exp_exp(x: float) -> float:
    return math.log(-math.log(x))


class ExpExpRadius(Radius):
    def calculate_min_max(self, r_min: float, r_max: float, **kwargs) -> tp.Tuple[float, float]:
        assert 0.0 < r_min <= r_max < self.r_max
        r_min = inv_exp_exp(r_min / self.r_max)
        r_max = inv_exp_exp(r_max / self.r_max)
        return r_max, r_min     # decreasing function -> reverse the order

    def forward(self, x: Tensor) -> Tensor:
        return self.r_max * torch.exp(-torch.exp(x))


class Theta(nn.Module):
    def __init__(
        self, channels: int, init_kwargs: tp.Dict[str, tp.Any],
        device=None, dtype=None
    ) -> None:
        super().__init__()
        theta = torch.empty(channels, device=device, dtype=dtype)
        self.initialize_theta(theta, **init_kwargs)
        self.weight = nn.Parameter(theta)
    
    def initialize_theta(
        self, theta: Tensor, theta_min: tp.Optional[float] = None,
        theta_max: tp.Optional[float] = None, freq_min: tp.Optional[int] = 20,
        freq_max: tp.Optional[int] = 12000, fs: tp.Optional[int] = 24000,
        theta_log: bool = True, **kwargs
    ) -> None:
        if theta_min is None or theta_max is None:
            theta_min = 2 * math.pi * freq_min / fs
            theta_max = 2 * math.pi * freq_max / fs
        assert 0 <= theta_min <= theta_max <= math.pi
        if theta_log:
            assert theta_min > 0
            theta_min = math.log(theta_min)
            theta_max = math.log(theta_max)
        theta.uniform_(theta_min, theta_max)
        if theta_log:
            theta.exp_()
        
        # Since theta is a phase, it is randomly flipped.
        prob = torch.ones_like(theta) * 0.5
        mask = torch.bernoulli(prob)
        theta.mul_(2 * mask - 1)
    
    def get(self) -> Tensor:
        return self.weight


@jit.script
def calculate_denom_scale(r: Tensor, theta: Tensor) -> tp.Tuple[Tensor, Tensor]:
    r_square = r.square()
    denom = torch.stack([-r_square, 2*torch.cos(theta)*r], dim=1)    # [C, 2]
    scale = (1 - r) * (1 - 2*r*torch.cos(2*theta) + r_square).sqrt()
    return denom, scale


@jit.script
def calculate_denom_scale_delay(r: Tensor, theta: Tensor) -> tp.Tuple[Tensor, Tensor, Tensor]:
    r_square = r.square()
    denom = torch.stack([-r_square, 2*torch.cos(theta)*r], dim=1)    # [C, 2]
    r_cos_2theta = r * torch.cos(2 * theta)
    tmp = 1 - 2*r_cos_2theta + r_square
    scale = (1 - r) * tmp.sqrt()
    delay = r / (1 - r) + (r_cos_2theta - r_square) / tmp
    return denom, scale, delay.to(torch.int32) + 1


DEFAULT_INIT_KWARGS = dict(
    r_min = 0.0,
    r_max = 0.9,
    theta_min = 1e-2,
    theta_max = 3.141592,
    freq_min = None,
    freq_max = None,
    fs = None,
    theta_log = True,
)


class IIR2O(nn.Module):
    ''' 2nd-order IIR Layer.
    More specifically, it is composed of a 2nd-order all-pole layer (cuda kernel implemented),
    followed by a 2nd-order all-zero layer (Depthwise Conv1d with kernel_size=3).
    Arguments:
        channels(int): Number of channels.
        bias(bool): If True, bias is added.
        bidirectional(bool): If True, forward and backward iir layers are learned.
            More specifically, it is composed of 3 layers:
                - 2nd-order forward all-pole layer
                - 2nd-order backward all-pole layer
                - 4th-order all-zero layer (Depthwise Conv1d with kernel_size=5 and padding=2)
                    (Efficiently model a cascade of two 2nd-order all-zero layers)
        filtfilt(bool): If True, apply filtfilt operation.
            1. Zero-pad input to the right
            2. Forward all-pole iir filtering
            3. Backward all-pole iir filtering with the same weight
                (If bidirectional is set True, the backward uses a different filter
                from a forward filter. Otherwise, backward/forward use the same filter.)
            4. Forward all-zero filtering (Depthwise Conv1d with kernel_size=5 and padding=2)
            5. Discard the right part of the output to match the length of the input.
        unpad_delay(bool): If True, unpad group_delay to the input.
            1. Calculate group_delay for each channel
            2. Pad max(group_delay) to the input
            3. All-pole iir filtering
            4. Unpad group_delay for each channel
            5. All-zero fir filtering
        r_max(float): Maximum pole radius. 0 < r_max <= 1.
        r_activation(str): Activation function for pole radius.
            linear: Linear + clipping to [-r_max, r_max]
            tanh: r_max * tanh(x)
            sigmoid: r_max * sigmoid(x)
            exp-exp: r_max * exp(-exp(x))
        init_kwargs(dict): Initialization parameters.
            Note that only one of 'theta' and 'freq' is used.
            'freq' is ignored if 'theta' is used.
            If 'freq' is used, 'theta_min' and 'theta_max' are internally calculated
            using 'freq_min', 'freq_max', and 'fs'.
            Arguments:
                r_min(float): Min(pole radius).
                r_max(float): Max(pole radius).
                theta_min(float|None): Min(pole angle).
                theta_max(float|None): Max(pole angle).
                freq_min(int|None): Min(frequency).
                freq_max(int|None): Max(frequency).
                fs(int|None): Sampling frequency.
                theta_log(bool): If True, theta (or freq) is log-uniformly distributed.
    '''
    def __init__(
        self,
        channels: int,
        bias: bool = True,
        bidirectional: bool = False,
        filtfilt: bool = False,
        unpad_delay: bool = False,
        r_max: float = 1.0,
        r_activation: str = "tanh",
        fir_weight_norm: bool = True,
        fir_weight_norm_kwargs: tp.Dict[str, tp.Any] = dict(),
        fir_init: str = "linear",   # linear | relu | ""
        init_kwargs: tp.Dict[str, tp.Any] = dict(),
        device = None,
        dtype = None,
        _torch_native: bool = False,    # for debugging
    ):
        assert 0.0 < r_max <= 1.0
        assert bidirectional + filtfilt + unpad_delay <= 1, \
            f"Only one of bidirectional, filtfilt, and unpad_delay can be True, " +\
            f"but got bidirectional: {bidirectional}, filtfilt: {filtfilt}, " +\
            f"unpad_delay: {unpad_delay}."
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.bidirectional = bidirectional
        self.filtfilt = filtfilt
        self.unpad_delay = unpad_delay
        self.channels = channels
        self._torch_native = _torch_native
        
        default_init_kwargs = DEFAULT_INIT_KWARGS.copy()
        default_init_kwargs.update(init_kwargs)
        init_kwargs = default_init_kwargs
        
        if bidirectional:
            channels = 2 * channels
        
        if r_activation == "linear":
            self.radius = LinearRadius(channels, r_max, init_kwargs, **factory_kwargs)
        elif r_activation == "tanh":
            self.radius = TanhRadius(channels, r_max, init_kwargs, **factory_kwargs)
        elif r_activation == "sigmoid":
            self.radius = SigmoidRadius(channels, r_max, init_kwargs, **factory_kwargs)
        elif r_activation == "exp-exp":
            self.radius = ExpExpRadius(channels, r_max, init_kwargs, **factory_kwargs)
        else:
            raise ValueError(f"Unknown IIR.r_activation: {r_activation}")
        self.theta = Theta(channels, init_kwargs, **factory_kwargs)
        
        self.register_buffer("weight_iir", torch.empty(0))
        self.register_buffer("scale_iir", torch.empty(0))
        # self.register_buffer("delay_iir", torch.empty(0))
        self.weight_iir: Tensor
        self.scale_iir: Tensor
        # self.delay_iir: Tensor
        
        kernel_size = 5 if filtfilt else 3
        fir = nn.Conv1d(channels, channels, kernel_size,
                        groups=channels, bias=bias, **factory_kwargs)
        
        # fir initialization
        if fir_init:
            nn.init.kaiming_normal_(fir.weight.data, nonlinearity=fir_init)
            if fir.bias is not None:
                fir.bias.data.zero_()
        
        if fir_weight_norm:
            self.fir = weight_norm(fir, **fir_weight_norm_kwargs)
        else:
            self.fir = fir
    
    def calculate_denom_scale_radius(self) -> tp.Tuple[Tensor, Tensor, Tensor]:
        if self.weight_iir.numel() > 0:
            return self.weight_iir, self.scale_iir, self.radius.get()
        r = self.radius.get()
        theta = self.theta.get()
        return *calculate_denom_scale(r, theta), r
    
    def calculate_denom_scale_delay(self) -> tp.Tuple[Tensor, Tensor, Tensor]:
        if self.weight_iir.numel() > 0:
            return self.weight_iir, self.scale_iir, self.delay
        r = self.radius.get()
        theta = self.theta.get()
        return calculate_denom_scale_delay(r, theta)
    
    def remove_weight_parametrization(self) -> None:
        if self.unpad_delay:
            weight_iir, scale_iir, delay = self.calculate_denom_scale_delay()
            delattr(self, "delay_iir")
            self.delay_iir = delay
        else:
            weight_iir, scale_iir, _ = self.calculate_denom_scale_radius()
        delattr(self, "weight_iir")
        delattr(self, "scale_iir")
        self.weight_iir = nn.Parameter(weight_iir)
        self.scale_iir = nn.Parameter(scale_iir)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, T]
        if self.unpad_delay:
            denom, scale, delay = self.calculate_denom_scale_delay()
            max_delay = delay.max().item()
            x = F.pad(x, (0, max_delay))    # [B, C, T+max_delay]
        else:
            denom, scale, radius = self.calculate_denom_scale_radius()
            if self.bidirectional:
                x = torch.cat((x, x.flip(2)), dim=1)
                scale = scale / math.sqrt(2)
            elif self.filtfilt:
                T = x.size(2)
                P = int(math.log(0.05) / math.log(radius.detach().max().item()))
                x = F.pad(x, (0, P))

        if self._torch_native:
            B, C = x.size(0), x.size(1)
            y = iir_default(x, denom, scale, x.new_zeros(B, C, 2))
        else:
            y = iir(x, denom, scale)    # [B, C, 2+T]
        
        if self.filtfilt:
            y = y.flip(2)               # [B, C, padding+T+2]
            y = iir(y, denom, scale)    # [B, C, 2+padding+T+2]
            y = y[:, :, -T-4:].flip(2)  # [B, C, 2+T+2]
        elif self.unpad_delay:
            y = unpad_delay(y, delay, max_delay)    # [B, C, 2+T+max_delay] -> [B, C, 2+T]

        y = self.fir(y)    # [B, C, T]
        
        if self.bidirectional:
            y = y[:, :self.channels, :] + y[:, self.channels:, :].flip(2)
        return y


def debug_diff(a: Tensor, b: Tensor, title: str = "") -> None:
    EPS = 1e-12
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    value, idx = ((a - b).abs() / (b.abs() + EPS)).max(dim=0)
    print(title)
    print("\trelative diff: ", value.item(), a[idx].item(), b[idx].item())
    value, idx = (a - b).abs().max(dim=0)
    print("\tabsolute diff: ", value.item(), a[idx].item(), b[idx].item())


def simple_forward_test():
    dtype = torch.float32
    device = 'cuda'
    B, C, T = 1, 2, 7
    x = torch.arange(T, dtype=dtype, device=device).view(1, 1, T).expand(B, C, T)
    x = x.contiguous()
    for i in range(1, C):
        x[:, i, :] *= (i + 1)
    print("=== input ===")
    print(x)
    weight = torch.ones(C, 2, device=device, dtype=dtype)
    weight[:, 0] = -1
    for i in range(1, C):
        weight[i] *= i / C
    scale = torch.ones(C, device=device, dtype=dtype)
    
    out = iir_default(x, weight, scale)
    print("=== pytorch native ===")
    print(out)
    
    out = iir(x, weight, scale)
    torch.cuda.synchronize()
    print("=== custom cuda kernel ===")
    print(out[:, :, 2:])


def simple_test():
    dtype = torch.float32
    device = 'cuda'
    B, C, T = 1, 2, 7
    x = torch.arange(T, dtype=dtype, device=device).view(1, 1, T).expand(B, C, T)
    x = x.contiguous()
    for i in range(1, C):
        x[:, i, :] *= (i + 1)
    print("=== input ===")
    print(x)
    x.requires_grad_(True)
    x.retain_grad()
    weight = torch.ones(C, 2, device=device, dtype=dtype)
    weight[:, 0] = -1
    for i in range(1, C):
        weight[i] *= i / C
    weight.requires_grad_(True)
    weight.retain_grad()
    scale = torch.ones(C, device=device, dtype=dtype)
    for i in range(1, C):
        scale[i] /= (i+1)
    scale.requires_grad_(True)
    scale.retain_grad()
    
    out = iir_default(x, weight, scale)
    print("=== out, weight.grad, scale.grad, x.grad ===")
    print(out)
    out.square().sum().backward()
    print(weight.grad.view(-1))
    print(scale.grad)
    print(x.grad)
    o = out.detach().clone()
    wg = weight.grad.data.clone()
    sg = scale.grad.data.clone()
    xg = x.grad.data.clone()
    
    x.grad = None
    weight.grad = None
    scale.grad = None
    out = iir(x, weight, scale)
    torch.cuda.synchronize()
    print("=== out, weight.grad, scale.grad, x.grad ===")
    print(out[:, :, 2:])
    out.square().sum().backward()
    print(weight.grad.view(-1))
    print(scale.grad)
    print(x.grad)
    assert torch.allclose(o, out[:, :, 2:])
    assert torch.allclose(weight.grad, wg)
    assert torch.allclose(scale.grad, sg)
    assert torch.allclose(x.grad, xg)


def random_test():
    import matplotlib.pyplot as plt
    
    dtype = torch.float32
    device = 'cuda'
    factory_kwargs = {"dtype": dtype, "device": device}
    B, C, T = 5, 11, 1601
    x = torch.randn(B, C, T, **factory_kwargs)
    x.requires_grad_(True)
    x.retain_grad()
    
    iir_layer = IIR(C, **factory_kwargs)
    with torch.no_grad():
        weight, scale = iir_layer.calculate_denom_scale()
    weight = weight.data.clone().requires_grad_(True)
    weight.retain_grad()
    scale = scale.data.clone().requires_grad_(True)
    scale.retain_grad()

    out = iir_default(x, weight, scale)
    out.square().mean().backward()
    o = out.detach().clone()
    wg = weight.grad.clone()
    sg = scale.grad.clone()
    xg = x.grad.clone()
    assert torch.all(torch.isfinite(o))
    assert torch.all(torch.isfinite(wg))
    assert torch.all(torch.isfinite(sg))
    assert torch.all(torch.isfinite(xg))
    plot_idx = 0
    plt.plot(o[0, plot_idx, :].detach().cpu().numpy(), label='out_defalt')
    
    x.grad = None
    weight.grad = None
    scale.grad = None
    with torch.cuda.amp.autocast():
        out = iir(x.half(), weight, scale)[:, :, 2:]
    torch.cuda.synchronize()
    out.square().mean().backward()
    torch.cuda.synchronize()
    
    # assert torch.allclose(o, out, atol=1e-3)
    # assert torch.allclose(x.grad, xg, atol=1e-3)
    # assert torch.allclose(weight.grad, wg, atol=1e-3), f"{weight.grad.view(-1)}\n{wg.view(-1)}"
    debug_diff(o, out, "output")
    debug_diff(x.grad, xg, "x.grad")
    debug_diff(weight.grad, wg, "weight.grad")
    debug_diff(scale.grad, sg, "scale.grad")
    plt.plot((out[0, plot_idx, :].detach()-o[0, plot_idx, :].detach()).cpu().numpy(), label='diff')
    plt.legend()
    plt.savefig("delete_it.png")


def iir_module_test():
    device = 'cuda'
    dtype = torch.float32
    factory_kwargs = {"device": device, "dtype": dtype}
    B, C, T = 33, 11, 111
    for idx in tqdm(range(10)):
        x = torch.randn(B-idx, C, T, requires_grad=True, **factory_kwargs)
        x.retain_grad()
        
        iir1 = IIR(C, False, **factory_kwargs)
        iir2 = IIR(C, False, _torch_native=True, **factory_kwargs)
        with torch.no_grad():
            iir2.load_state_dict(iir1.state_dict())
        
        y1 = iir1(x)
        y1.square().mean().backward()
        xg = x.grad.clone()
        x.grad = None
        
        y2 = iir2(x)
        y2.square().mean().backward()
        assert torch.all(torch.isfinite(y2.data))
        assert torch.allclose(y1.data, y2.data, rtol=1e-3)
        assert torch.allclose(iir1.radius.weight.grad, iir2.radius.weight.grad, rtol=1e-3)
        assert torch.allclose(iir1.theta.weight.grad, iir2.theta.weight.grad, rtol=1e-3)
        assert torch.allclose(iir1.fir.weight_g.grad, iir2.fir.weight_g.grad, rtol=1e-3)
        assert torch.allclose(iir1.fir.weight_v.grad, iir2.fir.weight_v.grad, rtol=1e-3)
        assert torch.allclose(xg, x.grad, rtol=1e-3)


def train_speed_test():
    device = 'cuda'
    dtype = torch.float32
    warmup, iterations = 5, 10
    B = 32
    Cs = [32, 64, 128, 256]
    Ts = [24000, 12000, 3000, 600]
    bidirectional = True
    
    factory_kwargs = {"device": device, "dtype": dtype}
    
    fir_total, iir_total = 0., 0.
    for idx in range(4):
        C = Cs[idx]
        T = Ts[idx]
        print(f"==== C: {C}, T: {T} ====")
        x = torch.randn(B, C, T, requires_grad=True, **factory_kwargs)
        x.retain_grad()
        iir = IIR(C, bidirectional=bidirectional, **factory_kwargs)
        kernel_size = 9 if bidirectional else 5
        fir = nn.Conv1d(C, C, kernel_size, groups=C,
                        padding=(kernel_size-1)//2, **factory_kwargs)
        ellapsed = 0.
        for i in range(warmup + iterations):
            start = time.perf_counter()
            y = iir(x)
            y.square().mean().backward()
            torch.cuda.synchronize()
            if i >= warmup:
                ellapsed += time.perf_counter() - start
                iir_total += ellapsed
        print(f"iir: {ellapsed:.3f} s")
        
        ellapsed = 0.
        for i in range(iterations):
            start = time.perf_counter()
            y = fir(x)
            y.square().mean().backward()
            torch.cuda.synchronize()
            ellapsed += time.perf_counter() - start
            fir_total += ellapsed
        print(f"fir: {ellapsed:.3f} s")
    print(f"Total - iir: {iir_total:.3f} s / fir: {fir_total:.3f} s")


def simple_delay_test():
    device = 'cuda'
    dtype = torch.float32
    B, C, T = 2, 2, 10
    factory_kwargs = {"device": device, "dtype": dtype}
    x = torch.arange(1, T+1, **factory_kwargs).view(1, 1, T) * \
        torch.arange(1, C+1, **factory_kwargs).view(1, C, 1)
    x = x.expand(B, C, T).contiguous()
    delay = (torch.randn(C, **factory_kwargs) * 3).abs().to(torch.int32) + 1
    output = torch.empty(B, C, T+delay.max().item(), **factory_kwargs)
    delay_cuda.pad(x, output, delay)
    print("input:", x)
    print("delay:", delay)
    print("padded:", output)
    y = torch.empty(B, C, T, **factory_kwargs)
    delay_cuda.unpad(output, y, delay)
    print("unpadded:", y)


def random_delay_test():
    device = 'cuda'
    dtype = torch.float16
    factory_kwargs = {"device": device, "dtype": dtype}
    B, C, T = 5, 11, 1601
    x = torch.randn(B, C, T, **factory_kwargs)
    x.requires_grad_(True)
    x.retain_grad()
    
    delay = (torch.randn(C, **factory_kwargs) * 20).abs().to(torch.int32) + 1
    max_delay = delay.max().item()
    
    out = unpad_delay_torch(x, delay, max_delay)
    out.square().mean().backward()
    o = out.detach().clone()
    xg = x.grad.clone()
    x.grad = None
    
    out = unpad_delay(x, delay, max_delay)
    out.square().mean().backward()
    debug_diff(o, out, "output")
    debug_diff(x.grad, xg, "x.grad")


def speed_delay_test():
    device = 'cuda'
    dtype = torch.float16
    warmup, iterations = 5, 10
    B = 32
    Cs = [32, 64, 128, 256]
    Ts = [24000, 12000, 3000, 600]
    bidirectional = True
    
    factory_kwargs = {"device": device, "dtype": dtype}
    
    fir_total, iir_total = 0., 0.
    for idx in range(4):
        C = Cs[idx]
        T = Ts[idx]
        print(f"==== C: {C}, T: {T} ====")
        x = torch.randn(B, C, T, requires_grad=True, **factory_kwargs)
        x.retain_grad()
        delay = torch.randn(C, **factory_kwargs).abs().mul(55).to(torch.int32) + 1
        max_delay = delay.max().item()
        ellapsed = 0.
        for i in range(warmup + iterations):
            start = time.perf_counter()
            y = unpad_delay(x, delay, max_delay)
            y.square().mean().backward()
            torch.cuda.synchronize()
            if i >= warmup:
                ellapsed += time.perf_counter() - start
                iir_total += ellapsed
        print(f"delay: {ellapsed:.3f} s")
        
        ellapsed = 0.
        for i in range(iterations):
            start = time.perf_counter()
            y = x[:, :, 3:-max_delay+3]
            y.square().mean().backward()
            torch.cuda.synchronize()
            ellapsed += time.perf_counter() - start
            fir_total += ellapsed
        print(f"pad: {ellapsed:.3f} s")
    print(f"Total - delay: {iir_total:.3f} s / pad: {fir_total:.3f} s")

if __name__ == "__main__":
    # simple_forward_test()
    # simple_test()
    # random_test()
    # iir_module_test()
    train_speed_test()
