import typing as tp
import time
import math

import torch
from torch import distributed as dist
from torch import jit
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm


_EXTENSION_AVAILABLE = False
try:
    from .kernels import iir_cuda
    _EXTENSION_AVAILABLE = True
except ImportError:
    pass
try:
    from kernels import iir_cuda
    _EXTENSION_AVAILABLE = True
except ImportError:
    pass

if not _EXTENSION_AVAILABLE:
    raise RuntimeError("IIR cuda extension not found.")


def iir_torch(x: Tensor, denom_flipped: Tensor, out: Tensor) -> None:
    N = denom_flipped.size(1)
    if N == 1:
        iir_torch_first_order(x, denom_flipped, out)
    elif N == 2:
        iir_torch_second_order(x, denom_flipped, out)
    else:
        raise RuntimeError(denom_flipped.shape)


@jit.script
def iir_torch_second_order(x: Tensor, denom_flipped: Tensor, out: Tensor) -> None:
    '''input:
        x: [T, C, B], denom_flipped: [C, 2]
    output: [2+T, C, B]
    In-place calculation. Faster, but backward is not supported.'''
    T = x.size(0)
    assert x.size(1) == denom_flipped.size(0) == out.size(1)
    assert denom_flipped.size(1) == 2
    assert x.size(2) + 2 == out.size(0)
    
    # reshape
    denom_flipped = denom_flipped.unsqueeze(1)     # [C, 1, 2]
    x = x.transpose(0, 1)       # [C, T, B]
    out = out.transpose(0, 1)   # [C, 2+T, B]
    
    # iir filtering
    for t in range(0, T):
        torch.baddbmm(  # [C, 1, B] + [C, 1, 2] @ [C, 2, B] = [C, 1, B]
            x[:, t:t+1, :],             # [C, 1, B]
            denom_flipped,              # [C, 1, 2]
            out[:, t:t+2, :],           # [C, 2, B]
            out=out[:, t+2:t+3, :]      # [C, 1, B]
        )


@jit.script
def iir_torch_first_order(x: Tensor, denom_flipped: Tensor, out: Tensor) -> None:
    '''input:
        x: [T, C, B], denom_flipped: [C, 1]
    output: [1+T, C, B]
    In-place calculation. Faster, but backward is not supported.'''
    T = x.size(0)
    assert x.size(1) == denom_flipped.size(0) == out.size(1)
    assert denom_flipped.size(1) == 1
    assert x.size(0) + 1 == out.size(0)
    
    # reshape
    denom_flipped = denom_flipped.unsqueeze(1)     # [C, 1, 2]
    x = x.transpose(0, 1)       # [C, T, B]
    out = out.transpose(0, 1)   # [C, 1+T, B]
    
    # iir filtering
    for t in range(0, T):
        torch.baddbmm(  # [C, 1, B] + [C, 1, 1] @ [C, 1, B] = [C, 1, B]
            x[:, t:t+1, :],             # [C, 1, B]
            denom_flipped,              # [C, 1, 1]
            out[:, t:t+1, :],           # [C, 1, B]
            out=out[:, t+1:t+2, :]      # [C, 1, B]
        )


def iir_default(x: Tensor, denom_flipped: Tensor, out: Tensor) -> Tensor:
    N = denom_flipped.size(1)
    if N == 1:
        return iir_default_first_order(x, denom_flipped, out)
    elif N == 2:
        return iir_default_second_order(x, denom_flipped, out)
    else:
        raise RuntimeError(denom_flipped.shape)


def iir_default_second_order(x: Tensor, denom_flipped: Tensor, out: Tensor) -> Tensor:
    '''input:
        x: [B, C, T], denom_flipped: [C, 2], out: [B, C, 2+T]
    output: [B, C, 2+T]
    Slower, but backward is supported.'''
    B, C, T = x.shape
    denom_flipped = denom_flipped.reshape(C, 2, 1).transpose(0, 1)    # [2, C, 1]
    x = x.transpose(0, 2).contiguous()      # [T, C, B]
    
    cache = out[:, :, :2].transpose(0, 2)   # [2, C, B]
    out_list = [cache]
    for t in range(T):
        _x = (cache * denom_flipped).sum(0, keepdim=True)   # [2, C, B], [2, C, 1] -> [1, C, B]
        _x = _x + x[t:t+1, :, :]                            # [1, C, B]
        out_list.append(_x)
        cache = torch.cat([cache[1:, :, :], _x], dim=0)     # [2, C, B]
    out = torch.cat(out_list, dim=0)            # [2+T, C, B]
    out = out.transpose(0, 2)                   # [B, C, 2+T]
    return out


def iir_default_first_order(x: Tensor, denom_flipped: Tensor, out: Tensor) -> Tensor:
    '''input:
        x: [B, C, T], denom_flipped: [C, 1], out: [B, C, 1+T]
    output: [B, C, 1+T]
    Slower, but backward is supported.'''
    B, C, T = x.shape
    denom_flipped = denom_flipped.reshape(C, 1, 1).transpose(0, 1)    # [1, C, 1]
    x = x.transpose(0, 2).contiguous()      # [T, C, B]
    
    cache = out[:, :, :1].transpose(0, 2)   # [1, C, B]
    out_list = [cache]
    for t in range(T):
        _x = cache * denom_flipped  # [1, C, B], [1, C, 1] -> [1, C, B]
        _x = _x + x[t:t+1, :, :]    # [1, C, B]
        out_list.append(_x)
        cache = _x                  # [1, C, B]
    out = torch.cat(out_list, dim=0)            # [1+T, C, B]
    out = out.transpose(0, 2)                   # [B, C, 1+T]
    return out


def iir_backward_grad_weight(out: Tensor, grad_out: Tensor, weight: Tensor) -> Tensor:
    '''input:
        out: [B, C, N+T]
        grad_out: [B, C, T]
        weight: [C, N]
    output: [C, N]
    
    Assume that N = 2. Then,
    out[t:t+2] * weight + in = out[t+2]
    <=> out[2:] = conv(out[:-1], weight) + in
    Therefore, grad_weight = conv_backward(grad_out, out[:-1], weight))
    ex)    o2          o6 : output
            ↑           ↑
        w0 w1       w0 w1 : weight convolution
        o0 o1 o2 o3 o4 o5 : input
    <=> o2~o6 = conv(o0~o5, w)
    <=> grad_weight = conv_backward(grad_out, o0~o5, w)'''
    C = out.size(1)
    N = weight.size(1)
    return torch.ops.aten.convolution_backward(     # type: ignore
        grad_out,               # grad_output
        out[:, :, :-1],         # input
        weight.unsqueeze(1),    # weight
        None,                   # bias_size
        (1,),                   # stride
        (0,),                   # padding
        (1,),                   # dilation
        False,                  # transposed
        (0,),                   # output_padding
        C,                      # groups
        (False, True, False)    # mask: (input, weight, bias)
    )[1].squeeze(1)


class IIRTorchFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, out: Tensor) -> Tensor:
        # input:
        #   x: [B, C, T]
        #   weight: [C, N]
        #   out: [B, C, N+T]
        xt = x.transpose(0, 2)      # [T, C, B]
        out = out.transpose(0, 2)   # [N+T, C, B]
        iir_torch(xt, weight, out)
        out = out.transpose(0, 2)   # [B, C, N+T]
        ctx.save_for_backward(weight, out)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tp.Tuple[Tensor, tp.Optional[Tensor], None]:
        # grad_out: [B, C, N+T]
        weight, out = ctx.saved_tensors
        B, C, T_N = out.shape
        N = weight.size(1)
        T = T_N - N
        grad_out = grad_out[:, :, N:].flip(2).transpose(0, 2)   # [T, C, B]

        grad_in = torch.zeros(T_N, C, B, device=out.device, dtype=out.dtype)
        iir_torch(grad_out, weight, grad_in)                    # [N+T, C, B]
        grad_in = grad_in.flip(0).transpose(0, 2)[:, :, :T]     # [B, C, T]
        if ctx.needs_input_grad[1]:
            grad_weight = iir_backward_grad_weight(out, grad_in, weight)
        else:
            grad_weight = None

        return grad_in, grad_weight, None


class IIRCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, out: Tensor) -> Tensor:
        # input:
        #   x: [B, C, T]
        #   weight: [C, N]
        #   out: [B, C, N+T]
        # return output with inplace operation
        iir_cuda.forward(x, weight, out)    # [B, C, N+T]
        ctx.save_for_backward(weight, out)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tp.Tuple[Tensor, tp.Optional[Tensor], None]:
        # grad_out: [B, C, N+T]
        weight, out = ctx.saved_tensors
        B, C, T_N = out.shape
        N = weight.size(1)
        T = T_N - N
        grad_out = grad_out[:, :, N:].flip(2)       # [B, C, T]

        grad_in = torch.zeros(B, C, T_N, device=out.device, dtype=out.dtype)
        iir_cuda.forward(grad_out, weight, grad_in) # [B, C, N+T]
        grad_in = grad_in.flip(2)[:, :, :T]         # [B, C, T]
        if ctx.needs_input_grad[1]:
            grad_weight = iir_backward_grad_weight(out, grad_in, weight)
        else:
            grad_weight = None

        return grad_in, grad_weight, None


def iir(x: Tensor, weight: Tensor, out: Tensor) -> Tensor:
    ''' inputs:
        x: [B, C, T]
        weight: [C, N]
        out: [B, C, N+T]
    output: [B, C, T]
    '''
    if x.is_cuda and _EXTENSION_AVAILABLE:
        x = IIRCudaFunction.apply(x, weight, out)
    else:
        x = IIRTorchFunction.apply(x, weight, out)
    N = weight.size(1)
    return x[:, :, N:]


def inv_sigmoid(x: float) -> float:
    assert 0. < x < 1.
    return math.log(x / (1 - x))


class EMAFixed(nn.Module):
    '''First-order IIR filter layer.
    weight: [C]
    y[:, :, i+1] = y[:, :, i] * gamma + (1 - gamma) * x[:, :, i+1]
    '''
    
    def __init__(
        self,
        channels: int,
        gamma: float = 0.9,
        reversed: bool = False,
        use_cache: bool = False,
        device = None,
        dtype = torch.float32,
    ):
        assert 0 < gamma <= 1.0, gamma
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.channels = channels
        self.reversed = reversed
        
        weight = torch.empty(channels, **factory_kwargs).fill_(gamma)
        self.register_buffer("weight", weight)
        self.weight: Tensor
        
        self.cache = torch.empty(0)
        self.use_cache = use_cache
    
    def empty_cache(self):
        self.cache = torch.empty(0)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, T]
        gamma = self.weight.unsqueeze(1)
        
        x = x * (1.0 - gamma.view(1, self.channels, 1))     # [B, C, T]
        if self.reversed:
            x = x.flip(2)
            
        out = x.new_zeros(x.size(0), self.channels, x.size(2)+1)    # [B, C, T+1]
        B = min(self.cache.size(0), x.size(0))
        if B > 0:
            out[:B, :, :1] = self.cache[:B, :, :1]
        x = iir(x, gamma, out)       # [B, C, T]
        if self.use_cache:
            self.cache = x.detach()[:, :, -1:]
        if self.reversed:
            x = x.flip(2)
        return x


class EMA(nn.Module):
    '''First-order IIR filter layer.
    weight: [C]
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
        use_cache: bool = False,
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
        
        self.cache = torch.empty(0)
        self.use_cache = use_cache
    
    def empty_cache(self):
        self.cache = torch.empty(0)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, T]
        gamma = self.r_max * torch.sigmoid(self.weight)     # [C]
        gamma = gamma.unsqueeze(1)  # [C, 1]
        
        x = x * (1.0 - gamma.view(1, self.channels, 1))     # [B, C, T]
        if self.reversed:
            x = x.flip(2)
        
        out = x.new_zeros(x.size(0), self.channels, x.size(2)+1)    # [B, C, T+1]
        B = min(self.cache.size(0), x.size(0))
        if B > 0:
            out[:B, :, :1] = self.cache[:B, :, :1]
        x = iir(x, gamma, out)       # [B, C, T]
        if self.use_cache:
            self.cache = x.detach()[:, :, -1:]
        if self.reversed:
            x = x.flip(2)
        return x


class EMATorch(EMA):
    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, T]
        gamma = self.r_max * torch.sigmoid(self.weight)     # [C]
        gamma = gamma.unsqueeze(1)  # [C, 1]
        
        x = x * (1.0 - gamma.view(1, self.channels, 1))     # [B, C, T]
        if self.reversed:
            x = x.flip(2)
        
        out = x.new_zeros(x.size(0), self.channels, x.size(2)+1)    # [B, C, T+1]
        B = min(self.cache.size(0), x.size(0))
        if B > 0:
            out[:B, :, :1] = self.cache[:B, :, :1]
        x = iir_default(x, gamma, out)[:, :, 1:]    # [B, C, T]
        if self.use_cache:
            self.cache = x.detach()[:, :, -1:]
        if self.reversed:
            x = x.flip(2)
        return x


def debug_diff(a: Tensor, b: Tensor, title: str = "") -> None:
    EPS = 1e-12
    a = a.reshape(-1)
    b = b.reshape(-1)
    value, idx = ((a - b).abs() / (b.abs() + EPS)).max(dim=0)
    print(title)
    print("\trelative diff: ", value.item(), a[idx].item(), b[idx].item())
    value, idx = (a - b).abs().max(dim=0)
    print("\tabsolute diff: ", value.item(), a[idx].item(), b[idx].item())


def simple_test():
    dtype = torch.float64
    device = 'cuda'
    B, C, T, N = 1, 1, 11, 1
    Start = 0.1
    
    x = torch.arange(T, dtype=dtype, device=device).view(1, 1, T).expand(B, C, T)
    x = x.contiguous()
    for i in range(1, C):
        x[:, i, :] *= (i + 1)
    x.requires_grad_(True)
    x.retain_grad()
    
    weight = torch.ones(C, N, device=device, dtype=dtype) * 0.9
    if weight.size(1) > 1:
        weight[:, :, 0] = 0
    for i in range(1, C):
        weight[i] *= i / C
    weight.requires_grad_(True)
    weight.retain_grad()
    
    out = torch.zeros(B, C, N+T, dtype=dtype, device=device)
    out[:, :, 0] = Start
    out = iir_default(x, weight, out)[:, :, N:]
    print("out default", out)
    out.square().sum().backward()
    print("w_g default", weight.grad)
    print("x_g default", x.grad)
    o = out.detach().clone()
    wg = weight.grad.data.clone()
    xg = x.grad.data.clone()
    
    out = torch.zeros(B, C, N+T, dtype=dtype, device=device)
    out[:, :, 0] = Start
    x.grad = None
    weight.grad = None
    out = iir(x, weight, out)
    torch.cuda.synchronize()
    print("o kernel", out)
    out.square().sum().backward()
    print("w_g kernel", weight.grad)
    print("x_g kernel", x.grad)
    assert torch.allclose(o, out)
    assert torch.allclose(weight.grad, wg)
    assert torch.allclose(x.grad, xg)


def random_test():
    dtype = torch.float64
    device = 'cuda'
    factory_kwargs = {"dtype": dtype, "device": device}
    B, C, T = 102, 27, 1601
    N, Start = 1, 0.1
    x = torch.randn(B, C, T, **factory_kwargs)
    x.requires_grad_(True)
    x.retain_grad()
    
    weight = torch.randn(C, N, **factory_kwargs)
    weight = torch.tanh(weight)
    weight.requires_grad_(True)
    weight.retain_grad()

    out = torch.zeros(B, C, N+T, dtype=dtype, device=device)
    out[:, :, 0] = Start
    out = iir_default(x, weight, out)
    o = out[:, :, N:]
    o.square().mean().backward()
    o = o.detach().clone()
    wg = weight.grad.clone()
    xg = x.grad.clone()
    assert torch.all(torch.isfinite(o))
    assert torch.all(torch.isfinite(wg))
    assert torch.all(torch.isfinite(xg))
    # plot_idx = 0
    # plt.plot(o[0, plot_idx, :].detach().cpu().numpy(), label='out_defalt')
    
    x.grad = None
    weight.grad = None
    out = torch.zeros(B, C, N+T, dtype=dtype, device=device)
    out[:, :, 0] = Start
    out = iir(x, weight, out)
    torch.cuda.synchronize()
    out.square().mean().backward()
    torch.cuda.synchronize()
    
    debug_diff(o, out, "output")
    debug_diff(x.grad, xg, "x.grad")
    debug_diff(weight.grad, wg, "weight.grad")
    # plt.plot((out[0, plot_idx, :].detach()-o[0, plot_idx, :].detach()).cpu().numpy(), label='diff')
    # plt.legend()
    # plt.savefig("delete_it.png")


def iir_module_test():
    device = 'cuda'
    dtype = torch.float64
    factory_kwargs = {"device": device, "dtype": dtype}
    B, C, T = 11, 17, 501
    for idx in tqdm(range(1)):
        if B-idx < 1:
            break
        x = torch.randn(B-idx, C, T, requires_grad=True, **factory_kwargs)
        x.retain_grad()
        
        iir1 = EMATorch(C, init_value=0.9, **factory_kwargs)
        iir2 = EMA(C, init_value=0.9, **factory_kwargs)
        with torch.no_grad():
            iir2.load_state_dict(iir1.state_dict())
        
        y1 = iir1(x)
        y1.square().mean().backward()
        xg = x.grad.clone()
        x.grad = None
        y2 = iir2(x)
        y2.square().mean().backward()
        assert torch.all(torch.isfinite(y2.data))
        # print(y1.squeeze(), y2.squeeze())
        # assert torch.allclose(y1.data, y2.data, rtol=1e-3)
        # assert torch.allclose(iir1.weight.grad, iir2.weight.grad, rtol=1e-3)
        # assert torch.allclose(xg, x.grad, rtol=1e-3)
        debug_diff(y1.data, y2.data, "output")
        debug_diff(iir1.weight.grad, iir2.weight.grad, "weight grad")
        debug_diff(xg, x.grad, "input grad")


def train_speed_test():
    device = 'cuda'
    dtype = torch.float32
    warmup, iterations = 5, 10
    B = 32
    Cs = [32, 64, 128, 256]
    Ts = [16000, 8000, 2000, 400]
    
    factory_kwargs = {"device": device, "dtype": dtype}
    
    fir_total, iir_total = 0., 0.
    for idx in range(4):
        C = Cs[idx]
        T = Ts[idx]
        print(f"==== C: {C}, T: {T} ====")
        x = torch.randn(B, C, T, requires_grad=True, **factory_kwargs)
        x.retain_grad()
        fir = nn.Conv1d(C, C, 2, padding=1, groups=C, bias=False, **factory_kwargs)
        # iir = EMA(C, r_max=0.93, **factory_kwargs)
        iir = EMAFixed(C, gamma=0.90, **factory_kwargs)
        ellapsed = 0.
        for i in range(warmup + iterations):
            start = time.time()
            y = iir(x)
            y.square().mean().backward()
            torch.cuda.synchronize()
            if i >= warmup:
                ellapsed += time.time() - start
        iir_total += ellapsed
        print(f"iir: {ellapsed:.3f} s")
        
        ellapsed = 0.
        for i in range(warmup + iterations):
            start = time.time()
            y = fir(x)
            y.square().mean().backward()
            torch.cuda.synchronize()
            if i >= warmup:
                ellapsed += time.time() - start
        fir_total += ellapsed
        print(f"fir: {ellapsed:.3f} s")
    print(f"Total - iir: {iir_total:.3f} s / fir: {fir_total:.3f} s")


if __name__ == "__main__":
    # simple_test()
    # random_test()
    iir_module_test()
    # train_speed_test()
