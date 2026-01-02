from typing import Tuple

import torch
from torch import Tensor

try:
    from .kernels import customquant_cuda
    _KERNEL_AVAILABLE = True
except ImportError:
    print("Warning: Customquant cuda kernel is not available. Using torch implementation.")
    _KERNEL_AVAILABLE = False

class CustomQuantCuda(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        scale: Tensor,
        q: float,
        inplace: bool = False,
    ) -> Tensor:
        assert scale.numel() == 1
        output, dx_ds, mask = customquant_cuda.forward(x, scale, q, inplace)
        ctx.save_for_backward(dx_ds, mask)
        return output

    @staticmethod
    def backward(
        ctx,
        grad_output: Tensor
    ) -> Tuple[Tensor, Tensor, None, None]:
        dx_ds, mask = ctx.saved_tensors
        return grad_output * mask, dx_ds.mul_(grad_output).sum().view(1), None, None


class RoundSTE(torch.autograd.Function):
    """ Straight-through estimator of rounding. """
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        return x.round()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output


def customquant_torch(x: Tensor, scale: Tensor, q: float) -> Tensor:
    with torch.amp.autocast("cuda", enabled=False):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.float()
        limit = scale.detach() * q
        clamped = torch.clamp(x, -limit, limit)
        rounded = RoundSTE.apply(clamped / scale)
        return rounded * scale


def customquant(x: Tensor, scale: Tensor, q: float, inplace: bool = False) -> Tensor:
    """ output = round(x / scale) * scale """
    if _KERNEL_AVAILABLE and x.is_cuda and scale.is_cuda:
        return CustomQuantCuda.apply(x.contiguous(), scale, q, inplace)
    else:
        return customquant_torch(x, scale, q)


if __name__ == "__main__":
    # cd ../
    # python -m quantization.customquant
    DTYPE = torch.float64
    import copy
    x1 = torch.randn(1024, device="cuda", dtype=DTYPE, requires_grad=True)
    x1.data.mul_(64)
    scale1 = torch.randn(1, device="cuda", dtype=torch.float32, requires_grad=True)
    scale1.data.add_(5).div_(5)

    x2 = torch.empty(1024, device="cuda", dtype=DTYPE, requires_grad=True)
    x2.data.copy_(x1.data)
    scale2 = torch.empty(1, device="cuda", dtype=torch.float32, requires_grad=True)
    scale2.data.copy_(scale1.data)

    q = float(2**7 - 1) - 0.25

    with torch.amp.autocast("cuda", enabled=True):
        y1 = x1 * 2
    output1 = customquant_torch(y1, scale1, q)
    output1 = output1.to(DTYPE)
    output1.retain_grad()
    output1.sum().backward()

    y2 = x2 * 2
    output2 = CustomQuantCuda.apply(y2, scale2, q, True)
    output2.retain_grad()
    output2.sum().backward()

    # print(x1.grad)
    # print(scale1.grad)
    # print(x2.grad)
    # print(scale2.grad)

    print(torch.allclose(output1, output2))
    print(torch.allclose(output1.grad, output2.grad))
    print(torch.allclose(x1.grad, x2.grad))
    print(torch.allclose(scale1.grad, scale2.grad))
    breakpoint()
