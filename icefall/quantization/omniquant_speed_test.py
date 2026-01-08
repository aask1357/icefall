from .kernels import omniquant_fast_cuda
from .kernels import omniquant_cuda

import torch
from torch import Tensor
from typing import Tuple


class OmniQuantFastCuda(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        scale: Tensor,
        q: float,
        inplace: bool = False,
    ) -> Tensor:
        assert scale.numel() == 1
        output, dx_ds, mask = omniquant_fast_cuda.forward(x, scale, q, inplace)
        ctx.save_for_backward(dx_ds, mask)
        return output

    @staticmethod
    def backward(
        ctx,
        grad_output: Tensor
    ) -> Tuple[Tensor, Tensor, None, None]:
        dx_ds, mask = ctx.saved_tensors
        return grad_output * mask, dx_ds.mul_(grad_output).float().sum().view(1), None, None


class OmniQuantCuda(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        scale: Tensor,
        q: float,
        inplace: bool = False,
    ) -> Tensor:
        assert scale.numel() == 1
        output, dx_ds, mask = omniquant_cuda.forward(x, scale, q, inplace)
        ctx.save_for_backward(dx_ds, mask)
        return output

    @staticmethod
    def backward(
        ctx,
        grad_output: Tensor
    ) -> Tuple[Tensor, Tensor, None, None]:
        dx_ds, mask = ctx.saved_tensors
        return grad_output * mask, dx_ds.mul_(grad_output).float().sum().view(1), None, None


class RoundSTE(torch.autograd.Function):
    """ Straight-through estimator of rounding. """
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        return x.round()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output


def omniquant_torch(x: Tensor, scale: Tensor, q: float) -> Tensor:
    with torch.amp.autocast("cuda", enabled=False):
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.float()
        rounded = RoundSTE.apply(x / scale)
        clamped = torch.where(
            rounded < -q - 0.5,
            -q,
            rounded
        )
        clamped = torch.where(
            rounded > q + 0.5,
            q,
            clamped
        )
        return clamped * scale


if __name__ == "__main__":
    DTYPE = torch.float16
    N = 10
    import time
    x = torch.randn(64, 1024, 800, device="cuda", dtype=DTYPE, requires_grad=True)
    x.data.mul_(64)
    scale = torch.randn(1, device="cuda", dtype=torch.float32, requires_grad=True)
    scale.data.add_(5).div_(5)
    w = torch.randn(256, 1024, 1, device="cuda", dtype=torch.float32, requires_grad=True)
    w.data.mul_(4)

    # Warm up
    for _ in range(N):
        OmniQuantFastCuda.apply(x, scale, float(2**7 - 1), True)
        OmniQuantCuda.apply(w, scale, float(2**3 - 1), False)
        OmniQuantCuda.apply(x, scale, float(2**7 - 1), True)
        OmniQuantCuda.apply(w, scale, float(2**3 - 1), False)
    torch.cuda.synchronize()

    # Test OmniQuantCuda
    start_time = time.perf_counter()
    for _ in range(N):
        y2 = OmniQuantCuda.apply(x, scale, float(2**7 - 1), True)
        w2 = OmniQuantCuda.apply(w, scale, float(2**3 - 1), False)
        # loss2 = y2.sum() + w2.sum()
        # loss2.backward()
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    print(f"OmniQuantCuda: {(end_time - start_time) / N * 1000:.2f} ms per run")

    # Test OmniQuantFastCuda
    start_time = time.perf_counter()
    for _ in range(N):
        y1 = OmniQuantFastCuda.apply(x, scale, float(2**7 - 1), True)
        w1 = OmniQuantFastCuda.apply(w, scale, float(2**3 - 1), False)
        # loss1 = y1.sum() + w1.sum()
        # loss1.backward()
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    print(f"OmniQuantFastCuda: {(end_time - start_time) / N * 1000:.2f} ms per run")

    # Test omniquant_torch
    start_time = time.perf_counter()
    for _ in range(N):
        y3 = omniquant_torch(x, scale, float(2**7 - 1))
        w3 = omniquant_torch(w, scale, float(2**3 - 1))
        # loss3 = y3.sum() + w3.sum()
        # loss3.backward()
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    print(f"omniquant_torch: {(end_time - start_time) / N * 1000:.2f} ms per run")
