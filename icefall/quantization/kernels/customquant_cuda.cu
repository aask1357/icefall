#include <tuple>
#include <iostream>
#include <stdio.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/detail/IndexUtils.cuh>


template <typename scalar_t, typename index_t>
__global__ void customquant_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ s,
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ dx_ds,
    uint8_t* __restrict__ mask,
    const float q,
    const index_t size
) {
    using acc_t = at::acc_type<scalar_t, true>;
    acc_t s_val = *s;

    index_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    index_t stride = blockDim.x * gridDim.x;

    const acc_t q_val = static_cast<acc_t>(q);
    for (index_t i = idx; i < size; i += stride) {
        acc_t x_val = static_cast<acc_t>(x[i]);

        acc_t div = x_val / s_val;
        acc_t clamped = (div < -q_val) ? -q_val : (div > q_val) ? q_val : div;
        acc_t rounded = roundf(clamped);
        uint8_t mask_val = (div >= -q_val) && (div <= q_val);

        acc_t dx_ds_val = rounded - clamped;
        acc_t out_val = rounded * s_val;

        out[i] = static_cast<scalar_t>(out_val);
        dx_ds[i] = static_cast<scalar_t>(dx_ds_val);
        mask[i] = mask_val;
    }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> customquant_cuda(
    at::Tensor x,
    at::Tensor scale,
    const float q,
    const bool inplace
) {
    CHECK_INPUT(x);
    CHECK_INPUT(scale);
    TORCH_CHECK(scale.numel() == 1, "scale must be a scalar tensor");
    TORCH_CHECK(scale.dtype() == at::ScalarType::Float, "scale must be float32");
  
    auto options = at::TensorOptions().dtype(at::kBool).device(x.device());
    auto out = inplace ? x : at::empty_like(x);
    auto dx_ds = at::empty_like(x);
    auto mask = at::empty(x.sizes(), options);

    const int64_t size = x.numel();
    const int threads = 512;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "customquant_cuda",
        ([&] {
            if (at::cuda::detail::canUse32BitIndexMath(x)) {
                customquant_kernel<scalar_t, int32_t><<<blocks, threads>>>(
                    x.data_ptr<scalar_t>(),
                    scale.data_ptr<float>(),
                    out.data_ptr<scalar_t>(),
                    dx_ds.data_ptr<scalar_t>(),
                    (uint8_t*)mask.data_ptr<bool>(),
                    q,
                    size
                );
            } else {
                customquant_kernel<scalar_t, int64_t><<<blocks, threads>>>(
                    x.data_ptr<scalar_t>(),
                    scale.data_ptr<float>(),
                    out.data_ptr<scalar_t>(),
                    dx_ds.data_ptr<scalar_t>(),
                    (uint8_t*)mask.data_ptr<bool>(),
                    q,
                    size
                );
            }
        })
    );

    return {out, dx_ds, mask};
}
