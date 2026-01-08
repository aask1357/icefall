#include <tuple>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/detail/IndexUtils.cuh>

template <typename scalar_t>
__global__ void omniquant_kernel_optimized(
    const scalar_t* __restrict__ x,
    const float* __restrict__ s,
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ dx_ds,
    uint8_t* __restrict__ mask,
    const float q,
    const int64_t size
) {
    using acc_t = at::acc_type<scalar_t, true>;
    
    // [최적화 1] Scale을 레지스터에 로드하고 '역수'를 미리 계산 (나눗셈 제거)
    // __ldg를 써서 Read-only cache를 태웁니다. (스칼라 1개라 부담 없음)
    const acc_t s_val = static_cast<acc_t>(__ldg(s));
    const acc_t s_inv = static_cast<acc_t>(1.0f) / s_val; 

    // 상수 미리 계산
    const acc_t q_val = static_cast<acc_t>(q);
    const acc_t q_plus_eps = q_val + 0.5f;
    const acc_t neg_q_plus_eps = -q_plus_eps;
    const acc_t neg_q_val = -q_val;

    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;

    // [최적화 2] Unrolling을 통해 메모리 대역폭 활용 극대화 (ILP)
    // 컴파일러가 알아서 2개 또는 4개씩 묶어서 로드(LDG.64/128)하도록 유도합니다.
    #pragma unroll 4
    for (int64_t i = idx; i < size; i += stride) {
        acc_t x_val = static_cast<acc_t>(x[i]);

        // [최적화 3] 나눗셈(/)을 곱셈(*)으로 변경 -> 수십 cycle 절약
        acc_t div = x_val * s_inv;
        
        // [최적화 4] rintf (Intrinsic) 사용
        acc_t rounded = rintf(div);
        
        // Clamp 로직 간소화
        acc_t clamped = rounded;
        if (rounded < neg_q_plus_eps) clamped = neg_q_val;
        else if (rounded > q_plus_eps) clamped = q_val;

        // 범위 체크
        bool is_in_range = (rounded >= neg_q_plus_eps) && (rounded <= q_plus_eps);

        acc_t dx_ds_val = is_in_range ? (clamped - div) : clamped;
        acc_t out_val = clamped * s_val;

        out[i] = static_cast<scalar_t>(out_val);
        dx_ds[i] = static_cast<scalar_t>(dx_ds_val);
        mask[i] = static_cast<uint8_t>(is_in_range);
    }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Host Wrapper
std::vector<at::Tensor> omniquant_fast_cuda(
    at::Tensor x,
    at::Tensor scale,
    const float q,
    const bool inplace
) {
    CHECK_INPUT(x);
    CHECK_INPUT(scale);
    TORCH_CHECK(scale.numel() == 1, "scale must be a scalar");

    auto options = at::TensorOptions().dtype(at::kBool).device(x.device());
    auto out = inplace ? x : at::empty_like(x);
    auto dx_ds = at::empty_like(x);
    auto mask = at::empty(x.sizes(), options); // Note: pytorch bool is 1 byte (uint8)

    const int64_t size = x.numel();
    const int threads = 512; // 256 or 512 are sweet spots
    const int blocks = std::min((int64_t)65535, (size + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "omniquant_cuda",
        ([&] {
             omniquant_kernel_optimized<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                scale.data_ptr<float>(),
                out.data_ptr<scalar_t>(),
                dx_ds.data_ptr<scalar_t>(),
                (uint8_t*)mask.data_ptr<bool>(),
                q,
                size
            );
        })
    );

    return {out, dx_ds, mask};
}