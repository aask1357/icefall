#include <tuple>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h> 

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/detail/IndexUtils.cuh>

// ------------------------------------------------------------------
// [수정 1] 벡터화 타입을 위한 구조체 정의
// ------------------------------------------------------------------
template <typename scalar_t, int vec_size>
struct VectorizedType {
    scalar_t val[vec_size];
};

// ------------------------------------------------------------------
// [수정 2] __ldg 호환성을 위한 헬퍼 함수 추가
// 사용자 정의 구조체 포인터를 int4*(128bit)로 속여서 로드합니다.
// ------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ T load_vector(const T* ptr) {
    // 16바이트(128비트)인 경우 int4 사용
    if constexpr (sizeof(T) == 16) {
        const int4* alias_ptr = reinterpret_cast<const int4*>(ptr);
        int4 v = __ldg(alias_ptr);
        return *reinterpret_cast<T*>(&v);
    }
    // 8바이트(64비트)인 경우 int2 사용 (만약 vec_size가 작을 경우 대비)
    else if constexpr (sizeof(T) == 8) {
        const int2* alias_ptr = reinterpret_cast<const int2*>(ptr);
        int2 v = __ldg(alias_ptr);
        return *reinterpret_cast<T*>(&v);
    }
    // 4바이트(32비트)인 경우 int 사용
    else if constexpr (sizeof(T) == 4) {
        const int* alias_ptr = reinterpret_cast<const int*>(ptr);
        int v = __ldg(alias_ptr);
        return *reinterpret_cast<T*>(&v);
    }
    else {
        // Fallback: 일반 로드 (캐시 힌트 없음)
        return *ptr;
    }
}

// ------------------------------------------------------------------
// 커널 구현
// ------------------------------------------------------------------
template <typename scalar_t, int vec_size>
__global__ void omniquant_kernel_vectorized(
    const scalar_t* __restrict__ x,
    const float* __restrict__ s,
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ dx_ds,
    uint8_t* __restrict__ mask,
    const float q,
    const int64_t size
) {
    using acc_t = at::acc_type<scalar_t, true>;
    // `typename ... ::type` 제거하고 직접 구조체 사용
    using VecT = VectorizedType<scalar_t, vec_size>;
    using MaskVecT = VectorizedType<uint8_t, vec_size>;

    const acc_t s_val = static_cast<acc_t>(*s);
    const acc_t s_inv = static_cast<acc_t>(1.0f) / s_val;

    const acc_t q_val = static_cast<acc_t>(q);
    const acc_t q_plus_eps = q_val + 0.5f;
    const acc_t neg_q_plus_eps = -q_plus_eps;
    const acc_t neg_q_val = -q_val;

    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;
    
    const int64_t vec_loop_end = size / vec_size;
    
    const VecT* x_vec = reinterpret_cast<const VecT*>(x);
    VecT* out_vec = reinterpret_cast<VecT*>(out);
    VecT* dx_ds_vec = reinterpret_cast<VecT*>(dx_ds);
    MaskVecT* mask_vec = reinterpret_cast<MaskVecT*>(mask);

    for (int64_t i = idx; i < vec_loop_end; i += stride) {
        // [수정 3] __ldg 대신 직접 만든 load_vector 사용
        VecT x_v = load_vector(&x_vec[i]); 
        
        VecT out_v;
        VecT dx_ds_v;
        MaskVecT mask_v;

        #pragma unroll
        for (int j = 0; j < vec_size; ++j) {
            acc_t x_val_acc = static_cast<acc_t>(x_v.val[j]);
            
            acc_t div = x_val_acc * s_inv;
            acc_t rounded = rintf(div);
            
            acc_t clamped;
            if (rounded < neg_q_plus_eps) clamped = neg_q_val;
            else if (rounded > q_plus_eps) clamped = q_val;
            else clamped = rounded;

            bool is_in_range = (rounded >= neg_q_plus_eps) && (rounded <= q_plus_eps);
            
            acc_t dx_ds_val = is_in_range ? (clamped - div) : clamped;
            acc_t out_val = clamped * s_val;

            out_v.val[j] = static_cast<scalar_t>(out_val);
            dx_ds_v.val[j] = static_cast<scalar_t>(dx_ds_val);
            mask_v.val[j] = static_cast<uint8_t>(is_in_range);
        }

        out_vec[i] = out_v;
        dx_ds_vec[i] = dx_ds_v;
        mask_vec[i] = mask_v;
    }

    int64_t remainder_start = vec_loop_end * vec_size;
    for (int64_t i = remainder_start + idx; i < size; i += stride) {
        acc_t x_val = static_cast<acc_t>(x[i]);
        acc_t div = x_val * s_inv;
        acc_t rounded = rintf(div);
        
        acc_t clamped;
        if (rounded < neg_q_plus_eps) clamped = neg_q_val;
        else if (rounded > q_plus_eps) clamped = q_val;
        else clamped = rounded;
        
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
    const int threads = 256; // 256 or 512 are sweet spots

    // 각 스레드가 처리할 벡터 크기 결정 (128bit load를 위해)
    // float32(4byte) -> vec_size=4
    // half(2byte) -> vec_size=8 (최대 float4 크기에 맞춤)
    // bfloat16(2byte) -> vec_size=8

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "omniquant_cuda",
        ([&] {
            // 128 bit alignment check logic 생략 (보통 텐서는 정렬되어 있음)
            constexpr int vec_size = 16 / sizeof(scalar_t); 

            // Blocks calculation needs to account for vectorization
            // Each thread handles 'vec_size' elements per iteration
            const int64_t num_vec_elements = (size + vec_size - 1) / vec_size;
            const int blocks = std::min((int64_t)65535, (num_vec_elements + threads - 1) / threads);

            omniquant_kernel_vectorized<scalar_t, vec_size><<<blocks, threads>>>(
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