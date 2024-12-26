#include <tuple>
#include <iostream>
#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/detail/IndexUtils.cuh>

#include "cuda_utils.cuh"

template <typename T, size_t N>
using CudaAcsr = at::PackedTensorAccessor32<T, N, at::RestrictPtrTraits>;

template <typename scalar_t, typename weight_t, typename index_t>
__global__ void iir_kernel(
        const CudaAcsr<scalar_t, 3> input,
        const CudaAcsr<weight_t, 2> denom,
        const CudaAcsr<weight_t, 1> scale,
        CudaAcsr<scalar_t, 3> output) {
    using acc_t = at::acc_type<scalar_t, true>;
    const int BATCH_SIZE = input.size(0);
    const int CHANNELS = input.size(1);
    const int TIME = input.size(2);

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int m = idx % 2;
    const int c = (idx / 2) % CHANNELS;

    // load weights
    acc_t a = denom[c][1-m];
    acc_t b = scale[c];

    // load inputs to shared memory (prefetch)
    const int d_size = blockDim.x / 2;
    extern __shared__ char s[];
    scalar_t *input_sm = (scalar_t*)&s;
    scalar_t *output_sm = &((scalar_t*)&s)[d_size * 34];
    const int d_start = blockIdx.x * (blockDim.x / 2);
    for (int i = threadIdx.x; i < d_size * 32; i += blockDim.x) {
        const int d_input = i/32 + d_start;
        const int t_input = i % 32;
        if (d_input < BATCH_SIZE*CHANNELS && t_input < TIME) {
            input_sm[(i/32)*34 + i%32] = input.data()[d_input*TIME + t_input];
        }
    }
    __syncthreads();

    scalar_t input_reg[16];
    
    acc_t y_prev = (idx/2 < BATCH_SIZE * CHANNELS) ?
        static_cast<acc_t>(output.data()[(idx/2)*(TIME+2) + m]) : acc_t(0);
    acc_t y = (m == 0) ? a * y_prev : acc_t(0);     // y = (m==0) a0 * y[0]; (m==1) 0;
    y_prev = WARP_SHFL(y_prev, 1, 2);               // y_prev = y[1];

    const int sm_start = (threadIdx.x/2)*34;
    for (int t = 0; t < TIME; t += 32) {
        // load input (prefetch)
        #pragma unroll
        for (int l = 0; l < 16; l++) {
            input_reg[l] = input_sm[sm_start + l*2 + m];
        }
        __syncthreads();
        #pragma unroll
        for (int i = threadIdx.x; i < d_size * 32; i += blockDim.x) {
            const int d_input = i/32 + d_start;
            const int t_input = t + 32 + i % 32;
            if (d_input < BATCH_SIZE*CHANNELS && t_input < TIME) {
                index_t input_idx = static_cast<index_t>(d_input)*TIME + t_input;
                input_sm[(i/32)*34 + i%32] = input.data()[input_idx];
            }
        }

        // calculate output
        #pragma unroll
        for (int l = 0; l < 16; l++) {
            y += b * input_reg[l] + a * y_prev;     // (m==0) y_{t+2*l} completed
            acc_t y_ex = WARP_SHFL_XOR(y, 1, 2);    // y_exchange
            y_prev = (m == 0) ? y : y_ex;
            y = (m == 0) ? y_ex : acc_t(0);
            y += a * y_prev;                        // (m==0) y_{t+2*l+1} completed
            y_ex = WARP_SHFL_XOR(y, 1, 2);
            acc_t y_out = (m == 0) ? y_prev : y_ex;
            y_prev = (m == 0) ? y : y_ex;
            y = (m == 0) ? y_ex : acc_t(0);

            output_sm[sm_start + l*2 + m] = y_out;
        }

        // store output
        __syncthreads();
        #pragma unroll
        for (int i = threadIdx.x; i < d_size * 32; i += blockDim.x) {
            int d_output = i/32 + d_start;
            const int t_output = t + i % 32;
            if (d_output < BATCH_SIZE*CHANNELS && t_output < TIME) {
                index_t idx = static_cast<index_t>(d_output)*(TIME+2) + t_output + 2;
                output.data()[idx] = output_sm[(i/32)*34 + i%32];
            }
        }
    }
}


template <typename scalar_t, typename weight_t, typename index_t>
void iir_forward_template(
        at::Tensor& input,
        at::Tensor& denom,
        at::Tensor& scale,
        at::Tensor& output) {
    const int BATCH_SIZE = input.size(0);
    const int CHANNELS = input.size(1);
    const int TIME = input.size(2);

    assert(denom.size(0) == CHANNELS);
    assert(denom.size(1) == 2);
    assert(scale.size(0) == CHANNELS);
    assert(output.size(1) == CHANNELS);
    assert(output.size(2) == TIME + 2);
    
    auto input_acsr = input.packed_accessor32<scalar_t,3,at::RestrictPtrTraits>();
    auto denom_acsr = denom.packed_accessor32<weight_t,2,at::RestrictPtrTraits>();
    auto scale_acsr = scale.packed_accessor32<weight_t,1,at::RestrictPtrTraits>();
    auto output_acsr = output.packed_accessor32<scalar_t,3,at::RestrictPtrTraits>();

    // input / output: D x Time where D = Batch x Channels. We can parallelize over D.
    // shared memory: D' x (32 + 2) where D' = blockDim.x / 2
    //    32: (load -> calculate -> store) 32 samples at once for efficient Gloal Mem load/store
    //    2: padding of Shared Mem (SM) for efficient SM load/store
    int blockDim = getBlockDim(2);
    const int size = BATCH_SIZE * CHANNELS * 2;
    const int gridDim = (size + blockDim - 1) / blockDim;
    auto sm_size = (blockDim/2) * (2*16+2) * 2 * sizeof(scalar_t);
    // std::cout << "blockDim: " << blockDim << ", gridDim: " << gridDim << ", sm_size: " << sm_size << std::endl;
    iir_kernel<scalar_t, weight_t, index_t><<<gridDim, blockDim, sm_size>>>(
            input_acsr, denom_acsr, scale_acsr, output_acsr);
}

void iir_forward_cuda(
    at::Tensor input,   // B x C x T
    at::Tensor denom,   // C x 2
    at::Tensor scale,   // C
    at::Tensor output   // B x C x (2+T)
) {
    #define DISPATCH_INDEX_DTYPE(scalar_t, weight_t) {              \
        if (at::cuda::detail::canUse32BitIndexMath(output)) {       \
                iir_forward_template<scalar_t, weight_t, int32_t>(  \
                    input, denom, scale, output);                   \
            } else {                                                \
                iir_forward_template<scalar_t, weight_t, int64_t>(  \
                    input, denom, scale, output);                   \
            }                                                       \
    }
    
    at::ScalarType input_t = ::detail::scalar_type(input.scalar_type());
    at::ScalarType weight_t = ::detail::scalar_type(denom.scalar_type());

    switch(input_t) {
        case at::ScalarType::Half:
            // support mixed precision
            if (weight_t == at::ScalarType::Half) {
                DISPATCH_INDEX_DTYPE(at::Half, at::Half)
            } else if (weight_t == at::ScalarType::Float) {
                DISPATCH_INDEX_DTYPE(at::Half, float)
            } else {
                throw std::runtime_error("Unsupported scalar type.");
            }
            break;
        case at::ScalarType::Float:
            DISPATCH_INDEX_DTYPE(float, float)
            break;
        case at::ScalarType::Double:
            DISPATCH_INDEX_DTYPE(double, double)
            break;
        default:
            throw std::runtime_error("Unsupported scalar type.");
    }
}
