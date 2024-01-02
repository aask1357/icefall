#include <tuple>
#include <iostream>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/detail/IndexUtils.cuh>

#include "cuda_utils.cuh"

template <typename T, size_t N>
using CudaAcsr = at::PackedTensorAccessor32<T, N, at::RestrictPtrTraits>;

#define CHUNKSIZE 32
#define DIVIDE 1
#define PADDING 1

template <typename scalar_t, typename index_t>
__global__ void iir_first_order_kernel(
        const CudaAcsr<scalar_t, 3> input,
        const CudaAcsr<scalar_t, 2> denom,
        CudaAcsr<scalar_t, 3> output) {
    using acc_t = at::acc_type<scalar_t, true>;
    const int BATCH_SIZE = input.size(0);
    const int CHANNELS = input.size(1);
    const int TIME = input.size(2);

    const int bc = threadIdx.x + blockIdx.x * blockDim.x;
    const int b = (bc / CHANNELS) % BATCH_SIZE;
    const int c = bc % CHANNELS;

    // load weights
    acc_t a = denom[c][0];

    // load inputs to shared memory (prefetch)
    extern __shared__ char s[];
    scalar_t *input_sm = (scalar_t*)&s;
    scalar_t *output_sm = &((scalar_t*)&s)[blockDim.x * (CHUNKSIZE + PADDING)];
    const int bc_start = blockIdx.x * blockDim.x;
    for (int i = threadIdx.x; i < blockDim.x * CHUNKSIZE; i += blockDim.x) {
        const int bc_sm = i / CHUNKSIZE;
        const int t_sm = i % CHUNKSIZE;
        const int bc_input = bc_sm + bc_start;
        const int t_input = t_sm;
        if (bc_input < BATCH_SIZE*CHANNELS && t_input < TIME) {
            input_sm[bc_sm*(CHUNKSIZE+PADDING) + t_sm] = input.data()[bc_input*TIME + t_input];
        }
    }
    __syncthreads();

    scalar_t x[CHUNKSIZE];
    acc_t y = (b < BATCH_SIZE) ? output[b][c][0] : acc_t(0);
    const int sm_start = threadIdx.x*(CHUNKSIZE+PADDING);
    for (int t = 0; t < TIME; t += CHUNKSIZE) {
        // load input (prefetch)
        #pragma unroll
        for (int l = 0; l < CHUNKSIZE; l++) {
            x[l] = input_sm[sm_start + l];  // Pad(1) enables faster shared-memory access.
        }
        __syncthreads();
        for (int i = threadIdx.x; i < blockDim.x * CHUNKSIZE; i += blockDim.x) {
            const int bc_sm = i / CHUNKSIZE;
            const int t_sm = i % CHUNKSIZE;
            const int bc_input = bc_sm + bc_start;
            const int t_input = t + CHUNKSIZE + t_sm;
            if (bc_input < BATCH_SIZE*CHANNELS && t_input < TIME) {
                input_sm[bc_sm*(CHUNKSIZE+PADDING) + t_sm] = input.data()[bc_input*TIME + t_input];
            }
        }

        // Calculate output
        #pragma unroll
        for (int l = 0; l < CHUNKSIZE; l++) {
            y = (acc_t)x[l] + y * a;
            output_sm[sm_start + l] = (scalar_t)y;    // Pad(1) enables faster shared-memory access.
        }

        // save output
        __syncthreads();
        #pragma unroll
        for (int i = threadIdx.x; i < blockDim.x * CHUNKSIZE; i += blockDim.x) {
            const int bc_sm = i / CHUNKSIZE;
            const int t_sm = i % CHUNKSIZE;
            const int bc_output = bc_sm + bc_start;
            const int t_output = t + 1 + t_sm;
            if (bc_output < BATCH_SIZE*CHANNELS && t_output <= TIME) {
                output.data()[bc_output*(TIME+1) + t_output] = output_sm[bc_sm*(CHUNKSIZE+PADDING) + t_sm];
            }
        }
    }
}

template<typename scalar_t, typename index_t>
void iir_forward_template(
        at::Tensor& input,
        at::Tensor& denom,
        at::Tensor& output) {
    const int BATCH_SIZE = input.size(0);
    const int CHANNELS = input.size(1);
    const int TIME = input.size(2);
    const int N_ORDER = denom.size(1);

    assert(CHANNELS == denom.size(0));
    assert(CHANNELS == output.size(1));
    assert(output.size(2) == N_ORDER + TIME);
    
    auto input_acsr = input.packed_accessor32<scalar_t,3,at::RestrictPtrTraits>();
    auto denom_acsr = denom.packed_accessor32<scalar_t,2,at::RestrictPtrTraits>();
    auto output_acsr = output.packed_accessor32<scalar_t,3,at::RestrictPtrTraits>();

    if (N_ORDER == 1) {
        const int BC = BATCH_SIZE * CHANNELS * DIVIDE;       // maximum parallizable unit
        int blockDim = 32;
        const int gridDim = (BC + blockDim - 1) / blockDim;

        // load/calculate/store every 32 frames. Padding 1 for load/store speedup.
        auto sm_size = blockDim / DIVIDE * (CHUNKSIZE+PADDING) * 2 * sizeof(scalar_t);
        // std::cout << "blockDim: " << blockDim << ", gridDim: " << gridDim << ", smsize: " << sm_size << std::endl;

        iir_first_order_kernel<scalar_t, index_t><<<gridDim, blockDim, sm_size>>>(
            input_acsr, denom_acsr, output_acsr);
    } else if (N_ORDER == 2) {
        throw std::runtime_error("N_ORDER 2 is currently not implemented.");
    } else {
        throw std::runtime_error("N_ORDER must be 1 or 2.");
    }
}

void iir_forward_cuda(
    at::Tensor input,   // B x C x T
    at::Tensor denom,   // C x N where N = 1 or 2
    at::Tensor output   // B x C x (N+T)
) {
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "iir_cuda", ([&] {
    if (at::cuda::detail::canUse32BitIndexMath(output) && \
        at::cuda::detail::canUse32BitIndexMath(denom)) {
      iir_forward_template<scalar_t, int32_t>(input, denom, output);
    } else {
      iir_forward_template<scalar_t, int64_t>(input, denom, output);
    }
  }));
}
