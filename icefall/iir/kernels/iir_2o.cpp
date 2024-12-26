#include <torch/extension.h>
#include <ATen/ATen.h>

// CUDA forward declarations

void iir_forward_cuda(
    at::Tensor input,
    at::Tensor denom,
    at::Tensor scale,
    at::Tensor output);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void iir_forward_impl(
        at::Tensor input,
        at::Tensor denom,
        at::Tensor scale,
        at::Tensor output) {
    CHECK_INPUT(input);
    CHECK_INPUT(denom);
    CHECK_INPUT(scale);
    CHECK_INPUT(output);

    iir_forward_cuda(input, denom, scale, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &iir_forward_impl, "IIR CUDA Forward");
}