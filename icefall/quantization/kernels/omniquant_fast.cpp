#include <torch/extension.h>
#include <ATen/ATen.h>

// CUDA forward declarations

std::vector<at::Tensor> omniquant_fast_cuda(
    at::Tensor x,
    at::Tensor scale,
    const float q,
    const bool inplace
);

// C++ interface

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &omniquant_fast_cuda, "Round CUDA");
}