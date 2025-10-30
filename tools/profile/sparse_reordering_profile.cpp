#include <torch/extension.h>

#include <ATen/ATen.h>
#include <utils/spconv/spconv/reordering.h>
#include <utils/spconv/tensorview/tensorview.h>

#include "xpu/spconv_utils_xpu.h"

namespace {

template <typename scalar_t>
torch::Tensor sparse_gather_impl(torch::Tensor features,
                                 torch::Tensor indices) {
  auto options = features.options();
  auto num_hot = static_cast<int>(indices.size(0));
  auto num_planes = static_cast<int>(features.size(1));
  auto buffer = torch::empty({num_hot, num_planes}, options);
  functor::SparseGatherFunctor<tv::TorchGPU, scalar_t, int>()(
      tv::TorchGPU(), tv::torch2tv<scalar_t>(buffer),
      tv::torch2tv<const scalar_t>(features),
      tv::torch2tv<const int>(indices), num_hot);
  return buffer;
}

template <typename scalar_t>
void sparse_scatter_add_impl(torch::Tensor out, torch::Tensor buffer,
                             torch::Tensor indices) {
  auto num_hot = static_cast<int>(indices.size(0));
  functor::SparseScatterAddFunctor<tv::TorchGPU, scalar_t, int>()(
      tv::TorchGPU(), tv::torch2tv<scalar_t>(out),
      tv::torch2tv<const scalar_t>(buffer),
      tv::torch2tv<const int>(indices), num_hot, true);
}

void check_inputs(torch::Tensor features, torch::Tensor indices) {
  TORCH_CHECK(features.device().type() == at::kXPU,
              "features must live on XPU" );
  TORCH_CHECK(indices.device().type() == at::kXPU,
              "indices must live on XPU" );
  TORCH_CHECK(indices.dtype() == torch::kInt32,
              "indices must be int32" );
  TORCH_CHECK(features.dim() == 2,
              "features must be 2D [N, C]");
  TORCH_CHECK(indices.dim() == 1,
              "indices must be 1D" );
}

torch::Tensor sparse_gather(torch::Tensor features, torch::Tensor indices) {
  check_inputs(features, indices);
  torch::Tensor result;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.scalar_type(), "sparse_gather_impl", [&] {
        result = sparse_gather_impl<scalar_t>(features, indices);
      });
  return result;
}

void sparse_scatter_add(torch::Tensor out, torch::Tensor buffer,
                        torch::Tensor indices) {
  check_inputs(out, indices);
  TORCH_CHECK(buffer.device().type() == at::kXPU,
              "buffer must live on XPU");
  TORCH_CHECK(buffer.scalar_type() == out.scalar_type(),
              "dtype mismatch between out and buffer");
  TORCH_CHECK(buffer.dim() == 2,
              "buffer must be 2D [num_hot, num_planes]");
  TORCH_CHECK(static_cast<int64_t>(buffer.size(1)) == out.size(1),
              "buffer feature dim must match out feature dim");
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      out.scalar_type(), "sparse_scatter_add_impl", [&] {
        sparse_scatter_add_impl<scalar_t>(out, buffer, indices);
      });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_gather", &sparse_gather, "Sparse gather on XPU");
  m.def("sparse_scatter_add", &sparse_scatter_add,
        "Sparse scatter add on XPU");
}

}  // namespace
