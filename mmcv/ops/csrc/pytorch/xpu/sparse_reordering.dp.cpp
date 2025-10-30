// Copyright 2019 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <sycl/sycl.hpp>

#include <ATen/ATen.h>
#include <cstdint>
#include <type_traits>
#include <utils/spconv/spconv/reordering.h>
#include <utils/spconv/tensorview/helper_launch.h>
#include <utils/spconv/tensorview/tensorview.h>

#include "spconv_utils_xpu.h"

namespace functor {
namespace {

template <typename scalar_t, typename Index>
struct SparseGatherKernelTag {};

template <typename scalar_t, typename Index>
struct SparseScatterKernelTag {};

template <typename scalar_t>
struct AtomicAdd {
  static inline void apply(scalar_t* addr, scalar_t value) {
    sycl::atomic_ref<scalar_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        ref(*addr);
    ref.fetch_add(value);
  }
};

template <>
struct AtomicAdd<at::Half> {
  static inline void apply(at::Half* addr, at::Half value) {
    static_assert(sizeof(at::Half) == sizeof(sycl::half),
                  "Expected at::Half to match sycl::half storage");
    auto half_ptr = reinterpret_cast<sycl::half*>(addr);
    sycl::half increment = static_cast<sycl::half>(static_cast<float>(value));
  sycl::atomic_ref<sycl::half, sycl::memory_order::relaxed,
           sycl::memory_scope::device,
           sycl::access::address_space::global_space>
    ref(*half_ptr);
    ref.fetch_add(increment);
  }
};

template <typename scalar_t, typename Index>
void gather_impl(const tv::TorchGPU& d, scalar_t* dst, const scalar_t* src,
                 const Index* indices, int num_rows, int num_cols) {
  if (num_rows == 0 || num_cols == 0) {
    return;
  }
  constexpr int kLocalRows = 4;
  constexpr int kLocalCols = 64;
  const size_t global_row_count =
    static_cast<size_t>(tv::launch::DivUp(num_rows, kLocalRows)) *
    static_cast<size_t>(kLocalRows);
  const size_t global_col_count =
    static_cast<size_t>(tv::launch::DivUp(num_cols, kLocalCols)) *
    static_cast<size_t>(kLocalCols);
  auto global_rows = sycl::range<1>(global_row_count);
  auto global_cols = sycl::range<1>(global_col_count);
  auto event = d.getQueue().submit([&](sycl::handler& cgh) {
    cgh.parallel_for<SparseGatherKernelTag<scalar_t, Index>>(
        sycl::nd_range<2>(sycl::range<2>(global_rows[0], global_cols[0]),
                          sycl::range<2>(kLocalRows, kLocalCols)),
        [=](sycl::nd_item<2> item) {
          int row = item.get_global_id(0);
          int col = item.get_global_id(1);
          if (row >= num_rows || col >= num_cols) {
            return;
          }
          auto src_offset = static_cast<std::int64_t>(indices[row]) *
                            static_cast<std::int64_t>(num_cols) + col;
          auto dst_offset = static_cast<std::int64_t>(row) *
                            static_cast<std::int64_t>(num_cols) + col;
          dst[dst_offset] = src[src_offset];
        });
  });
  event.wait();
}

template <typename scalar_t, typename Index>
void scatter_add_impl_serial(const tv::TorchGPU& d, scalar_t* out,
                             const scalar_t* buffer, const Index* indices,
                             int num_rows, int num_cols) {
  if (num_rows == 0 || num_cols == 0) {
    return;
  }
  auto event = d.getQueue().submit([&](sycl::handler& cgh) {
    cgh.single_task([=]() {
      for (int row = 0; row < num_rows; ++row) {
        const auto out_offset = static_cast<std::int64_t>(indices[row]) *
                                static_cast<std::int64_t>(num_cols);
        const auto buf_offset = static_cast<std::int64_t>(row) *
                                static_cast<std::int64_t>(num_cols);
        for (int col = 0; col < num_cols; ++col) {
          out[out_offset + col] =
              out[out_offset + col] + buffer[buf_offset + col];
        }
      }
    });
  });
  event.wait();
}

template <typename scalar_t, typename Index>
void scatter_add_impl(const tv::TorchGPU& d, scalar_t* out,
                      const scalar_t* buffer, const Index* indices,
                      int num_rows, int num_cols) {
  if (num_rows == 0 || num_cols == 0) {
    return;
  }
  if constexpr (std::is_same_v<scalar_t, at::Half>) {
    if (!d.getQueue().get_device().has(sycl::aspect::ext_oneapi_atomic16)) {
      scatter_add_impl_serial(d, out, buffer, indices, num_rows, num_cols);
      return;
    }
  }
  constexpr int kLocalRows = 4;
  constexpr int kLocalCols = 64;
  const size_t global_row_count =
    static_cast<size_t>(tv::launch::DivUp(num_rows, kLocalRows)) *
    static_cast<size_t>(kLocalRows);
  const size_t global_col_count =
    static_cast<size_t>(tv::launch::DivUp(num_cols, kLocalCols)) *
    static_cast<size_t>(kLocalCols);
  auto global_rows = sycl::range<1>(global_row_count);
  auto global_cols = sycl::range<1>(global_col_count);
  auto event = d.getQueue().submit([&](sycl::handler& cgh) {
    cgh.parallel_for<SparseScatterKernelTag<scalar_t, Index>>(
        sycl::nd_range<2>(sycl::range<2>(global_rows[0], global_cols[0]),
                          sycl::range<2>(kLocalRows, kLocalCols)),
        [=](sycl::nd_item<2> item) {
          int row = item.get_global_id(0);
          int col = item.get_global_id(1);
          if (row >= num_rows || col >= num_cols) {
            return;
          }
          auto out_offset = static_cast<std::int64_t>(indices[row]) *
                            static_cast<std::int64_t>(num_cols) + col;
          auto buf_offset = static_cast<std::int64_t>(row) *
                            static_cast<std::int64_t>(num_cols) + col;
          AtomicAdd<scalar_t>::apply(out + out_offset, buffer[buf_offset]);
        });
  });
  event.wait();
}
}  // namespace

template <typename scalar_t, typename Index>
struct SparseGatherFunctor<tv::TorchGPU, scalar_t, Index> {
  void operator()(const tv::TorchGPU& d, tv::TensorView<scalar_t> buffer,
                  tv::TensorView<const scalar_t> features,
                  tv::TensorView<const Index> indices, int size) {
    if (size <= 0) {
      return;
    }
    gather_impl<scalar_t, Index>(d, buffer.data(), features.data(),
                                 indices.data(), size, features.dim(1));
  }
};

template <typename scalar_t, typename Index>
struct SparseScatterAddFunctor<tv::TorchGPU, scalar_t, Index> {
  void operator()(const tv::TorchGPU& d, tv::TensorView<scalar_t> out_features,
                  tv::TensorView<const scalar_t> buffer,
                  tv::TensorView<const Index> indices, int size,
                  bool /*stable*/) {
    if (size <= 0) {
      return;
    }
    scatter_add_impl<scalar_t, Index>(d, out_features.data(), buffer.data(),
                                      indices.data(), size,
                                      out_features.dim(1));
  }
};

#define DECLARE_XPU_SPECS_T_INDEX(scalar_t, Index)                             \
  template struct SparseGatherFunctor<tv::TorchGPU, scalar_t, Index>;          \
  template struct SparseScatterAddFunctor<tv::TorchGPU, scalar_t, Index>;

#define DECLARE_XPU_SPECS(scalar_t) DECLARE_XPU_SPECS_T_INDEX(scalar_t, int)

DECLARE_XPU_SPECS(float);
DECLARE_XPU_SPECS(double);
DECLARE_XPU_SPECS(at::Half);

#undef DECLARE_XPU_SPECS
#undef DECLARE_XPU_SPECS_T_INDEX

}  // namespace functor
