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
#include <c10/xpu/XPUStream.h>
// clang-format off
// TODO: make spconv_utils.h order agnostic
#include "spconv_utils_xpu.h"
// clang-format on
#include <utils/spconv/spconv/indice.h>
#include <utils/spconv/spconv/mp_helper.h>
#include <utils/spconv/tensorview/helper_launch.h>
#include <utils/spconv/tensorview/tensorview.h>

#include <chrono>
#include <limits>
#include <type_traits>
#include <xpu/indice.dp.hpp>

namespace sycl {
template <typename T, int Rank>
struct is_device_copyable<tv::TensorView<T, Rank>> : std::true_type {};
template <typename T, int Rank>
struct is_device_copyable<const tv::TensorView<T, Rank>> : std::true_type {};
template <typename T, size_t MaxDim>
struct is_device_copyable<tv::SimpleVector<T, MaxDim>> : std::true_type {};
template <typename T, size_t MaxDim>
struct is_device_copyable<const tv::SimpleVector<T, MaxDim>> : std::true_type {};
}

namespace {

template <typename Index, typename IndexGrid, unsigned NDim, int KernelVolume>
struct PrepareDeConvIndicePairsKernelTag {};

template <typename Index, typename IndexGrid, unsigned NDim, int KernelVolume>
struct PrepareIndicePairsKernelTag {};

template <typename Index, typename IndexGrid, unsigned NDim>
struct AssignGridAndIndiceOutKernelTag {};

template <typename Index, typename IndexGrid, unsigned NDim>
struct AssignIndicePairsKernelTag {};

template <typename Index, typename IndexGrid, unsigned NDim>
struct ResetGridKernelTag {};

template <typename Index, typename IndexGrid, unsigned NDim>
struct PrepareSubMGridKernelTag {};

template <typename Index, typename IndexGrid, unsigned NDim, int KernelVolume>
struct GetSubMIndicePairsKernelTag {};

template <typename Index, typename IndexGrid, unsigned NDim>
struct ResetGridSubMKernelTag {};

}  // namespace

namespace functor {
template <typename Index, typename IndexGrid, unsigned NDim>
struct CreateConvIndicePairFunctorP1<tv::TorchGPU, Index, IndexGrid, NDim> {
  Index operator()(const tv::TorchGPU &d, tv::TensorView<const Index> indicesIn,
                   tv::TensorView<Index> indicesOut,
                   tv::TensorView<IndexGrid> gridsOut,
                   tv::TensorView<Index> indicePairs,
                   tv::TensorView<Index> indiceNum,
                   tv::TensorView<Index> indicePairUnique,
                   const tv::SimpleVector<Index, NDim> kernelSize,
                   const tv::SimpleVector<Index, NDim> stride,
                   const tv::SimpleVector<Index, NDim> padding,
                   const tv::SimpleVector<Index, NDim> dilation,
                   const tv::SimpleVector<Index, NDim> outSpatialShape,
                   bool transpose) {
    Index batchSize = gridsOut.dim(0);
    auto numActIn = indicesIn.dim(0);
    if (numActIn == 0) return 0;
    auto& queue = d.getQueue();
    if (transpose)
      queue.parallel_for<PrepareDeConvIndicePairsKernelTag<Index, IndexGrid,
                                                           NDim, 4096>>(
          sycl::nd_range<3>(
              sycl::range<3>(1, 1, tv::launch::getBlocks(numActIn)) *
                  sycl::range<3>(1, 1, tv::launch::CUDA_NUM_THREADS),
              sycl::range<3>(1, 1, tv::launch::CUDA_NUM_THREADS)),
          [=](sycl::nd_item<3> item_ct1) {
      prepareDeConvIndicePairsKernel<Index, IndexGrid, NDim, 4096>(
        item_ct1, indicesIn, indicesOut, gridsOut, indicePairs,
        indiceNum,
                indicePairUnique, kernelSize, stride, padding, dilation,
                outSpatialShape);
          });
    else
      queue.parallel_for<PrepareIndicePairsKernelTag<Index, IndexGrid, NDim,
                                                     4096>>(
          sycl::nd_range<3>(
              sycl::range<3>(1, 1, tv::launch::getBlocks(numActIn)) *
                  sycl::range<3>(1, 1, tv::launch::CUDA_NUM_THREADS),
              sycl::range<3>(1, 1, tv::launch::CUDA_NUM_THREADS)),
          [=](sycl::nd_item<3> item_ct1) {
      prepareIndicePairsKernel<Index, IndexGrid, NDim, 4096>(
        item_ct1, indicesIn, indicesOut, gridsOut, indicePairs,
        indiceNum,
                indicePairUnique, kernelSize, stride, padding, dilation,
                outSpatialShape);
          });
    return 1;
  }
};

template <typename Index, typename IndexGrid, unsigned NDim>
struct CreateConvIndicePairFunctorP2<tv::TorchGPU, Index, IndexGrid, NDim> {
  Index operator()(const tv::TorchGPU &d, tv::TensorView<const Index> indicesIn,
                   tv::TensorView<Index> indicesOut,
                   tv::TensorView<IndexGrid> gridsOut,
                   tv::TensorView<Index> indicePairs,
                   tv::TensorView<Index> indiceNum,
                   tv::TensorView<Index> indicePairUnique,
                   const tv::SimpleVector<Index, NDim> outSpatialShape,
                   bool transpose, bool resetGrid) {
    Index batchSize = gridsOut.dim(0);
    auto kernelVolume = indicePairs.dim(0);
    auto numActIn = indicesIn.dim(0);
    if (numActIn == 0) return 0;
  Index numAct = indicePairUnique.dim(0) - 1;
  auto& queue = d.getQueue();
  queue.parallel_for<AssignGridAndIndiceOutKernelTag<Index, IndexGrid, NDim>>(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, tv::launch::getBlocks(numAct)) *
                sycl::range<3>(1, 1, tv::launch::CUDA_NUM_THREADS),
            sycl::range<3>(1, 1, tv::launch::CUDA_NUM_THREADS)),
        [=](sycl::nd_item<3> item_ct1) {
      assignGridAndIndiceOutKernel<Index, IndexGrid, NDim>(
        item_ct1, indicesOut, gridsOut, numAct, indicePairs,
        indicePairUnique, outSpatialShape, batchSize);
        });
  queue.parallel_for<AssignIndicePairsKernelTag<Index, IndexGrid, NDim>>(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, tv::launch::getBlocks(numActIn)) *
                sycl::range<3>(1, 1, tv::launch::CUDA_NUM_THREADS),
            sycl::range<3>(1, 1, tv::launch::CUDA_NUM_THREADS)),
        [=](sycl::nd_item<3> item_ct1) {
      assignIndicePairsKernel<Index, IndexGrid, NDim>(
        item_ct1, indicesOut, gridsOut, numActIn, indicePairs,
        indicePairUnique, outSpatialShape);
        });

    if (resetGrid) {
      queue.submit([&](sycl::handler& cgh) {
        auto indicePairUnique_data_ct0 = indicePairUnique.data();

        cgh.parallel_for<ResetGridKernelTag<Index, IndexGrid, NDim>>(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, tv::launch::getBlocks(numAct)) *
                    sycl::range<3>(1, 1, tv::launch::CUDA_NUM_THREADS),
                sycl::range<3>(1, 1, tv::launch::CUDA_NUM_THREADS)),
            [=](sycl::nd_item<3> item_ct1) {
        resetGridKernel<Index, IndexGrid, NDim>(
          item_ct1, indicePairUnique_data_ct0, gridsOut, numAct);
            });
      });
    }
    return numAct;
  }
};

template <typename Index, typename IndexGrid, unsigned NDim>
struct CreateSubMIndicePairFunctor<tv::TorchGPU, Index, IndexGrid, NDim> {
  Index operator()(const tv::TorchGPU &d, tv::TensorView<const Index> indicesIn,
                   tv::TensorView<IndexGrid> gridsOut,
                   tv::TensorView<Index> indicePairs,
                   tv::TensorView<Index> indiceNum,
                   const tv::SimpleVector<Index, NDim> kernelSize,
                   const tv::SimpleVector<Index, NDim> stride,
                   const tv::SimpleVector<Index, NDim> padding,
                   const tv::SimpleVector<Index, NDim> dilation,
                   const tv::SimpleVector<Index, NDim> outSpatialShape,
                   bool transpose, bool resetGrid) {
  auto numActIn = indicesIn.dim(0);
  if (numActIn == 0) return 0;
  auto& queue = d.getQueue();
  queue.parallel_for<PrepareSubMGridKernelTag<Index, IndexGrid, NDim>>(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, tv::launch::getBlocks(numActIn)) *
                sycl::range<3>(1, 1, tv::launch::CUDA_NUM_THREADS),
            sycl::range<3>(1, 1, tv::launch::CUDA_NUM_THREADS)),
        [=](sycl::nd_item<3> item_ct1) {
      prepareSubMGridKernel<Index, IndexGrid, NDim>(
        item_ct1, indicesIn, gridsOut, outSpatialShape);
        });
  queue.parallel_for<
    GetSubMIndicePairsKernelTag<Index, IndexGrid, NDim, 4096>>(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, tv::launch::getBlocks(numActIn)) *
                sycl::range<3>(1, 1, tv::launch::CUDA_NUM_THREADS),
            sycl::range<3>(1, 1, tv::launch::CUDA_NUM_THREADS)),
        [=](sycl::nd_item<3> item_ct1) {
      getSubMIndicePairsKernel<Index, IndexGrid, NDim, 4096>(
        item_ct1, indicesIn, gridsOut, indicePairs, indiceNum,
        kernelSize, stride, padding, dilation, outSpatialShape);
        });

    if (resetGrid) {
      queue.submit([&](sycl::handler& cgh) {
        auto indicesIn_data_ct0 = indicesIn.data();

        cgh.parallel_for<ResetGridSubMKernelTag<Index, IndexGrid, NDim>>(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, tv::launch::getBlocks(numActIn)) *
                    sycl::range<3>(1, 1, tv::launch::CUDA_NUM_THREADS),
                sycl::range<3>(1, 1, tv::launch::CUDA_NUM_THREADS)),
            [=](sycl::nd_item<3> item_ct1) {
        resetGridSubMKernel<Index, IndexGrid, NDim>(
          item_ct1, indicesIn_data_ct0, gridsOut, outSpatialShape,
          numActIn);
            });
      });
    }
    return numActIn;
  }
};
}  // namespace functor

#define DECLARE_GPU_SPECS_INDEX_NDIM(Index, NDIM)                             \
  template struct functor::CreateConvIndicePairFunctor<tv::TorchGPU, Index,   \
                                                       int, NDIM>;            \
  template struct functor::CreateConvIndicePairFunctorP1<tv::TorchGPU, Index, \
                                                         int, NDIM>;          \
  template struct functor::CreateConvIndicePairFunctorP2<tv::TorchGPU, Index, \
                                                         int, NDIM>;          \
  template struct functor::CreateSubMIndicePairFunctor<tv::TorchGPU, Index,   \
                                                       int, NDIM>;

#define DECLARE_GPU_INDEX(Index)          \
  DECLARE_GPU_SPECS_INDEX_NDIM(Index, 1); \
  DECLARE_GPU_SPECS_INDEX_NDIM(Index, 2); \
  DECLARE_GPU_SPECS_INDEX_NDIM(Index, 3); \
  DECLARE_GPU_SPECS_INDEX_NDIM(Index, 4);

DECLARE_GPU_INDEX(int);

#undef DECLARE_GPU_INDEX
#undef DECLARE_GPU_SPECS_INDEX_NDIM
