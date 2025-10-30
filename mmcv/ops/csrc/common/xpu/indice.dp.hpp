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

#ifndef INDICE_CU_H_
#define INDICE_CU_H_
#include <sycl/sycl.hpp>
#include <utils/spconv/spconv/geometry.h>
#include <utils/spconv/tensorview/tensorview.h>

namespace tv {
namespace detail {

template <typename Index>
inline Index thread_start_x(sycl::nd_item<3> item_ct1) {
  return static_cast<Index>(item_ct1.get_group(2)) *
             static_cast<Index>(item_ct1.get_local_range(2)) +
         static_cast<Index>(item_ct1.get_local_id(2));
}

template <typename Index>
inline Index thread_stride_x(sycl::nd_item<3> item_ct1) {
  return static_cast<Index>(item_ct1.get_group_range(2)) *
         static_cast<Index>(item_ct1.get_local_range(2));
}

template <typename T>
inline T atomicAddGlobal(T* addr, T val) {
#if defined(__CUDA_ARCH__)
  return atomicAdd(addr, val);
#else
  sycl::atomic_ref<T, sycl::memory_order::relaxed,
                   sycl::memory_scope::device,
                   sycl::access::address_space::global_space>
      ref(*addr);
  return ref.fetch_add(val);
#endif
}

}  // namespace detail
}  // namespace tv

template <typename Index, typename IndexGrid, unsigned NDim,
          int KernelMaxVolume = 256>
void prepareIndicePairsKernel(
  sycl::nd_item<3> item_ct1,
    tv::TensorView<const Index> indicesIn, tv::TensorView<Index> indicesOut,
    tv::TensorView<IndexGrid> gridsOut, tv::TensorView<Index> indicePairs,
    tv::TensorView<Index> indiceNum, tv::TensorView<Index> indicePairUnique,
    const tv::SimpleVector<Index, NDim> kernelSize,
    const tv::SimpleVector<Index, NDim> stride,
    const tv::SimpleVector<Index, NDim> padding,
    const tv::SimpleVector<Index, NDim> dilation,
    const tv::SimpleVector<Index, NDim> outSpatialShape) {
  auto numActIn = indicesIn.dim(0);
  Index spatialVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }
  Index kernelVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    kernelVolume *= kernelSize[i];
  }
  Index numValidPoints = 0;
  Index validPoints[KernelMaxVolume * (NDim + 1)];
  Index *pointPtr = nullptr;
  auto indicePairsDim2 = indicePairs.dim(2);
  Index index;
  auto thread_start =
    tv::detail::thread_start_x<Index>(item_ct1);
  auto thread_stride =
    tv::detail::thread_stride_x<Index>(item_ct1);
  for (Index ix = thread_start; ix < numActIn; ix += thread_stride) {
    numValidPoints = getValidOutPos<Index, NDim>(
        indicesIn.data() + ix * (NDim + 1) + 1, kernelSize.data(),
        stride.data(), padding.data(), dilation.data(), outSpatialShape.data(),
        validPoints);
    for (Index i = 0; i < numValidPoints; ++i) {
      pointPtr = validPoints + i * (NDim + 1);
      auto offset = pointPtr[NDim];
  auto oldNum = tv::detail::atomicAddGlobal(indiceNum.data() + offset,
                   Index(1));
      indicePairs(offset, 0, oldNum) = ix;
      index = tv::rowArrayIdx<Index, NDim>(pointPtr, outSpatialShape.data()) +
              spatialVolume * indicesIn(ix, 0);
      indicePairs(offset, 1, oldNum) = index;
      indicePairUnique[offset * indicePairsDim2 + oldNum] = index;
    }
  }
}

template <typename Index, typename IndexGrid, unsigned NDim,
          int KernelMaxVolume = 256>
void prepareDeConvIndicePairsKernel(
  sycl::nd_item<3> item_ct1,
    tv::TensorView<const Index> indicesIn, tv::TensorView<Index> indicesOut,
    tv::TensorView<IndexGrid> gridsOut, tv::TensorView<Index> indicePairs,
    tv::TensorView<Index> indiceNum, tv::TensorView<Index> indicePairUnique,
    const tv::SimpleVector<Index, NDim> kernelSize,
    const tv::SimpleVector<Index, NDim> stride,
    const tv::SimpleVector<Index, NDim> padding,
    const tv::SimpleVector<Index, NDim> dilation,
    const tv::SimpleVector<Index, NDim> outSpatialShape) {
  auto numActIn = indicesIn.dim(0);
  Index spatialVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }
  Index kernelVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    kernelVolume *= kernelSize[i];
  }
  Index numValidPoints = 0;
  Index validPoints[KernelMaxVolume * (NDim + 1)];
  Index *pointPtr = nullptr;
  auto indicePairsDim2 = indicePairs.dim(2);
  Index index;
  auto thread_start =
    tv::detail::thread_start_x<Index>(item_ct1);
  auto thread_stride =
    tv::detail::thread_stride_x<Index>(item_ct1);
  for (Index ix = thread_start; ix < numActIn; ix += thread_stride) {
    numValidPoints = getValidOutPosTranspose<Index, NDim>(
        indicesIn.data() + ix * (NDim + 1) + 1, kernelSize.data(),
        stride.data(), padding.data(), dilation.data(), outSpatialShape.data(),
        validPoints);
    for (Index i = 0; i < numValidPoints; ++i) {
      pointPtr = validPoints + i * (NDim + 1);
      auto offset = pointPtr[NDim];
  auto oldNum = tv::detail::atomicAddGlobal(indiceNum.data() + offset,
                   Index(1));
      indicePairs(offset, 0, oldNum) = ix;
      index = tv::rowArrayIdx<Index, NDim>(pointPtr, outSpatialShape.data()) +
              spatialVolume * indicesIn(ix, 0);
      indicePairs(offset, 1, oldNum) = index;
      indicePairUnique[offset * indicePairsDim2 + oldNum] = index;
    }
  }
}

template <typename Index, typename IndexGrid, unsigned NDim>
void assignGridAndIndiceOutKernel(
    sycl::nd_item<3> item_ct1,
    tv::TensorView<Index> indicesOut, tv::TensorView<IndexGrid> gridsOut,
    int numAct, tv::TensorView<Index> indicePairs,
    tv::TensorView<Index> indicePairUnique,
    const tv::SimpleVector<Index, NDim> outSpatialShape, int batchSize) {
  Index index;
  auto indicesOutPtr = indicesOut.data();
  auto thread_start = tv::detail::thread_start_x<int>(item_ct1);
  auto thread_stride = tv::detail::thread_stride_x<int>(item_ct1);
  for (int ix = thread_start; ix < numAct; ix += thread_stride) {
    index = indicePairUnique[ix];
    gridsOut[index] = ix;
    index = tv::rowArrayIdxInv<Index, NDim>(
        index, indicesOutPtr + ix * (NDim + 1) + 1, outSpatialShape.data());
    indicesOut[ix * (NDim + 1)] = index % batchSize;
  }
}

template <typename Index, typename IndexGrid, unsigned NDim>
void assignIndicePairsKernel(
    sycl::nd_item<3> item_ct1,
    tv::TensorView<Index> indicesOut, tv::TensorView<IndexGrid> gridsOut,
    int numActIn, tv::TensorView<Index> indicePairs,
    tv::TensorView<Index> indicePairUnique,
    const tv::SimpleVector<Index, NDim> outSpatialShape) {
  Index index;
  int kernelVolume = indicePairs.dim(0);
  auto thread_start = tv::detail::thread_start_x<int>(item_ct1);
  auto thread_stride = tv::detail::thread_stride_x<int>(item_ct1);
  for (int ix = thread_start; ix < numActIn; ix += thread_stride) {
    for (int i = 0; i < kernelVolume; ++i) {
      index = indicePairs(i, 1, ix);
      if (index > -1) {
        indicePairs(i, 1, ix) = gridsOut[index];
      }
    }
  }
}

template <typename Index, typename IndexGrid, unsigned NDim>
void prepareSubMGridKernel(
    sycl::nd_item<3> item_ct1,
    tv::TensorView<const Index> indicesIn, tv::TensorView<IndexGrid> gridsOut,
    const tv::SimpleVector<Index, NDim> outSpatialShape) {
  auto numActIn = indicesIn.dim(0);
  Index spatialVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }
  Index index = 0;
  auto thread_start =
    tv::detail::thread_start_x<int>(item_ct1);
  auto thread_stride =
    tv::detail::thread_stride_x<int>(item_ct1);
  for (int ix = thread_start; ix < numActIn; ix += thread_stride) {
    index = tv::rowArrayIdx<Index, NDim>(indicesIn.data() + ix * (NDim + 1) + 1,
                                         outSpatialShape.data()) +
            spatialVolume * indicesIn(ix, 0);
    gridsOut[index] = ix;
  }
}

template <typename Index, typename IndexGrid, unsigned NDim,
          int KernelMaxVolume = 256>
void getSubMIndicePairsKernel(
  sycl::nd_item<3> item_ct1,
    tv::TensorView<const Index> indicesIn, tv::TensorView<IndexGrid> gridsOut,
    tv::TensorView<Index> indicePairs, tv::TensorView<Index> indiceNum,
    const tv::SimpleVector<Index, NDim> kernelSize,
    const tv::SimpleVector<Index, NDim> stride,
    const tv::SimpleVector<Index, NDim> padding,
    const tv::SimpleVector<Index, NDim> dilation,
    const tv::SimpleVector<Index, NDim> outSpatialShape) {
  auto numActIn = indicesIn.dim(0);
  Index spatialVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }
  Index numValidPoints = 0;
  Index validPoints[KernelMaxVolume * (NDim + 1)];
  Index *pointPtr = nullptr;
  Index index = 0;
  auto thread_start =
    tv::detail::thread_start_x<int>(item_ct1);
  auto thread_stride =
    tv::detail::thread_stride_x<int>(item_ct1);
  for (int ix = thread_start; ix < numActIn; ix += thread_stride) {
    numValidPoints = getValidOutPos<Index, NDim>(
        indicesIn.data() + ix * (NDim + 1) + 1, kernelSize.data(),
        stride.data(), padding.data(), dilation.data(), outSpatialShape.data(),
        validPoints);
    for (int i = 0; i < numValidPoints; ++i) {
      pointPtr = validPoints + i * (NDim + 1);
      auto offset = pointPtr[NDim];
      index = tv::rowArrayIdx<Index, NDim>(pointPtr, outSpatialShape.data()) +
              spatialVolume * indicesIn(ix, 0);
      if (gridsOut[index] > -1) {
  auto oldNum = tv::detail::atomicAddGlobal(indiceNum.data() + offset,
              Index(1));
        indicePairs(offset, 1, oldNum) = gridsOut[index];
        indicePairs(offset, 0, oldNum) = ix;
      }
    }
  }
}

template <typename Index, typename IndexGrid, unsigned NDim>
void resetGridKernel(sycl::nd_item<3> item_ct1, const Index *indicePairUnique,
                     tv::TensorView<IndexGrid> gridsOut, int numAct) {
  auto thread_start = tv::detail::thread_start_x<int>(item_ct1);
  auto thread_stride = tv::detail::thread_stride_x<int>(item_ct1);
  for (int ix = thread_start; ix < numAct; ix += thread_stride) {
    gridsOut[indicePairUnique[ix]] = -1;
  }
}

template <typename Index, typename IndexGrid, unsigned NDim>
void resetGridSubMKernel(
    sycl::nd_item<3> item_ct1, const Index *indices,
    tv::TensorView<IndexGrid> gridsOut,
    const tv::SimpleVector<Index, NDim> outSpatialShape, int numAct) {
  int outSpatialShapeReg[NDim];
  for (int i = 0; i < NDim; ++i) {
    outSpatialShapeReg[i] = outSpatialShape[i];
  }
  Index spatialVolume = 1;
  auto indsPtr = indices;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }
  Index index;
  auto thread_start = tv::detail::thread_start_x<int>(item_ct1);
  auto thread_stride = tv::detail::thread_stride_x<int>(item_ct1);
  for (int ix = thread_start; ix < numAct; ix += thread_stride) {
    indsPtr = indices + ix * (NDim + 1);
    index = tv::rowArrayIdx<Index, NDim>(indsPtr + 1, outSpatialShapeReg);
    gridsOut[index + spatialVolume * indsPtr[0]] = -1;
  }
}

#endif
