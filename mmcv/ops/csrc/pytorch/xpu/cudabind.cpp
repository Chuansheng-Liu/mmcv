#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

#ifdef MMCV_WITH_XPU

torch::Tensor IndiceConvForwardCUDAKernelLauncher(
    torch::Tensor features, torch::Tensor filters, torch::Tensor indicePairs,
    torch::Tensor indiceNum, int64_t numActOut, int64_t _inverse,
    int64_t _subM);

std::vector<torch::Tensor> IndiceConvBackwardCUDAKernelLauncher(
    torch::Tensor features, torch::Tensor filters, torch::Tensor outGrad,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t _inverse,
    int64_t _subM);

static torch::Tensor indice_conv_forward_xpu(
    torch::Tensor features, torch::Tensor filters, torch::Tensor indicePairs,
    torch::Tensor indiceNum, int64_t numActOut, int64_t _inverse,
    int64_t _subM) {
  return IndiceConvForwardCUDAKernelLauncher(features, filters, indicePairs,
                                             indiceNum, numActOut, _inverse,
                                             _subM);
}

torch::Tensor indice_conv_forward_impl(torch::Tensor features,
                                                                             torch::Tensor filters,
                                                                             torch::Tensor indicePairs,
                                                                             torch::Tensor indiceNum,
                                                                             int64_t numActOut, int64_t _inverse,
                                                                             int64_t _subM);

static std::vector<torch::Tensor> indice_conv_backward_xpu(
    torch::Tensor features, torch::Tensor filters, torch::Tensor outGrad,
    torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t _inverse,
    int64_t _subM) {
  return IndiceConvBackwardCUDAKernelLauncher(features, filters, outGrad,
                                              indicePairs, indiceNum, _inverse,
                                              _subM);
}

std::vector<torch::Tensor> indice_conv_backward_impl(
        torch::Tensor features, torch::Tensor filters, torch::Tensor outGrad,
        torch::Tensor indicePairs, torch::Tensor indiceNum, int64_t _inverse,
        int64_t _subM);

REGISTER_DEVICE_IMPL(indice_conv_forward_impl, XPU, indice_conv_forward_xpu);
REGISTER_DEVICE_IMPL(indice_conv_backward_impl, XPU, indice_conv_backward_xpu);

#endif  // MMCV_WITH_XPU
