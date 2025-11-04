# Porting MMCV Sparse Ops to Intel® XPU (SYCL)

## Overview
This document summarizes the porting of MMCV's sparse convolution and reordering operators to Intel® XPU (SYCL), describes the validation and profiling process, and outlines next steps for further development and upstreaming.

## What Was Done
- **Device-agnostic build system**: `setup.py` now detects XPU and builds with SYCL flags and XPU sources.
- **XPU kernel implementations**: All major sparse ops (indice conv, gather, scatter, reordering) have SYCL/XPU versions in `mmcv/ops/csrc/pytorch/xpu/` and `mmcv/ops/csrc/common/xpu/`.
- **dpct stubs**: Minimal CUDA/THC header stubs added for dpct migration compatibility.
- **Python compatibility**: Top-level `mmcv/__init__.py` ensures correct import behavior for downstream projects.
- **Profiling and test scripts**: Added scripts for benchmarking XPU ops and validating correctness/performance.
- **Core file updates**: Minor changes to core spconv and helper files for device-agnostic logic.

## How to Build and Run
> **Note:** For the latest Intel® XPU (SYCL) support, please use the `xpu_ops_porting` branch from [https://github.com/Chuansheng-Liu/mmcv](https://github.com/Chuansheng-Liu/mmcv).

### Prerequisites
- Intel® oneAPI DPC++/SYCL toolchain (2025+ recommended)
- PyTorch with XPU support (2.4+)
- Python 3.9+
- Export `MMCV_ROOT` (and optionally `MMDET3D_ROOT`) to point at your local checkouts.


### Build MMCV with XPU Support (Recommended)

#### Method 1: (Recommended for native code changes)
```bash
cd "$MMCV_ROOT"
FORCE_XPU=1 CXX=icpx CC=icpx python setup.py build_ext --inplace
export PYTHONPATH="${MMCV_ROOT}${MMDET3D_ROOT:+:${MMDET3D_ROOT}}${PYTHONPATH:+:${PYTHONPATH}}"
```
This method ensures all native (C++/SYCL) code is rebuilt and available for import. Use this especially if you modify native code and want to avoid issues with develop mode not always rebuilding extensions.

#### Method 2: (Editable Python, but may not always rebuild native code)
```bash
cd "$MMCV_ROOT"
FORCE_XPU=1 CXX=icpx CC=icpx python setup.py develop
```
This command installs MMCV in development (editable) mode, so Python changes are reflected immediately. However, in some cases, changes in native code may not trigger a rebuild. Use Method 1 above if you encounter this issue, until further investigation.

### Run Unit Tests (XPU)
```bash
# Example: run all tests (ensure XPU device is available)
pytest tests/ --disable-warnings

# Or run specific test modules for sparse ops
pytest tests/test_ops/test_spconv.py -k xpu
```

### Profile XPU Sparse Ops
```bash
python tools/profile/profile_sparse_reordering.py --workloads 4096 16384 65536 --channels 128 --dtype float32 --iters 20
```

## Validation
- All major sparse ops (indice conv, gather, scatter) pass correctness tests on XPU.
- Profiling scripts confirm functional and performance parity with CUDA for supported workloads.
- Device selection is runtime-configurable; all ops fall back to CPU if XPU is unavailable.

## Next Steps
- Upstream these changes via a well-documented pull request.
- Expand test coverage for additional ops and edge cases.
- Collaborate with upstream maintainers for review and integration.
- Continue performance tuning and support for new XPU features as PyTorch evolves.

## Contact
For questions or collaboration, please open an issue or PR on the project repository.
