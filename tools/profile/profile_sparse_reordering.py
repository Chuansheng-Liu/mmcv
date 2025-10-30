#!/usr/bin/env python3
"""Profile XPU sparse gather/scatter functors.

Builds a small torch extension that exposes the gather/scatter helpers and
collects simple runtime statistics across different workloads.
"""

import argparse
import os
import pathlib
import time
from typing import List, Tuple

import torch
from torch.utils.cpp_extension import load

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "tools" / "profile" / "sparse_reordering_profile.cpp"


def build_extension(verbose: bool = False):
    env = os.environ
    env.setdefault("CC", "icx")
    env.setdefault("CXX", "icpx")
    extra_include_paths = [
        str(REPO_ROOT / "mmcv" / "ops" / "csrc" / "common"),
        str(REPO_ROOT / "mmcv" / "ops" / "csrc" / "pytorch"),
        str(REPO_ROOT / "mmcv" / "ops" / "csrc" / "pytorch" / "xpu"),
    ]
    build_dir = REPO_ROOT / "build" / "profile_sparse_reordering"
    build_dir.mkdir(parents=True, exist_ok=True)
    mmcv_sparse_reordering = REPO_ROOT / "mmcv" / "ops" / "csrc" / "pytorch" / "xpu" / "sparse_reordering.dp.cpp"
    return load(
        name="sparse_reordering_profile",
        sources=[str(SRC_PATH), str(mmcv_sparse_reordering)],
        extra_cflags=["-std=c++17", "-fsycl", "-fsycl-unnamed-lambda"],
        extra_ldflags=["-fsycl"],
        extra_include_paths=extra_include_paths,
        build_directory=str(build_dir),
        verbose=verbose,
    )


def _synchronize():
    if hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    else:
        torch.cuda.synchronize()


def benchmark_case(module, n_hot: int, num_planes: int, dtype: torch.dtype,
                    iters: int) -> Tuple[float, float]:
    device = torch.device("xpu")
    num_features = n_hot * 2
    features = torch.randn(num_features, num_planes, device=device, dtype=dtype)
    indices = torch.randint(0, num_features, (n_hot,), device=device, dtype=torch.int32)

    # Warm-up
    for _ in range(3):
        module.sparse_gather(features, indices)
        _synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        module.sparse_gather(features, indices)
    _synchronize()
    gather_time = (time.perf_counter() - start) / iters

    buffer = module.sparse_gather(features, indices)
    out = torch.zeros_like(features)

    for _ in range(3):
        out.zero_()
        module.sparse_scatter_add(out, buffer, indices)
        _synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        out.zero_()
        module.sparse_scatter_add(out, buffer, indices)
    _synchronize()
    scatter_time = (time.perf_counter() - start) / iters

    return gather_time, scatter_time


def run_benchmarks(workloads: List[int], num_planes: int, dtype: torch.dtype,
                   iters: int, verbose: bool):
    module = build_extension(verbose=verbose)
    print(f"Profiling dtype={dtype}, channels={num_planes}, iterations={iters}")
    print(f"{'n_hot':>10} | {'gather (ms)':>12} | {'scatter_add (ms)':>17}")
    print("-" * 45)
    for n_hot in workloads:
        gather_time, scatter_time = benchmark_case(module, n_hot, num_planes,
                                                   dtype, iters)
        print(f"{n_hot:10d} | {gather_time * 1e3:12.3f} | {scatter_time * 1e3:17.3f}")



def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workloads", type=int, nargs="*",
                        default=[4_096, 16_384, 65_536, 262_144],
                        help="Number of active rows to profile")
    parser.add_argument("--channels", type=int, default=128,
                        help="Feature dimension per row")
    parser.add_argument("--dtype", choices=["float32", "float16"],
                        default="float32")
    parser.add_argument("--iters", type=int, default=20,
                        help="Averaging iterations per measurement")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose build output")
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    run_benchmarks(args.workloads, args.channels, dtype, args.iters,
                   verbose=args.verbose)


if __name__ == "__main__":
    main()
