"""Micro-benchmark generation handler — infrastructure layer.

Generates CUDA kernel source code based on benchmark type and parameters.
Uses template-based code generation when no model caller is available.
"""
from __future__ import annotations

import os
from typing import Any


def generate_microbenchmark_handler(arguments: dict[str, Any]) -> dict[str, Any]:
    """Auto-generate CUDA micro-benchmark kernels.

    Args (from input_schema):
        benchmark_type: str — type of benchmark ("pointer_chase", "working_set", "timing_loop", "stream")
        parameters: dict — benchmark-specific parameters

    Returns (from output_schema):
        source_code: str — generated CUDA source
        file_path: str — path where source was written (empty string if not saved)
    """
    benchmark_type = arguments.get("benchmark_type", "pointer_chase")
    parameters = arguments.get("parameters", {})

    generators = {
        "pointer_chase": _pointer_chasing_kernel,
        "working_set": _working_set_kernel,
        "timing_loop": _timing_loop_kernel,
        "stream": _stream_kernel,
    }

    generator = generators.get(benchmark_type, _generic_kernel)
    source_code = generator(parameters)

    return {
        "source_code": source_code,
        "file_path": "",  # Caller decides where to write
    }


def _pointer_chasing_kernel(params: dict[str, Any]) -> str:
    iterations = params.get("iterations", 100000)
    return f"""\
// Pointer-chasing kernel for latency measurement
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void pointer_chase(uint32_t* next, uint32_t* latency) {{
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t start = idx;
    uint32_t iterations = {iterations};

    clock_t t0 = clock();
    for (uint32_t i = 0; i < iterations; i++) {{
        idx = next[idx];
    }}
    clock_t t1 = clock();

    latency[start] = t1 - t0;
}}
"""


def _working_set_kernel(params: dict[str, Any]) -> str:
    size = params.get("size", 1024)
    return f"""\
// Working-set sweep kernel for capacity measurement
#include <cuda_runtime.h>

__global__ void working_set(int* data, int* result, int size) {{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int sum = 0;
    for (int stride = 1; stride < size; stride *= 2) {{
        sum += data[(idx + stride) % size];
    }}
    result[idx] = sum;
}}
"""


def _timing_loop_kernel(params: dict[str, Any]) -> str:
    iterations = params.get("iterations", 1000000)
    return f"""\
// Timing loop kernel for clock measurement
#include <cuda_runtime.h>

__global__ void timing_loop(uint64_t* out) {{
    clock_t t0 = clock();
    for (volatile int i = 0; i < {iterations}; i++) {{}}
    clock_t t1 = clock();
    out[threadIdx.x] = t1 - t0;
}}
"""


def _stream_kernel(params: dict[str, Any]) -> str:
    return """\
// Stream kernel for bandwidth measurement
#include <cuda_runtime.h>

__global__ void stream_copy(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] * 2.0f + 1.0f;
    }
}
"""


def _generic_kernel(params: dict[str, Any]) -> str:
    benchmark_type = params.get("benchmark_type", "generic")
    return f"""\
// Generic kernel for {benchmark_type}
#include <cuda_runtime.h>

__global__ void generic_kernel(int* out) {{
    out[threadIdx.x] = threadIdx.x;
}}
"""
