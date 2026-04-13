"""Shared memory bandwidth probe.

Measures shared memory throughput by having a kernel read/write
shared memory at varying access patterns. Bandwidth = bytes / gpu_time_ns.

The kernel reads N floats from shared memory, writes N floats back,
using all threads in a block cooperatively. This saturates the shared
memory bus. Timed via ncu wall-clock profiling.

No cudaGetDeviceProperties used — all measurements from micro-benchmarks.
"""
from __future__ import annotations

import os
import shutil
from typing import Any

from src.infrastructure.probing.probe_helpers import (
    compile_and_run,
    parse_ncu_gpu_time,
    parse_nvcc_output,
    _assess_confidence,
)
from src.infrastructure.sandbox import LocalSandbox, SandboxConfig, SandboxRunner


def _shmem_bandwidth_kernel(block_threads: int = 256) -> str:
    """Generate CUDA source for shared memory bandwidth measurement."""
    return f"""
#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE {block_threads}

// Simple shared memory bandwidth kernel
__global__ void shmem_bw_kernel(int iterations) {{
    extern __shared__ float shmem[];
    int tid = threadIdx.x;
    int size = blockDim.x;

    // Initialize shared memory
    for (int i = tid; i < size; i += blockDim.x) {{
        shmem[i] = (float)(i * 0.001f);
    }}
    __syncthreads();

    // Repeatedly read/write shared memory
    for (int iter = 0; iter < iterations; iter++) {{
        // Read all elements, write back transformed
        float sum = 0;
        for (int i = tid; i < size; i += blockDim.x) {{
            float val = shmem[i];
            val = val * 1.001f + 0.0001f;
            sum += val;
        }}
        // Prevent dead-code elimination
        if (sum > 1e9f) shmem[0] = sum;

        // Write back
        for (int i = tid; i < size; i += blockDim.x) {{
            shmem[i] = (float)(sum + i);
        }}
        __syncthreads();
    }}
}}

int main() {{
    int iterations = 10000;
    int block_threads = {block_threads};
    int shmem_bytes = block_threads * sizeof(float);

    // Set max dynamic shared memory
    cudaFuncSetAttribute(shmem_bw_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes);

    // Launch
    shmem_bw_kernel<<<1, block_threads, shmem_bytes>>>(iterations);
    cudaDeviceSynchronize();

    printf("shmem_bytes: %d\\n", shmem_bytes);
    printf("block_threads: %d\\n", block_threads);
    printf("iterations: %d\\n", iterations);

    // Calculate total bytes transferred (reads + writes per iteration)
    // Each iteration: block_threads reads + block_threads writes
    int total_bytes = iterations * block_threads * 2 * sizeof(float);
    printf("total_bytes_transferred: %d\\n", total_bytes);

    return 0;
}}
"""


# Binary name used by compile_and_run — kept in sync with probe_helpers.py
_BINARY_NAME = "probe_binary"


def probe_shmem_bandwidth(
    sandbox: SandboxRunner | None = None,
) -> dict[str, Any] | None:
    """Measure shared memory bandwidth.

    Strategy:
    1. Compile shmem_bw kernel once
    2. Profile with ncu to get GPU execution time
    3. If ncu fails, reuse the compiled binary + cudaEvents fallback
    4. bandwidth = total_bytes / gpu_time_ns * 1e9

    New-8 fix: Compile kernel once; reuse the binary for all timing paths.

    Returns dict with:
        shmem_bandwidth_gbps: float — shared memory bandwidth in GB/s
        total_bytes: int — total bytes transferred
        gpu_time_ns: float — GPU execution time in ns
        method: str — methodology
    """
    source = _shmem_bandwidth_kernel()
    result = compile_and_run(source, sandbox=sandbox)
    if not result or not result.success:
        return None

    parsed = parse_nvcc_output(result.stdout)
    total_bytes = parsed.get("total_bytes_transferred", 0)
    if not total_bytes:
        return None

    # Profile with ncu first
    work_dir = getattr(sandbox, "sandbox_root", None) or getattr(sandbox, "_sandbox_root", None) if sandbox else None
    if not work_dir:
        return None

    binary_path = os.path.join(work_dir, _BINARY_NAME)
    ncu = shutil.which("ncu")

    if ncu is not None:
        # P1-5: Run ncu 3 times, take minimum gpu_time_ns
        # P2-9: Save raw ncu output from best run
        try:
            min_gpu_time_ns = None
            best_ncu_raw = None
            for _ in range(3):
                ncu_result = sandbox.run(
                    command=ncu,
                    args=["--profile-fromstart", "on", "--print-summary", "per_kernel",
                          binary_path],
                    work_dir=work_dir,
                )
                if ncu_result and ncu_result.success:
                    gpu_time_ns = parse_ncu_gpu_time(ncu_result.stdout + ncu_result.stderr)
                    if gpu_time_ns and gpu_time_ns > 0:
                        if min_gpu_time_ns is None or gpu_time_ns < min_gpu_time_ns:
                            min_gpu_time_ns = gpu_time_ns
                            best_ncu_raw = ncu_result.stdout + ncu_result.stderr
            if min_gpu_time_ns:
                result = _build_result(total_bytes, parsed, min_gpu_time_ns)
                if result and best_ncu_raw:
                    result["_ncu_raw_output"] = best_ncu_raw
                return result
        except Exception:
            pass

    # ncu not available or failed — fall back to cudaEvents
    # Reuse the already-compiled probe_binary for total_bytes, compile
    # only the event-timed wrapper for timing
    gpu_time_ns = _measure_shmem_with_cuda_events(sandbox)
    return _build_result(total_bytes, parsed, gpu_time_ns)


def _measure_shmem_with_cuda_events(sandbox) -> float | None:
    """P2: Measure shmem bandwidth using cudaEventElapsedTime."""
    import os
    import shutil

    runner = sandbox
    work_dir = getattr(runner, "sandbox_root", None) or getattr(runner, "_sandbox_root", None) if runner else None
    nvcc = shutil.which("nvcc")
    if nvcc is None or not runner or not work_dir:
        return None

    from src.infrastructure.probing.probe_helpers import _detect_arch
    arch = _detect_arch(runner)

    source = """
#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void shmem_bw_kernel(int iterations) {
    extern __shared__ float shmem[];
    int tid = threadIdx.x;
    int size = blockDim.x;

    for (int i = tid; i < size; i += blockDim.x) {
        shmem[i] = (float)(i * 0.001f);
    }
    __syncthreads();

    for (int iter = 0; iter < iterations; iter++) {
        float sum = 0;
        for (int i = tid; i < size; i += blockDim.x) {
            float val = shmem[i];
            val = val * 1.001f + 0.0001f;
            sum += val;
        }
        if (sum > 1e9f) shmem[0] = sum;
        for (int i = tid; i < size; i += blockDim.x) {
            shmem[i] = (float)(sum + i);
        }
        __syncthreads();
    }
}

int main() {
    int iterations = 10000;
    int block_threads = BLOCK_SIZE;
    int shmem_bytes = block_threads * sizeof(float);

    cudaFuncSetAttribute(shmem_bw_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes);

    // Warmup
    shmem_bw_kernel<<<1, block_threads, shmem_bytes>>>(iterations);
    cudaDeviceSynchronize();

    // Measurement with cudaEventElapsedTime
    cudaEvent_t evt_start, evt_stop;
    cudaEventCreate(&evt_start);
    cudaEventCreate(&evt_stop);

    cudaEventRecord(evt_start);
    shmem_bw_kernel<<<1, block_threads, shmem_bytes>>>(iterations);
    cudaEventRecord(evt_stop);
    cudaEventSynchronize(evt_stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, evt_start, evt_stop);

    int total_bytes = iterations * block_threads * 2 * sizeof(float);
    printf("gpu_time_ms: %.4f\\n", elapsed_ms);
    printf("total_bytes_transferred: %d\\n", total_bytes);

    cudaEventDestroy(evt_start);
    cudaEventDestroy(evt_stop);
    return 0;
}
"""

    compile_result = runner.run(
        source_code=source,
        command=nvcc,
        args=["-o", "shmem_event", "source.cu", f"-arch={arch}"],
        work_dir=work_dir,
    )
    if not compile_result.success:
        return None

    timed_binary = os.path.join(work_dir, "shmem_event")

    # Run 3 times, take minimum
    min_elapsed_ms = None
    for _ in range(3):
        r = runner.run(
            command=timed_binary,
            args=[],
            work_dir=work_dir,
        )
        if r and r.success:
            p = parse_nvcc_output(r.stdout)
            elapsed_ms = p.get("gpu_time_ms", 0)
            if elapsed_ms > 0:
                if min_elapsed_ms is None or elapsed_ms < min_elapsed_ms:
                    min_elapsed_ms = elapsed_ms

    if min_elapsed_ms and min_elapsed_ms > 0:
        return min_elapsed_ms * 1_000_000  # ms → ns

    return None


def _build_result(total_bytes: int, parsed: dict, gpu_time_ns: float | None) -> dict[str, Any] | None:
    results: dict[str, Any] = {
        "method": "shmem_cooperative_read_write",
        "total_bytes_transferred": int(total_bytes),
    }

    if "shmem_bytes" in parsed:
        results["shmem_block_size_bytes"] = int(parsed["shmem_bytes"])
    if "block_threads" in parsed:
        results["block_threads"] = int(parsed["block_threads"])

    if gpu_time_ns and gpu_time_ns > 0:
        results["gpu_time_ns"] = round(gpu_time_ns, 2)
        bw_gbps = total_bytes / gpu_time_ns
        results["shmem_bandwidth_gbps"] = round(bw_gbps, 2)
        # Confidence: shmem bandwidth typically 500-3000 GB/s
        results["_confidence"] = round(
            _assess_confidence(bw_gbps, low=100.0, high=5000.0,
                               ideal_low=500.0, ideal_high=3000.0), 2
        )

    return results if results.get("total_bytes_transferred") else None
