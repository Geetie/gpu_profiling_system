"""DRAM bandwidth probe — measures memory bandwidth in GB/s.

Uses a STREAM copy pattern (dst[i] = src[i]) to saturate the memory bus.
Wall-clock timing via ncu profiling — no device-side clock() needed.

Bandwidth = bytes_copied / gpu_time_ns * 1e9 (GB/s).

This is frequency-independent — no SM clock assumptions.
"""
from __future__ import annotations

import os
import shutil
from typing import Any

from src.infrastructure.probing.kernel_templates import stream_copy_kernel
from src.infrastructure.probing.probe_helpers import (
    compile_and_run,
    parse_ncu_gpu_time,
    parse_nvcc_output,
    _assess_confidence,
)


def probe_dram_bandwidth(
    sandbox=None,
    actual_freq_mhz: float | None = None,
) -> dict[str, Any] | None:
    """Measure DRAM bandwidth via STREAM copy + ncu profiling.

    Strategy:
    1. Compile and run stream_copy kernel
    2. Profile with ncu to get GPU execution time
    3. bandwidth = bytes / gpu_time_ns * 1e9

    Returns dict with:
        dram_bandwidth_gbps: float — bandwidth in GB/s
        bytes_copied: int — total bytes transferred
        gpu_time_ns: float — GPU execution time in ns
        elements_copied: int — number of elements
        method: str — methodology
    """
    ncu = shutil.which("ncu")
    if ncu is None:
        # ncu not available — return minimal data
        return _run_without_ncu(sandbox)

    kernel = stream_copy_kernel(size_elements=32 * 1024 * 1024)
    result = compile_and_run(kernel.source, sandbox=sandbox)
    if not result or not result.success:
        return None

    parsed = parse_nvcc_output(result.stdout)
    elements = parsed.get("elements", 0)
    bytes_copied = parsed.get("bytes_copied", 0)

    if not elements or not bytes_copied:
        return None

    # Profile with ncu to get GPU time
    work_dir = getattr(sandbox, "sandbox_root", None) or getattr(sandbox, "_sandbox_root", None) if sandbox else None
    binary_path = None
    if work_dir:
        import os
        for name in ["probe_binary", "stream_copy"]:
            p = os.path.join(work_dir, name)
            if os.path.exists(p):
                binary_path = p
                break

    if not binary_path:
        return _build_result(elements, bytes_copied, parsed, None)

    # P1-5: Run ncu 3 times, take minimum gpu_time_ns to reduce
    # scheduling noise on multi-tenant GPUs
    # P2-9: Save raw ncu output from best run for evidence
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
            result = _build_result(elements, bytes_copied, parsed, min_gpu_time_ns)
            if result and best_ncu_raw:
                result["_ncu_raw_output"] = best_ncu_raw
            return result
    except Exception:
        pass

    return _build_result(elements, bytes_copied, parsed, None)


def _measure_with_cuda_events(
    size_elements: int,
    sandbox,
) -> float | None:
    """P2: Measure GPU time using cudaEventElapsedTime as ncu fallback.

    Creates a wrapper binary that times the stream_copy kernel with
    cudaEventRecord/cudaEventElapsedTime.
    """
    import os
    import shutil

    runner = sandbox
    work_dir = getattr(runner, "sandbox_root", None) or getattr(runner, "_sandbox_root", None) if runner else None
    nvcc = shutil.which("nvcc")
    if nvcc is None or not runner or not work_dir:
        return None

    from src.infrastructure.probing.probe_helpers import _detect_arch
    arch = _detect_arch(runner)

    # Source: stream copy with event timing
    source = f"""
#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void stream_copy_kernel(float* dst, const float* src, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {{
        dst[i] = src[i];
    }}
}}

int main() {{
    int n = {size_elements};
    size_t bytes = n * sizeof(float);

    float *d_src, *d_dst;
    cudaMalloc(&d_src, bytes);
    cudaMalloc(&d_dst, bytes);

    // Initialize
    float* h_src = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) h_src[i] = (float)i;
    cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice);

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks > 65535) blocks = 65535;

    // Warmup
    stream_copy_kernel<<<blocks, BLOCK_SIZE>>>(d_dst, d_src, n);
    cudaDeviceSynchronize();

    // Measurement with cudaEventElapsedTime
    cudaEvent_t evt_start, evt_stop;
    cudaEventCreate(&evt_start);
    cudaEventCreate(&evt_stop);

    cudaEventRecord(evt_start);
    stream_copy_kernel<<<blocks, BLOCK_SIZE>>>(d_dst, d_src, n);
    cudaEventRecord(evt_stop);
    cudaEventSynchronize(evt_stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, evt_start, evt_stop);

    printf("elements: %d\\n", n);
    printf("bytes_copied: %zu\\n", bytes);
    printf("gpu_time_ms: %.4f\\n", elapsed_ms);

    cudaEventDestroy(evt_start);
    cudaEventDestroy(evt_stop);
    free(h_src);
    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
}}
"""

    compile_result = runner.run(
        source_code=source,
        command=nvcc,
        args=["-o", "stream_event", "source.cu", f"-arch={arch}"],
        work_dir=work_dir,
    )
    if not compile_result.success:
        return None

    timed_binary = os.path.join(work_dir, "stream_event")

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
        # Convert ms to ns
        return min_elapsed_ms * 1_000_000  # ms → ns

    return None


def _run_without_ncu(sandbox) -> dict[str, Any] | None:
    """Run stream_copy without ncu — falls back to cudaEventElapsedTime [P2]."""
    # P2: Use cudaEventElapsedTime instead of returning basic data only
    size = 32 * 1024 * 1024
    gpu_time_ns = _measure_with_cuda_events(size, sandbox)

    # Also run the basic kernel to get element/byte counts
    kernel = stream_copy_kernel(size_elements=size)
    result = compile_and_run(kernel.source, sandbox=sandbox)
    if not result or not result.success:
        return None
    parsed = parse_nvcc_output(result.stdout)
    elements = parsed.get("elements", 0)
    bytes_copied = parsed.get("bytes_copied", 0)
    if not elements:
        return None
    return _build_result(elements, bytes_copied, parsed, gpu_time_ns)


def _build_result(
    elements: int,
    bytes_copied: int,
    parsed: dict,
    gpu_time_ns: float | None,
) -> dict[str, Any] | None:
    """Build bandwidth result dict."""
    results: dict[str, Any] = {
        "method": "stream_copy_with_ncu_profiling",
        "elements_copied": int(elements),
        "bytes_copied": int(bytes_copied),
    }

    if "blocks" in parsed:
        results["launch_blocks"] = int(parsed["blocks"])

    if gpu_time_ns and gpu_time_ns > 0:
        results["gpu_time_ns"] = round(gpu_time_ns, 2)
        # BW = bytes / ns * 1e9 bytes/s / 1e9 = bytes / ns GB/s
        bw_gbps = bytes_copied / gpu_time_ns
        results["dram_bandwidth_gbps"] = round(bw_gbps, 2)
        # Confidence: DRAM bandwidth 50-2000 GB/s is plausible
        results["_confidence"] = round(
            _assess_confidence(bw_gbps, low=50.0, high=2000.0,
                               ideal_low=100.0, ideal_high=1600.0), 2
        )

    return results if results.get("elements_copied") else None
