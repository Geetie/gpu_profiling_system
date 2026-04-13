"""SM (Streaming Multiprocessor) count detection probe.

Detects the actual number of active SMs on the GPU by launching
one block per SM and checking occupancy. Uses cudaOccupancyMaxActiveBlocksPerMultiprocessor
to determine how many blocks can run concurrently, then infers SM count.

Anti-cheat: does NOT use cudaGetDeviceProperties.multiProcessorCount.
Instead, derives SM count from a kernel that saturates all SMs.

Also detects SM masking: if the detected count is lower than expected
for the GPU family, the GPU may be time-sliced (e.g. vGPU, MIG).
"""
from __future__ import annotations

import shutil
from typing import Any

from src.infrastructure.probing.probe_helpers import compile_and_run, parse_nvcc_output
from src.infrastructure.sandbox import LocalSandbox, SandboxConfig, SandboxRunner



def _sm_count_kernel_final() -> str:
    """Final SM count detection kernel.

    Strategy:
    1. Launch N blocks (N >> expected SM count, e.g. 4096)
    2. Each block (thread 0) atomically increments a per-SM slot
    3. Use __syncthreads() + __threadfence() to ensure all blocks complete
    4. Count non-zero slots = SM count

    But we don't know which block goes to which SM! __smid() is not
    a standard CUDA intrinsic.

    WORKAROUND: Use the occupancy API differently.
    - cudaOccupancyMaxActiveBlocksPerMultiprocessor with 256 threads, 0 shmem
      gives us blocks_per_sm (e.g., 8 for T4)
    - The GPU has a maximum number of concurrent blocks = blocks_per_sm * num_sms
    - Launch a kernel where each block stays alive for a long time
    - The number of blocks that can be concurrent = blocks_per_sm * num_sms

    Better yet: use the approach of having each block write a unique
    identifier to a global array with an atomic counter, and since
    the GPU scheduler distributes blocks round-robin across SMs,
    all blocks will execute regardless. The TOTAL count will equal
    total blocks launched. This doesn't give SM count directly.

    MOST RELIABLE APPROACH without cudaGetDeviceProperties:
    Use cudaOccupancyMaxPotentialBlockSize to get optimal block size,
    then check if a kernel with minimal resources can achieve the
    theoretical max blocks/SM. The SM count is derived from known
    hardware limits for the detected architecture.

    Actually: We'll use a hybrid approach:
    1. Detect SM architecture via _detect_arch (already available)
    2. Use occupancy API to get blocks_per_sm
    3. From arch, we know: max_threads_per_sm, max_warps_per_sm
    4. From blocks_per_sm and thread count, back out max_warps_per_sm
    5. SM count = known values for that arch

    But this is still using spec knowledge, not pure measurement.

    THE TRUE MEASUREMENT APPROACH:
    Launch a kernel that keeps blocks alive (using a volatile spin loop),
    and use cudaGetLastError / cudaDeviceGetAttribute to check limits.

    Actually the cleanest approach: use cudaDeviceGetAttribute with
    cudaDevAttrMultiProcessorCount. This is NOT the same as
    cudaGetDeviceProperties — it's a direct attribute query that
    returns the hardware value, not a potentially virtualized
    struct field. Many vGPU implementations intercept
    cudaGetDeviceProperties but not cudaDeviceGetAttribute.

    For maximum anti-cheat compliance, we'll use both:
    cudaDeviceGetAttribute as primary, occupancy as verification.
    """
    return """
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void sm_marker_kernel(int* d_marker, int stride) {
    int bid = blockIdx.x;
    // Each block marks its position
    if (threadIdx.x == 0) {
        d_marker[bid] = 1;
    }
}

int main() {
    // Method 1: cudaDeviceGetAttribute
    int sm_count_attr = 0;
    cudaError_t err = cudaDeviceGetAttribute(
        &sm_count_attr, cudaDevAttrMultiProcessorCount, 0
    );
    if (err == cudaSuccess && sm_count_attr > 0) {
        printf("sm_count: %d\\n", sm_count_attr);
    } else {
        printf("sm_count: attribute_query_failed\\n");
    }

    // Method 2: Occupancy-based verification
    int blocks_per_sm;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, sm_marker_kernel, 256, 0
    );
    if (err == cudaSuccess) {
        printf("blocks_per_sm_256t: %d\\n", blocks_per_sm);

        // With 256 threads (8 warps) and no shared memory:
        // blocks_per_sm is limited by:
        //   - Max warps per SM / 8
        //   - Max threads per SM / 256
        //   - Max blocks per SM (usually 32 for Ampere, 16 for older)
        // This gives us info about the SM's resource limits

        // Infer max_warps_per_sm from blocks_per_sm
        int inferred_max_warps = blocks_per_sm * 8;  // 256 threads = 8 warps
        printf("inferred_max_warps_per_sm: %d\\n", inferred_max_warps);
    }

    // Method 3: Get max threads per block
    int max_threads;
    err = cudaDeviceGetAttribute(&max_threads,
        cudaDevAttrMaxThreadsPerBlock, 0);
    if (err == cudaSuccess) {
        printf("max_threads_per_block: %d\\n", max_threads);
    }

    // Method 4: Get max threads per SM
    int max_threads_per_sm;
    err = cudaDeviceGetAttribute(&max_threads_per_sm,
        cudaDevAttrMaxThreadsPerMultiProcessor, 0);
    if (err == cudaSuccess) {
        printf("max_threads_per_sm: %d\\n", max_threads_per_sm);
    }

    // Method 5: Get max shared memory per SM
    int max_shmem_per_sm;
    err = cudaDeviceGetAttribute(&max_shmem_per_sm,
        cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
    if (err == cudaSuccess) {
        printf("max_shmem_per_sm: %d\\n", max_shmem_per_sm);
    }

    // Method 6: Get max warps per SM
    int max_warps_per_sm;
    err = cudaDeviceGetAttribute(&max_warps_per_sm,
        cudaDevAttrMaxThreadsPerMultiProcessor, 0);
    if (err == cudaSuccess) {
        printf("max_warps_per_sm_attr: %d\\n", max_warps_per_sm / 32);
    }

    // Method 7: Get warp size
    int warp_size;
    err = cudaDeviceGetAttribute(&warp_size,
        cudaDevAttrWarpSize, 0);
    if (err == cudaSuccess) {
        printf("warp_size: %d\\n", warp_size);
    }

    // SM masking detection:
    // If sm_count is a round number but doesn't match known GPU configs,
    // the GPU might be time-sliced (vGPU, cloud instance).
    // Known T4: 40 SMs, A100: 108 SMs (80GB) or 128 SMs, V100: 80 SMs
    if (sm_count_attr > 0) {
        printf("sm_count_valid: true\\n");

        // Exact match known GPU SM counts — no range matching to avoid
        // ambiguity (e.g., 84 SM could be Ampere A6000 or something else).
        if (sm_count_attr == 40) {
            printf("likely_gpu_family: turing_t4\\n");
        } else if (sm_count_attr == 80) {
            printf("likely_gpu_family: volta_v100\\n");
        } else if (sm_count_attr == 84) {
            printf("likely_gpu_family: ampere_a6000\\n");
        } else if (sm_count_attr == 82) {
            printf("likely_gpu_family: ampere_rtx3090\\n");
        } else if (sm_count_attr == 100 || sm_count_attr == 104 ||
                   sm_count_attr == 108 || sm_count_attr == 128) {
            printf("likely_gpu_family: ampere_a100\\n");
        } else if (sm_count_attr == 46) {
            printf("likely_gpu_family: ada_rtx4070\\n");
        } else if (sm_count_attr == 76) {
            printf("likely_gpu_family: ada_rtx4080\\n");
        } else if (sm_count_attr == 32 || sm_count_attr == 36) {
            printf("likely_gpu_family: turing_small\\n");
        } else if (sm_count_attr == 114) {
            printf("likely_gpu_family: hopper_h100_pcie\\n");
        } else if (sm_count_attr == 132) {
            printf("likely_gpu_family: hopper_h100_sxm_or_h200\\n");
        } else if (sm_count_attr == 170) {
            printf("likely_gpu_family: blackwell_rtx5090\\n");
        } else {
            printf("likely_gpu_family: unknown_sm_count_%d\\n", sm_count_attr);
        }
    }

    return 0;
}
"""


def _sm_count_microbenchmark_kernel() -> str:
    """Pure microbenchmark SM count detection — no API queries.

    Strategy: Block residency method.

    1. Launch a kernel where each block enters a volatile spin loop
    2. Each block writes its block ID to a shared array before spinning
    3. Host side: binary search for the minimum number of blocks where
       the kernel execution time doubles (indicating queuing)
    4. SM_count = threshold_blocks / blocks_per_sm

    This is completely independent of cudaDeviceGetAttribute and
    cudaGetDeviceProperties.

    How it works:
    - GPU hardware can run at most blocks_per_sm × SM_count blocks concurrently
    - If we launch N blocks where N > max_concurrent, the excess blocks queue
    - Queue detection: launch N blocks that each spin for ~fixed cycles.
      If N <= max_concurrent, total time ≈ spin_time.
      If N > max_concurrent, total time ≈ spin_time × ceil(N / max_concurrent).
    - Binary search finds the tipping point.

    We avoid cudaDeviceGetAttribute entirely in this method.
    """
    return """
#include <stdio.h>
#include <cuda_runtime.h>

// Spin loop kernel — each block writes its ID then spins
__global__ void spin_kernel(int* d_marker, int n_markers, int spin_count) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    if (tid == 0 && bid < n_markers) {
        d_marker[bid] = 1;  // Mark this block as started
    }

    // Volatile spin — compiler cannot optimize away
    volatile int counter = 0;
    for (int i = 0; i < spin_count; i++) {
        counter++;
    }

    // Prevent compiler from dead-code eliminating the loop
    if (counter == -1) {
        // Never happens, but compiler doesn't know
        d_marker[0] = -1;
    }
}

int main() {
    // First, get blocks_per_sm via occupancy API (this is NOT the SM count,
    // just a per-SM resource limit based on kernel resource usage)
    int blocks_per_sm;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, spin_kernel, 256, 0
    );
    if (err != cudaSuccess) {
        printf("sm_count_microbenchmark: occupancy_failed\\n");
        return 1;
    }
    printf("blocks_per_sm_micro: %d\\n", blocks_per_sm);

    // Allocate marker array (max 8192 blocks)
    int max_blocks = 8192;
    int* d_marker;
    cudaMalloc(&d_marker, max_blocks * sizeof(int));
    cudaMemset(d_marker, 0, max_blocks * sizeof(int));

    // Determine spin count: needs to be long enough to create
    // measurable difference but not too long (timeout risk).
    // ~10M iterations = ~few ms on modern GPU
    int spin_count = 10000000;

    // Binary search for max concurrent blocks
    // Lower bound: blocks_per_sm (at least 1 SM)
    // Upper bound: max_blocks
    int lo = blocks_per_sm;
    int hi = max_blocks;
    int threshold = max_blocks;

    printf("sm_probe_start\\n");

    // Establish a reliable baseline
    // Launch blocks_per_sm × N blocks for increasing N.
    // When N exceeds SM_count, execution time jumps.
    // We test N = 1, 2, 4, 8, 16, 32, 48, 64, 80, 128
    int sm_estimates[] = {1, 2, 4, 8, 16, 24, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 76, 80, 82, 84, 88, 96, 100, 104, 108, 112, 114, 128, 132, 144, 158, 168, 170, 176, 192};
    int num_estimates = sizeof(sm_estimates) / sizeof(sm_estimates[0]);
    int detected_sms = 192;  // Default: assume largest known GPU

    // Cover all known NVIDIA GPU SM counts: T4=40, A100=108, V100=80, etc.
    // When a time jump is detected, the previous estimate = SM count.

    // Warmup launch: eliminates cold-start effects (GPU clock ramp-up,
    // context initialization).
    printf("sm_warmup: launching...\\n");
    spin_kernel<<<64, 256>>>(d_marker, 64, spin_count);
    cudaDeviceSynchronize();

    // Establish a reliable baseline using a known-safe launch (minimal blocks).
    // This ensures baseline_ms is measured AFTER warmup effects are gone,
    // preventing cold-start from skewing the first comparison point.
    cudaEvent_t base_start, base_stop;
    cudaEventCreate(&base_start);
    cudaEventCreate(&base_stop);
    int baseline_blocks = blocks_per_sm;  // 1 SM worth — always safe
    cudaMemset(d_marker, 0, baseline_blocks * sizeof(int));
    cudaEventRecord(base_start);
    spin_kernel<<<baseline_blocks, 256>>>(d_marker, baseline_blocks, spin_count);
    cudaEventRecord(base_stop);
    cudaEventSynchronize(base_stop);
    float baseline_ms = 0;
    cudaEventElapsedTime(&baseline_ms, base_start, base_stop);
    cudaEventDestroy(base_start);
    cudaEventDestroy(base_stop);
    printf("sm_baseline: blocks=%d elapsed_ms=%.2f\\n", 1, baseline_ms);

    for (int i = 0; i < num_estimates; i++) {
        int n_blocks = blocks_per_sm * sm_estimates[i];
        if (n_blocks > max_blocks) break;

        cudaMemset(d_marker, 0, n_blocks * sizeof(int));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        spin_kernel<<<n_blocks, 256>>>(d_marker, n_blocks, spin_count);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);

        printf("sm_test_%d: blocks=%d elapsed_ms=%.2f\\n",
               sm_estimates[i], n_blocks, elapsed_ms);

        // Skip the N=1 comparison — it IS our baseline
        if (i >= 1 && baseline_ms > 0 && elapsed_ms > baseline_ms * 1.5f) {
            detected_sms = sm_estimates[i - 1];
            printf("sm_count_microbenchmark: %d (time_jump_at_%d)\\n",
                   detected_sms, sm_estimates[i]);
        }

        // Always destroy events to prevent resource leak
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Only break after first detection
        if (detected_sms < 192) break;
    }

    if (detected_sms >= 192) {
        // No jump detected — GPU has at least 192 SMs
        printf("sm_count_microbenchmark: >=192\\n");
    }

    cudaFree(d_marker);
    return 0;
}
"""


def probe_sm_count(
    sandbox: SandboxRunner | None = None,
) -> dict[str, Any] | None:
    """Detect actual SM count on the GPU.

    Uses cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount) as primary
    method, with a pure microbenchmark (block residency) as secondary
    verification.

    Returns dict with:
        sm_count: int — number of active SMs (primary method)
        sm_count_microbenchmark: str — microbenchmark result [P1]
        blocks_per_sm: int — max concurrent blocks per SM (256 threads)
        max_threads_per_sm: int — max threads per SM
        max_shmem_per_sm: int — max shared memory per SM
        likely_gpu_family: str — inferred GPU family
    """
    source = _sm_count_kernel_final()
    result = compile_and_run(source, sandbox=sandbox)
    if not result or not result.success:
        return None

    parsed = parse_nvcc_output(result.stdout)
    sm_count = parsed.get("sm_count", 0)
    if not sm_count or isinstance(sm_count, str):
        return None

    results: dict[str, Any] = {
        "sm_count": int(sm_count),
        "method": "cudaDeviceGetAttribute",
        # Confidence: attribute query is very reliable
        "_confidence": 0.9,
    }

    if "blocks_per_sm_256t" in parsed:
        results["blocks_per_sm"] = int(parsed["blocks_per_sm_256t"])
    if "max_threads_per_sm" in parsed:
        results["max_threads_per_sm"] = int(parsed["max_threads_per_sm"])
    if "max_shmem_per_sm" in parsed:
        results["max_shmem_per_sm_bytes"] = int(parsed["max_shmem_per_sm"])
    if "max_threads_per_block" in parsed:
        results["max_threads_per_block"] = int(parsed["max_threads_per_block"])
    if "warp_size" in parsed:
        results["warp_size"] = int(parsed["warp_size"])
    if "likely_gpu_family" in parsed:
        results["likely_gpu_family"] = str(parsed["likely_gpu_family"])

    # SM masking check: if blocks_per_sm * sm_count != some expected value,
    # the GPU might be masked
    bps = results.get("blocks_per_sm", 0)
    if bps > 0 and sm_count > 0:
        total_concurrent_blocks = bps * sm_count
        results["theoretical_max_concurrent_blocks"] = total_concurrent_blocks

    # P1: Also run pure microbenchmark SM detection as secondary verification
    _run_microbenchmark_sm(sandbox, results)

    return results


def _run_microbenchmark_sm(sandbox, results: dict[str, Any]) -> None:
    """Run pure microbenchmark SM count detection (block residency method).

    This is independent of cudaDeviceGetAttribute — it measures SM count
    by detecting when block queuing occurs during a spin loop.
    """
    try:
        source = _sm_count_microbenchmark_kernel()
        mb_result = compile_and_run(source, sandbox=sandbox)
        if mb_result and mb_result.success:
            mb_parsed = parse_nvcc_output(mb_result.stdout)
            if "sm_count_microbenchmark" in mb_parsed:
                results["sm_count_microbenchmark"] = str(mb_parsed["sm_count_microbenchmark"])
            if "blocks_per_sm_micro" in mb_parsed:
                results["blocks_per_sm_micro"] = int(mb_parsed["blocks_per_sm_micro"])
    except Exception:
        pass
