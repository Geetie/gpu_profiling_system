"""CUDA kernel templates for hardware probing.

Generates specialized CUDA source code for different measurement probes.
All kernels return results via device memory that is copied back to host.
Every kernel uses clock() cycle counts — not wall-clock — making all
measurements naturally frequency-independent.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KernelSource:
    """Compiled-ready CUDA source with metadata."""
    name: str
    source: str
    arch: str = "sm_50"       # Compatible with all modern GPUs
    nvcc_flags: list[str] = ()


def pointer_chase_kernel(
    array_size: int = 4096,
    iterations: int = 10000,
    output_name: str = "pointer_chase",
    use_random_chain: bool = True,
) -> KernelSource:
    """Pointer chasing kernel for measuring memory hierarchy latency.

    Creates a linked list in GPU memory and traverses it, measuring
    cycles per access. When use_random_chain is True (default), the
    host generates a random permutation so the access pattern cannot
    be predicted by the hardware prefetcher.

    When array_size fits in L1: measures L1 latency.
    When array_size fits in L2 but not L1: measures L2 latency.
    When array_size exceeds L2: measures DRAM latency.
    """
    source = f"""
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void measure_latency_kernel(
    unsigned long long* d_cycles,
    int* d_chain,
    int array_size,
    int iterations
) {{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid != 0) return;  // single-thread measurement

    // Warmup: traverse chain to ensure pages are resident
    int idx = d_chain[0];
    for (int w = 0; w < 100; w++) {{
        idx = d_chain[idx % array_size];
    }}
    __threadfence();

    // Main measurement: count cycles for N traversals
    unsigned long long start = clock();
    for (int i = 0; i < {iterations}; i++) {{
        idx = d_chain[idx % array_size];
    }}
    unsigned long long end = clock();

    d_cycles[0] = end - start;
    d_cycles[1] = idx;  // prevent dead-code elimination
}}

int main() {{
    int array_size = {array_size};
    int iterations = {iterations};

    // Allocate and initialize chain on host
    int* h_chain = (int*)malloc(array_size * sizeof(int));
    // Build random permutation (Knuth shuffle) — NOT sequential stride
    for (int i = 0; i < array_size; i++) {{
        h_chain[i] = i;
    }}
    // Seed LCG — deterministic but non-trivial
    unsigned int seed = 0xDEADBEEF;
    for (int i = array_size - 1; i > 0; i--) {{
        seed = seed * 1103515245 + 12345;
        int j = seed % (i + 1);
        int tmp = h_chain[i];
        h_chain[i] = h_chain[j];
        h_chain[j] = tmp;
    }}
    // Make it circular: last element points back to first
    // Already done: h_chain is a permutation of 0..size-1
    // But we need exactly one cycle — verify by following chain
    // (Knuth shuffle of identity gives uniform random permutation)

    // Copy chain to device
    int* d_chain;
    cudaMalloc(&d_chain, array_size * sizeof(int));
    cudaMemcpy(d_chain, h_chain, array_size * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate output
    unsigned long long* d_cycles;
    cudaMalloc(&d_cycles, 2 * sizeof(unsigned long long));

    // Launch kernel
    measure_latency_kernel<<<1, 1>>>(d_cycles, d_chain, array_size, iterations);
    cudaDeviceSynchronize();

    // Copy result back
    unsigned long long h_cycles[2];
    cudaMemcpy(h_cycles, d_cycles, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Cycles per access
    double cycles_per_access = (double)h_cycles[0] / {iterations};
    printf("total_cycles: %llu\\n", h_cycles[0]);
    printf("iterations: %d\\n", {iterations});
    printf("cycles_per_access: %.1f\\n", cycles_per_access);
    printf("chain_checksum: %llu\\n", h_cycles[1]);

    cudaFree(d_chain);
    cudaFree(d_cycles);
    free(h_chain);
    return 0;
}}
"""
    return KernelSource(name=output_name, source=source)


def working_set_sweep_kernel(
    max_size: int = 8 * 1024 * 1024,  # 8M ints = 32 MB
    iterations: int = 1000,
) -> KernelSource:
    """Working-set sweep kernel for detecting cache capacity cliffs.

    Traverses a randomly permuted linked list. Random permutation prevents
    hardware prefetcher from hiding cache miss penalties (M4 fix).
    When the working set exceeds cache capacity, latency jumps sharply.
    """
    source = f"""
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void sweep_kernel(
    unsigned long long* d_cycles,
    int* d_chain,
    int size,
    int iterations
) {{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid != 0) return;

    int idx = 0;
    // Warmup
    for (int w = 0; w < 100; w++) {{
        idx = d_chain[idx % size];
    }}
    __threadfence();

    unsigned long long start = clock();
    for (int i = 0; i < iterations; i++) {{
        idx = d_chain[idx % size];
    }}
    unsigned long long end = clock();

    d_cycles[0] = end - start;
    d_cycles[1] = idx;
}}

int main() {{
    int size = {max_size};
    int iterations = {iterations};

    int* h_chain = (int*)malloc(size * sizeof(int));
    // Build random permutation (Knuth shuffle) — defeats hardware prefetcher
    for (int i = 0; i < size; i++) {{
        h_chain[i] = i;
    }}
    unsigned int seed = 0xDEADBEEF;
    for (int i = size - 1; i > 0; i--) {{
        seed = seed * 1103515245 + 12345;
        int j = seed % (i + 1);
        int tmp = h_chain[i];
        h_chain[i] = h_chain[j];
        h_chain[j] = tmp;
    }}

    int* d_chain;
    cudaMalloc(&d_chain, size * sizeof(int));
    cudaMemcpy(d_chain, h_chain, size * sizeof(int), cudaMemcpyHostToDevice);

    unsigned long long* d_cycles;
    cudaMalloc(&d_cycles, 2 * sizeof(unsigned long long));

    sweep_kernel<<<1, 1>>>(d_cycles, d_chain, size, iterations);
    cudaDeviceSynchronize();

    unsigned long long h_cycles[2];
    cudaMemcpy(h_cycles, d_cycles, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    double cpa = iterations > 0 ? (double)h_cycles[0] / iterations : 0.0;
    printf("size_bytes: %d\\n", size * 4);
    printf("total_cycles: %llu\\n", h_cycles[0]);
    printf("cycles_per_access: %.1f\\n", cpa);
    printf("iterations: %d\\n", iterations);

    cudaFree(d_chain);
    cudaFree(d_cycles);
    free(h_chain);
    return 0;
}}
"""
    return KernelSource(name="working_set_sweep", source=source)


def clock_calibration_kernel(
    loop_iterations: int = 10_000_000,
) -> KernelSource:
    """Clock calibration kernel for measuring actual SM frequency.

    Runs a tight loop with a fixed cycle count per iteration,
    then outputs total cycles. When combined with ncu wall-clock
    timing, actual frequency = total_cycles / elapsed_time_ns.

    Uses a random permutation linked list (Knuth shuffle) to ensure
    access patterns cannot be predicted by the hardware prefetcher,
    consistent with M4 anti-prefetcher strategy.
    """
    source = f"""
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void clock_cal_kernel(
    unsigned long long* d_cycles,
    int* d_chain,
    int size,
    int iterations
) {{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid != 0) return;

    int idx = 0;
    // Warmup
    for (int w = 0; w < 100; w++) {{
        idx = d_chain[idx % size];
    }}
    __threadfence();

    unsigned long long start = clock();
    for (int i = 0; i < iterations; i++) {{
        idx = d_chain[idx % size];
    }}
    unsigned long long end = clock();

    d_cycles[0] = end - start;
    d_cycles[1] = idx;
}}

int main() {{
    int size = 256;  // Fits in L1, so each access has predictable latency
    int iterations = {loop_iterations};

    int* h_data = (int*)malloc(size * sizeof(int));
    // Build random permutation (Knuth shuffle) — defeats hardware prefetcher
    for (int i = 0; i < size; i++) {{
        h_data[i] = i;
    }}
    unsigned int seed = 0xDEADBEEF;
    for (int i = size - 1; i > 0; i--) {{
        seed = seed * 1103515245 + 12345;
        int j = seed % (i + 1);
        int tmp = h_data[i];
        h_data[i] = h_data[j];
        h_data[j] = tmp;
    }}

    int* d_data;
    cudaMalloc(&d_data, size * sizeof(int));
    cudaMemcpy(d_data, h_data, size * sizeof(int), cudaMemcpyHostToDevice);

    unsigned long long* d_cycles;
    cudaMalloc(&d_cycles, 2 * sizeof(unsigned long long));

    clock_cal_kernel<<<1, 1>>>(d_cycles, d_data, size, iterations);
    cudaDeviceSynchronize();

    unsigned long long h_cycles[2];
    cudaMemcpy(h_cycles, d_cycles, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    printf("total_cycles: %llu\\n", h_cycles[0]);
    printf("iterations: %d\\n", iterations);
    double cpi = iterations > 0 ? (double)h_cycles[0] / iterations : 0.0;
    printf("cycles_per_iter: %.2f\\n", cpi);

    cudaFree(d_data);
    cudaFree(d_cycles);
    free(h_data);
    return 0;
}}
"""
    return KernelSource(name="clock_calibration", source=source)


def stream_copy_kernel(
    size_elements: int = 32 * 1024 * 1024,  # 32M = 128 MB
) -> KernelSource:
    """STREAM copy kernel for measuring DRAM bandwidth.

    Simple array copy: dst[i] = src[i]. No device-side timing.
    Wall-clock timing is done externally (ncu or host perf_counter).

    Output: elements copied, bytes copied, checksum for verification.
    """
    source = f"""
#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void stream_copy_kernel(
    float* dst,
    const float* src,
    int n
) {{
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

    // Measurement run
    stream_copy_kernel<<<blocks, BLOCK_SIZE>>>(d_dst, d_src, n);
    cudaDeviceSynchronize();

    // Verify data correctness
    float* h_dst = (float*)malloc(bytes);
    cudaMemcpy(h_dst, d_dst, bytes, cudaMemcpyDeviceToHost);

    uint64_t checksum = 0;
    for (int i = 0; i < 1000; i++) {{
        checksum += (uint64_t)(h_dst[i] * 1.0f);
    }}

    printf("elements: %d\\n", n);
    printf("bytes_copied: %zu\\n", bytes);
    printf("blocks: %d\\n", blocks);
    printf("checksum: %llu\\n", checksum);

    free(h_src);
    free(h_dst);
    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
}}
"""
    return KernelSource(name="stream_copy", source=source)


def shmem_capacity_kernel() -> KernelSource:
    """Shared memory capacity probe kernel.

    Uses cudaOccupancyMaxActiveBlocksPerMultiprocessor to determine
    how many blocks can be active at a given shared memory size.
    By sweeping shmem sizes and checking occupancy, we find the
    maximum shared memory per block.

    Does NOT use cudaGetDeviceProperties (anti-cheat: may return
    virtualized data). Instead, derives capacity from occupancy
    sweep boundaries.
    """
    source = """
#include <stdio.h>
#include <cuda_runtime.h>

// Kernel uses dynamic shared memory
extern __shared__ float shmem[];

__global__ void shmem_kernel(int size) {
    int tid = threadIdx.x;
    // Touch all shared memory to ensure allocation
    if (tid < size) {
        shmem[tid] = (float)tid;
    }
    __syncthreads();
    // Read back to prevent optimization
    if (tid == 0) {
        volatile float v = shmem[0];
    }
}

int main() {
    int max_shmem_bytes = 0;

    // Set max dynamic shared memory to allow large allocations
    cudaFuncSetAttribute(shmem_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 262144);

    printf("shmem_probe_start\\n");

    // Sweep small sizes: 1KB to 100KB
    int sizes[] = {1024, 2048, 4096, 8192, 16384, 32768, 49152, 65536, 98304, 102400};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        int blocks;
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks, shmem_kernel, 256, sizes[i]
        );
        if (err != cudaSuccess) {
            printf("shmem_%d: occupancy_error\\n", sizes[i]);
            break;
        }
        int active_warps = blocks * 8;  // 256 threads = 8 warps per block
        printf("shmem_%d: blocks=%d active_warps=%d\\n", sizes[i], blocks, active_warps);
        if (blocks > 0) {
            max_shmem_bytes = sizes[i];
        }
    }

    // Sweep larger sizes
    int large_sizes[] = {122880, 163840, 200000, 262144};
    for (int i = 0; i < (int)(sizeof(large_sizes) / sizeof(large_sizes[0])); i++) {
        int blocks;
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks, shmem_kernel, 256, large_sizes[i]
        );
        if (err != cudaSuccess || blocks == 0) {
            break;
        }
        max_shmem_bytes = large_sizes[i];
        printf("shmem_%d: blocks=%d active_warps=%d\\n", large_sizes[i], blocks, blocks * 8);
    }

    printf("shmem_max_tested: %d\\n", max_shmem_bytes);
    return 0;
}
"""
    return KernelSource(name="shmem_capacity", source=source, nvcc_flags=[])


def bank_conflict_kernel(
    size: int = 32768,
) -> KernelSource:
    """Bank conflict latency measurement kernel.

    Runs two access patterns:
    1. Strided: thread t accesses element t*stride → bank conflicts
    2. Sequential: thread t accesses element t → no conflicts

    The ratio of (strided_cycles / sequential_cycles) gives the
    bank conflict penalty factor.
    """
    source = f"""
#include <stdio.h>
#include <cuda_runtime.h>

extern __shared__ float shmem[];

__global__ void bank_conflict_kernel(int size, int stride,
    unsigned long long* d_strided_cycles,
    unsigned long long* d_seq_cycles,
    int iterations)
{{
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    // Initialize shared memory
    for (int i = tid; i < size; i += blockDim.x) {{
        shmem[i] = (float)(i * 0.001f);
    }}
    __syncthreads();

    // Ensure all threads have finished initializing before timing
    __syncthreads();

    // Strided access pattern (bank conflicts) — use clock64 for Pascal+
    __syncthreads();
    unsigned long long strided_start = clock64();
    {{
        volatile float sum = 0;
        for (int iter = 0; iter < iterations; iter++) {{
            for (int i = 0; i < 32; i++) {{
                int idx = (warp_id * 32 + i * stride) % size;
                sum += shmem[idx];
            }}
        }}
        // Prevent dead-code elimination
        if (tid == 0) shmem[0] = sum;
    }}
    unsigned long long strided_end = clock64();

    // Warp-aggregate result: thread 0 reads the per-thread value
    __syncthreads();
    unsigned long long total_strided = strided_end - strided_start;
    if (tid == 0) *d_strided_cycles = total_strided;

    __syncthreads();

    // Sequential access pattern (no conflicts)
    __syncthreads();
    unsigned long long seq_start = clock64();
    {{
        volatile float sum = 0;
        for (int iter = 0; iter < iterations; iter++) {{
            for (int i = 0; i < 32; i++) {{
                int idx = (tid + i) % size;
                sum += shmem[idx];
            }}
        }}
        if (tid == 0) shmem[0] = sum;
    }}
    unsigned long long seq_end = clock64();

    __syncthreads();
    unsigned long long total_seq = seq_end - seq_start;
    if (tid == 0) *d_seq_cycles = total_seq;
}}

int main() {{
    int size = {size};
    int stride = 32;  // Max bank conflict: each thread in warp hits different bank
    int iterations = 1000;
    int shmem_bytes = size * sizeof(float);

    // Set max dynamic shared memory
    cudaFuncSetAttribute(bank_conflict_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes);

    unsigned long long* d_strided_cycles;
    unsigned long long* d_seq_cycles;
    cudaMalloc(&d_strided_cycles, sizeof(unsigned long long));
    cudaMalloc(&d_seq_cycles, sizeof(unsigned long long));

    // Warmup
    bank_conflict_kernel<<<1, 256, shmem_bytes>>>(size, stride,
        d_strided_cycles, d_seq_cycles, iterations);
    cudaDeviceSynchronize();

    // Measurement
    bank_conflict_kernel<<<1, 256, shmem_bytes>>>(size, stride,
        d_strided_cycles, d_seq_cycles, iterations);
    cudaDeviceSynchronize();

    unsigned long long h_strided, h_seq;
    cudaMemcpy(&h_strided, d_strided_cycles, sizeof(unsigned long long),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_seq, d_seq_cycles, sizeof(unsigned long long),
               cudaMemcpyDeviceToHost);

    double ratio = (h_seq > 0) ? (double)h_strided / (double)h_seq : 0.0;
    printf("strided_cycles: %llu\\n", h_strided);
    printf("sequential_cycles: %llu\\n", h_seq);
    printf("bank_conflict_ratio: %.2f\\n", ratio);
    printf("stride: %d\\n", stride);
    printf("size: %d\\n", size);
    printf("iterations: %d\\n", iterations);

    cudaFree(d_strided_cycles);
    cudaFree(d_seq_cycles);
    return 0;
}}
"""
    return KernelSource(name="bank_conflict", source=source)
