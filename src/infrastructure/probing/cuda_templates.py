"""Pre-defined CUDA code templates for common GPU profiling targets.

These templates provide fallback code when the LLM fails to generate correct
measurements. Each template includes the full CUDA source code, compile flags,
and the expected stdout format.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CUDATemplate:
    """A CUDA code template for a specific measurement target."""
    target_name: str
    source_code: str
    compile_flags: list[str]
    description: str
    expected_output_prefix: str
    num_trials: int = 3


def _get_launch_sm_count_template() -> CUDATemplate:
    """Template for measuring SM count."""
    return CUDATemplate(
        target_name="launch__sm_count",
        source_code=r'''#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <algorithm>

__global__ void flag_kernel(volatile int* flags, int num_blocks) {
    if (blockIdx.x < num_blocks) {
        flags[blockIdx.x] = 1;
    }
}

int main() {
    int sm_count = 0;
    cudaError_t err = cudaDeviceGetAttribute(
        &sm_count, cudaDevAttrMultiProcessorCount, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetAttribute failed: %s\n",
                cudaGetErrorString(err));
        return 1;
    }
    printf("launch__sm_count: %d\n", sm_count);

    // Cross-validation: block ID sweep
    const int max_blocks = 256;
    volatile int* d_flags = nullptr;
    cudaMalloc((void**)&d_flags, max_blocks * sizeof(int));
    cudaMemset((void*)d_flags, 0, max_blocks * sizeof(int));

    flag_kernel<<<max_blocks, 1>>>(d_flags, max_blocks);
    cudaDeviceSynchronize();

    int h_flags[max_blocks];
    cudaMemcpy(h_flags, (void*)d_flags, max_blocks * sizeof(int),
               cudaMemcpyDeviceToHost);

    int executed_blocks = 0;
    for (int i = 0; i < max_blocks; i++) {
        if (h_flags[i] == 1) executed_blocks++;
    }
    printf("Cross-validation: %d blocks executed out of %d launched\n",
           executed_blocks, max_blocks);

    cudaFree((void*)d_flags);
    return 0;
}
''',
        compile_flags=["-O3", "-arch=native"],
        description="SM count via cudaDeviceGetAttribute + block ID sweep",
        expected_output_prefix="launch__sm_count",
    )


def _get_dram_bytes_read_template() -> CUDATemplate:
    """Template for measuring DRAM read bandwidth."""
    return CUDATemplate(
        target_name="dram__bytes_read.sum.per_second",
        source_code=r'''#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <cstdlib>

#define BUFFER_SIZE (64 * 1024 * 1024)  // 64MB

__global__ void read_only_kernel(const float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    volatile float sink = 0.0f;
    while (idx < n) {
        sink += __ldg(&data[idx]);
        idx += stride;
    }
}

int main() {
    float* d_data = nullptr;
    cudaMalloc((void**)&d_data, BUFFER_SIZE);
    cudaMemset(d_data, 0x42, BUFFER_SIZE);

    int threads = 256;
    int blocks = 65535;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double max_bandwidth = 0.0;

    for (int trial = 0; trial < 3; trial++) {
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        read_only_kernel<<<blocks, threads>>>(d_data, BUFFER_SIZE / sizeof(float));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        double seconds = ms / 1000.0;
        double bytes = (double)(BUFFER_SIZE / sizeof(float)) * sizeof(float);
        double bandwidth = bytes / seconds;

        if (bandwidth > max_bandwidth) max_bandwidth = bandwidth;
    }

    printf("dram__bytes_read.sum.per_second: %.2f\n", max_bandwidth);

    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
''',
        compile_flags=["-O3", "-arch=native", "-Xptxas=-dlcm=ca"],
        description="DRAM read bandwidth via read-only STREAM kernel",
        expected_output_prefix="dram__bytes_read.sum.per_second",
    )


def _get_dram_bytes_write_template() -> CUDATemplate:
    """Template for measuring DRAM write bandwidth."""
    return CUDATemplate(
        target_name="dram__bytes_write.sum.per_second",
        source_code=r'''#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <cstdlib>

#define BUFFER_SIZE (64 * 1024 * 1024)  // 64MB

__global__ void write_only_kernel(volatile float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < n) {
        data[idx] = 1.23456f;
        idx += stride;
    }
}

int main() {
    volatile float* d_data = nullptr;
    cudaMalloc((void**)&d_data, BUFFER_SIZE);

    int threads = 256;
    int blocks = 65535;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double max_bandwidth = 0.0;

    for (int trial = 0; trial < 3; trial++) {
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        write_only_kernel<<<blocks, threads>>>((float*)d_data, BUFFER_SIZE / sizeof(float));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        double seconds = ms / 1000.0;
        double bytes = (double)(BUFFER_SIZE / sizeof(float)) * sizeof(float);
        double bandwidth = bytes / seconds;

        if (bandwidth > max_bandwidth) max_bandwidth = bandwidth;
    }

    printf("dram__bytes_write.sum.per_second: %.2f\n", max_bandwidth);

    cudaFree((void*)d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
''',
        compile_flags=["-O3", "-arch=native"],
        description="DRAM write bandwidth via write-only STREAM kernel",
        expected_output_prefix="dram__bytes_write.sum.per_second",
    )


def _get_device_max_gpu_freq_template() -> CUDATemplate:
    """Template for measuring max GPU clock frequency."""
    return CUDATemplate(
        target_name="device__attribute_max_gpu_frequency_khz",
        source_code=r'''#include <cuda_runtime.h>
#include <cstdio>

int main() {
    int clock_rate = 0;
    cudaError_t err = cudaDeviceGetAttribute(
        &clock_rate, cudaDevAttrClockRate, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetAttribute failed: %s\n",
                cudaGetErrorString(err));
        return 1;
    }
    // clock_rate is in kHz
    printf("device__attribute_max_gpu_frequency_khz: %d\n", clock_rate);
    return 0;
}
''',
        compile_flags=["-O3", "-arch=native"],
        description="Max GPU clock frequency via cudaDevAttrClockRate",
        expected_output_prefix="device__attribute_max_gpu_frequency_khz",
    )


def _get_device_max_mem_freq_template() -> CUDATemplate:
    """Template for measuring max memory clock frequency."""
    return CUDATemplate(
        target_name="device__attribute_max_mem_frequency_khz",
        source_code=r'''#include <cuda_runtime.h>
#include <cstdio>

int main() {
    int mem_clock = 0;
    cudaError_t err = cudaDeviceGetAttribute(
        &mem_clock, cudaDevAttrMemoryClockRate, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetAttribute failed: %s\n",
                cudaGetErrorString(err));
        return 1;
    }
    // mem_clock is in kHz
    printf("device__attribute_max_mem_frequency_khz: %d\n", mem_clock);
    return 0;
}
''',
        compile_flags=["-O3", "-arch=native"],
        description="Max memory clock frequency via cudaDevAttrMemoryClockRate",
        expected_output_prefix="device__attribute_max_mem_frequency_khz",
    )


def _get_device_fb_bus_width_template() -> CUDATemplate:
    """Template for measuring frame buffer bus width."""
    return CUDATemplate(
        target_name="device__attribute_fb_bus_width",
        source_code=r'''#include <cuda_runtime.h>
#include <cstdio>

int main() {
    int bus_width = 0;
    cudaError_t err = cudaDeviceGetAttribute(
        &bus_width, cudaDevAttrMemoryBusWidth, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceGetAttribute failed: %s\n",
                cudaGetErrorString(err));
        return 1;
    }
    // bus_width is in bits
    printf("device__attribute_fb_bus_width: %d\n", bus_width);
    return 0;
}
''',
        compile_flags=["-O3", "-arch=native"],
        description="Memory bus width via cudaDevAttrMemoryBusWidth",
        expected_output_prefix="device__attribute_fb_bus_width",
    )


def _get_sm_throughput_template() -> CUDATemplate:
    """Template for measuring SM throughput."""
    return CUDATemplate(
        target_name="sm__throughput.avg.pct_of_peak_sustained_elapsed",
        source_code=r'''#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

__global__ void compute_intensive_kernel(volatile double* sink) {
    double a = 1.23456789012345;
    double b = 9.87654321098765;
    double c = 3.14159265358979;
    double result = 0.0;
    #pragma unroll 1
    for (int i = 0; i < 10000000; i++) {
        result += a * b + c;
    }
    *sink = result;
}

int main() {
    volatile double* d_sink = nullptr;
    cudaMalloc((void**)&d_sink, sizeof(double));

    int threads = 256;
    int blocks = 65535;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double max_utilization = 0.0;

    for (int trial = 0; trial < 3; trial++) {
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        compute_intensive_kernel<<<blocks, threads>>>(d_sink);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        // Compute SM utilization approximation
        int sm_count = 0;
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

        // Total FLOPs: blocks * threads * 2 (FMA) * 10000000 (iterations) * 2 (FMA = 2 flops)
        double total_flops = (double)blocks * threads * 10000000.0 * 2.0 * 2.0;
        double seconds = ms / 1000.0;
        double flops_per_sec = total_flops / seconds;

        // Peak FLOPS (DP): sm_count * 2 (DP units) * clock_rate(GHz) * 2 (FMA)
        int clock_khz = 0;
        cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
        double peak_flops = sm_count * 2.0 * (clock_khz / 1e6) * 2.0;

        double utilization = (flops_per_sec / peak_flops) * 100.0;
        if (utilization > max_utilization) max_utilization = utilization;
    }

    printf("sm__throughput.avg.pct_of_peak_sustained_elapsed: %.2f\n", max_utilization);

    cudaFree((void*)d_sink);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
''',
        compile_flags=["-O3", "-arch=native", "-Xptxas=-v"],
        description="SM throughput via compute-intensive FMA kernel",
        expected_output_prefix="sm__throughput.avg.pct_of_peak_sustained_elapsed",
    )


def _get_gpu_compute_mem_throughput_template() -> CUDATemplate:
    """Template for measuring GPU compute-memory throughput."""
    return CUDATemplate(
        target_name="gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        source_code=r'''#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#define BUFFER_SIZE (64 * 1024 * 1024)  // 64MB

__global__ void fused_kernel(const float* input, volatile float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < n) {
        float val = __ldg(&input[idx]);
        double result = 0.0;
        #pragma unroll 1
        for (int i = 0; i < 100; i++) {
            result += val * 1.23456 + 0.789012;
        }
        output[idx] = (float)result;
        idx += stride;
    }
}

int main() {
    float* d_input = nullptr;
    volatile float* d_output = nullptr;
    cudaMalloc((void**)&d_input, BUFFER_SIZE);
    cudaMalloc((void**)&d_output, BUFFER_SIZE);
    cudaMemset(d_input, 0x42, BUFFER_SIZE);

    int threads = 256;
    int blocks = 65535;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double max_utilization = 0.0;

    for (int trial = 0; trial < 3; trial++) {
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        fused_kernel<<<blocks, threads>>>(d_input, (float*)d_output, BUFFER_SIZE / sizeof(float));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        // Approximate utilization based on effective bandwidth + compute
        double seconds = ms / 1000.0;
        double bytes = (double)(BUFFER_SIZE / sizeof(float)) * sizeof(float);
        double bandwidth = bytes / seconds;

        // Peak bandwidth approximation (depends on GPU)
        int mem_clock = 0, bus_width = 0;
        cudaDeviceGetAttribute(&mem_clock, cudaDevAttrMemoryClockRate, 0);
        cudaDeviceGetAttribute(&bus_width, cudaDevAttrMemoryBusWidth, 0);
        double peak_bw = 2.0 * (mem_clock / 1e3) * (bus_width / 8.0) / 1e6;  // MB/s

        double bw_util = (bandwidth / (peak_bw * 1e6)) * 100.0;
        if (bw_util > max_utilization) max_utilization = bw_util;
    }

    printf("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed: %.2f\n", max_utilization);

    cudaFree(d_input);
    cudaFree((void*)d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
''',
        compile_flags=["-O3", "-arch=native", "-Xptxas=-dlcm=ca"],
        description="GPU compute-memory throughput via fused read-compute-write kernel",
        expected_output_prefix="gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    )


# Registry of all templates - populated at module load time
_TEMPLATE_REGISTRY: dict[str, CUDATemplate] = {}


def _register_templates():
    """Initialize the template registry."""
    templates = [
        _get_launch_sm_count_template(),
        _get_dram_bytes_read_template(),
        _get_dram_bytes_write_template(),
        _get_device_max_gpu_freq_template(),
        _get_device_max_mem_freq_template(),
        _get_device_fb_bus_width_template(),
        _get_sm_throughput_template(),
        _get_gpu_compute_mem_throughput_template(),
    ]
    for t in templates:
        _TEMPLATE_REGISTRY[t.target_name] = t


def get_template(target_name: str) -> Optional[CUDATemplate]:
    """Get a pre-defined CUDA template for the given target name.

    Args:
        target_name: The measurement target name.

    Returns:
        The CUDATemplate if available, None otherwise.
    """
    if not _TEMPLATE_REGISTRY:
        _register_templates()
    return _TEMPLATE_REGISTRY.get(target_name)


def get_template_for_targets(targets: list[str]) -> list[CUDATemplate]:
    """Get templates for a list of targets.

    Returns only templates that are available for the given targets.
    """
    if not _TEMPLATE_REGISTRY:
        _register_templates()
    result = []
    for t in targets:
        template = _TEMPLATE_REGISTRY.get(t)
        if template:
            result.append(template)
    return result


def has_template(target_name: str) -> bool:
    """Check if a template exists for the given target."""
    if not _TEMPLATE_REGISTRY:
        _register_templates()
    return target_name in _TEMPLATE_REGISTRY


# Initialize registry at module load time
_register_templates()
