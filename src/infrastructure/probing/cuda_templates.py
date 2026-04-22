"""CUDA code pattern templates for GPU profiling targets.

These templates provide PARTIAL code snippets, key API patterns, and measurement
methodology hints. The LLM MUST complete the full CUDA code generation.
Templates are guidance only — they do NOT provide fully compilable code.
This complies with P5: compile-time elimination rather than runtime gating.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CUDAPatternTemplate:
    """A partial code pattern template for a specific measurement target."""
    target_name: str
    key_api_pattern: str  # The core CUDA API/pattern to use
    measurement_skeleton: str  # Partial code structure (not fully compilable)
    expected_output_format: str  # How to print the result
    measurement_methodology: str  # Description of the approach
    critical_notes: list[str]  # Important implementation notes


def _get_launch_sm_count_pattern() -> CUDAPatternTemplate:
    """Pattern for measuring SM count."""
    return CUDAPatternTemplate(
        target_name="launch__sm_count",
        key_api_pattern=(
            "int sm_count = 0;\n"
            "cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);\n"
            "printf(\"launch__sm_count: %d\\n\", sm_count);"
        ),
        measurement_skeleton=(
            "#include <cuda_runtime.h>\n"
            "#include <cstdio>\n\n"
            "__global__ void flag_kernel(volatile int* flags) {\n"
            "    flags[blockIdx.x] = 1;\n"
            "}\n\n"
            "int main() {\n"
            "    // TODO: Query SM count using cudaDeviceGetAttribute\n"
            "    // TODO: Launch flag_kernel with enough blocks to fill all SMs\n"
            "    // TODO: Copy back and count executed blocks for cross-validation\n"
            "    // TODO: Print: printf(\"launch__sm_count: %d\\n\", sm_count);\n"
            "    return 0;\n"
            "}"
        ),
        expected_output_format="launch__sm_count: <integer>",
        measurement_methodology=(
            "Use cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount) to query SM count. "
            "Cross-validate by launching enough thread blocks to fill all SMs and counting "
            "how many blocks actually executed by checking a shared memory flag array."
        ),
        critical_notes=[
            "cudaDeviceGetAttribute returns the actual SM count, not theoretical max",
            "Block ID sweep should launch at least 256 blocks to ensure full coverage",
            "Use volatile int* for flags to prevent compiler optimization"
        ]
    )


def _get_dram_bytes_read_pattern() -> CUDAPatternTemplate:
    """Pattern for measuring DRAM read bandwidth."""
    return CUDAPatternTemplate(
        target_name="dram__bytes_read.sum.per_second",
        key_api_pattern=(
            "__global__ void read_only_kernel(const float* data, int n) {\n"
            "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "    int stride = blockDim.x * gridDim.x;\n"
            "    volatile float sink = 0.0f;\n"
            "    while (idx < n) {\n"
            "        sink += __ldg(&data[idx]);  // __ldg forces read through load path\n"
            "        idx += stride;\n"
            "    }\n"
            "}"
        ),
        measurement_skeleton=(
            "#include <cuda_runtime.h>\n"
            "#include <cstdio>\n\n"
            "// TODO: Implement read-only kernel that reads from global memory\n"
            "// TODO: Allocate large buffer (64MB) on device\n"
            "// TODO: Launch with 65535 blocks x 256 threads\n"
            "// TODO: Use cudaEventElapsedTime for timing\n"
            "// TODO: Compute bandwidth = total_bytes_read / elapsed_seconds\n"
            "// TODO: Print: printf(\"dram__bytes_read.sum.per_second: %.2f\\n\", bandwidth);\n"
        ),
        expected_output_format="dram__bytes_read.sum.per_second: <float>",
        measurement_methodology=(
            "Perform a read-only STREAM benchmark: allocate a large buffer, "
            "read sequentially using __ldg() to bypass L1 cache and go through "
            "the load path. Measure elapsed time with cudaEventElapsedTime. "
            "Bandwidth = total_bytes / seconds. Run multiple trials, report max."
        ),
        critical_notes=[
            "Use __ldg() to ensure reads go through the load path (not L1)",
            "Buffer size should be large enough (>64MB) to exceed L2 cache",
            "Use volatile sink to prevent dead-code elimination",
            "Kernel must be read-only (no writes to the buffer)"
        ]
    )


def _get_dram_bytes_write_pattern() -> CUDAPatternTemplate:
    """Pattern for measuring DRAM write bandwidth."""
    return CUDAPatternTemplate(
        target_name="dram__bytes_write.sum.per_second",
        key_api_pattern=(
            "__global__ void write_only_kernel(volatile float* data, int n) {\n"
            "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "    int stride = blockDim.x * gridDim.x;\n"
            "    while (idx < n) {\n"
            "        data[idx] = 1.23456f;  // Pure write, no read\n"
            "        idx += stride;\n"
            "    }\n"
            "}"
        ),
        measurement_skeleton=(
            "#include <cuda_runtime.h>\n"
            "#include <cstdio>\n\n"
            "// TODO: Implement write-only kernel that writes to global memory\n"
            "// TODO: Allocate large buffer (64MB) on device\n"
            "// TODO: Launch with 65535 blocks x 256 threads\n"
            "// TODO: Use cudaEventElapsedTime for timing\n"
            "// TODO: Compute bandwidth = total_bytes_written / elapsed_seconds\n"
            "// TODO: Print: printf(\"dram__bytes_write.sum.per_second: %.2f\\n\", bandwidth);\n"
        ),
        expected_output_format="dram__bytes_write.sum.per_second: <float>",
        measurement_methodology=(
            "Perform a write-only STREAM benchmark: allocate a large buffer, "
            "write sequentially without reading. Use volatile pointer to prevent "
            "compiler optimization. Measure elapsed time. "
            "Bandwidth = total_bytes / seconds. Run multiple trials, report max."
        ),
        critical_notes=[
            "Use volatile pointer to prevent compiler from optimizing away writes",
            "Kernel must be write-only (no reads from the buffer being written)",
            "Buffer size should be large enough to exceed L2 cache"
        ]
    )


def _get_device_max_gpu_freq_pattern() -> CUDAPatternTemplate:
    """Pattern for measuring max GPU clock frequency."""
    return CUDAPatternTemplate(
        target_name="device__attribute_max_gpu_frequency_khz",
        key_api_pattern=(
            "int clock_rate = 0;\n"
            "cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, 0);\n"
            "printf(\"device__attribute_max_gpu_frequency_khz: %d\\n\", clock_rate);"
        ),
        measurement_skeleton=(
            "#include <cuda_runtime.h>\n"
            "#include <cstdio>\n\n"
            "int main() {\n"
            "    // TODO: Query clock rate using cudaDeviceGetAttribute(cudaDevAttrClockRate)\n"
            "    // TODO: clock_rate is returned in kHz\n"
            "    // TODO: Print: printf(\"device__attribute_max_gpu_frequency_khz: %d\\n\", clock_rate);\n"
            "    return 0;\n"
            "}"
        ),
        expected_output_format="device__attribute_max_gpu_frequency_khz: <integer_khz>",
        measurement_methodology=(
            "Query the GPU clock rate using cudaDeviceGetAttribute with cudaDevAttrClockRate. "
            "The value is returned in kHz."
        ),
        critical_notes=[
            "cudaDevAttrClockRate returns value in kHz, not Hz or MHz",
            "This is the maximum clock rate, not the current clock rate"
        ]
    )


def _get_device_max_mem_freq_pattern() -> CUDAPatternTemplate:
    """Pattern for measuring max memory clock frequency."""
    return CUDAPatternTemplate(
        target_name="device__attribute_max_mem_frequency_khz",
        key_api_pattern=(
            "int mem_clock = 0;\n"
            "cudaDeviceGetAttribute(&mem_clock, cudaDevAttrMemoryClockRate, 0);\n"
            "printf(\"device__attribute_max_mem_frequency_khz: %d\\n\", mem_clock);"
        ),
        measurement_skeleton=(
            "#include <cuda_runtime.h>\n"
            "#include <cstdio>\n\n"
            "int main() {\n"
            "    // TODO: Query memory clock rate using cudaDeviceGetAttribute(cudaDevAttrMemoryClockRate)\n"
            "    // TODO: mem_clock is returned in kHz\n"
            "    // TODO: Print: printf(\"device__attribute_max_mem_frequency_khz: %d\\n\", mem_clock);\n"
            "    return 0;\n"
            "}"
        ),
        expected_output_format="device__attribute_max_mem_frequency_khz: <integer_khz>",
        measurement_methodology=(
            "Query the memory clock rate using cudaDeviceGetAttribute with cudaDevAttrMemoryClockRate. "
            "The value is returned in kHz."
        ),
        critical_notes=[
            "cudaDevAttrMemoryClockRate returns value in kHz",
            "This is the peak memory clock, not the current clock"
        ]
    )


def _get_device_fb_bus_width_pattern() -> CUDAPatternTemplate:
    """Pattern for measuring frame buffer bus width."""
    return CUDAPatternTemplate(
        target_name="device__attribute_fb_bus_width",
        key_api_pattern=(
            "int bus_width = 0;\n"
            "cudaDeviceGetAttribute(&bus_width, cudaDevAttrMemoryBusWidth, 0);\n"
            "printf(\"device__attribute_fb_bus_width: %d\\n\", bus_width);"
        ),
        measurement_skeleton=(
            "#include <cuda_runtime.h>\n"
            "#include <cstdio>\n\n"
            "int main() {\n"
            "    // TODO: Query memory bus width using cudaDeviceGetAttribute(cudaDevAttrMemoryBusWidth)\n"
            "    // TODO: bus_width is returned in bits\n"
            "    // TODO: If attribute not available, infer from other attributes\n"
            "    // TODO: Print: printf(\"device__attribute_fb_bus_width: %d\\n\", bus_width);\n"
            "    return 0;\n"
            "}"
        ),
        expected_output_format="device__attribute_fb_bus_width: <integer_bits>",
        measurement_methodology=(
            "Query the memory bus width using cudaDeviceGetAttribute with cudaDevAttrMemoryBusWidth. "
            "The value is returned in bits. Common values: 256, 384, 512, 4096."
        ),
        critical_notes=[
            "cudaDevAttrMemoryBusWidth returns value in bits",
            "If attribute not supported, can infer from known GPU configs",
            "Typical values: RTX3090=384, RTX3070=256, A100=4096"
        ]
    )


def _get_sm_throughput_pattern() -> CUDAPatternTemplate:
    """Pattern for measuring SM throughput."""
    return CUDAPatternTemplate(
        target_name="sm__throughput.avg.pct_of_peak_sustained_elapsed",
        key_api_pattern=(
            "__global__ void compute_intensive_kernel(volatile double* sink) {\n"
            "    double a = 1.23456789012345, b = 9.87654321098765, c = 3.14159265358979;\n"
            "    double result = 0.0;\n"
            "    #pragma unroll 1\n"
            "    for (int i = 0; i < 10000000; i++) {\n"
            "        result += a * b + c;  // FMA: 2 DP FLOPs per iteration\n"
            "    }\n"
            "    *sink = result;\n"
            "}"
        ),
        measurement_skeleton=(
            "#include <cuda_runtime.h>\n"
            "#include <cstdio>\n\n"
            "// TODO: Implement compute-intensive kernel with FMA operations\n"
            "// TODO: Launch with 65535 blocks x 256 threads\n"
            "// TODO: Use cudaEventElapsedTime for timing\n"
            "// TODO: Query sm_count and clock_rate for peak FLOPs calculation\n"
            "// TODO: Compute utilization = (actual_flops / peak_flops) * 100\n"
            "// TODO: Print: printf(\"sm__throughput.avg.pct_of_peak_sustained_elapsed: %.2f\\n\", util);\n"
        ),
        expected_output_format="sm__throughput.avg.pct_of_peak_sustained_elapsed: <float_percent>",
        measurement_methodology=(
            "Launch a purely compute-bound kernel with no memory bottleneck. "
            "Each thread performs 10M FMA (fused multiply-add) operations. "
            "Use volatile double sink to prevent dead-code elimination. "
            "Calculate SM utilization = actual_FLOPs / peak_FLOPs * 100. "
            "Peak FLOPs = sm_count * DP_units_per_SM * clock_GHz * 2 (for FMA)."
        ),
        critical_notes=[
            "Kernel must be purely compute-bound (no global memory after init)",
            "Use #pragma unroll 1 to prevent loop optimization",
            "Use volatile double to prevent compiler from eliminating the loop",
            "Peak DP FLOPs = sm_count * 2 * clock_GHz * 2"
        ]
    )


def _get_gpu_compute_mem_throughput_pattern() -> CUDAPatternTemplate:
    """Pattern for measuring GPU compute-memory throughput."""
    return CUDAPatternTemplate(
        target_name="gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        key_api_pattern=(
            "__global__ void fused_kernel(const float* input, volatile float* output, int n) {\n"
            "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "    int stride = blockDim.x * gridDim.x;\n"
            "    while (idx < n) {\n"
            "        float val = __ldg(&input[idx]);  // Read\n"
            "        double result = 0.0;\n"
            "        #pragma unroll 1\n"
            "        for (int i = 0; i < 100; i++) {  // Compute\n"
            "            result += val * 1.23456 + 0.789012;\n"
            "        }\n"
            "        output[idx] = (float)result;  // Write\n"
            "        idx += stride;\n"
            "    }\n"
            "}"
        ),
        measurement_skeleton=(
            "#include <cuda_runtime.h>\n"
            "#include <cstdio>\n\n"
            "// TODO: Implement fused read-compute-write kernel\n"
            "// TODO: Allocate large buffer (64MB) for input and output\n"
            "// TODO: Launch with 65535 blocks x 256 threads\n"
            "// TODO: Each thread: read, compute FMA loop, write back\n"
            "// TODO: Use cudaEventElapsedTime for timing\n"
            "// TODO: Compute utilization based on effective bandwidth + compute\n"
            "// TODO: Print: printf(\"gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed: %.2f\\n\", util);\n"
        ),
        expected_output_format="gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed: <float_percent>",
        measurement_methodology=(
            "Launch a kernel that reads from global memory, performs arithmetic, "
            "and writes back. This exercises both compute and memory paths. "
            "Calculate effective utilization as a percentage of peak sustained "
            "throughput based on measured bandwidth and compute rate."
        ),
        critical_notes=[
            "Kernel must do both reads and writes with computation in between",
            "Use __ldg() for reads to go through load path",
            "Use volatile for writes to prevent optimization",
            "Balance compute intensity to avoid being purely compute or memory bound"
        ]
    )


# Registry of all pattern templates
_PATTERN_REGISTRY: dict[str, CUDAPatternTemplate] = {}


def _register_patterns():
    """Initialize the pattern registry."""
    patterns = [
        _get_launch_sm_count_pattern(),
        _get_dram_bytes_read_pattern(),
        _get_dram_bytes_write_pattern(),
        _get_device_max_gpu_freq_pattern(),
        _get_device_max_mem_freq_pattern(),
        _get_device_fb_bus_width_pattern(),
        _get_sm_throughput_pattern(),
        _get_gpu_compute_mem_throughput_pattern(),
    ]
    for p in patterns:
        _PATTERN_REGISTRY[p.target_name] = p


def get_pattern(target_name: str) -> Optional[CUDAPatternTemplate]:
    """Get a pattern template for the given target name.

    Returns a PARTIAL code pattern — NOT a fully compilable implementation.
    The LLM must complete the full code generation.
    """
    if not _PATTERN_REGISTRY:
        _register_patterns()
    return _PATTERN_REGISTRY.get(target_name)


def get_pattern_for_targets(targets: list[str]) -> list[CUDAPatternTemplate]:
    """Get pattern templates for a list of targets."""
    if not _PATTERN_REGISTRY:
        _register_patterns()
    result = []
    for t in targets:
        pattern = _PATTERN_REGISTRY.get(t)
        if pattern:
            result.append(pattern)
    return result


def has_pattern(target_name: str) -> bool:
    """Check if a pattern template exists for the given target."""
    if not _PATTERN_REGISTRY:
        _register_patterns()
    return target_name in _PATTERN_REGISTRY
