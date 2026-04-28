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
            "// TODO: Define __global__ flag_kernel(volatile int* flags)\n"
            "// TODO: In flag_kernel, set flags[blockIdx.x] = 1\n"
            "// TODO: In main(), query SM count using cudaDeviceGetAttribute\n"
            "// TODO: Allocate flag array on device\n"
            "// TODO: Launch flag_kernel with enough blocks to fill all SMs\n"
            "// TODO: Copy back and count executed blocks for cross-validation\n"
            "// TODO: Print: printf(\"launch__sm_count: %d\\n\", sm_count);\n"
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
            "APPROACH: Write a __global__ read-only kernel that:\n"
            "- Takes const float* data and int n parameters\n"
            "- Each thread reads data[idx] using __ldg() to force read through load path\n"
            "- Accumulates into volatile float sink to prevent dead-code elimination\n"
            "- Uses stride loop: idx += blockDim.x * gridDim.x\n"
            "In main(): allocate 64MB buffer, launch 65535 blocks x 256 threads,\n"
            "time with cudaEventElapsedTime, compute bandwidth = bytes / seconds."
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
            "APPROACH: Write a __global__ write-only kernel that:\n"
            "- Takes volatile float* data and int n parameters\n"
            "- Each thread writes a constant value to data[idx] (pure write, no read)\n"
            "- Uses volatile pointer to prevent compiler optimization\n"
            "- Uses stride loop: idx += blockDim.x * gridDim.x\n"
            "In main(): allocate 64MB buffer, launch 65535 blocks x 256 threads,\n"
            "time with cudaEventElapsedTime, compute bandwidth = bytes / seconds."
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
    """Pattern for measuring SM throughput.

    sm__throughput.avg.pct_of_peak_sustained_elapsed measures SM compute
    throughput as a percentage of peak. The CUDA program must compute this
    empirically using cudaDeviceGetAttribute for hardware parameters:
    achieved_FLOPS / peak_FLOPS * 100.
    The harness adds a runtime clamp [0,100] as safety net.
    """
    return CUDAPatternTemplate(
        target_name="sm__throughput.avg.pct_of_peak_sustained_elapsed",
        key_api_pattern=(
            "APPROACH: Write a __global__ compute-intensive kernel that:\n"
            "- Uses double-precision FMA operations (a * b + c) in a tight loop\n"
            "- Declares volatile double* sink parameter to prevent dead-code elimination\n"
            "- Uses #pragma unroll 1 before the compute loop to prevent unrolling\n"
            "- Performs at least 10M iterations per thread\n"
            "- Has NO global memory access after initialization (purely compute-bound)\n"
            "- Records clock64() before/after FMA loop, outputs cycle count via uint64_t* cycle_out\n"
            "In main(): query SM count and compute capability via cudaDeviceGetAttribute,\n"
            "launch sm_count*4 blocks x 256 threads, warmup, then timed measurement.\n"
            "Determine fp64_per_sm from compute capability: SM70=32, SM80=32, SM90=64, SM75=2, SM86+=2\n"
            "COMPUTE: actual_freq_mhz = cycle_count / (elapsed_ms * 1000.0)\n"
            "COMPUTE: peak_flops = sm_count * fp64_per_sm * actual_freq_mhz * 1e6 * 2\n"
            "COMPUTE: achieved_flops = total_fma_ops / elapsed_seconds\n"
            "COMPUTE: pct = (achieved_flops / peak_flops) * 100.0\n"
            "Do NOT use cudaDevAttrClockRate for peak — it may report base clock, not boost!\n"
            "printf(\"sm__throughput.avg.pct_of_peak_sustained_elapsed: %.2f\\n\", pct);"
        ),
        measurement_skeleton=(
            "#include <cuda_runtime.h>\n"
            "#include <cstdio>\n\n"
            "// TODO: Implement compute-intensive kernel with FMA operations\n"
            "// TODO: Use volatile double* sink to prevent dead-code elimination\n"
            "// TODO: Use #pragma unroll 1 to prevent loop optimization\n"
            "// TODO: Inside kernel: record clock64() before/after FMA loop\n"
            "//   uint64_t start_cycle = clock64();\n"
            "//   // ... FMA loop ...\n"
            "//   uint64_t end_cycle = clock64();\n"
            "//   if (threadIdx.x == 0 && blockIdx.x == 0) *cycle_out = end_cycle - start_cycle;\n"
            "// TODO: Launch with sm_count*4 blocks x 256 threads\n"
            "// TODO: Include warmup run before timed measurement\n"
            "// TODO: Use cudaEventElapsedTime for timing\n"
            "// TODO: cudaMemcpy cycle_count from device to host\n"
            "// TODO: Query SM_count and compute capability via cudaDeviceGetAttribute\n"
            "//   cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0)\n"
            "//   cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0)\n"
            "//   cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0)\n"
            "//   Determine fp64_per_sm: SM70=32, SM80=32, SM90=64, SM75=2, SM86+=2\n"
            "// COMPUTE: actual_freq_mhz = cycle_count / (elapsed_ms * 1000.0)\n"
            "// COMPUTE: peak_flops = sm_count * fp64_per_sm * actual_freq_mhz * 1e6 * 2\n"
            "// COMPUTE: achieved_flops = total_fma_ops / elapsed_seconds\n"
            "// COMPUTE: pct = (achieved_flops / peak_flops) * 100.0\n"
            "// printf(\"sm__throughput.avg.pct_of_peak_sustained_elapsed: %.2f\\n\", pct);\n"
        ),
        expected_output_format="sm__throughput.avg.pct_of_peak_sustained_elapsed: <float_percent>",
        measurement_methodology=(
            "Launch a purely compute-bound kernel with no memory bottleneck. "
            "Each thread performs 10M FMA (fused multiply-add) operations. "
            "Use volatile double sink to prevent dead-code elimination. "
            "Inside kernel: record clock64() before/after FMA loop, output cycle count. "
            "Compute actual_freq_mhz = cycle_count / (elapsed_ms * 1000.0). "
            "Compute actual percentage: achieved_FLOPS / peak_FLOPS * 100. "
            "peak_FLOPS = sm_count * fp64_per_sm * actual_freq_mhz * 1e6 * 2. "
            "fp64_per_sm depends on compute capability: V100=32, A100=32, H100=64, T4=2, consumer=2. "
            "Do NOT use cudaDevAttrClockRate for peak — it may report base clock, not boost! "
            "The harness adds a runtime clamp [0,100] as safety net."
        ),
        critical_notes=[
            "Kernel must be purely compute-bound (no global memory after init)",
            "Use #pragma unroll 1 to prevent loop optimization",
            "Use volatile double to prevent compiler from eliminating the loop",
            "COMPUTE actual percentage using clock64() for actual frequency + cudaDeviceGetAttribute for SM count",
            "Do NOT use cudaDevAttrClockRate for peak_flops — it may report base clock, not boost!",
            "The harness adds a runtime clamp [0,100] as safety net",
            "Kernel must run long enough (>10us) for accurate measurement",
            "Use at least sm_count*4 blocks with 256 threads each"
        ]
    )


def _get_gpu_compute_mem_throughput_pattern() -> CUDAPatternTemplate:
    """Pattern for measuring GPU compute-memory throughput.

    gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
    measures combined compute and memory throughput. The CUDA program must
    compute this empirically using cudaDeviceGetAttribute for hardware
    parameters: achieved_BW / peak_BW * 100.
    The harness adds a runtime clamp [0,100] as safety net.
    """
    return CUDAPatternTemplate(
        target_name="gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        key_api_pattern=(
            "APPROACH: Write a __global__ fused read-compute-write kernel that:\n"
            "- Takes const float* __restrict__ input and volatile float* output parameters\n"
            "- Each thread reads input[idx] — this value MUST be used in the FMA chain\n"
            "- Performs a compute loop (e.g., 8 FMA iterations) USING the read value\n"
            "- Writes result to output[idx] via volatile pointer\n"
            "- Uses #pragma unroll 1 before the compute loop\n"
            "- Balances compute intensity to stress both memory and compute paths\n"
            "- CRITICAL: FMA chain MUST depend on value read from memory, NOT register-only!\n"
            "  WRONG: val = val * 1.0001f + 0.001f where val is register-only → 0.09%!\n"
            "  RIGHT: val = input[i]; then val = val * 1.0001f + 0.001f; then output[i] = val;\n"
            "In main(): allocate large buffers (>= 16M floats = 64MB),\n"
            "query mem_clock_khz and bus_width_bits via cudaDeviceGetAttribute,\n"
            "launch sm_count*4 blocks x 256 threads, warmup, then timed run.\n"
            "COMPUTE: peak_bw = (mem_clock_khz / 1000.0) * 1e6 * (bus_width_bits / 8) * 2 / 1e9\n"
            "COMPUTE: achieved_bw = (2.0 * buffer_size_bytes) / elapsed_seconds / 1e9\n"
            "  (2x because each element is read AND written)\n"
            "COMPUTE: pct = (achieved_bw / peak_bw) * 100.0\n"
            "If pct < 5%, the kernel is FUNDAMENTALLY WRONG — not stressing memory at all!\n"
            "printf(\"gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed: %.2f\\n\", pct);"
        ),
        measurement_skeleton=(
            "#include <cuda_runtime.h>\n"
            "#include <cstdio>\n\n"
            "// TODO: Implement fused read-compute-write kernel\n"
            "// TODO: Allocate large buffers (at least 16M floats = 64MB)\n"
            "// TODO: Each thread: read input[idx], compute FMA loop USING read value, volatile write\n"
            "// TODO: Launch with sm_count*4 blocks x 256 threads\n"
            "// TODO: Include warmup run before timed measurement\n"
            "// TODO: Use cudaEventElapsedTime for timing\n"
            "// TODO: Query mem_clock_khz and bus_width_bits via cudaDeviceGetAttribute\n"
            "//   cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0)\n"
            "//   cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, 0)\n"
            "// COMPUTE: peak_bw = (mem_clock_khz / 1000.0) * 1e6 * (bus_width_bits / 8) * 2 / 1e9\n"
            "// COMPUTE: achieved_bw = (2.0 * buffer_size_bytes) / elapsed_seconds / 1e9\n"
            "//   NOTE: 2x because each element is read AND written (input+output)\n"
            "// COMPUTE: pct = (achieved_bw / peak_bw) * 100.0\n"
            "// printf(\"gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed: %.2f\\n\", pct);\n"
        ),
        expected_output_format="gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed: <float_percent>",
        measurement_methodology=(
            "Launch a kernel that reads from global memory, performs arithmetic, "
            "and writes back. This exercises both compute and memory paths. "
            "Compute actual percentage: achieved_BW / peak_BW * 100. "
            "peak_BW = (mem_clock_khz / 1000.0) * 1e6 * (bus_width_bits / 8) * 2 / 1e9 GB/s. "
            "achieved_BW = (2.0 * buffer_size_bytes) / elapsed_seconds / 1e9 (2x for read+write). "
            "The harness adds a runtime clamp [0,100] as safety net."
        ),
        critical_notes=[
            "Kernel must do both reads and writes with computation in between",
            "Use __ldg() for reads to go through load path",
            "Use volatile for writes to prevent optimization",
            "Balance compute intensity to avoid being purely compute or memory bound",
            "COMPUTE actual percentage using cudaDeviceGetAttribute for peak calculation",
            "The harness adds a runtime clamp [0,100] as safety net",
            "Kernel must run long enough (>10us) for accurate measurement",
            "Use at least sm_count*4 blocks with 256 threads each"
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
