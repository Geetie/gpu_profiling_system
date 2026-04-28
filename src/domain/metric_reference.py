"""NCU Metric Reference Documentation for LLM Code Generation.

This module provides detailed explanations for each NCU metric target,
including what it measures, how to measure it, expected output format,
and common pitfalls. It is automatically injected into the CodeGen agent's
context when processing targets it may not understand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MetricDoc:
    """Complete documentation for a single NCU metric target."""
    target_name: str
    what_it_measures: str
    measurement_approach: str
    key_api_calls: list[str]
    kernel_type: str
    thread_config: str
    expected_output_format: str
    expected_range: str
    anti_cheat_notes: list[str] = field(default_factory=list)
    common_pitfalls: list[str] = field(default_factory=list)
    code_template: str = ""


# ============================================================
# NCU METRIC DOCUMENTS
# ============================================================

_METRIC_DOCS: dict[str, MetricDoc] = {}


def _build_docs():
    """Build the metric documentation registry."""

    # 1. Launch SM Count
    _METRIC_DOCS["launch__sm_count"] = MetricDoc(
        target_name="launch__sm_count",
        what_it_measures=(
            "The actual number of Streaming Multiprocessors (SMs) available on the GPU. "
            "Each SM is a compute unit that can execute thread blocks in parallel. "
            "This is the fundamental unit of GPU compute capacity."
        ),
        measurement_approach=(
            "1. Use cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount) — simplest and most reliable.\n"
            "2. Cross-validate with a block ID sweep: launch many blocks, each writes a flag at blockIdx.x,\n"
            "   then count how many flags are set. This verifies the hardware actually has that many SMs."
        ),
        key_api_calls=[
            "cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id)",
        ],
        kernel_type="Query API (no kernel needed for primary measurement), optional flag kernel for cross-validation",
        thread_config="For cross-validation: launch 256 blocks x 1 thread",
        expected_output_format="launch__sm_count: <integer>",
        expected_range="8-256 SMs (P100=56, V100=80, A100=108, H100=114)",
        anti_cheat_notes=[
            "SM count may be reduced by CUDA_VISIBLE_DEVICES or MIG partitions",
            "Block ID sweep must launch more blocks than SMs to ensure all SMs are used"
        ],
        common_pitfalls=[
            "Using cudaGetDeviceProperties instead of cudaDeviceGetAttribute",
            "Not checking cudaError_t return value"
        ],
        code_template=(
            '// TODO: Include necessary headers\n'
            '// TODO: Declare flag_kernel that signals from each block\n'
            '// TODO: In main(), query SM count via cudaDeviceGetAttribute\n'
            '// TODO: Print result as "launch__sm_count: <value>"\n'
        ),
    )

    # 2. DRAM Read Bandwidth
    _METRIC_DOCS["dram__bytes_read.sum.per_second"] = MetricDoc(
        target_name="dram__bytes_read.sum.per_second",
        what_it_measures=(
            "The sustained DRAM (global memory) read throughput in bytes per second. "
            "This measures how fast the GPU can read data from VRAM (video RAM). "
            "The 'sum' means total bytes read across all SMs during execution."
        ),
        measurement_approach=(
            "1. Allocate a large buffer (>64MB) on the device, fill with arbitrary data.\n"
            "2. Launch a READ-ONLY kernel: each thread reads array[idx] using __ldg() for cache bypass.\n"
            "3. Use cudaEventElapsedTime to measure wall-clock time.\n"
            "4. bandwidth = total_bytes_read / elapsed_seconds.\n"
            "5. Run 3 trials, report MAXIMUM.\n"
            "CRITICAL: The kernel must only READ (no writes), otherwise it measures read+write."
        ),
        key_api_calls=[
            "cudaMalloc(&d_data, BUFFER_SIZE)",
            "cudaEventRecord(start); kernel<<<...>>>; cudaEventRecord(stop);",
            "cudaEventElapsedTime(&ms, start, stop)",
        ],
        kernel_type="Read-only STREAM kernel",
        thread_config="65535 blocks x 256 threads (max parallelism to saturate memory bus)",
        expected_output_format="dram__bytes_read.sum.per_second: <float_bytes_per_second>",
        expected_range="200-1600 GB/s (depends on GPU, A100=1555 GB/s)",
        anti_cheat_notes=[
            "Buffer must be larger than L2 cache to avoid measuring L2 bandwidth",
            "Must use __ldg() to bypass L1 and read from global memory",
            "Kernel must be read-only — no writes to the measured buffer"
        ],
        common_pitfalls=[
            "Not using __ldg() — data stays in L1 cache, underestimates DRAM bandwidth",
            "Buffer too small — stays in L2 cache, measures L2 not DRAM bandwidth",
            "Using clock64() instead of cudaEventElapsedTime for wall-clock timing"
        ],
        code_template=(
            '// TODO: Include cuda_runtime.h and cstdio\n'
            '// TODO: Define BUFFER_SIZE as 64*1024*1024 (64MB)\n'
            '// TODO: Write __global__ read_kernel(const float* data, int n)\n'
            '//   - Each thread reads data[idx] using __ldg() for cache bypass\n'
            '//   - Accumulate into volatile float sink to prevent dead-code elimination\n'
            '//   - Use stride loop: idx += blockDim.x * gridDim.x\n'
            '// TODO: In main(): cudaMalloc buffer, cudaMemset to initialize\n'
            '// TODO: Use cudaEventElapsedTime for wall-clock timing\n'
            '// TODO: Compute bandwidth = total_bytes / elapsed_seconds\n'
            '// TODO: Print: printf("dram__bytes_read.sum.per_second: %.2f\\n", bandwidth)\n'
        ),
    )

    # 3. DRAM Write Bandwidth
    _METRIC_DOCS["dram__bytes_write.sum.per_second"] = MetricDoc(
        target_name="dram__bytes_write.sum.per_second",
        what_it_measures=(
            "The sustained DRAM (global memory) write throughput in bytes per second. "
            "This measures how fast the GPU can write data to VRAM."
        ),
        measurement_approach=(
            "1. Allocate a large buffer (>64MB) on the device.\n"
            "2. Launch a WRITE-ONLY kernel: each thread writes to array[idx].\n"
            "3. Use cudaEventElapsedTime to measure wall-clock time.\n"
            "4. bandwidth = total_bytes_written / elapsed_seconds.\n"
            "5. Run 3 trials, report MAXIMUM.\n"
            "CRITICAL: The kernel must only WRITE (no reads of the data being written)."
        ),
        key_api_calls=[
            "cudaMalloc(&d_data, BUFFER_SIZE)",
            "cudaEventRecord(start); kernel<<<...>>>; cudaEventRecord(stop);",
            "cudaEventElapsedTime(&ms, start, stop)",
        ],
        kernel_type="Write-only STREAM kernel",
        thread_config="65535 blocks x 256 threads",
        expected_output_format="dram__bytes_write.sum.per_second: <float_bytes_per_second>",
        expected_range="200-1600 GB/s",
        anti_cheat_notes=[
            "Buffer must be larger than L2 cache",
            "Kernel must be write-only — no reads from the measured buffer"
        ],
        common_pitfalls=[
            "Reading before writing — turns it into read+write benchmark",
            "Buffer too small — stays in L2 cache"
        ],
        code_template=(
            '// TODO: Include cuda_runtime.h and cstdio\n'
            '// TODO: Define BUFFER_SIZE as 64*1024*1024 (64MB)\n'
            '// TODO: Write __global__ write_kernel(volatile float* data, int n)\n'
            '//   - Each thread writes a constant value to data[idx]\n'
            '//   - Use volatile float* to prevent compiler optimization\n'
            '//   - Use stride loop: idx += blockDim.x * gridDim.x\n'
            '// TODO: In main(): cudaMalloc buffer\n'
            '// TODO: Use cudaEventElapsedTime for wall-clock timing\n'
            '// TODO: Compute bandwidth = total_bytes / elapsed_seconds\n'
            '// TODO: Print: printf("dram__bytes_write.sum.per_second: %.2f\\n", bandwidth)\n'
        ),
    )

    # 4. Device Max GPU Frequency
    _METRIC_DOCS["device__attribute_max_gpu_frequency_khz"] = MetricDoc(
        target_name="device__attribute_max_gpu_frequency_khz",
        what_it_measures=(
            "The maximum GPU core clock frequency in kilohertz (kHz). "
            "This is the peak operating frequency of the GPU's compute cores."
        ),
        measurement_approach=(
            "Simple API query using cudaDeviceGetAttribute(cudaDevAttrClockRate).\n"
            "The returned value is already in kHz — print it directly."
        ),
        key_api_calls=[
            "cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, device_id)",
        ],
        kernel_type="No kernel needed — pure API query",
        thread_config="N/A",
        expected_output_format="device__attribute_max_gpu_frequency_khz: <integer_khz>",
        expected_range="1000000-2500000 kHz (1000-2500 MHz, A100=1410000 kHz)",
        anti_cheat_notes=[
            "Frequency may be locked to non-standard values during evaluation",
            "This returns maximum, not current frequency"
        ],
        common_pitfalls=[
            "Confusing kHz with MHz — the API returns kHz, don't multiply/divide",
            "Using cudaGetDeviceProperties instead of cudaDeviceGetAttribute"
        ],
        code_template=(
            '// TODO: Include cuda_runtime.h and cstdio\n'
            '// TODO: In main(), declare int clock_rate = 0;\n'
            '// TODO: Call cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, 0)\n'
            '// TODO: Print: printf("device__attribute_max_gpu_frequency_khz: %d\\n", clock_rate)\n'
        ),
    )

    # 5. Device Max Memory Frequency
    _METRIC_DOCS["device__attribute_max_mem_frequency_khz"] = MetricDoc(
        target_name="device__attribute_max_mem_frequency_khz",
        what_it_measures=(
            "The maximum memory (VRAM) clock frequency in kilohertz (kHz). "
            "This is the peak operating frequency of the GPU's memory subsystem."
        ),
        measurement_approach=(
            "Simple API query using cudaDeviceGetAttribute(cudaDevAttrMemoryClockRate).\n"
            "The returned value is already in kHz — print it directly."
        ),
        key_api_calls=[
            "cudaDeviceGetAttribute(&mem_clock, cudaDevAttrMemoryClockRate, device_id)",
        ],
        kernel_type="No kernel needed — pure API query",
        thread_config="N/A",
        expected_output_format="device__attribute_max_mem_frequency_khz: <integer_khz>",
        expected_range="500000-1600000 kHz (500-1600 MHz, A100=1215000 kHz)",
        anti_cheat_notes=[
            "Memory frequency may be locked to non-standard values during evaluation"
        ],
        common_pitfalls=[
            "Confusing with GPU core clock (cudaDevAttrClockRate vs cudaDevAttrMemoryClockRate)"
        ],
        code_template=(
            '// TODO: Include cuda_runtime.h and cstdio\n'
            '// TODO: In main(), declare int mem_clock = 0;\n'
            '// TODO: Call cudaDeviceGetAttribute(&mem_clock, cudaDevAttrMemoryClockRate, 0)\n'
            '// TODO: Print: printf("device__attribute_max_mem_frequency_khz: %d\\n", mem_clock)\n'
        ),
    )

    # 6. Device Frame Buffer Bus Width
    _METRIC_DOCS["device__attribute_fb_bus_width"] = MetricDoc(
        target_name="device__attribute_fb_bus_width",
        what_it_measures=(
            "The memory bus width in bits. This determines the maximum bandwidth "
            "between the GPU and VRAM: bandwidth = bus_width/8 * 2 * memory_clock."
        ),
        measurement_approach=(
            "Simple API query using cudaDeviceGetAttribute(cudaDevAttrMemoryBusWidth).\n"
            "The returned value is in bits — print it directly.\n"
            "If the attribute is not supported, infer from other GPU properties."
        ),
        key_api_calls=[
            "cudaDeviceGetAttribute(&bus_width, cudaDevAttrMemoryBusWidth, device_id)",
        ],
        kernel_type="No kernel needed — pure API query",
        thread_config="N/A",
        expected_output_format="device__attribute_fb_bus_width: <integer_bits>",
        expected_range="256-4096 bits (A100=4096, RTX3090=384, RTX3070=256)",
        anti_cheat_notes=[],
        common_pitfalls=[
            "cudaDevAttrMemoryBusWidth may not be available on older CUDA versions",
            "Value is in bits, not bytes — don't divide by 8"
        ],
        code_template=(
            '// TODO: Include cuda_runtime.h and cstdio\n'
            '// TODO: In main(), declare int bus_width = 0;\n'
            '// TODO: Call cudaDeviceGetAttribute(&bus_width, cudaDevAttrMemoryBusWidth, 0)\n'
            '// TODO: Print: printf("device__attribute_fb_bus_width: %d\\n", bus_width)\n'
        ),
    )

    # 7. SM Throughput
    _METRIC_DOCS["sm__throughput.avg.pct_of_peak_sustained_elapsed"] = MetricDoc(
        target_name="sm__throughput.avg.pct_of_peak_sustained_elapsed",
        what_it_measures=(
            "The SM (Streaming Multiprocessor) compute throughput as a percentage "
            "of the peak sustained throughput. This is an NCU-native metric — "
            "it is measured by Nsight Compute, NOT calculated by the CUDA program."
        ),
        measurement_approach=(
            "1. Launch a COMPUTE-INTENSIVE kernel with NO memory bottleneck.\n"
            "2. Each thread performs millions of FMA (fused multiply-add) operations.\n"
            "3. Use volatile double to prevent dead-code elimination.\n"
            "4. Compute actual percentage: achieved_FLOPS / peak_FLOPS * 100.\n"
            "5. Query sm_count and compute capability via cudaDeviceGetAttribute at runtime.\n"
            "6. Inside kernel: record clock64() before/after FMA loop, output cycle count.\n"
            "7. Compute actual_freq_mhz = cycle_count / (elapsed_ms * 1000.0).\n"
            "8. peak_FLOPS = sm_count * fp64_per_sm * actual_freq_mhz * 1e6 * 2.\n"
            "   fp64_per_sm depends on compute capability: SM70=32, SM80=32, SM90=64, SM75=2, SM86+=2.\n"
            "9. achieved_FLOPS = total_FMA_ops / elapsed_seconds.\n"
            "10. ALWAYS run a WARMUP kernel before the timed measurement to reach steady-state clock.\n"
            "11. Do NOT use cudaDevAttrClockRate for peak — it may report base clock, not boost!\n"
            "KERNEL STRUCTURE: __global__ void compute_kernel(volatile double* sink, uint64_t* cycle_out)\n"
            "  - All variables in registers: double a=1.0123, b=0.9876, c=0.1234, result=0.0\n"
            "  - uint64_t start_cycle = clock64();\n"
            "  - #pragma unroll 1 before the FMA loop (10M+ iterations)\n"
            "  - result += a * b + c; // pure FMA, NO global memory access\n"
            "  - uint64_t end_cycle = clock64();\n"
            "  - *sink = result; + asm volatile('' : '+d'(sink) : : 'memory')\n"
            "  - if (threadIdx.x == 0 && blockIdx.x == 0) *cycle_out = end_cycle - start_cycle;\n"
            "  - Launch: sm_count*4 blocks x 256 threads\n"
            "  - Warmup: run kernel once before timed measurement"
        ),
        key_api_calls=[
            "cudaEventRecord(start); kernel<<<...>>>; cudaEventRecord(stop);",
            "cudaEventElapsedTime(&ms, start, stop)",
        ],
        kernel_type="Compute-intensive FMA kernel (pure compute, no global memory after init)",
        thread_config="sm_count*4 blocks x 256 threads (maximize SM utilization)",
        expected_output_format="sm__throughput.avg.pct_of_peak_sustained_elapsed: <float_percent>",
        expected_range="70-99% (well-tuned compute kernel should achieve >80%)",
        anti_cheat_notes=[
            "Kernel must be purely compute-bound — any global memory access will lower the percentage",
            "Use #pragma unroll 1 to prevent compiler from unrolling the FMA loop",
            "Must use volatile to prevent the compiler from eliminating the loop entirely",
            "COMPUTE actual percentage using clock64() for actual frequency + cudaDeviceGetAttribute for SM count",
            "Do NOT use cudaDevAttrClockRate for peak_flops — it may report base clock, not boost!",
            "The harness adds a runtime clamp [0,100] as safety net"
        ],
        common_pitfalls=[
            "Reading/writing global memory in the kernel — makes it memory-bound, not compute-bound",
            "Not using #pragma unroll 1 — compiler unrolls loop, reduces iteration count",
            "Outputting 0.0 as placeholder instead of computing actual percentage",
            "Using hardcoded peak values instead of cudaDeviceGetAttribute",
            "Using cudaDevAttrClockRate for peak_flops — may report base clock → pct > 100% → clamped to 100%"
        ],
        code_template=(
            '// TODO: Include cuda_runtime.h and cstdio\n'
            '// TODO: Write __global__ compute_kernel(volatile double* sink, uint64_t* cycle_out)\n'
            '//   - Declare local doubles a, b, c with arbitrary values\n'
            '//   - Initialize result = 0.0\n'
            '//   - uint64_t start_cycle = clock64();\n'
            '//   - Use #pragma unroll 1 before a loop of 10M iterations\n'
            '//   - In loop: result += a * b + c (FMA operation)\n'
            '//   - uint64_t end_cycle = clock64();\n'
            '//   - Write result to *sink to prevent dead-code elimination\n'
            '//   - if (threadIdx.x == 0 && blockIdx.x == 0) *cycle_out = end_cycle - start_cycle;\n'
            '// TODO: In main(): cudaMalloc for sink and cycle_out, query SM count and compute capability\n'
            '//   - cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0)\n'
            '//   - cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0)\n'
            '//   - cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0)\n'
            '//   - Determine fp64_per_sm: SM70=32, SM80=32, SM90=64, SM75=2, SM86+=2\n'
            '// TODO: Launch sm_count*4 blocks x 256 threads\n'
            '// TODO: Include warmup run before timed measurement\n'
            '// TODO: Use cudaEventElapsedTime for timing\n'
            '// TODO: cudaMemcpy cycle_count from device to host\n'
            '// COMPUTE: actual_freq_mhz = cycle_count / (elapsed_ms * 1000.0)\n'
            '// COMPUTE: peak_flops = sm_count * fp64_per_sm * actual_freq_mhz * 1e6 * 2\n'
            '// COMPUTE: achieved_flops = total_fma_ops / elapsed_seconds\n'
            '// COMPUTE: pct = (achieved_flops / peak_flops) * 100.0\n'
            '// printf("sm__throughput.avg.pct_of_peak_sustained_elapsed: %.2f\\n", pct);\n'
        ),
    )

    # 8. GPU Compute Memory Throughput
    _METRIC_DOCS["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"] = MetricDoc(
        target_name="gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        what_it_measures=(
            "The combined compute and memory throughput as a percentage of peak "
            "sustained throughput. This is an NCU-native metric — it is measured "
            "by Nsight Compute, NOT calculated by the CUDA program."
        ),
        measurement_approach=(
            "1. Launch a FUSED kernel that does: READ → COMPUTE → WRITE.\n"
            "2. Each thread reads from global memory, performs FMA operations, writes back.\n"
            "3. Compute actual percentage: achieved_BW / peak_BW * 100.\n"
            "4. Query mem_clock_khz and bus_width_bits via cudaDeviceGetAttribute at runtime.\n"
            "5. peak_BW = (mem_clock_khz / 1000.0) * 1e6 * (bus_width_bits / 8) * 2 / 1e9 GB/s.\n"
            "6. achieved_BW = total_bytes / elapsed_seconds / 1e9 GB/s.\n"
            "7. ALWAYS run a WARMUP kernel before the timed measurement to reach steady-state clock.\n"
            "KERNEL STRUCTURE: __global__ void fused_kernel(const float* __restrict__ input, volatile float* output, int n)\n"
            "  - Each thread: stride loop over elements\n"
            "  - Read: val = input[i] (use __ldg for read-only path)\n"
            "  - Compute: #pragma unroll 1; for(j=0;j<8;j++) val = val*1.0001f + 0.001f;\n"
            "  - Write: output[i] = val; (volatile prevents dead-code elimination)\n"
            "  - Buffer: >= 64MB to exceed L2 cache\n"
            "  - Launch: sm_count*4 blocks x 256 threads\n"
            "  - Warmup: run kernel once before timed measurement"
        ),
        key_api_calls=[
            "cudaMalloc(&d_input, BUFFER_SIZE); cudaMalloc(&d_output, BUFFER_SIZE)",
            "cudaEventRecord(start); kernel<<<...>>>; cudaEventRecord(stop);",
            "cudaEventElapsedTime(&ms, start, stop)",
        ],
        kernel_type="Fused read-compute-write kernel (exercises both memory and compute)",
        thread_config="sm_count*4 blocks x 256 threads",
        expected_output_format="gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed: <float_percent>",
        expected_range="50-95%",
        anti_cheat_notes=[
            "Kernel must do BOTH reads and writes with computation in between",
            "Use __ldg() for reads to go through load path",
            "Use volatile for writes to prevent optimization",
            "Balance compute intensity — not purely compute-bound or memory-bound",
            "COMPUTE actual percentage using cudaDeviceGetAttribute for peak calculation",
            "The harness adds a runtime clamp [0,100] as safety net"
        ],
        common_pitfalls=[
            "Only reading or only writing — must do both for combined throughput",
            "Too much compute — becomes purely compute-bound, not combined",
            "Too little compute — becomes purely memory-bound",
            "Outputting 0.0 as placeholder instead of computing actual percentage",
            "Using hardcoded peak values instead of cudaDeviceGetAttribute"
        ],
        code_template=(
            '// TODO: Include cuda_runtime.h and cstdio\n'
            '// TODO: Define BUFFER_SIZE (at least 64MB)\n'
            '// TODO: Write __global__ fused_kernel(const float* input, volatile float* output, int n)\n'
            '//   - Each thread reads input[idx] using __ldg()\n'
            '//   - Performs FMA compute loop (e.g., 100 iterations) on the read value\n'
            '//   - Writes result to output[idx] via volatile pointer\n'
            '//   - Use #pragma unroll 1 before the compute loop\n'
            '// TODO: In main(): cudaMalloc input/output buffers, initialize input\n'
            '// TODO: Query mem_clock_khz and bus_width_bits via cudaDeviceGetAttribute\n'
            '//   - cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0)\n'
            '//   - cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, 0)\n'
            '// TODO: Launch sm_count*4 blocks x 256 threads\n'
            '// TODO: Include warmup run before timed measurement\n'
            '// TODO: Use cudaEventElapsedTime for timing\n'
            '// COMPUTE: peak_bw = (mem_clock_khz / 1000.0) * 1e6 * (bus_width_bits / 8) * 2 / 1e9\n'
            '// COMPUTE: achieved_bw = (2.0 * buffer_size_bytes) / elapsed_seconds / 1e9\n'
            '//   NOTE: 2x because each element is read AND written (input+output)\n'
            '// COMPUTE: pct = (achieved_bw / peak_bw) * 100.0\n'
            '// printf("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed: %.2f\\n", pct);\n'
        ),
    )


def get_metric_doc(target_name: str) -> Optional[MetricDoc]:
    """Get the metric documentation for a given target name."""
    if not _METRIC_DOCS:
        _build_docs()
    return _METRIC_DOCS.get(target_name)


def get_metric_docs_for_targets(targets: list[str]) -> list[MetricDoc]:
    """Get metric documentation for a list of target names."""
    if not _METRIC_DOCS:
        _build_docs()
    return [_METRIC_DOCS.get(t) for t in targets if t in _METRIC_DOCS]


def has_metric_doc(target_name: str) -> bool:
    """Check if metric documentation exists for a given target."""
    if not _METRIC_DOCS:
        _build_docs()
    return target_name in _METRIC_DOCS


def format_metric_context(target_name: str) -> str:
    """Format metric documentation as context text for LLM injection."""
    doc = get_metric_doc(target_name)
    if not doc:
        return ""

    parts = [
        f"📊 NCU METRIC REFERENCE: {target_name}",
        f"=" * 60,
        f"",
        f"WHAT IT MEASURES:",
        f"  {doc.what_it_measures}",
        f"",
        f"MEASUREMENT APPROACH:",
        f"  {doc.measurement_approach}",
        f"",
        f"KERNEL TYPE: {doc.kernel_type}",
        f"THREAD CONFIG: {doc.thread_config}",
        f"EXPECTED OUTPUT FORMAT: {doc.expected_output_format}",
        f"EXPECTED RANGE: {doc.expected_range}",
        f"",
    ]

    if doc.key_api_calls:
        parts.append("KEY CUDA API CALLS:")
        for api in doc.key_api_calls:
            parts.append(f"  • {api}")
        parts.append("")

    if doc.anti_cheat_notes:
        parts.append("⚠️ ANTI-CHEAT NOTES:")
        for note in doc.anti_cheat_notes:
            parts.append(f"  • {note}")
        parts.append("")

    if doc.common_pitfalls:
        parts.append("❌ COMMON PITFALLS:")
        for pitfall in doc.common_pitfalls:
            parts.append(f"  • {pitfall}")
        parts.append("")

    if doc.code_template:
        parts.append("📝 REFERENCE CODE (adapt this for your implementation):")
        parts.append(f"```cuda\n{doc.code_template}\n```")
        parts.append("")
        parts.append("⚠️ This is a reference implementation. You MUST generate your own code.")
        parts.append("   Use this as methodology guidance, not a copy-paste template.")

    return "\n".join(parts)
