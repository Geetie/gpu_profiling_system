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
            '#include <cuda_runtime.h>\n'
            '#include <cstdio>\n\n'
            '__global__ void flag_kernel(volatile int* flags) {\n'
            '    flags[blockIdx.x] = 1;\n'
            '}\n\n'
            'int main() {\n'
            '    int sm_count = 0;\n'
            '    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);\n'
            '    printf("launch__sm_count: %d\\n", sm_count);\n'
            '    return 0;\n'
            '}'
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
            '#include <cuda_runtime.h>\n'
            '#include <cstdio>\n\n'
            '#define BUFFER_SIZE (64 * 1024 * 1024)  // 64MB\n\n'
            '__global__ void read_kernel(const float* data, int n) {\n'
            '    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n'
            '    int stride = blockDim.x * gridDim.x;\n'
            '    volatile float sink = 0.0f;\n'
            '    while (idx < n) {\n'
            '        sink += __ldg(&data[idx]);  // Force global memory read\n'
            '        idx += stride;\n'
            '    }\n'
            '}\n\n'
            'int main() {\n'
            '    float* d_data = nullptr;\n'
            '    cudaMalloc(&d_data, BUFFER_SIZE);\n'
            '    cudaMemset(d_data, 0x42, BUFFER_SIZE);\n\n'
            '    cudaEvent_t start, stop;\n'
            '    cudaEventCreate(&start); cudaEventCreate(&stop);\n\n'
            '    cudaEventRecord(start);\n'
            '    read_kernel<<<65535, 256>>>(d_data, BUFFER_SIZE / sizeof(float));\n'
            '    cudaEventRecord(stop);\n'
            '    cudaEventSynchronize(stop);\n\n'
            '    float ms = 0;\n'
            '    cudaEventElapsedTime(&ms, start, stop);\n\n'
            '    double seconds = ms / 1000.0;\n'
            '    double bytes = (double)(BUFFER_SIZE / sizeof(float)) * sizeof(float);\n'
            '    double bandwidth = bytes / seconds;\n'
            '    printf("dram__bytes_read.sum.per_second: %.2f\\n", bandwidth);\n'
            '    return 0;\n'
            '}'
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
            '#include <cuda_runtime.h>\n'
            '#include <cstdio>\n\n'
            '#define BUFFER_SIZE (64 * 1024 * 1024)  // 64MB\n\n'
            '__global__ void write_kernel(volatile float* data, int n) {\n'
            '    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n'
            '    int stride = blockDim.x * gridDim.x;\n'
            '    while (idx < n) {\n'
            '        data[idx] = 1.23456f;  // Pure write, no read\n'
            '        idx += stride;\n'
            '    }\n'
            '}\n\n'
            'int main() {\n'
            '    volatile float* d_data = nullptr;\n'
            '    cudaMalloc((void**)&d_data, BUFFER_SIZE);\n\n'
            '    cudaEvent_t start, stop;\n'
            '    cudaEventCreate(&start); cudaEventCreate(&stop);\n\n'
            '    cudaEventRecord(start);\n'
            '    write_kernel<<<65535, 256>>>((float*)d_data, BUFFER_SIZE / sizeof(float));\n'
            '    cudaEventRecord(stop);\n'
            '    cudaEventSynchronize(stop);\n\n'
            '    float ms = 0;\n'
            '    cudaEventElapsedTime(&ms, start, stop);\n\n'
            '    double seconds = ms / 1000.0;\n'
            '    double bytes = (double)(BUFFER_SIZE / sizeof(float)) * sizeof(float);\n'
            '    double bandwidth = bytes / seconds;\n'
            '    printf("dram__bytes_write.sum.per_second: %.2f\\n", bandwidth);\n'
            '    return 0;\n'
            '}'
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
            '#include <cuda_runtime.h>\n'
            '#include <cstdio>\n\n'
            'int main() {\n'
            '    int clock_rate = 0;\n'
            '    cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, 0);\n'
            '    printf("device__attribute_max_gpu_frequency_khz: %d\\n", clock_rate);\n'
            '    return 0;\n'
            '}'
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
            '#include <cuda_runtime.h>\n'
            '#include <cstdio>\n\n'
            'int main() {\n'
            '    int mem_clock = 0;\n'
            '    cudaDeviceGetAttribute(&mem_clock, cudaDevAttrMemoryClockRate, 0);\n'
            '    printf("device__attribute_max_mem_frequency_khz: %d\\n", mem_clock);\n'
            '    return 0;\n'
            '}'
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
            '#include <cuda_runtime.h>\n'
            '#include <cstdio>\n\n'
            'int main() {\n'
            '    int bus_width = 0;\n'
            '    cudaDeviceGetAttribute(&bus_width, cudaDevAttrMemoryBusWidth, 0);\n'
            '    printf("device__attribute_fb_bus_width: %d\\n", bus_width);\n'
            '    return 0;\n'
            '}'
        ),
    )

    # 7. SM Throughput
    _METRIC_DOCS["sm__throughput.avg.pct_of_peak_sustained_elapsed"] = MetricDoc(
        target_name="sm__throughput.avg.pct_of_peak_sustained_elapsed",
        what_it_measures=(
            "The SM (Streaming Multiprocessor) compute throughput as a percentage "
            "of the peak sustained throughput. This measures how efficiently the "
            "GPU's compute cores are being utilized."
        ),
        measurement_approach=(
            "1. Launch a COMPUTE-INTENSIVE kernel with NO memory bottleneck.\n"
            "2. Each thread performs millions of FMA (fused multiply-add) operations.\n"
            "3. Use volatile double to prevent dead-code elimination.\n"
            "4. Measure elapsed time with cudaEventElapsedTime.\n"
            "5. Calculate: actual_FLOPs / peak_FLOPs * 100 = utilization %\n"
            "   peak_FLOPs = sm_count * 2(DP_units) * clock_GHz * 2(FMA)\n"
            "   actual_FLOPs = blocks * threads * iterations * 2(FMA) * 2(flops_per_FMA)"
        ),
        key_api_calls=[
            "cudaEventRecord(start); kernel<<<...>>>; cudaEventRecord(stop);",
            "cudaEventElapsedTime(&ms, start, stop)",
            "cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0)",
            "cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, 0)",
        ],
        kernel_type="Compute-intensive FMA kernel (pure compute, no global memory after init)",
        thread_config="65535 blocks x 256 threads (maximize SM utilization)",
        expected_output_format="sm__throughput.avg.pct_of_peak_sustained_elapsed: <float_percent>",
        expected_range="70-99% (well-tuned compute kernel should achieve >80%)",
        anti_cheat_notes=[
            "Kernel must be purely compute-bound — any global memory access will lower the percentage",
            "Use #pragma unroll 1 to prevent compiler from unrolling the FMA loop",
            "Must use volatile to prevent the compiler from eliminating the loop entirely"
        ],
        common_pitfalls=[
            "Reading/writing global memory in the kernel — makes it memory-bound, not compute-bound",
            "Not using #pragma unroll 1 — compiler unrolls loop, reduces iteration count",
            "Using wrong peak FLOPs formula (must account for DP vs SP)"
        ],
        code_template=(
            '#include <cuda_runtime.h>\n'
            '#include <cstdio>\n\n'
            '__global__ void compute_kernel(volatile double* sink) {\n'
            '    double a = 1.234, b = 9.876, c = 3.141;\n'
            '    double result = 0.0;\n'
            '    #pragma unroll 1\n'
            '    for (int i = 0; i < 10000000; i++) {\n'
            '        result += a * b + c;  // FMA: 2 DP FLOPs per iteration\n'
            '    }\n'
            '    *sink = result;\n'
            '}\n\n'
            'int main() {\n'
            '    volatile double* d_sink = nullptr;\n'
            '    cudaMalloc((void**)&d_sink, sizeof(double));\n\n'
            '    cudaEvent_t start, stop;\n'
            '    cudaEventCreate(&start); cudaEventCreate(&stop);\n\n'
            '    cudaEventRecord(start);\n'
            '    compute_kernel<<<65535, 256>>>(d_sink);\n'
            '    cudaEventRecord(stop);\n'
            '    cudaEventSynchronize(stop);\n\n'
            '    float ms = 0;\n'
            '    cudaEventElapsedTime(&ms, start, stop);\n\n'
            '    // Calculate utilization\n'
            '    int sm_count = 0, clock_khz = 0;\n'
            '    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);\n'
            '    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);\n\n'
            '    double total_flops = 65535.0 * 256.0 * 10000000.0 * 2.0 * 2.0;\n'
            '    double seconds = ms / 1000.0;\n'
            '    double peak_flops = sm_count * 2.0 * (clock_khz / 1e6) * 2.0;\n'
            '    double util = (total_flops / seconds) / peak_flops * 100.0;\n'
            '    printf("sm__throughput.avg.pct_of_peak_sustained_elapsed: %.2f\\n", util);\n'
            '    return 0;\n'
            '}'
        ),
    )

    # 8. GPU Compute Memory Throughput
    _METRIC_DOCS["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"] = MetricDoc(
        target_name="gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        what_it_measures=(
            "The combined compute and memory throughput as a percentage of peak "
            "sustained throughput. This measures how efficiently the GPU utilizes "
            "BOTH its compute cores AND memory subsystem simultaneously."
        ),
        measurement_approach=(
            "1. Launch a FUSED kernel that does: READ → COMPUTE → WRITE.\n"
            "2. Each thread reads from global memory, performs FMA operations, writes back.\n"
            "3. Use cudaEventElapsedTime for wall-clock timing.\n"
            "4. Calculate effective throughput = (memory_bytes + compute_flops) / elapsed_time.\n"
            "5. Express as percentage of peak sustained throughput."
        ),
        key_api_calls=[
            "cudaMalloc(&d_input, BUFFER_SIZE); cudaMalloc(&d_output, BUFFER_SIZE)",
            "cudaEventRecord(start); kernel<<<...>>>; cudaEventRecord(stop);",
            "cudaEventElapsedTime(&ms, start, stop)",
        ],
        kernel_type="Fused read-compute-write kernel (exercises both memory and compute)",
        thread_config="65535 blocks x 256 threads",
        expected_output_format="gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed: <float_percent>",
        expected_range="50-95%",
        anti_cheat_notes=[
            "Kernel must do BOTH reads and writes with computation in between",
            "Use __ldg() for reads to go through load path",
            "Use volatile for writes to prevent optimization",
            "Balance compute intensity — not purely compute-bound or memory-bound"
        ],
        common_pitfalls=[
            "Only reading or only writing — must do both for combined throughput",
            "Too much compute — becomes purely compute-bound, not combined",
            "Too little compute — becomes purely memory-bound"
        ],
        code_template=(
            '#include <cuda_runtime.h>\n'
            '#include <cstdio>\n\n'
            '#define BUFFER_SIZE (64 * 1024 * 1024)  // 64MB\n\n'
            '__global__ void fused_kernel(const float* input, volatile float* output, int n) {\n'
            '    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n'
            '    int stride = blockDim.x * gridDim.x;\n'
            '    while (idx < n) {\n'
            '        float val = __ldg(&input[idx]);  // Read from global memory\n'
            '        double result = 0.0;\n'
            '        #pragma unroll 1\n'
            '        for (int i = 0; i < 100; i++) {  // Compute (FMA)\n'
            '            result += val * 1.234 + 0.567;\n'
            '        }\n'
            '        output[idx] = (float)result;  // Write back to global memory\n'
            '        idx += stride;\n'
            '    }\n'
            '}\n\n'
            'int main() {\n'
            '    float* d_input = nullptr;\n'
            '    volatile float* d_output = nullptr;\n'
            '    cudaMalloc(&d_input, BUFFER_SIZE);\n'
            '    cudaMalloc((void**)&d_output, BUFFER_SIZE);\n'
            '    cudaMemset(d_input, 0x42, BUFFER_SIZE);\n\n'
            '    cudaEvent_t start, stop;\n'
            '    cudaEventCreate(&start); cudaEventCreate(&stop);\n\n'
            '    cudaEventRecord(start);\n'
            '    fused_kernel<<<65535, 256>>>(d_input, (float*)d_output, BUFFER_SIZE / sizeof(float));\n'
            '    cudaEventRecord(stop);\n'
            '    cudaEventSynchronize(stop);\n\n'
            '    float ms = 0;\n'
            '    cudaEventElapsedTime(&ms, start, stop);\n\n'
            '    double seconds = ms / 1000.0;\n'
            '    double bytes = (double)(BUFFER_SIZE / sizeof(float)) * sizeof(float);\n'
            '    double bandwidth = bytes / seconds;\n\n'
            '    // Peak bandwidth approximation\n'
            '    int mem_clock = 0, bus_width = 0;\n'
            '    cudaDeviceGetAttribute(&mem_clock, cudaDevAttrMemoryClockRate, 0);\n'
            '    cudaDeviceGetAttribute(&bus_width, cudaDevAttrMemoryBusWidth, 0);\n'
            '    double peak_bw = 2.0 * (mem_clock / 1e3) * (bus_width / 8.0);  // bytes/s\n\n'
            '    double util = (bandwidth / peak_bw) * 100.0;\n'
            '    printf("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed: %.2f\\n", util);\n'
            '    return 0;\n'
            '}'
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
