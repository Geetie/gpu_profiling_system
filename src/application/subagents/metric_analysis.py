"""Metric Analysis Agent — GPU performance profiling and bottleneck identification.

Performs active tool-based analysis:
1. Extracts binary paths from CodeGen's output
2. Profiles binaries with Nsight Compute (ncu) when available
3. Performs Roofline model analysis on ncu metrics
4. Falls back to rule-based analysis on printf output when ncu unavailable
5. Generates targeted optimization recommendations
6. Provides cross-validation between CodeGen measurements and ncu data
7. Code quality review: accuracy, efficiency, resource consumption, compatibility, maintainability
8. Anti-cheat environment detection: frequency lock, SM masking, API interception
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
from typing import Any

from src.application.context import ContextManager, Role

logger = logging.getLogger(__name__)
from src.domain.permission import PermissionMode
from src.domain.subagent import (
    AgentRole,
    BaseSubAgent,
    CollaborationMessage,
    SubAgentResult,
    SubAgentStatus,
)
from src.domain.tool_contract import ToolRegistry


_METRIC_SELECTION_MAP: dict[str, list[str]] = {
    "dram_latency_cycles": [
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "l2__throughput.avg.pct_of_peak_sustained_elapsed",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
        "smsp__sass_inst_executed_op_global_ld.sum",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    ],
    "l2_latency_cycles": [
        "l2__throughput.avg.pct_of_peak_sustained_elapsed",
        "l1tex__t_sectors.sum",
        "l1tex__t_sectors_hit.sum",
        "l1tex__data_pipe_lsu_mem_global_op_ld.sum",
        "lts__t_sectors_op_read.sum",
        "lts__t_sectors_op_write.sum",
    ],
    "l1_latency_cycles": [
        "l1tex__t_sectors.sum",
        "l1tex__t_sectors_hit.sum",
        "l1tex__data_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__data_pipe_lsu_mem_local_op_ld.sum",
        "smsp__sass_inst_executed_op_global_ld.sum",
    ],
    "dram_bandwidth_gbps": [
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__bytes.sum",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "lts__t_sectors_op_read.sum",
        "lts__t_sectors_op_write.sum",
        "smsp__sass_inst_executed_op_global_st.sum",
    ],
    "shmem_bandwidth_gbps": [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__inst_mem_shared_op_ld.sum",
        "sm__inst_mem_shared_op_st.sum",
        "l1tex__data_pipe_surf_op_ld.sum",
        "l1tex__data_pipe_surf_op_st.sum",
    ],
    "l2_cache_size_mb": [
        "l2__throughput.avg.pct_of_peak_sustained_elapsed",
        "l1tex__t_sectors.sum",
        "l1tex__t_sectors_hit.sum",
        "lts__t_sectors_op_read.sum",
        "lts__t_sectors_op_write.sum",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    ],
    "l1_cache_size_kb": [
        "l1tex__t_sectors.sum",
        "l1tex__t_sectors_hit.sum",
        "l1tex__data_pipe_lsu_mem_global_op_ld.sum",
        "lts__t_sectors_op_read.sum",
    ],
    "actual_boost_clock_mhz": [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__cycles_active.sum",
        "sm__cycles_elapsed.sum",
        "sm__inst_executed.sum",
    ],
    "sm_count": [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__cycles_active.sum",
        "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
        "launch__blocks_per_sm.avg",
    ],
    "max_shmem_per_block_kb": [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__inst_mem_shared_op_ld.sum",
        "sm__inst_mem_shared_op_st.sum",
        "launch__blocks_per_sm.avg",
    ],
    "bank_conflict_penalty_ratio": [
        "l1tex__data_pipe_bank_conflicts_st_bank.sum",
        "l1tex__data_pipe_bank_conflicts_ld_bank.sum",
        "sm__inst_mem_shared_op_ld.sum",
        "sm__inst_mem_shared_op_st.sum",
    ],
    "shmem_bank_conflict_penalty_ns": [
        "l1tex__data_pipe_bank_conflicts_st_bank.sum",
        "l1tex__data_pipe_bank_conflicts_ld_bank.sum",
        "sm__inst_mem_shared_op_ld.sum",
        "sm__inst_mem_shared_op_st.sum",
    ],
    "default": [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "l2__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__pipe_tensor_op_hmma_cycle_active.avg.pct_of_peak_sustained_active",
        "l1tex__t_sectors.sum",
        "l1tex__t_sectors_hit.sum",
    ],
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__cycles_active.sum",
        "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
        "sm__pipe_fma_cycle_active.avg.pct_of_peak_sustained_active",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    ],
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": [
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "lts__t_sectors_op_read.sum",
        "lts__t_sectors_op_write.sum",
    ],
}

_RECOMMENDATIONS: dict[str, dict[str, list[str]]] = {
    "memory_bound": {
        "dram": [
            "Reduce global memory accesses by using shared memory for frequently accessed data",
            "Implement tiling to improve data reuse (e.g., 128x128 tiles for matrix operations)",
            "Check if memory accesses are coalesced (contiguous 32/64/128-byte access pattern per warp)",
            "Consider using async copy (cp.async) on Ampere/Hopper architectures to overlap transfers",
            "Increase arithmetic intensity to hide memory latency (more compute per memory access)",
            "Use __ldg() intrinsic for read-only data to leverage read-only data cache",
        ],
        "l2": [
            "Optimize data layout to improve L2 hit rate (spatial locality within 128-byte cache lines)",
            "Reduce working set size if possible (e.g., process data in cache-sized chunks)",
            "Check for bank conflicts in shared memory that may cause unnecessary L2 traffic",
            "Use __ldg() intrinsic for read-only data to leverage texture cache path",
            "Consider software prefetching with __prefetch_global_l1() or __prefetch_global_l2()",
        ],
        "l1": [
            "Use shared memory to cache frequently accessed data with irregular access patterns",
            "Implement software-managed caching for data that does not fit L1 naturally",
            "Check if data can be kept in registers instead of spilling to local memory",
            "Adjust -Xptxas -v flags to monitor register pressure and local memory usage",
            "Consider using __launch_bounds__ to control register allocation",
        ],
    },
    "compute_bound": {
        "tensor_core": [
            "Ensure using WMMA or CUTLASS library for matrix operations to utilize Tensor Cores",
            "Check if data types are optimal (FP16/BF16 vs FP32) for Tensor Core utilization",
            "Verify Tensor Core occupancy is high (use cudaOccupancy API to check)",
            "Consider mixed precision if accuracy allows (FP16 accumulation with FP32 output)",
            "Use wmma::load_matrix_sync and wmma::store_matrix_sync for efficient data movement",
        ],
        "fp32": [
            "Consider mixed precision (FP16/BF16) if accuracy allows for 2-4x throughput gain",
            "Check instruction mix (FMA vs ADD/MUL) — FMA provides 2x throughput over separate ops",
            "Optimize register usage to improve occupancy (reduce register pressure with --ptxas-options=-v)",
            "Use __shfl_sync() for warp-level communication instead of shared memory round-trips",
            "Consider using __funnelshift_l() or __funnelshift_r() for combined shift operations",
        ],
        "fp64": [
            "Check if FP64 computation is necessary — FP32 may suffice for some calculations",
            "Verify GPU has high FP64 throughput (data center GPUs vs consumer GPUs differ significantly)",
            "Consider iterative refinement: FP32 main computation with FP64 correction passes",
        ],
        "sm_throughput": [
            "CRITICAL: sm__throughput kernel MUST be PURELY compute-bound with ZERO global memory access in the FMA loop",
            "Use double-precision FMA (result += a * b + c) with ALL variables in registers — NOT float",
            "Initialize a, b, c as register doubles BEFORE the timed loop to prevent constant-folding",
            "Use volatile double* sink to prevent dead-code elimination of the FMA chain",
            "Add asm volatile('' : '+d'(sink) : : 'memory') AFTER the FMA loop",
            "Use #pragma unroll 1 before the FMA loop for consistent compiler behavior",
            "Launch sm_count*4 blocks x 256 threads for full SM utilization",
            "Run a WARMUP kernel before timing — GPU power ramping reduces first-run throughput by 20-30%",
            "Inside kernel: record clock64() before/after FMA loop, output cycle count",
            "COMPUTE actual_freq_mhz = cycle_count / (elapsed_ms * 1000.0)",
            "COMPUTE actual pct: achieved_flops / (sm_count * fp64_per_sm * actual_freq_mhz * 1e6 * 2) * 100",
            "Do NOT use cudaDevAttrClockRate for peak — it may report base clock, not boost!",
        ],
        "compute_memory_throughput": [
            "CRITICAL: gpu__compute_memory_throughput requires a FUSED read-compute-write kernel",
            "Read input[i] with const float* __restrict__ pointer, compute 8+ FMA per element USING the read value, write to volatile output[i]",
            "Use 64MB+ buffer (16M+ floats) to ensure data goes through DRAM (not just L2 cache)",
            "Use volatile float* output to prevent dead-code elimination of writes",
            "FMA chain MUST use the value read from memory — register-only FMA does NOT stress memory → 0.09%!",
            "WRONG: val = val * 1.0001f + 0.001f where val is register-only",
            "RIGHT: val = input[i]; then val = val * 1.0001f + 0.001f; then output[i] = val;",
            "Launch sm_count*4 blocks x 256 threads for full memory+compute utilization",
            "Run a WARMUP kernel before timing — GPU power ramping reduces first-run throughput",
            "Balance compute and memory: each thread should do 4-8 FMA operations per memory access",
            "COMPUTE actual pct: achieved_bw / peak_bw * 100",
            "  peak_bw = (mem_clock_khz/1000.0) * 1e6 * (bus_width_bits/8) * 2 / 1e9",
            "  achieved_bw = (2.0 * buffer_size_bytes) / elapsed_seconds / 1e9  // 2x for read+write",
            "If pct < 5%, the kernel is FUNDAMENTALLY WRONG — not stressing memory at all",
        ],
    },
    "latency_bound": [
        "Use more threads to hide latency (increase occupancy to >50% if possible)",
        "Implement software pipelining to overlap memory accesses with computation",
        "Consider using async operations (async copy, async transpose on Ampere+)",
        "Check if warp scheduling is optimal (avoid warp divergence within warps)",
        "Use prefetching if access pattern is predictable (__prefetch_global_l1/l2)",
        "Increase the number of concurrent warps per SM to improve latency hiding",
    ],
    "cache_capacity": [
        "Reduce working set size to fit within cache boundary (e.g., tile to L2-sized blocks)",
        "Implement blocking/tiling to process data in cache-sized chunks",
        "Use cache-aware algorithms (e.g., cache-oblivious matrix transpose)",
        "Consider using shared memory as software-managed cache for irregular access patterns",
        "Adjust cache configuration with cudaDeviceSetCacheConfig to prefer L1 over shared memory",
    ],
    "balanced": [
        "Workload is balanced between compute and memory — optimization may yield limited gains",
        "Consider algorithmic improvements rather than micro-optimizations",
        "Profile individual kernel phases to find phase-specific bottlenecks",
    ],
}


class MetricAnalysisAgent(BaseSubAgent):
    """GPU performance profiling agent with active tool calling and Roofline analysis.

    Capabilities:
    - Extracts binary paths from CodeGen output and profiles with ncu
    - Performs Roofline model analysis on hardware counter data
    - Identifies bottleneck type and sub-type (e.g., memory_bound/dram)
    - Generates targeted optimization recommendations
    - Cross-validates CodeGen measurements against ncu data
    - Falls back to printf analysis when ncu is unavailable
    """

    def __init__(
        self,
        context_manager: ContextManager | None = None,
        tool_registry: ToolRegistry | None = None,
        state_dir: str = ".state",
        permission_mode: PermissionMode = PermissionMode.DEFAULT,
        max_tokens: int = 16000,
        sandbox: Any | None = None,
    ) -> None:
        super().__init__(
            role=AgentRole.METRIC_ANALYSIS,
            context_manager=context_manager or ContextManager(max_tokens=max_tokens),
            tool_registry=tool_registry or ToolRegistry(),
            state_dir=state_dir,
            permission_mode=permission_mode,
            max_tokens=max_tokens,
        )
        self._sandbox = sandbox
        self._ncu_available: bool | None = None

    def _process(self, message: CollaborationMessage) -> SubAgentResult:
        """Analyze metrics with active tool calling and Roofline analysis.

        Strategy:
        1. Extract binary paths from CodeGen output
        2. If binary paths exist, profile with ncu
        3. Perform Roofline model analysis on ncu data
        4. Fall back to printf analysis if no binaries or ncu unavailable
        
        T5 FIX #3: Enhanced tool restrictions to prevent compile_cuda KeyError
        """
        # T5 FIX #3: Inject tool restriction guidance to prevent compile_cuda errors
        tool_restriction_guidance = (
            "🚨🚨🚨 CRITICAL STAGE IDENTIFICATION 🚨🚨🚨\n\n"
            "You are in the **METRIC ANALYSIS** stage.\n"
            "This is NOT the CodeGen stage. You CANNOT compile or execute code.\n\n"

            "✅ **YOUR ONLY AVAILABLE TOOLS:**\n"
            "  1. **run_ncu** - Profile existing binaries with Nsight Compute\n"
            "     Usage: Call run_ncu with the binary path from CodeGen stage\n"
            "  2. **read_file** - Read measurement output files\n\n"

            "❌❌❌ **ABSOLUTELY FORBIDDEN (WILL CAUSE IMMEDIATE ERRORS):**\n"
            "  • **compile_cuda** - This tool DOES NOT EXIST in this stage!\n"
            "    If you call it, you will get: KeyError \"Tool 'compile_cuda' is not registered\"\n"
            "  • **execute_binary** - This tool DOES NOT EXIST in this stage!\n"
            "    If you call it, you will get: KeyError \"Tool 'execute_binary' is not registered\"\n\n"

            "🎯 **YOUR ACTUAL TASK (STEP BY STEP):**\n"
            "1. Check if NCU is available (call run_ncu on any binary)\n"
            "2. If NCU returns ERR_NVGPUCTRPERM → NCU is unavailable in this environment\n"
            "   → Immediately switch to TEXT-BASED ANALYSIS of existing measurements\n"
            "3. Read the measurement values from execute_binary results (already done by CodeGen)\n"
            "4. Analyze if values are reasonable for this GPU architecture\n"
            "5. Provide confidence assessment and potential error sources\n\n"

            "⛔⛔⛔ **VIOLATION CONSEQUENCES:**\n"
            "If you attempt to call compile_cuda or execute_binary:\n"
            "  • The system will throw a KeyError exception\n"
            "  • Your analysis will FAIL completely\n"
            "  • You are WASTING valuable API calls and time\n"
            "  • The pipeline will stall or produce incorrect results\n\n"

            "💡 **REMEMBER:** CodeGen already compiled and executed the code.\n"
            "   Your job is to ANALYZE the results, not re-do the work!\n"
            "   If you need new measurements, that's CodeGen's job, not yours.\n"
        )
        
        self.context_manager.add_entry(
            Role.SYSTEM,
            tool_restriction_guidance,
            token_count=200,  # High visibility
        )

        # P0-1 FIX: Intelligent NCU Degradation - Check availability BEFORE starting
        # T11 ENHANCEMENT: Re-check on EVERY call (not just first time)
        # This ensures persistent guidance even if LLM "forgets" in later turns
        try:
            from src.infrastructure.tools.run_ncu import _ncu_permission_cache

            if _ncu_permission_cache.get("allowed") == False:
                # NCU is confirmed unavailable - inject FORCE SKIP guidance
                ncu_skip_guidance = (
                    "🚨🚨🚨 IMMEDIATE ACTION REQUIRED - NCU PERMANENTLY UNAVAILABLE 🚨🚨🚨\n\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "⚠️  ENVIRONMENT CONSTRAINT DETECTED\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

                    "❌ **FORBIDDEN ACTIONS (will waste time or cause errors):**\n"
                    "  1. Do NOT call run_ncu - it will ALWAYS return 'Permission denied'\n"
                    "  2. Do NOT call compile_cuda - it is NOT registered in this stage\n"
                    "  3. Do NOT call execute_binary - it is NOT registered in this stage\n"
                    "  4. Do NOT attempt to re-measure targets - CodeGen already completed that\n\n"

                    "✅ **MANDATORY ACTIONS (complete these in order):**\n"
                    "  **Step 1**: Use read_file to load measurement output files from CodeGen\n"
                    "    Look for files like: benchmark_*.stdout, results.json, or execution.log\n\n"
                    "  **Step 2**: Parse numeric values from the output\n"
                    "    Expected format: 'target_name: value' (e.g., 'dram_latency_cycles: 485')\n\n"
                    "  **Step 3**: Validate values against known GPU specifications\n"
                    "    • dram_latency_cycles: should be 100-2000 cycles\n"
                    "    • l2_cache_size_mb: should be 0.25-100 MB (varies by GPU)\n"
                    "    • actual_boost_clock_mhz: should be 500-4000 MHz\n\n"
                    "  **Step 4**: Generate text-based analysis report\n"
                    "    Classify bottleneck: compute_bound | memory_bound | latency_bound | cache_capacity\n"
                    "    Provide evidence and confidence assessment (0.3-0.6 range without NCU)\n\n"
                    "  **Step 5**: Output structured JSON with your analysis\n"
                    "{\n"
                    '  "bottleneck_type": "...",\n'
                    '  "bottleneck_sub_type": "...",\n'
                    '  "parsed_metrics": {...},\n'
                    '  "confidence": 0.4,\n'
                    '  "analysis_method": "text_based_fallback"\n'
                    "}\n\n"

                    "⏱️  TIME BUDGET: You have MAXIMUM 8 turns to complete this task.\n"
                    "   Do NOT waste turns on unavailable tools!\n\n"

                    "💡  REMINDER: CodeGen stage has ALREADY:\n"
                    "   ✅ Compiled CUDA code for all targets\n"
                    "   ✅ Executed the binaries successfully\n"
                    "   ✅ Produced measurement values in stdout\n"
                    "   YOUR ONLY job is to ANALYZE those existing results!\n"
                )
                self.context_manager.add_entry(
                    Role.SYSTEM,
                    ncu_skip_guidance,
                    token_count=250,  # Highest priority - override all other guidance
                )
                logger.warning(
                    "[MetricAnalysis] 🚨 P0-1 ACTIVE: NCU unavailable - "
                    "forcing text-based analysis mode (expected ~300s vs ~1062s)"
                )
        except ImportError:
            logger.debug("[MetricAnalysis] run_ncu module not found - skipping pre-check")
        except Exception as e:
            logger.warning(f"[MetricAnalysis] P0-1 pre-check error (non-fatal): {e}")

        prev_result = message.payload.get("prev_result", {})
        target_spec = message.payload.get("target_spec", {})

        if isinstance(prev_result, dict):
            prev_data = prev_result.get("data", {})
        elif hasattr(prev_result, "data"):
            prev_data = prev_result.data if prev_result.data else {}
        else:
            prev_data = {}

        raw_output = prev_data.get("raw_output", "")
        tool_results = prev_data.get("tool_results", [])
        target = target_spec.get("target", "unknown")

        binary_paths = self._extract_binary_paths(prev_data, tool_results)

        ncu_results: list[dict[str, Any]] = []
        ncu_success = False

        if binary_paths:
            ncu_results, ncu_success = self._profile_with_ncu(binary_paths, target)

        if ncu_success and ncu_results:
            return self._analyze_ncu_results(ncu_results, target, target_spec, raw_output)

        if raw_output:
            return self._analyze_raw_output(raw_output, target, target_spec)

        return SubAgentResult(
            agent_role=self.role,
            status=SubAgentStatus.FAILED,
            error="No binary paths or raw output to analyze",
        )

    def _extract_binary_paths(
        self, prev_data: dict[str, Any], tool_results: list[dict[str, Any]]
    ) -> list[str]:
        """Extract binary paths from CodeGen's tool_results or output data."""
        binary_paths: list[str] = []

        direct_path = prev_data.get("binary_path", "")
        if direct_path and isinstance(direct_path, str) and len(direct_path.strip()) > 0:
            binary_paths.append(direct_path.strip())

        for result in tool_results:
            if not isinstance(result, dict):
                continue

            bp = result.get("binary_path", "")
            if bp and isinstance(bp, str) and len(bp.strip()) > 0:
                binary_paths.append(bp.strip())

            if result.get("tool") == "compile_cuda" and result.get("success") is True:
                bp2 = result.get("binary_path", "")
                if bp2 and isinstance(bp2, str) and len(bp2.strip()) > 0:
                    binary_paths.append(bp2.strip())

            if result.get("tool") == "execute_binary":
                exe = result.get("executable", "") or result.get("binary_path", "")
                if exe and isinstance(exe, str) and len(exe.strip()) > 0:
                    binary_paths.append(exe.strip())

        seen: set[str] = set()
        unique: list[str] = []
        for p in binary_paths:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique

    def _profile_with_ncu(
        self, binary_paths: list[str], target: str
    ) -> tuple[list[dict[str, Any]], bool]:
        """Profile binaries with Nsight Compute.

        T11 FIX: Enhanced with early-exit on first NCU failure.
        If the first binary fails due to NCU unavailability, immediately stop
        profiling remaining binaries to save ~35s per binary.

        Returns:
            Tuple of (list of ncu result dicts, whether any profiling succeeded).
        """
        metrics = self._select_metrics_for_target(target)
        ncu_results: list[dict[str, Any]] = []
        any_success = False
        ncu_unavailable_detected = False  # T11 FIX: Track NCU availability

        for binary_path in binary_paths:
            # T11 FIX: Early exit if NCU already detected as unavailable
            if ncu_unavailable_detected:
                import logging as _logging
                logger = _logging.getLogger(__name__)
                logger.warning(
                    f"[MetricAnalysis] ⚡ SKIPPING remaining binary (NCU unavailable)\n"
                    f"  Binary: {binary_path}\n"
                    f"  Saved ~35s by early exit"
                )
                continue

            try:
                result = self._call_tool("run_ncu", {
                    "executable": binary_path,
                    "metrics": metrics,
                })
                if isinstance(result, dict):
                    parsed = result.get("parsed_metrics", {})
                    error = parsed.get("error", "") if isinstance(parsed, dict) else ""

                    # T11 FIX: Detect NCU unavailability and set flag
                    if parsed.get("fast_fail") or parsed.get("cached_result"):
                        ncu_unavailable_detected = True
                        logger.warning(
                            f"[MetricAnalysis] ⚠️ NCU unavailable detected for '{binary_path}'\n"
                            f"  Error: {error[:100]}\n"
                            f"  Will skip remaining {len(binary_paths) - binary_paths.index(binary_path) - 1} binaries"
                        )

                    if not error:
                        ncu_results.append(result)
                        any_success = True
                        self._persister.log_entry(
                            action="ncu_profile_success",
                            details={"binary": binary_path, "metric_count": len(parsed)},
                        )
                    else:
                        self._persister.log_entry(
                            action="ncu_profile_error",
                            details={"binary": binary_path, "error": error},
                        )
                else:
                    ncu_results.append({"raw_output": str(result), "parsed_metrics": {}})
            except Exception as e:
                self._persister.log_entry(
                    action="ncu_profile_failed",
                    details={"binary": binary_path, "error": str(e)},
                )

        return ncu_results, any_success

    def _call_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        """Call a registered tool through the infrastructure layer.

        T11 FIX: Enhanced with NCU permission pre-check to avoid wasted API calls.
        If NCU is already marked as unavailable, returns cached error immediately (<1ms)
        instead of attempting actual execution (which would waste ~35s per call).
        
        CRITICAL FIX: Added support for read_file tool which is documented in prompts
        but was not actually implemented, causing all file read operations to fail.
        """
        from src.infrastructure.tools.run_ncu import (
            run_ncu_handler,
            _ncu_permission_cache,
        )

        if tool_name == "run_ncu":
            # T11 FIX: Pre-check NCU permission cache BEFORE executing
            if _ncu_permission_cache.get("checked") and not _ncu_permission_cache.get("allowed"):
                import logging as _logging
                logger = _logging.getLogger(__name__)
                logger.warning(
                    f"[MetricAnalysis] ⚡ FAST FAIL: NCU cached as unavailable\n"
                    f"  Reason: {_ncu_permission_cache.get('error_message', 'unknown')}\n"
                    f"  Saved ~35s by skipping actual ncu execution"
                )
                return {
                    "raw_output": "",
                    "parsed_metrics": {
                        "error": f"NCU unavailable (cached): {_ncu_permission_cache.get('error_message', 'unknown')}",
                        "hint": "NCU permission denied. Use text-based analysis instead.",
                        "cached_result": True,
                        "fast_fail": True,  # Flag for _profile_with_ncu to detect
                    },
                }

            return run_ncu_handler(args, sandbox=self._sandbox)

        # CRITICAL FIX: Support read_file tool which is documented in prompts
        if tool_name == "read_file":
            from src.infrastructure.tools.file_tools import make_read_file_handler
            from src.infrastructure.file_ops import FileOperations
            
            file_ops = FileOperations(self._sandbox.root if self._sandbox else "/workspace")
            handler = make_read_file_handler(file_ops)
            return handler(args)

        if self.tool_registry.has_tool(tool_name):
            self._persister.log_entry(
                action="tool_call_unsupported",
                details={"tool": tool_name, "error": "No handler registered"},
            )

        return {"error": f"Tool '{tool_name}' handler not available"}

    def _select_metrics_for_target(self, target: str) -> list[str]:
        """Select appropriate NCU metrics based on measurement target.

        Maps each profiling target to the specific hardware counters
        needed for Roofline analysis and bottleneck identification.
        """
        return _METRIC_SELECTION_MAP.get(target, _METRIC_SELECTION_MAP["default"])

    def _analyze_ncu_results(
        self,
        ncu_results: list[dict[str, Any]],
        target: str,
        target_spec: dict[str, Any],
        raw_output: str,
    ) -> SubAgentResult:
        """Analyze ncu profiling results with Roofline model."""
        merged_metrics: dict[str, float] = {}
        for ncu_result in ncu_results:
            parsed = ncu_result.get("parsed_metrics", {})
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    if isinstance(v, (int, float)):
                        merged_metrics[k] = float(v)

        if self._model_caller is not None:
            return self._llm_analyze_with_ncu(merged_metrics, target, target_spec, raw_output)

        roofline_result = self.analyze_roofline(merged_metrics, target)

        cross_validation = None
        if raw_output:
            codegen_metrics = self._parse_output(raw_output)
            cross_validation = self.cross_validate(codegen_metrics, merged_metrics, target)

        evidence = roofline_result.get("evidence", {})
        recommendations = roofline_result.get("recommendations", [])
        suggested_fixes = self._generate_suggested_fixes(recommendations)

        confidence, confidence_reason = self._assess_confidence_detailed(
            merged_metrics, ncu_available=True, cross_validation=cross_validation
        )

        code_quality = self._get_code_quality_review(target, raw_output)

        return SubAgentResult(
            agent_role=self.role,
            status=SubAgentStatus.SUCCESS,
            data={
                "bottleneck_type": roofline_result["bottleneck_type"],
                "bottleneck_sub_type": roofline_result.get("bottleneck_sub_type"),
                "parsed_metrics": merged_metrics,
                "measurements": merged_metrics,
                "evidence": evidence,
                "recommendations": recommendations,
                "suggested_fixes": suggested_fixes,
                "confidence": confidence,
                "confidence_reason": confidence_reason,
                "cross_validation": cross_validation,
                "analysis_method": "ncu_roofline",
                "code_quality": code_quality,
            },
            metadata={"target": target, "ncu_result_count": len(ncu_results)},
        )

    def _analyze_raw_output(
        self,
        raw_output: str,
        target: str,
        target_spec: dict[str, Any],
    ) -> SubAgentResult:
        """Analyze CodeGen's printf output when ncu is unavailable."""
        if self._model_caller is not None:
            return self._llm_analyze_raw(raw_output, target, target_spec)

        parsed_metrics = self._parse_output(raw_output)
        roofline_result = self.analyze_roofline(parsed_metrics, target)

        evidence = roofline_result.get("evidence", {})
        recommendations = roofline_result.get("recommendations", [])
        suggested_fixes = self._generate_suggested_fixes(recommendations)

        confidence, confidence_reason = self._assess_confidence_detailed(
            parsed_metrics, ncu_available=False
        )

        code_quality = self._get_code_quality_review(target, raw_output)

        return SubAgentResult(
            agent_role=self.role,
            status=SubAgentStatus.SUCCESS,
            data={
                "bottleneck_type": roofline_result["bottleneck_type"],
                "bottleneck_sub_type": roofline_result.get("bottleneck_sub_type"),
                "parsed_metrics": parsed_metrics,
                "measurements": parsed_metrics,
                "evidence": evidence,
                "recommendations": recommendations,
                "suggested_fixes": suggested_fixes,
                "confidence": confidence,
                "confidence_reason": confidence_reason,
                "cross_validation": None,
                "analysis_method": "printf_pattern_matching",
                "code_quality": code_quality,
            },
            metadata={"target": target, "ncu_available": False},
        )

    def analyze_roofline(
        self, metrics: dict[str, Any], target: str
    ) -> dict[str, Any]:
        """Perform Roofline model analysis on collected metrics.

        Implements the full Roofline methodology:
        1. Compare compute vs memory utilization
        2. If memory-bound, identify which memory level (DRAM, L2, L1)
        3. If compute-bound, identify which compute unit (Tensor Core, FP32, FP64)
        4. Detect latency-bound and cache-capacity patterns
        5. Generate evidence and recommendations
        """
        compute_util = self._extract_metric_value(
            metrics,
            ["sm__throughput.avg.pct_of_peak_sustained_elapsed",
             "sm__throughput.avg.pct_of_peak_sustained_active",
             "SM Throughput"],
        )

        memory_util = self._extract_metric_value(
            metrics,
            ["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
             "Memory Throughput",
             "dram__throughput.avg.pct_of_peak_sustained_elapsed"],
        )

        dram_util = self._extract_metric_value(
            metrics,
            ["dram__throughput.avg.pct_of_peak_sustained_elapsed",
             "DRAM Throughput"],
        )

        l2_util = self._extract_metric_value(
            metrics,
            ["l2__throughput.avg.pct_of_peak_sustained_elapsed",
             "L2 Throughput"],
        )

        tensor_util = self._extract_metric_value(
            metrics,
            ["sm__pipe_tensor_op_hmma_cycle_active.avg.pct_of_peak_sustained_active",
             "Tensor Core Utilization"],
        )

        has_ncu_data = compute_util > 0 or memory_util > 0

        if has_ncu_data:
            return self._roofline_from_ncu(
                compute_util, memory_util, dram_util, l2_util,
                tensor_util, metrics, target,
            )

        return self._roofline_from_keywords(metrics, target)

    def _roofline_from_ncu(
        self,
        compute_util: float,
        memory_util: float,
        dram_util: float,
        l2_util: float,
        tensor_util: float,
        metrics: dict[str, Any],
        target: str,
    ) -> dict[str, Any]:
        """Roofline analysis using actual ncu hardware counter data."""
        if compute_util > 70 and memory_util < 50:
            bottleneck = "compute_bound"
            sub_type = self._identify_compute_unit(tensor_util, metrics)
        elif memory_util > 70 and compute_util < 50:
            bottleneck = "memory_bound"
            sub_type = self._identify_memory_level(dram_util, l2_util, metrics)
        elif compute_util > 50 and memory_util > 50:
            bottleneck = "balanced"
            sub_type = None
        else:
            bottleneck, sub_type = self._classify_by_target(target, metrics)

        evidence = self._collect_evidence(metrics, bottleneck, sub_type, {
            "compute_utilization": compute_util,
            "memory_utilization": memory_util,
            "dram_utilization": dram_util,
            "l2_utilization": l2_util,
            "tensor_core_utilization": tensor_util,
        })

        recommendations = self._generate_recommendations(bottleneck, sub_type)

        return {
            "bottleneck_type": bottleneck,
            "bottleneck_sub_type": sub_type,
            "compute_utilization": compute_util,
            "memory_utilization": memory_util,
            "evidence": evidence,
            "recommendations": recommendations,
        }

    def _roofline_from_keywords(
        self, metrics: dict[str, Any], target: str
    ) -> dict[str, Any]:
        """Roofline analysis using keyword-based heuristics when ncu data unavailable."""
        bottleneck, sub_type = self._classify_by_target(target, metrics)

        if bottleneck == "unknown":
            bottleneck, sub_type = self._classify_by_keywords(metrics)

        evidence = self._collect_evidence(metrics, bottleneck, sub_type, {
            "compute_utilization": 0,
            "memory_utilization": 0,
            "note": "No ncu data available — classification based on target type and metric keywords",
        })

        recommendations = self._generate_recommendations(bottleneck, sub_type)

        return {
            "bottleneck_type": bottleneck,
            "bottleneck_sub_type": sub_type,
            "compute_utilization": 0,
            "memory_utilization": 0,
            "evidence": evidence,
            "recommendations": recommendations,
        }

    def _classify_by_target(
        self, target: str, metrics: dict[str, Any]
    ) -> tuple[str, str | None]:
        """Classify bottleneck based on the measurement target type."""
        latency_targets = {
            "dram_latency_cycles": ("latency_bound", "dram"),
            "l2_latency_cycles": ("latency_bound", "l2"),
            "l1_latency_cycles": ("latency_bound", "l1"),
        }
        bandwidth_targets = {
            "dram_bandwidth_gbps": ("memory_bound", "dram"),
            "shmem_bandwidth_gbps": ("memory_bound", "shmem"),
        }
        capacity_targets = {
            "l2_cache_size_mb": ("cache_capacity", "l2"),
            "l1_cache_size_kb": ("cache_capacity", "l1"),
            "max_shmem_per_block_kb": ("cache_capacity", "shmem"),
        }
        compute_targets = {
            "actual_boost_clock_mhz": ("compute_bound", "clock"),
            "sm_count": ("compute_bound", "sm"),
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": ("compute_bound", "sm_throughput"),
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": ("memory_bound", "compute_memory_throughput"),
        }
        conflict_targets = {
            "bank_conflict_penalty_ratio": ("latency_bound", "bank_conflict"),
            "shmem_bank_conflict_penalty_ns": ("latency_bound", "bank_conflict"),
        }

        for mapping in [latency_targets, bandwidth_targets, capacity_targets,
                        compute_targets, conflict_targets]:
            if target in mapping:
                return mapping[target]

        return "unknown", None

    def _classify_by_keywords(
        self, metrics: dict[str, Any]
    ) -> tuple[str, str | None]:
        """Classify bottleneck by examining metric key names."""
        keys_lower = {k.lower() for k in metrics}

        if any("latency" in k or "cycle" in k for k in keys_lower):
            return "latency_bound", None
        if any("bandwidth" in k or "throughput" in k for k in keys_lower):
            return "memory_bound", None
        if any("ipc" in k or "flop" in k or "tensor" in k for k in keys_lower):
            return "compute_bound", None
        if any("cache" in k or "miss" in k or "cliff" in k for k in keys_lower):
            return "cache_capacity", None

        values = [v for v in metrics.values() if isinstance(v, (int, float))]
        if len(values) >= 3:
            for i in range(1, len(values)):
                if values[i] > values[i - 1] * 2:
                    return "cache_capacity", None

        return "unknown", None

    def _identify_compute_unit(
        self, tensor_util: float, metrics: dict[str, Any]
    ) -> str:
        """Identify which compute unit is the bottleneck."""
        if tensor_util > 80:
            return "tensor_core"

        fp64_util = self._extract_metric_value(
            metrics,
            ["sm__pipe_fma_cycle_active.avg.pct_of_peak_sustained_active",
             "FP64 Utilization"],
        )
        if fp64_util > 60:
            return "fp64"

        return "fp32"

    def _identify_memory_level(
        self,
        dram_util: float,
        l2_util: float,
        metrics: dict[str, Any],
    ) -> str:
        """Identify which memory level is the bottleneck."""
        if dram_util > 80:
            return "dram"
        if l2_util > 80:
            return "l2"

        l1_hit_rate = self._compute_l1_hit_rate(metrics)
        if l1_hit_rate is not None and l1_hit_rate < 50:
            return "l1"
        if l2_util > dram_util and l2_util > 0:
            return "l2"

        return "dram"

    def _compute_l1_hit_rate(self, metrics: dict[str, Any]) -> float | None:
        """Compute L1 cache hit rate from ncu metrics."""
        sectors = self._extract_metric_value(
            metrics, ["l1tex__t_sectors.sum", "L1 Sectors"])
        hits = self._extract_metric_value(
            metrics, ["l1tex__t_sectors_hit.sum", "L1 Hits"])

        if sectors > 0 and hits > 0:
            return (hits / sectors) * 100.0
        return None

    def _collect_evidence(
        self,
        metrics: dict[str, Any],
        bottleneck: str,
        sub_type: str | None,
        util_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Collect supporting evidence for the bottleneck conclusion."""
        evidence: dict[str, Any] = {
            "bottleneck_type": bottleneck,
            "bottleneck_sub_type": sub_type,
            "utilization_data": util_data,
            "key_metrics": {},
        }

        relevant_metric_patterns = {
            "memory_bound": ["throughput", "bandwidth", "bytes", "sectors", "dram", "l2", "l1"],
            "compute_bound": ["throughput", "flop", "ipc", "tensor", "fma", "cycles_active"],
            "latency_bound": ["latency", "cycles", "elapsed", "duration"],
            "cache_capacity": ["cache", "miss", "hit", "cliff", "sectors"],
        }

        patterns = relevant_metric_patterns.get(bottleneck, [])
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                key_lower = key.lower()
                if any(p in key_lower for p in patterns):
                    evidence["key_metrics"][key] = value

        if not evidence["key_metrics"]:
            numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            evidence["key_metrics"] = dict(list(numeric_metrics.items())[:10])

        analysis_parts = [f"Identified bottleneck: {bottleneck}"]
        if sub_type:
            analysis_parts.append(f"Sub-type: {sub_type}")

        compute_util = util_data.get("compute_utilization", 0)
        memory_util = util_data.get("memory_utilization", 0)
        if compute_util > 0 or memory_util > 0:
            analysis_parts.append(
                f"Compute utilization: {compute_util:.1f}%, "
                f"Memory utilization: {memory_util:.1f}%"
            )
            if bottleneck == "compute_bound":
                analysis_parts.append(
                    f"Compute utilization ({compute_util:.1f}%) exceeds memory utilization "
                    f"({memory_util:.1f}%), indicating compute-bound workload"
                )
            elif bottleneck == "memory_bound":
                analysis_parts.append(
                    f"Memory utilization ({memory_util:.1f}%) exceeds compute utilization "
                    f"({compute_util:.1f}%), indicating memory-bound workload"
                )

        evidence["analysis"] = " ".join(analysis_parts)
        return evidence

    def _generate_recommendations(
        self, bottleneck: str, sub_type: str | None
    ) -> list[str]:
        """Generate targeted optimization recommendations based on bottleneck analysis."""
        if bottleneck in _RECOMMENDATIONS:
            bottleneck_recs = _RECOMMENDATIONS[bottleneck]
            if isinstance(bottleneck_recs, dict):
                if sub_type and sub_type in bottleneck_recs:
                    return list(bottleneck_recs[sub_type])
                for key in bottleneck_recs:
                    return list(bottleneck_recs[key])
            elif isinstance(bottleneck_recs, list):
                return list(bottleneck_recs)

        return [
            "Insufficient data for specific recommendations. "
            "Consider running ncu profiling with comprehensive metrics for deeper analysis."
        ]

    def _generate_suggested_fixes(self, recommendations: list[str]) -> list[str]:
        """Convert recommendations into actionable fix suggestions for CodeGen."""
        fixes: list[str] = []
        for rec in recommendations[:3]:
            fixes.append(rec)
        return fixes

    def cross_validate(
        self,
        codegen_metrics: dict[str, Any],
        ncu_metrics: dict[str, Any],
        target: str,
    ) -> dict[str, Any] | None:
        """Cross-validate CodeGen measurements against ncu profiling data.

        Compares the values reported by CodeGen's printf output with
        ncu's hardware counter measurements to detect discrepancies.
        """
        if not codegen_metrics or not ncu_metrics:
            return None

        codegen_numeric = {k: v for k, v in codegen_metrics.items() if isinstance(v, (int, float))}
        ncu_numeric = {k: v for k, v in ncu_metrics.items() if isinstance(v, (int, float))}

        if not codegen_numeric or not ncu_numeric:
            return None

        discrepancies: list[dict[str, Any]] = []
        agreements: list[dict[str, Any]] = []

        target_keywords = {
            "dram_latency_cycles": ["latency", "dram", "cycle"],
            "l2_latency_cycles": ["latency", "l2", "cycle"],
            "l1_latency_cycles": ["latency", "l1", "cycle"],
            "dram_bandwidth_gbps": ["bandwidth", "dram", "throughput"],
            "l2_cache_size_mb": ["cache", "l2", "size"],
            "actual_boost_clock_mhz": ["clock", "frequency", "mhz"],
        }

        keywords = target_keywords.get(target, ["value", "result", "measurement"])

        for cg_key, cg_value in codegen_numeric.items():
            cg_key_lower = cg_key.lower()
            if not any(kw in cg_key_lower for kw in keywords):
                continue

            for ncu_key, ncu_value in ncu_numeric.items():
                ncu_key_lower = ncu_key.lower()
                if not any(kw in ncu_key_lower for kw in keywords):
                    continue

                if ncu_value == 0:
                    continue

                diff_pct = abs(cg_value - ncu_value) / abs(ncu_value) * 100

                comparison = {
                    "codegen_key": cg_key,
                    "codegen_value": cg_value,
                    "ncu_key": ncu_key,
                    "ncu_value": ncu_value,
                    "difference_percent": round(diff_pct, 2),
                }

                if diff_pct > 20:
                    discrepancies.append(comparison)
                else:
                    agreements.append(comparison)

        result: dict[str, Any] = {
            "agreement": len(discrepancies) == 0,
            "agreements_count": len(agreements),
            "discrepancies_count": len(discrepancies),
        }

        if discrepancies:
            result["discrepancies"] = discrepancies
            result["recommendation"] = (
                "Significant discrepancy detected between CodeGen and ncu measurements. "
                "Re-run measurement with ncu for ground truth."
            )
        if agreements:
            result["confirmed_metrics"] = agreements

        return result

    def _assess_confidence_detailed(
        self,
        metrics: dict[str, Any],
        ncu_available: bool,
        cross_validation: dict[str, Any] | None = None,
    ) -> tuple[float, str]:
        """Assess confidence in the bottleneck identification with detailed reasoning.

        Returns:
            Tuple of (confidence score 0.0-1.0, human-readable reason).
        """
        if not metrics:
            return 0.0, "No metrics available for analysis"

        numeric_count = sum(1 for v in metrics.values() if isinstance(v, (int, float)))

        if ncu_available:
            base_confidence = 0.7
            metric_bonus = min(0.2, numeric_count / 20.0)

            if cross_validation:
                if cross_validation.get("agreement", False):
                    validation_bonus = 0.1
                    reason = (
                        f"High confidence: ncu profiling confirms with {numeric_count} consistent metrics. "
                        f"Cross-validation passed."
                    )
                else:
                    validation_bonus = -0.1
                    disc_count = cross_validation.get("discrepancies_count", 0)
                    reason = (
                        f"Medium confidence: ncu profiling available but {disc_count} discrepancy(ies) "
                        f"detected in cross-validation."
                    )
            else:
                validation_bonus = 0.0
                reason = (
                    f"High confidence: ncu profiling confirms with {numeric_count} metrics. "
                    f"No cross-validation performed."
                )

            confidence = min(1.0, base_confidence + metric_bonus + validation_bonus)
            return round(confidence, 2), reason

        base_confidence = 0.2
        metric_bonus = min(0.2, numeric_count / 10.0)
        confidence = min(0.5, base_confidence + metric_bonus)

        reason = (
            f"Low confidence: only CodeGen printf output available ({numeric_count} numeric metrics). "
            f"ncu profiling not available — recommend running ncu for higher confidence."
        )
        return round(confidence, 2), reason

    def _extract_metric_value(
        self, metrics: dict[str, Any], keys: list[str]
    ) -> float:
        """Extract a metric value trying multiple possible key names.

        NCU metric names vary across GPU architectures and versions.
        This method tries multiple possible key names and returns
        the first numeric value found.
        """
        for key in keys:
            if key in metrics:
                val = metrics[key]
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, str):
                    try:
                        cleaned = re.sub(r"[^\d.]", "", val.split()[0]) if val else ""
                        if cleaned:
                            return float(cleaned)
                    except (ValueError, IndexError):
                        continue

        for key in metrics:
            if len(key) < 4:
                continue
            key_lower = key.lower()
            for search_key in keys:
                search_lower = search_key.lower()
                if len(search_lower) >= 4 and (
                    search_lower in key_lower or key_lower in search_lower
                ):
                    val = metrics[key]
                    if isinstance(val, (int, float)):
                        return float(val)

        return 0.0

    def _llm_analyze_with_ncu(
        self,
        ncu_metrics: dict[str, float],
        target: str,
        target_spec: dict[str, Any],
        raw_output: str,
    ) -> SubAgentResult:
        """Use LLM to analyze ncu profiling results with Roofline context."""
        import json as _json

        user_msg = (
            f"Analyze this GPU profiling data using Roofline methodology.\n\n"
            f"Target: {target}\n"
            f"Target spec: {target_spec}\n\n"
            f"NCU metrics:\n{_json.dumps(ncu_metrics, indent=2)}\n\n"
            f"Raw benchmark output:\n{raw_output[:2000]}\n\n"
            f"Perform Roofline analysis:\n"
            f"1. Compare compute vs memory utilization\n"
            f"2. If memory-bound (memory_util > 70%, compute_util < 50%), identify level: dram, l2, or l1\n"
            f"3. If compute-bound (compute_util > 70%, memory_util < 50%), identify unit: tensor_core, fp32, or fp64\n"
            f"4. If latency > 500 cycles, classify as latency_bound\n"
            f"5. If performance cliff detected, classify as cache_capacity\n\n"
            f'Return JSON: {{"bottleneck_type": "...", "bottleneck_sub_type": "...", '
            f'"evidence": {{...}}, "recommendations": [...], "confidence": 0.0-1.0}}'
        )

        messages = self.context_manager.to_messages()
        messages.append({"role": "user", "content": user_msg})

        try:
            response = self._model_caller(messages)
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = _json.loads(response[start:end])
                bottleneck = parsed.get("bottleneck_type", "unknown")
                sub_type = parsed.get("bottleneck_sub_type")
                evidence = parsed.get("evidence", {})
                recommendations = parsed.get("recommendations", [])
                confidence = parsed.get("confidence", 0.5)

                suggested_fixes = self._generate_suggested_fixes(recommendations)
                code_quality = self._get_code_quality_review(target, raw_output)

                return SubAgentResult(
                    agent_role=self.role,
                    status=SubAgentStatus.SUCCESS,
                    data={
                        "bottleneck_type": bottleneck,
                        "bottleneck_sub_type": sub_type,
                        "parsed_metrics": ncu_metrics,
                        "measurements": ncu_metrics,
                        "evidence": evidence,
                        "recommendations": recommendations,
                        "suggested_fixes": suggested_fixes,
                        "confidence": confidence,
                        "confidence_reason": "LLM analysis with ncu data",
                        "cross_validation": None,
                        "analysis_method": "llm_ncu_roofline",
                        "code_quality": code_quality,
                    },
                    metadata={"target": target},
                )
        except Exception:
            pass

        roofline_result = self.analyze_roofline(ncu_metrics, target)
        code_quality = self._get_code_quality_review(target, raw_output)
        return SubAgentResult(
            agent_role=self.role,
            status=SubAgentStatus.SUCCESS,
            data={
                "bottleneck_type": roofline_result["bottleneck_type"],
                "bottleneck_sub_type": roofline_result.get("bottleneck_sub_type"),
                "parsed_metrics": ncu_metrics,
                "measurements": ncu_metrics,
                "evidence": roofline_result.get("evidence", {}),
                "recommendations": roofline_result.get("recommendations", []),
                "suggested_fixes": self._generate_suggested_fixes(
                    roofline_result.get("recommendations", [])
                ),
                "confidence": 0.6,
                "confidence_reason": "LLM parsing failed, fell back to rule-based Roofline",
                "cross_validation": None,
                "analysis_method": "fallback_rule_based",
                "code_quality": code_quality,
            },
            metadata={"target": target},
        )

    def _llm_analyze_raw(
        self,
        raw_output: str,
        target: str,
        target_spec: dict[str, Any],
    ) -> SubAgentResult:
        """Use LLM to analyze raw printf output when ncu is unavailable."""
        import json as _json

        user_msg = (
            f"Analyze this GPU benchmark output and identify the bottleneck.\n\n"
            f"Target: {target}\n"
            f"Target spec: {target_spec}\n\n"
            f"Raw output:\n{raw_output}\n\n"
            f"Classify the bottleneck as one of: compute_bound, memory_bound, "
            f"latency_bound, cache_capacity, balanced.\n"
            f"Provide a sub-type if possible (e.g., dram, l2, l1 for memory_bound; "
            f"tensor_core, fp32 for compute_bound).\n"
            f"Provide evidence and optimization recommendations.\n\n"
            f'Return JSON: {{"bottleneck_type": "...", "bottleneck_sub_type": "...", '
            f'"metrics": {{...}}, "evidence": {{...}}, "recommendations": [...]}}'
        )

        messages = self.context_manager.to_messages()
        messages.append({"role": "user", "content": user_msg})

        try:
            response = self._model_caller(messages)
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = _json.loads(response[start:end])
                metrics = parsed.get("metrics", {})
                bottleneck = parsed.get("bottleneck_type", "unknown")
                sub_type = parsed.get("bottleneck_sub_type")
                evidence = parsed.get("evidence", {})
                recommendations = parsed.get("recommendations", [])

                suggested_fixes = self._generate_suggested_fixes(recommendations)
                code_quality = self._get_code_quality_review(target, raw_output)

                return SubAgentResult(
                    agent_role=self.role,
                    status=SubAgentStatus.SUCCESS,
                    data={
                        "bottleneck_type": bottleneck,
                        "bottleneck_sub_type": sub_type,
                        "parsed_metrics": metrics,
                        "measurements": metrics,
                        "evidence": evidence,
                        "recommendations": recommendations,
                        "suggested_fixes": suggested_fixes,
                        "confidence": 0.4,
                        "confidence_reason": "LLM analysis without ncu — low confidence",
                        "cross_validation": None,
                        "analysis_method": "llm_printf",
                        "code_quality": code_quality,
                    },
                    metadata={"target": target},
                )
        except Exception:
            pass

        parsed_metrics = self._parse_output(raw_output)
        roofline_result = self.analyze_roofline(parsed_metrics, target)
        code_quality = self._get_code_quality_review(target, raw_output)

        return SubAgentResult(
            agent_role=self.role,
            status=SubAgentStatus.SUCCESS,
            data={
                "bottleneck_type": roofline_result["bottleneck_type"],
                "bottleneck_sub_type": roofline_result.get("bottleneck_sub_type"),
                "parsed_metrics": parsed_metrics,
                "measurements": parsed_metrics,
                "evidence": roofline_result.get("evidence", {}),
                "recommendations": roofline_result.get("recommendations", []),
                "suggested_fixes": self._generate_suggested_fixes(
                    roofline_result.get("recommendations", [])
                ),
                "confidence": 0.3,
                "confidence_reason": "LLM failed, rule-based analysis without ncu",
                "cross_validation": None,
                "analysis_method": "fallback_rule_based_printf",
                "code_quality": code_quality,
            },
            metadata={"target": target},
        )

    def _parse_output(self, raw: str) -> dict[str, Any]:
        """Parse raw output into structured metrics.

        Handles both ncu CSV output and simple numeric results.
        """
        metrics: dict[str, Any] = {}

        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("---"):
                continue

            match = re.match(r"^([^:]+):\s*(.+)$", line)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                try:
                    metrics[key] = float(value)
                except ValueError:
                    metrics[key] = value

            elif re.match(r"^\d+(\.\d+)?$", line):
                metrics.setdefault("result", []).append(float(line))

        if not metrics:
            metrics["raw"] = raw[:2000]

        return metrics

    def identify_bottleneck(self, metrics: dict[str, Any]) -> str:
        """Identify the bottleneck type from parsed metrics.

        Kept for backward compatibility. Use analyze_roofline() for
        full analysis with sub-type, evidence, and recommendations.
        """
        result = self.analyze_roofline(metrics, "unknown")
        return result["bottleneck_type"]

    def _assess_confidence(self, metrics: dict[str, Any]) -> float:
        """Assess confidence in the bottleneck identification.

        Kept for backward compatibility. Use _assess_confidence_detailed()
        for confidence with reasoning.
        """
        if not metrics:
            return 0.0
        numeric_count = sum(1 for v in metrics.values() if isinstance(v, (int, float)))
        return min(1.0, numeric_count / 5.0)

    def _get_code_quality_review(
        self, target: str, execution_output: str
    ) -> dict[str, Any] | None:
        """Attempt to retrieve source code and perform quality review.

        Tries to find the CUDA source from the sandbox or previous
        CodeGen output, then runs the 5-dimension quality review.
        """
        source_code = ""
        try:
            from src.infrastructure.file_ops import FileOperations
            file_ops = FileOperations("/workspace")
            cu_files = file_ops.list_files("/workspace", pattern="*.cu")
            if cu_files:
                latest_cu = sorted(cu_files)[-1]
                content = file_ops.read(latest_cu)
                if content:
                    source_code = content
        except Exception:
            pass

        if not source_code:
            try:
                sandbox_root = "/workspace/.sandbox"
                for root, dirs, files in os.walk(sandbox_root):
                    for f in files:
                        if f.endswith('.cu'):
                            path = os.path.join(root, f)
                            try:
                                with open(path, 'r') as fh:
                                    source_code = fh.read()
                                    break
                            except Exception:
                                continue
                    if source_code:
                        break
            except Exception:
                pass

        if not source_code:
            return None

        return self.review_code_quality(
            source_code=source_code,
            target_name=target,
            execution_output=execution_output,
        )

    def review_code_quality(
        self,
        source_code: str,
        target_name: str,
        execution_output: str = "",
    ) -> dict[str, Any]:
        """Comprehensive code quality review covering all 5 key dimensions.

        Dimensions:
        1. Code Accuracy: Correctness of measurement logic
        2. Execution Efficiency: Kernel optimization and throughput
        3. Resource Consumption: Register, shared memory, bandwidth usage
        4. Compatibility: Cross-architecture and anti-cheat resilience
        5. Maintainability: Code readability and modularity

        Returns:
            dict with keys: accuracy, efficiency, resource, compatibility,
            maintainability, overall_score, detailed_feedback, suggested_fixes
        """
        accuracy = self._review_accuracy(source_code, target_name, execution_output)
        efficiency = self._review_efficiency(source_code, target_name)
        resource = self._review_resource_consumption(source_code, target_name)
        compatibility = self._review_compatibility(source_code, target_name)
        maintainability = self._review_maintainability(source_code, target_name)

        scores = [accuracy["score"], efficiency["score"], resource["score"],
                  compatibility["score"], maintainability["score"]]
        overall = round(sum(scores) / len(scores), 2) if scores else 0.0

        all_fixes = []
        for dim in [accuracy, efficiency, resource, compatibility, maintainability]:
            all_fixes.extend(dim.get("fixes", []))

        return {
            "accuracy": accuracy,
            "efficiency": efficiency,
            "resource": resource,
            "compatibility": compatibility,
            "maintainability": maintainability,
            "overall_score": overall,
            "detailed_feedback": {
                "accuracy_feedback": accuracy.get("feedback", ""),
                "efficiency_feedback": efficiency.get("feedback", ""),
                "resource_feedback": resource.get("feedback", ""),
                "compatibility_feedback": compatibility.get("feedback", ""),
                "maintainability_feedback": maintainability.get("feedback", ""),
            },
            "suggested_fixes": all_fixes[:8],
        }

    def _review_accuracy(
        self, source: str, target: str, exec_output: str
    ) -> dict[str, Any]:
        """Review code accuracy: correct measurement logic, no logical errors."""
        issues = []
        fixes = []
        score = 1.0

        if "cudaGetDeviceProperties" in source:
            issues.append("Uses cudaGetDeviceProperties which may be intercepted in evaluation")
            fixes.append(
                "REPLACE cudaGetDeviceProperties with empirical measurement. "
                "Use clock64() + cudaEventElapsedTime for frequency, "
                "pointer-chasing for latency, occupancy API for SM count."
            )
            score -= 0.3

        if target and "latency" in target.lower():
            if "clock64()" not in source and "clock()" not in source:
                issues.append(f"Latency target '{target}' but no clock64() timing found")
                fixes.append("Add clock64() based timing for latency measurement")
                score -= 0.3

        if target and "bandwidth" in target.lower():
            if "cudaEventRecord" not in source and "cudaEventElapsedTime" not in source:
                issues.append(f"Bandwidth target '{target}' but no cudaEvent timing found")
                fixes.append("Use cudaEventRecord/cudaEventElapsedTime for bandwidth measurement")
                score -= 0.3

        if target and "clock" in target.lower() and "mhz" in target.lower():
            if "clock64()" not in source:
                issues.append("Clock frequency target requires clock64() for cycle counting")
                fixes.append("Use dual-timing: clock64() for cycles, cudaEventElapsedTime for wall time")
                score -= 0.3

        if "printf" in source:
            printf_matches = re.findall(r'printf\s*\(\s*"([^"]*)"', source)
            has_target_printf = any(target in p for p in printf_matches) if target else False
            if target and not has_target_printf:
                issues.append(f"No printf output matching target '{target}'")
                fixes.append(f"Ensure printf outputs '{target}: <value>' format for result extraction")
                score -= 0.2

        if exec_output and "0.0" in exec_output and target:
            lines = [l.strip() for l in exec_output.splitlines() if l.strip()]
            for line in lines:
                m = re.match(r'^([a-zA-Z_][\w.]*)\s*:\s*([\d.eE+-]+)', line)
                if m and m.group(1) == target and float(m.group(2)) == 0.0:
                    if "pct_of_peak" in target:
                        issues.append(
                            f"Measurement for '{target}' returned 0.0% — "
                            f"kernel failed to achieve measurable throughput. "
                            f"The FMA loop may be optimized away or peak calculation is wrong."
                        )
                        fixes.append(
                            "Ensure volatile double* sink prevents dead-code elimination, "
                            "use #pragma unroll 1, add warmup kernel, "
                            "and compute actual pct = achieved/peak * 100 (NOT placeholder 0.0)"
                        )
                    else:
                        issues.append(f"Measurement for '{target}' returned 0.0 — likely dead-code elimination")
                        fixes.append(
                            "Add 'volatile' qualifier on output pointer and "
                            "'asm volatile(\"\" ::: \"memory\")' barrier after compute loop"
                        )
                    score -= 0.3
                    break

        if "volatile" not in source and ("clock64()" in source or "cudaEvent" in source):
            has_asm_barrier = 'asm volatile' in source
            if not has_asm_barrier:
                issues.append("Missing volatile/asm barrier — compiler may optimize away measurement")
                fixes.append("Use 'volatile double* sink' or add asm volatile memory barrier")
                score -= 0.15

        score = max(0.0, score)
        feedback = "; ".join(issues) if issues else "Measurement logic appears correct"
        return {"score": score, "issues": issues, "fixes": fixes, "feedback": feedback}

    def _review_efficiency(
        self, source: str, target: str
    ) -> dict[str, Any]:
        """Review execution efficiency: kernel optimization, throughput, occupancy."""
        issues = []
        fixes = []
        score = 1.0

        is_compute_target = "sm__throughput" in target or "compute_memory_throughput" in target
        is_memory_target = any(kw in target for kw in ("bandwidth", "throughput", "bytes"))

        if is_compute_target:
            if "double" not in source and "float" in source:
                issues.append("Using float for sm__throughput — double-precision FMA achieves higher SM utilization")
                fixes.append("Change float to double in the compute loop for higher FMA throughput")
                score -= 0.2

            if "volatile" not in source:
                issues.append("Missing volatile qualifier — compiler may eliminate compute (dead-code)")
                fixes.append("Use 'volatile double* sink' or add asm volatile memory barrier after loop")
                score -= 0.2

            in_kernel = False
            has_global_read_in_loop = False
            lines = source.split('\n')
            loop_depth = 0
            for line in lines:
                s = line.strip()
                if '__global__' in s or '__device__' in s:
                    in_kernel = True
                    continue
                if in_kernel:
                    if '{' in s:
                        loop_depth += 1
                    if '}' in s:
                        loop_depth -= 1
                        if loop_depth <= 0:
                            in_kernel = False
                    if 'for' in s or 'while' in s:
                        next_lines = s
                        if any(kw in next_lines for kw in ('input[', 'data[', 'array[', 'global')):
                            has_global_read_in_loop = True

            if has_global_read_in_loop:
                issues.append("Global memory read inside timed compute loop — makes kernel memory-bound")
                fixes.append("Move all memory reads BEFORE the loop, use register variables only inside loop")
                score -= 0.2

            if "#pragma unroll" not in source:
                issues.append("No #pragma unroll — compiler may unpredictably unroll or optimize away iterations")
                fixes.append("Add '#pragma unroll 1' before the FMA loop for consistent behavior")
                score -= 0.1

        if is_memory_target:
            if "volatile" not in source:
                issues.append("Missing volatile on output pointer — compiler eliminates writes")
                fixes.append("Use 'volatile float* output' for the output buffer")
                score -= 0.2

        warmup_patterns = ['warmup', 'WARMUP', 'warm_up', 'dummy']
        has_warmup = any(p in source.lower() for p in warmup_patterns)
        if not has_warmup and ("cudaEventRecord" in source or "clock64()" in source):
            issues.append("No warmup kernel — first run measures lower throughput due to GPU power ramping")
            fixes.append("Run kernel once before starting cudaEventRecord timing")
            score -= 0.1

        if "cudaEventRecord" in source:
            if "cudaEventCreate" not in source:
                issues.append("cudaEventRecord used without cudaEventCreate — will crash")
                fixes.append("Add cudaEventCreate(&start); cudaEventCreate(&stop); before timing")
                score -= 0.2

        score = max(0.0, score)
        feedback = "; ".join(issues) if issues else "Kernel efficiency looks adequate"
        return {"score": score, "issues": issues, "fixes": fixes, "feedback": feedback}

    def _review_resource_consumption(
        self, source: str, target: str
    ) -> dict[str, Any]:
        """Review resource consumption: registers, shared memory, bandwidth."""
        issues = []
        fixes = []
        score = 1.0

        if "__shared__" in source:
            shmem_matches = re.findall(r'__shared__\s+\w+\s+(\w+)\s*\[', source)
            if shmem_matches and "bank_conflict" not in target:
                has_padding = any("padding" in source.lower() or "+ 1]" in source or "+ 2]" in source
                                 for _ in [1])
                if not has_padding and len(shmem_matches) > 0:
                    issues.append("Shared memory used without padding — potential bank conflicts")
                    fixes.append("Add padding to shared memory arrays (e.g., [256+1]) to avoid bank conflicts")
                    score -= 0.1

        if "__launch_bounds__" not in source and "__global__" in source:
            issues.append("No __launch_bounds__ specified — register allocation left to compiler default")
            fixes.append("Add __launch_bounds__(256, 2) to control register usage and occupancy")
            score -= 0.05

        if "register" in source.lower() or "local memory" in source.lower():
            pass

        if "memcpy" in source and "cudaMemcpy" not in source:
            issues.append("Using host memcpy instead of cudaMemcpy — may cause unnecessary D2H transfers")
            fixes.append("Use cudaMemcpy for device-to-host transfers")
            score -= 0.1

        if target and "bandwidth" in target.lower():
            block_matches = re.findall(r'<<<\s*(\d+)\s*,\s*(\d+)\s*>>>', source)
            if block_matches:
                blocks = int(block_matches[0][0])
                threads = int(block_matches[0][1])
                if blocks < 100 or threads < 128:
                    issues.append(
                        f"Low parallelism for bandwidth measurement: {blocks} blocks x {threads} threads"
                    )
                    fixes.append("Increase to at least 65535 blocks x 256 threads for full DRAM saturation")
                    score -= 0.2

        score = max(0.0, score)
        feedback = "; ".join(issues) if issues else "Resource usage appears reasonable"
        return {"score": score, "issues": issues, "fixes": fixes, "feedback": feedback}

    def _review_compatibility(
        self, source: str, target: str
    ) -> dict[str, Any]:
        """Review compatibility: cross-architecture, anti-cheat resilience, API independence."""
        issues = []
        fixes = []
        score = 1.0

        if "cudaGetDeviceProperties" in source:
            issues.append("DEPENDS on cudaGetDeviceProperties — may be intercepted/virtualized in evaluation")
            fixes.append(
                "Replace ALL cudaGetDeviceProperties calls with empirical measurement. "
                "Use clock64()+cudaEventElapsedTime for frequency, "
                "pointer-chasing for latency, occupancy API for SM count."
            )
            score -= 0.3

        if "cudaDeviceGetAttribute" in source:
            issues.append("Uses cudaDeviceGetAttribute — may return virtualized values in evaluation")
            fixes.append(
                "Cross-validate cudaDeviceGetAttribute results with empirical measurement. "
                "Do NOT rely on it as sole source for any target metric."
            )
            score -= 0.15

        sm_version_matches = re.findall(r'sm_(\d+)', source)
        if sm_version_matches:
            versions = set(sm_version_matches)
            if len(versions) == 1:
                issues.append(f"Hard-coded for sm_{list(versions)[0]} — may not work on other architectures")
                fixes.append("Use runtime architecture detection or make code architecture-agnostic")
                score -= 0.1

        if "cudaDevAttrMultiProcessorCount" in source and target != "launch__sm_count":
            issues.append("Measures SM count but target is not SM count — wrong measurement approach")
            fixes.append(f"Write code that specifically measures '{target}', not SM count")
            score -= 0.3

        if target and "clock" in target.lower() and "mhz" in target.lower():
            if "nvidia-smi" in source:
                issues.append("Uses nvidia-smi for clock — may report locked frequency, not actual")
                fixes.append("Use clock64()+cudaEventElapsedTime dual-timing to measure actual running frequency")
                score -= 0.2

        env_detection = self._detect_environment_interference()
        if env_detection.get("frequency_locked"):
            issues.append(
                f"GPU frequency appears LOCKED at {env_detection.get('reported_clock_mhz', '?')} MHz "
                f"— static lookup tables will give wrong results"
            )
            fixes.append("Use empirical measurement (clock64()+cudaEventElapsedTime) to detect actual frequency")
            score -= 0.1

        if env_detection.get("sm_masked"):
            issues.append(
                f"SM masking detected: API reports {env_detection.get('api_sm_count', '?')} SMs, "
                f"but actual active count may differ"
            )
            fixes.append("Use occupancy API + block ID sweep to detect actual active SM count")
            score -= 0.1

        score = max(0.0, score)
        feedback = "; ".join(issues) if issues else "Code appears compatible and anti-cheat resilient"
        return {"score": score, "issues": issues, "fixes": fixes, "feedback": feedback}

    def _review_maintainability(
        self, source: str, target: str
    ) -> dict[str, Any]:
        """Review maintainability: readability, modularity, error handling."""
        issues = []
        fixes = []
        score = 1.0

        lines = source.split('\n')
        total_lines = len(lines)
        if total_lines > 300:
            issues.append(f"Kernel code is very long ({total_lines} lines) — consider splitting into functions")
            fixes.append("Extract helper functions for initialization, measurement, and output")
            score -= 0.1

        if "__global__" in source:
            kernel_lines = 0
            in_kernel = False
            for line in lines:
                if '__global__' in line:
                    in_kernel = True
                if in_kernel:
                    kernel_lines += 1
                    if '}' in line and kernel_lines > 5:
                        brace_count = sum(1 for c in line if c == '}') - sum(1 for c in line if c == '{')
                        if brace_count > 0:
                            in_kernel = False
            if kernel_lines > 150:
                issues.append(f"Single kernel is {kernel_lines} lines — hard to debug and maintain")
                fixes.append("Break kernel into device helper functions for clarity")
                score -= 0.1

        has_error_check = any("cudaGetLastError" in l or "cudaPeekAtLastError" in l for l in lines)
        if "cudaMalloc" in source and not has_error_check:
            issues.append("CUDA operations without error checking — silent failures possible")
            fixes.append("Add cudaGetLastError() checks after kernel launches and cudaMalloc calls")
            score -= 0.1

        magic_numbers = re.findall(r'\b(\d{4,})\b', source)
        defined_constants = re.findall(r'#define\s+\w+\s+\d+', source)
        if len(magic_numbers) > 3 and len(defined_constants) < 2:
            issues.append(f"{len(magic_numbers)} magic numbers without #define constants")
            fixes.append("Define constants with #define or constexpr for key parameters")
            score -= 0.05

        if "fprintf(stderr" not in source and "printf" in source:
            has_diagnostic = any("error" in l.lower() or "fail" in l.lower() for l in lines if "printf" in l)
            if not has_diagnostic:
                issues.append("No diagnostic output for error conditions")
                fixes.append("Add fprintf(stderr, ...) for error conditions to aid debugging")
                score -= 0.05

        score = max(0.0, score)
        feedback = "; ".join(issues) if issues else "Code maintainability is acceptable"
        return {"score": score, "issues": issues, "fixes": fixes, "feedback": feedback}

    def _detect_environment_interference(self) -> dict[str, Any]:
        """Detect anti-cheat environment conditions: frequency lock, SM masking, API interception.

        Returns dict with detection results for each interference type.
        """
        result: dict[str, Any] = {
            "frequency_locked": False,
            "sm_masked": False,
            "api_interception": False,
            "reported_clock_mhz": None,
            "api_sm_count": None,
        }

        try:
            nvidia_smi = subprocess.run(
                ["nvidia-smi", "--query-gpu=clocks.max.sm,clocks.sm,multiprocessor_count",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if nvidia_smi.returncode == 0 and nvidia_smi.stdout.strip():
                parts = nvidia_smi.stdout.strip().split(",")
                if len(parts) >= 3:
                    try:
                        max_clock = float(parts[0].strip())
                        current_clock = float(parts[1].strip())
                        sm_count_api = int(parts[2].strip())
                        result["reported_clock_mhz"] = current_clock
                        result["api_sm_count"] = sm_count_api

                        if max_clock > 0 and current_clock > 0:
                            ratio = current_clock / max_clock
                            if ratio < 0.7:
                                result["frequency_locked"] = True
                                logger.warning(
                                    f"[env-detect] Frequency lock detected: "
                                    f"current={current_clock}MHz, max={max_clock}MHz, ratio={ratio:.2f}"
                                )
                    except (ValueError, IndexError):
                        pass
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        try:
            cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if cuda_env and cuda_env != "0" and "," not in cuda_env:
                try:
                    import torch
                    if torch.cuda.device_count() < 1:
                        result["sm_masked"] = True
                except ImportError:
                    pass
        except Exception:
            pass

        return result
