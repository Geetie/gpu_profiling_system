"""Metric Analysis Agent — GPU performance profiling and bottleneck identification.

Performs active tool-based analysis:
1. Extracts binary paths from CodeGen's output
2. Profiles binaries with Nsight Compute (ncu) when available
3. Performs Roofline model analysis on ncu metrics
4. Falls back to rule-based analysis on printf output when ncu unavailable
5. Generates targeted optimization recommendations
6. Provides cross-validation between CodeGen measurements and ncu data
"""
from __future__ import annotations

import re
from typing import Any

from src.application.context import ContextManager, Role
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

        Returns:
            Tuple of (list of ncu result dicts, whether any profiling succeeded).
        """
        metrics = self._select_metrics_for_target(target)
        ncu_results: list[dict[str, Any]] = []
        any_success = False

        for binary_path in binary_paths:
            try:
                result = self._call_tool("run_ncu", {
                    "executable": binary_path,
                    "metrics": metrics,
                })
                if isinstance(result, dict):
                    parsed = result.get("parsed_metrics", {})
                    error = parsed.get("error", "") if isinstance(parsed, dict) else ""
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
        """Call a registered tool through the infrastructure layer."""
        from src.infrastructure.tools.run_ncu import run_ncu_handler

        if tool_name == "run_ncu":
            return run_ncu_handler(args, sandbox=self._sandbox)

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

        return SubAgentResult(
            agent_role=self.role,
            status=SubAgentStatus.SUCCESS,
            data={
                "bottleneck_type": roofline_result["bottleneck_type"],
                "bottleneck_sub_type": roofline_result.get("bottleneck_sub_type"),
                "parsed_metrics": merged_metrics,
                "evidence": evidence,
                "recommendations": recommendations,
                "suggested_fixes": suggested_fixes,
                "confidence": confidence,
                "confidence_reason": confidence_reason,
                "cross_validation": cross_validation,
                "analysis_method": "ncu_roofline",
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

        return SubAgentResult(
            agent_role=self.role,
            status=SubAgentStatus.SUCCESS,
            data={
                "bottleneck_type": roofline_result["bottleneck_type"],
                "bottleneck_sub_type": roofline_result.get("bottleneck_sub_type"),
                "parsed_metrics": parsed_metrics,
                "evidence": evidence,
                "recommendations": recommendations,
                "suggested_fixes": suggested_fixes,
                "confidence": confidence,
                "confidence_reason": confidence_reason,
                "cross_validation": None,
                "analysis_method": "printf_pattern_matching",
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

                return SubAgentResult(
                    agent_role=self.role,
                    status=SubAgentStatus.SUCCESS,
                    data={
                        "bottleneck_type": bottleneck,
                        "bottleneck_sub_type": sub_type,
                        "parsed_metrics": ncu_metrics,
                        "evidence": evidence,
                        "recommendations": recommendations,
                        "suggested_fixes": suggested_fixes,
                        "confidence": confidence,
                        "confidence_reason": "LLM analysis with ncu data",
                        "cross_validation": None,
                        "analysis_method": "llm_ncu_roofline",
                    },
                    metadata={"target": target},
                )
        except Exception:
            pass

        roofline_result = self.analyze_roofline(ncu_metrics, target)
        return SubAgentResult(
            agent_role=self.role,
            status=SubAgentStatus.SUCCESS,
            data={
                "bottleneck_type": roofline_result["bottleneck_type"],
                "bottleneck_sub_type": roofline_result.get("bottleneck_sub_type"),
                "parsed_metrics": ncu_metrics,
                "evidence": roofline_result.get("evidence", {}),
                "recommendations": roofline_result.get("recommendations", []),
                "suggested_fixes": self._generate_suggested_fixes(
                    roofline_result.get("recommendations", [])
                ),
                "confidence": 0.6,
                "confidence_reason": "LLM parsing failed, fell back to rule-based Roofline",
                "cross_validation": None,
                "analysis_method": "fallback_rule_based",
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

                return SubAgentResult(
                    agent_role=self.role,
                    status=SubAgentStatus.SUCCESS,
                    data={
                        "bottleneck_type": bottleneck,
                        "bottleneck_sub_type": sub_type,
                        "parsed_metrics": metrics,
                        "evidence": evidence,
                        "recommendations": recommendations,
                        "suggested_fixes": suggested_fixes,
                        "confidence": 0.4,
                        "confidence_reason": "LLM analysis without ncu — low confidence",
                        "cross_validation": None,
                        "analysis_method": "llm_printf",
                    },
                    metadata={"target": target},
                )
        except Exception:
            pass

        parsed_metrics = self._parse_output(raw_output)
        roofline_result = self.analyze_roofline(parsed_metrics, target)

        return SubAgentResult(
            agent_role=self.role,
            status=SubAgentStatus.SUCCESS,
            data={
                "bottleneck_type": roofline_result["bottleneck_type"],
                "bottleneck_sub_type": roofline_result.get("bottleneck_sub_type"),
                "parsed_metrics": parsed_metrics,
                "evidence": roofline_result.get("evidence", {}),
                "recommendations": roofline_result.get("recommendations", []),
                "suggested_fixes": self._generate_suggested_fixes(
                    roofline_result.get("recommendations", [])
                ),
                "confidence": 0.3,
                "confidence_reason": "LLM failed, rule-based analysis without ncu",
                "cross_validation": None,
                "analysis_method": "fallback_rule_based_printf",
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
