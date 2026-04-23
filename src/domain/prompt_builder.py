"""Prompt construction for pipeline stages.

Strategy pattern: each stage type gets its own prompt-building logic,
keeping prompt engineering separate from orchestration.
"""
from __future__ import annotations

from typing import Any


class StagePromptBuilder:
    """Builds system and task prompts for each pipeline stage.

    Separates prompt engineering from pipeline orchestration.
    Each method returns the exact prompt string for a given stage.
    """

    def build_system_prompt(self, stage: "PipelineStage") -> str:
        """Return the system prompt for a pipeline stage."""
        builders = {
            "plan": self._plan_system,
            "code_gen": self._codegen_system,
            "metric_analysis": self._metric_system,
            "verification": self._verification_system,
        }
        stage_key = stage.value if hasattr(stage, "value") else str(stage)
        builder = builders.get(stage_key)
        if builder is None:
            raise ValueError(f"Unknown pipeline stage: {stage}")
        return builder()

    def build_task_prompt(
        self,
        stage: "PipelineStage",
        target_spec: dict[str, Any],
        prev_result: Any | None = None,
    ) -> str:
        """Return the task prompt for a pipeline stage."""
        builders = {
            "plan": self._plan_task,
            "code_gen": self._codegen_task,
            "metric_analysis": self._metric_task,
            "verification": self._verification_task,
        }
        stage_key = stage.value if hasattr(stage, "value") else str(stage)
        builder = builders.get(stage_key)
        if builder is None:
            raise ValueError(f"Unknown pipeline stage: {stage}")
        return builder(target_spec, prev_result)

    @staticmethod
    def _plan_system() -> str:
        return (
            "You are a GPU profiling expert. Your task is to create a detailed "
            "plan for measuring specific GPU hardware characteristics.\n\n"
            "You do NOT have any tools available. Your ONLY output should be a JSON array "
            "of task objects.\n\n"
            "DO NOT attempt to call any tools (write_file, compile_cuda, etc.) — "
            "you have NO tools. Just output your plan as JSON text.\n\n"
            "Your plan should include:\n"
            "1. Which GPU micro-architectural feature to measure\n"
            "2. What measurement technique to use (pointer-chasing, sweep, etc.)\n"
            "3. What anti-cheat strategies are needed\n"
            "4. Expected result ranges for validation\n\n"
            "OUTPUT FORMAT: A JSON array of task objects, each with:\n"
            '  "target": the measurement target name\n'
            '  "category": one of latency_measurement, capacity_measurement, '
            'clock_measurement, bandwidth_measurement, or unknown\n'
            '  "method": detailed description of the measurement approach\n'
        )

    @staticmethod
    def _codegen_system() -> str:
        return (
            "You are a CUDA kernel developer. Your task is to write a complete "
            "CUDA micro-benchmark that measures a specific GPU hardware characteristic.\n\n"
            "You MUST use the compile_cuda tool to compile your code, then "
            "execute_binary to run it. A response without tool calls is a failure.\n\n"
            "Rules:\n"
            "1. Write a COMPLETE .cu file with main() function\n"
            "2. Use clock64() for cycle-accurate timing\n"
            "3. Use cudaEventElapsedTime for wall-clock timing\n"
            "4. Output must be parseable: printf(\"key: value\\n\")\n"
            "5. Run at least 3 trials and report MEDIAN for latency, MAX for bandwidth\n"
            "6. Do NOT use cudaGetDeviceProperties as the sole measurement method\n"
        )

    @staticmethod
    def _metric_system() -> str:
        return (
            "You are a GPU performance analyst. Your task is to analyze the "
            "results of a CUDA micro-benchmark and extract meaningful metrics.\n\n"
            "You should:\n"
            "1. Parse the benchmark output for key-value pairs\n"
            "2. Validate results against expected ranges\n"
            "3. Identify any anomalies or measurement errors\n"
            "4. Provide a summary of findings\n"
        )

    @staticmethod
    def _verification_system() -> str:
        return (
            "You are a GPU benchmark verification expert. Your task is to "
            "independently verify the results of a GPU micro-benchmark.\n\n"
            "You must evaluate:\n"
            "1. Whether the measurement methodology is sound\n"
            "2. Whether results fall within expected ranges\n"
            "3. Whether anti-cheat measures were properly implemented\n"
            "4. Whether the benchmark actually measures what it claims\n\n"
            "IMPORTANT: You are the INDEPENDENT verifier. You must NOT have "
            "access to the code generation context. You only see the final "
            "results and must judge them on their own merits.\n\n"
            "End your analysis with a clear verdict: ACCEPT or REJECT.\n"
        )

    @staticmethod
    def _plan_task(target_spec: dict[str, Any], prev_result: Any | None) -> str:
        targets = target_spec.get("targets", [])
        target = target_spec.get("target", targets[0] if targets else "unknown")
        return (
            f"Create a detailed plan for measuring GPU characteristic: {target}\n\n"
            f"Target specification: {target_spec}\n\n"
            "Use the available tools to research this target. "
            "Your plan will be used by the CodeGen agent to write the actual benchmark."
        )

    @staticmethod
    def _codegen_task(target_spec: dict[str, Any], prev_result: Any | None) -> str:
        targets = target_spec.get("targets", [])
        target = target_spec.get("target", targets[0] if targets else "unknown")

        parts = [
            f"Write CUDA micro-benchmarks for: {target}",
        ]

        # 1. AUTO-INVOKE METRIC DOCUMENTATION (primary guidance)
        try:
            from src.domain.metric_reference import format_metric_context
            metric_context = format_metric_context(target)
            if metric_context:
                parts.append(f"\n{'=' * 60}")
                parts.append(f"📊 NCU METRIC DOCUMENTATION (auto-loaded for target: {target})")
                parts.append(f"{'=' * 60}")
                parts.append(metric_context)
        except (ImportError, KeyError):
            pass

        # 2. TARGET-SPECIFIC DESIGN PRINCIPLE (filtered to current target only)
        from src.domain.design_principles import get_design_principle
        principle = get_design_principle(target)
        if principle:
            parts.append(f"\n{'=' * 60}")
            parts.append(f"🎯 DESIGN PRINCIPLE FOR TARGET: {target}")
            parts.append(f"{'=' * 60}")
            parts.append(principle[:1500])

        # 3. PATTERN GUIDANCE (if available for this target)
        try:
            from src.infrastructure.probing.cuda_templates import get_pattern
            pattern = get_pattern(target)
            if pattern:
                parts.append(f"\n{'=' * 60}")
                parts.append(f"📝 CODE PATTERN REFERENCE FOR TARGET: {target}")
                parts.append(f"{'=' * 60}")
                parts.append("KEY API PATTERN (use this exact approach):")
                parts.append(f"```cuda\n{pattern.key_api_pattern}```\n")
                parts.append("CODE SKELETON (fill in the TODOs):")
                parts.append(f"```cuda\n{pattern.measurement_skeleton}```\n")
                parts.append(f"EXPECTED OUTPUT FORMAT: {pattern.expected_output_format}\n")
                parts.append(f"METHODOLOGY: {pattern.measurement_methodology}\n")
                if pattern.critical_notes:
                    parts.append("CRITICAL NOTES:")
                    for note in pattern.critical_notes:
                        parts.append(f"  • {note}")
                parts.append(
                    f"\n⚠️ You MUST write the COMPLETE CUDA source code.\n"
                    f"The skeleton above has TODOs — you must fill them in with working code.\n"
                    f"Do NOT submit code with TODOs still present!"
                )
        except ImportError:
            pass

        # 4. TARGET SPEC (concise)
        parts.append(f"\nTarget specification: {target_spec}")

        # 5. MULTI-TARGET WORKFLOW (only if multiple targets)
        if targets and len(targets) > 1:
            target_list_str = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(targets))
            parts.append(
                f"\n{'=' * 60}"
                f"📊 TARGET ASSIGNMENT (MEASURE ALL OF THESE IN ORDER)"
                f"{'=' * 60}\n\n"
                f"Targets to measure ({len(targets)} total):\n{target_list_str}\n\n"
                f"⚠️ You MUST measure ALL targets above in SEQUENTIAL ORDER.\n"
                f"Start with Target #1: **{targets[0]}**\n"
                f"After completing each target, move to the next number.\n"
                f"Do NOT skip any target. Do NOT reorder. Do NOT parallelize.\n\n"
                f"Current status:\n"
                f"  ☐ Target 1: {targets[0]} ← *** START HERE ***\n"
            )
            for i, t in enumerate(targets[1:], start=2):
                parts[-1] += f"  ☐ Target {i}: {t}\n"

            # Only include target type guidance for CURRENT target (filter out irrelevant)
            target_type_guidance = _get_target_type_guidance(target)
            parts[-1] += (
                f"\nWORKFLOW (repeat for EACH target):\n"
                f"  1. Write CUDA code SPECIFICALLY for the CURRENT target (unique kernel)\n"
                f"  2. compile_cuda(source=\"...\", flags=[\"-O3\"], target=\"CURRENT_TARGET_NAME\")\n"
                f"  3. execute_binary(binary_path=\"<exact path from compile_cuda>\")\n"
                f"  4. Record the measured value from stdout\n"
                f"  5. Wait for SYSTEM message → then move to NEXT target\n\n"
                f"⚠️ CRITICAL RULES:\n"
                f"  • Each target needs DIFFERENT CUDA code! Use the guidance below:\n"
                f"    — CURRENT TARGET ({target}): {target_type_guidance}\n\n"
                f"  • compile_cuda OVERWRITES the previous binary each time.\n"
                f"    So you MUST execute_binary IMMEDIATELY after each compile_cuda.\n"
                f"    Do NOT compile all targets first — compile+execute one at a time.\n\n"
                f"  • MAX 3 compilation attempts per target.\n"
                f"    After 3 failures, MOVE ON to next target (system will force switch).\n\n"
                f"  • Do NOT skip any target — pipeline WILL FAIL if any is missing!\n\n"
                f"  • The printf output MUST match the target name EXACTLY:\n"
                f"    printf(\"EXACT_TARGET_NAME: value\\n\")\n"
            )

        # 6. PREVIOUS STAGE RESULT (filtered to current target only)
        if prev_result is not None:
            prev_data = prev_result.data if hasattr(prev_result, "data") else (prev_result.get("data", {}) if isinstance(prev_result, dict) else {})
            if prev_data:
                plan_output = prev_data.get("final_output", "")
                tasks = prev_data.get("tasks", [])

                # Extract method for CURRENT target only
                method = ""
                for task in tasks:
                    if isinstance(task, dict) and task.get("target") == target:
                        method = task.get("method", "")
                        break

                if method:
                    parts.append(
                        f"\n{'=' * 60}"
                        f"📋 MEASUREMENT METHODOLOGY (from Planner, for target: {target})"
                        f"{'=' * 60}\n{method}"
                    )

                category = ""
                for task in tasks:
                    if isinstance(task, dict) and task.get("target") == target:
                        category = task.get("category", "")
                        break

                if category:
                    parts.append(f"\nTask category: {category}")

            if hasattr(prev_result, "error") and prev_result.error:
                parts.append(f"\n\n--- Plan from previous stage ---\n{prev_result.error}")
            elif isinstance(prev_result, dict) and prev_result.get("error"):
                parts.append(f"\n\n--- Plan from previous stage ---\n{prev_result['error']}")

        # 7. FINAL MANDATORY INSTRUCTION - STRONG TARGET AWARENESS
        parts.append(
            f"\n\n⚠️ URGENT: You are measuring **{target}**. "
            f"Do NOT generate code for any other target. "
            f"Do NOT use cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount). "
            f"Use the measurement approach described above for {target}."
        )

        return "\n".join(parts)

    @staticmethod
    def _metric_task(target_spec: dict[str, Any], prev_result: Any | None) -> str:
        from src.domain.design_principles import get_design_principle

        targets = target_spec.get("targets", [])
        target = target_spec.get("target", targets[0] if targets else "unknown")
        principle = get_design_principle(target)

        parts = [
            f"Analyze the benchmark results for: {target}",
            f"\nTarget specification: {target_spec}",
        ]

        if prev_result is not None:
            data = prev_result.data if hasattr(prev_result, "data") else (prev_result.get("data", {}) if isinstance(prev_result, dict) else {})
            if data:
                if "final_output" in data:
                    parts.append(f"\n\nBenchmark output:\n{data['final_output']}")
                if "tool_results" in data:
                    parts.append(f"\nTool results:\n{str(data['tool_results'])}")
                if "raw_output" in data:
                    parts.append(f"\nRaw benchmark output:\n{data['raw_output']}")
                if "binary_path" in data:
                    parts.append(f"\nCompiled binary path: {data['binary_path']}")
                if "measurements" in data:
                    parts.append(f"\nCodeGen measurements:\n{data['measurements']}")
                if "detected_arch" in data:
                    parts.append(f"\nDetected GPU architecture: {data['detected_arch']}")
                if "analysis_method" in data:
                    parts.append(f"\nCodeGen methodology: {data['analysis_method'][:1000]}")

        # NCU profiling instructions for current target only
        if targets and len(targets) > 1:
            target_metric_map = {
                "launch__sm_count": ["launch__sm_count", "sm__throughput.avg.pct_of_peak_sustained_elapsed"],
                "dram__bytes_read.sum.per_second": ["dram__throughput.avg.pct_of_peak_sustained_elapsed", "dram__bytes_read.sum.per_second", "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"],
                "dram__bytes_write.sum.per_second": ["dram__throughput.avg.pct_of_peak_sustained_elapsed", "dram__bytes_write.sum.per_second", "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"],
                "device__attribute_max_gpu_frequency_khz": ["sm__cycles_active.avg", "device__attribute_max_gpu_frequency_khz"],
                "device__attribute_max_mem_frequency_khz": ["dram__throughput.avg.pct_of_peak_sustained_elapsed", "device__attribute_max_mem_frequency_khz"],
                "device__attribute_fb_bus_width": ["dram__throughput.avg.pct_of_peak_sustained_elapsed", "device__attribute_fb_bus_width"],
                "sm__throughput.avg.pct_of_peak_sustained_elapsed": ["sm__throughput.avg.pct_of_peak_sustained_elapsed", "sm__cycles_active.avg", "sm__warps_active.avg.pct_of_peak_sustained_elapsed"],
                "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": ["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "dram__throughput.avg.pct_of_peak_sustained_elapsed", "sm__throughput.avg.pct_of_peak_sustained_elapsed"],
            }

            metrics_for_target = target_metric_map.get(target, ["sm__throughput.avg.pct_of_peak_sustained_elapsed", "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"])
            metrics_str = ", ".join(metrics_for_target)

            parts.append(
                f"\n\n{'=' * 60}"
                f"🎯 NCU PROFILING INSTRUCTIONS"
                f"{'=' * 60}\n\n"
                f"Current target: {target}\n"
                f"Use run_ncu with these metrics: [{metrics_str}]\n\n"
                f"⚠️ CRITICAL: Profile the binary for the CURRENT target ONLY.\n"
                f"Do NOT profile binaries from other targets — they are irrelevant.\n\n"
                f"WORKFLOW:\n"
                f"1. Use run_ncu(executable=\"<binary_path>\", metrics=[...]) to profile\n"
                f"2. Parse ncu output for actual metric values\n"
                f"3. Perform Roofline analysis\n"
                f"4. Classify bottleneck: compute_bound, memory_bound, latency_bound, or cache_capacity\n"
                f"5. Report confidence level with reasoning\n"
                f"6. Provide optimization recommendations\n\n"
                f"⚠️ If ncu returns ERR_NVGPUCTRPERM (permission denied):\n"
                f"  - Stop calling run_ncu immediately\n"
                f"  - Analyze the CodeGen measurements directly\n"
                f"  - Classify bottleneck based on measured values and design principle\n"
            )

        parts.append(
            "\n\nIMPORTANT: Perform Roofline analysis on the data above.\n"
            "1. If ncu metrics are available, compare compute vs memory utilization\n"
            "2. Identify bottleneck type AND sub-type (e.g., memory_bound/dram)\n"
            "3. Provide evidence supporting your conclusion\n"
            "4. Generate targeted optimization recommendations\n"
            "5. Assess confidence level with reasoning"
        )
        return "\n".join(parts)

    @staticmethod
    def _verification_task(target_spec: dict[str, Any], prev_result: Any | None) -> str:
        targets = target_spec.get("targets", [])
        target = target_spec.get("target", targets[0] if targets else "unknown")
        parts = [
            f"Verify the benchmark results for: {target}",
            f"\nRequested targets: {targets}",
            f"\nTarget specification: {target_spec}",
        ]

        data = {}
        if prev_result is not None:
            if hasattr(prev_result, "data"):
                data = prev_result.data
            elif isinstance(prev_result, dict):
                data = prev_result.get("data", {})

        if data:
            if "measurements" in data and isinstance(data["measurements"], dict):
                parts.append(f"\n\nMeasurements:")
                for k, v in data["measurements"].items():
                    parts.append(f"  {k}: {v}")
            if "final_output" in data:
                parts.append(f"\n\nFinal output:\n{data['final_output'][:2000]}")
            if "tool_results" in data and isinstance(data["tool_results"], list):
                parts.append(f"\n\nTool execution results:")
                for i, tr in enumerate(data["tool_results"]):
                    if isinstance(tr, dict):
                        stdout = tr.get("stdout", "")
                        if stdout:
                            parts.append(f"  Tool #{i+1} stdout:")
                            for line in stdout.splitlines()[:20]:
                                parts.append(f"    {line}")
                        if tr.get("return_code") is not None:
                            parts.append(f"  Tool #{i+1} return_code: {tr['return_code']}")
            if "analysis_method" in data:
                parts.append(f"\n\nMethodology:\n{data['analysis_method'][:1000]}")
            if "code_gen_output" in data:
                parts.append(f"\n\nCodeGen summary:\n{data['code_gen_output'][:500]}")

        parts.append(
            "\n\nProvide your verdict: ACCEPT or REJECT. "
            "Be specific about any concerns."
        )
        return "\n".join(parts)


# Helper function for target-specific guidance (extracted from _codegen_task for filtering)
def _get_target_type_guidance(target: str) -> str:
    """Return concise code guidance for a specific target. Context-filtered."""
    tl = target.lower()
    if "launch__sm_count" in target:
        return "Use cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount) AND block ID sweep"
    elif "dram__bytes_read" in target:
        return "Read-only STREAM kernel, 65535x256 threads, __ldg() for reads"
    elif "dram__bytes_write" in target:
        return "Write-only STREAM kernel, 65535x256 threads, volatile pointer writes"
    elif "device__attribute_max_gpu_frequency" in target:
        return "Use cudaDeviceGetAttribute(cudaDevAttrClockRate), result in kHz"
    elif "device__attribute_max_mem_frequency" in target:
        return "Use cudaDeviceGetAttribute(cudaDevAttrMemoryClockRate), result in kHz"
    elif "device__attribute_fb_bus_width" in target:
        return "Use cudaDeviceGetAttribute(cudaDevAttrMemoryBusWidth), result in bits"
    elif "sm__throughput" in target:
        return "Compute-intensive FMA kernel, 65535x256 threads, no global mem after init"
    elif "gpu__compute_memory_throughput" in target:
        return "Fused read-compute-write kernel, 65535x256 threads"
    elif "latency" in tl:
        return "Pointer-chasing kernel, clock64() timing, single thread"
    elif "cache_size" in tl or "l2" in tl or "l1" in tl:
        return "Working-set sweep kernel, clock64() timing"
    elif "bandwidth" in tl:
        return "STREAM copy kernel, cudaEventElapsedTime, 65535x256 threads"
    elif "clock" in tl:
        return "Cycle count / wall-clock time kernel"
    elif "sm_count" in tl:
        return "cudaDeviceGetAttribute + block ID sweep"
    else:
        return "See design principle and metric documentation above for measurement approach"
