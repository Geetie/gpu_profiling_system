"""Prompt construction for pipeline stages.

Strategy pattern: each stage type gets its own prompt-building logic,
keeping prompt engineering separate from orchestration.
"""
from __future__ import annotations

from typing import Any

from src.domain.subagent import PipelineStage


class StagePromptBuilder:
    """Builds system and task prompts for each pipeline stage.

    Separates prompt engineering from pipeline orchestration.
    Each method returns the exact prompt string for a given stage.
    """

    def build_system_prompt(self, stage: PipelineStage) -> str:
        """Return the system prompt for a pipeline stage."""
        builders = {
            PipelineStage.PLAN: self._plan_system,
            PipelineStage.CODE_GEN: self._codegen_system,
            PipelineStage.METRIC_ANALYSIS: self._metric_system,
            PipelineStage.VERIFICATION: self._verification_system,
        }
        builder = builders.get(stage)
        if builder is None:
            raise ValueError(f"Unknown pipeline stage: {stage}")
        return builder()

    def build_task_prompt(
        self,
        stage: PipelineStage,
        target_spec: dict[str, Any],
        prev_result: Any | None = None,
    ) -> str:
        """Return the task prompt for a pipeline stage."""
        builders = {
            PipelineStage.PLAN: self._plan_task,
            PipelineStage.CODE_GEN: self._codegen_task,
            PipelineStage.METRIC_ANALYSIS: self._metric_task,
            PipelineStage.VERIFICATION: self._verification_task,
        }
        builder = builders.get(stage)
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
        from src.domain.design_principles import get_design_principle

        targets = target_spec.get("targets", [])
        target = target_spec.get("target", targets[0] if targets else "unknown")
        principle = get_design_principle(target)

        # Check if there's a template available for this target
        template_source = ""
        try:
            from src.infrastructure.probing.cuda_templates import get_template
            tmpl = get_template(target)
            if tmpl:
                template_source = (
                    f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"📝 REFERENCE CUDA SOURCE CODE FOR TARGET: {target}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"Below is a reference implementation for this target.\n"
                    f"You MUST use this EXACT code (or improve upon it) for compilation.\n\n"
                    f"```cuda\n{tmpl.source_code}```\n\n"
                    f"⚠️ CRITICAL: This code is verified to work correctly.\n"
                    f"Use it directly - do NOT generate a different kernel!\n"
                    f"Compile flags: {tmpl.compile_flags}\n"
                )
        except ImportError:
            pass

        parts = [
            f"Write CUDA micro-benchmarks for: {target}",
            f"\nTarget specification: {target_spec}",
            f"\n{principle}",
            template_source,
        ]

        if targets and len(targets) > 1:
            target_list_str = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(targets))
            all_principles = []
            for t in targets:
                p = get_design_principle(t)
                brief = p[:600] if len(p) > 600 else p
                all_principles.append(f"  --- {t} ---\n{brief}")
            principles_str = "\n\n".join(all_principles)

            target_type_guidance = []
            for t in targets:
                tl = t.lower()
                if "launch__sm_count" in t:
                    target_type_guidance.append(
                        f"  • {t}: Use cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount) AND block ID sweep"
                    )
                elif "dram__bytes_read" in t:
                    target_type_guidance.append(
                        f"  • {t}: Read-only STREAM kernel, 65535x256 threads, __ldg() for reads"
                    )
                elif "dram__bytes_write" in t:
                    target_type_guidance.append(
                        f"  • {t}: Write-only STREAM kernel, 65535x256 threads, volatile pointer writes"
                    )
                elif "device__attribute_max_gpu_frequency" in t:
                    target_type_guidance.append(
                        f"  • {t}: Use cudaDeviceGetAttribute(cudaDevAttrClockRate), result in kHz"
                    )
                elif "device__attribute_max_mem_frequency" in t:
                    target_type_guidance.append(
                        f"  • {t}: Use cudaDeviceGetAttribute(cudaDevAttrMemoryClockRate), result in kHz"
                    )
                elif "device__attribute_fb_bus_width" in t:
                    target_type_guidance.append(
                        f"  • {t}: Use cudaDeviceGetAttribute(cudaDevAttrMemoryBusWidth), result in bits"
                    )
                elif "sm__throughput" in t:
                    target_type_guidance.append(
                        f"  • {t}: Compute-intensive FMA kernel, 65535x256 threads, no global mem after init"
                    )
                elif "gpu__compute_memory_throughput" in t:
                    target_type_guidance.append(
                        f"  • {t}: Fused read-compute-write kernel, 65535x256 threads"
                    )
                elif "latency" in tl:
                    target_type_guidance.append(
                        f"  • {t}: Pointer-chasing kernel, clock64() timing, single thread"
                    )
                elif "cache_size" in tl or "l2" in tl or "l1" in tl:
                    target_type_guidance.append(
                        f"  • {t}: Working-set sweep kernel, clock64() timing"
                    )
                elif "bandwidth" in tl:
                    target_type_guidance.append(
                        f"  • {t}: STREAM copy kernel, cudaEventElapsedTime, 65535x256 threads"
                    )
                elif "clock" in tl:
                    target_type_guidance.append(
                        f"  • {t}: Cycle count / wall-clock time kernel"
                    )
                elif "sm_count" in tl:
                    target_type_guidance.append(
                        f"  • {t}: cudaDeviceGetAttribute + block ID sweep"
                    )
                else:
                    target_type_guidance.append(
                        f"  • {t}: See design principle above for measurement approach"
                    )

            target_type_str = "\n".join(target_type_guidance)

            parts.append(
                f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 TARGET ASSIGNMENT (MEASURE ALL OF THESE IN ORDER)\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
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

            parts[-1] += (
                f"\nWORKFLOW (repeat for EACH target):\n"
                f"  1. Write CUDA code SPECIFICALLY for the CURRENT target (unique kernel)\n"
                f"  2. compile_cuda(source=\"...\", flags=[\"-O3\"], target=\"CURRENT_TARGET_NAME\")\n"
                f"  3. execute_binary(binary_path=\"<exact path from compile_cuda>\")\n"
                f"  4. Record the measured value from stdout\n"
                f"  5. Wait for SYSTEM message → then move to NEXT target\n\n"
                f"⚠️ CRITICAL RULES:\n"
                f"  • Each target needs DIFFERENT CUDA code! Use the guidance below:\n"
                f"{target_type_str}\n\n"
                f"  • compile_cuda OVERWRITES the previous binary each time.\n"
                f"    So you MUST execute_binary IMMEDIATELY after each compile_cuda.\n"
                f"    Do NOT compile all targets first — compile+execute one at a time.\n\n"
                f"  • MAX 3 compilation attempts per target.\n"
                f"    After 3 failures, MOVE ON to next target (system will force switch).\n\n"
                f"  • Do NOT skip any target — pipeline WILL FAIL if any is missing!\n\n"
                f"  • The printf output MUST match the target name EXACTLY:\n"
                f"    printf(\"EXACT_TARGET_NAME: value\\n\")\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📐 DESIGN PRINCIPLES FOR EACH TARGET\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{principles_str}"
            )

        if prev_result is not None:
            prev_data = prev_result.data if hasattr(prev_result, "data") else (prev_result.get("data", {}) if isinstance(prev_result, dict) else {})
            plan_output = ""
            if prev_data:
                plan_output = prev_data.get("final_output", "")

                tasks = prev_data.get("tasks", [])
                method = ""
                for task in tasks:
                    if isinstance(task, dict) and task.get("target") == target:
                        method = task.get("method", "")
                        break

                if method:
                    parts.append(
                        f"\n\n--- Measurement methodology from Planner ---\n{method}"
                    )

                category = ""
                for task in tasks:
                    if isinstance(task, dict) and task.get("target") == target:
                        category = task.get("category", "")
                        break

                if category:
                    parts.append(
                        f"\n\nTask category: {category}"
                    )

            if not plan_output and hasattr(prev_result, "error") and prev_result.error:
                plan_output = prev_result.error
            if isinstance(prev_result, dict) and not plan_output:
                plan_output = prev_result.get("error", "")
            if plan_output:
                parts.append(
                    f"\n\n--- Plan from previous stage ---\n{plan_output}"
                )

        parts.append(
            "\n\nIMPORTANT: You MUST call compile_cuda to compile your code, "
            "then execute_binary to run it. Text-only output without tool calls is a FAILURE."
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
            f"\n\n--- Design Principle for this target ---\n{principle[:2000]}",
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
                f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🎯 NCU PROFILING INSTRUCTIONS\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
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
