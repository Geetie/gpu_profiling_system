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
            "plan for measuring a specific GPU hardware characteristic.\n\n"
            "You must use the available tools to research and plan. "
            "Do NOT write CUDA code — that is the CodeGen agent's job.\n\n"
            "Your plan should include:\n"
            "1. Which GPU micro-architectural feature to measure\n"
            "2. What measurement technique to use (pointer-chasing, sweep, etc.)\n"
            "3. What anti-cheat strategies are needed\n"
            "4. Expected result ranges for validation\n\n"
            "IMPORTANT: You MUST call tools during your turn. "
            "A text-only response without tool calls is a failure."
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
        target = target_spec.get("target", "unknown")
        return (
            f"Create a detailed plan for measuring GPU characteristic: {target}\n\n"
            f"Target specification: {target_spec}\n\n"
            "Use the available tools to research this target. "
            "Your plan will be used by the CodeGen agent to write the actual benchmark."
        )

    @staticmethod
    def _codegen_task(target_spec: dict[str, Any], prev_result: Any | None) -> str:
        from src.domain.design_principles import get_design_principle

        target = target_spec.get("target", "unknown")
        targets = target_spec.get("targets", [])
        principle = get_design_principle(target)

        parts = [
            f"Write CUDA micro-benchmarks for: {target}",
            f"\nTarget specification: {target_spec}",
            f"\n{principle}",
        ]

        if targets and len(targets) > 1:
            parts.append(
                f"\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⚠️  CRITICAL: You MUST measure ALL {len(targets)} targets\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Targets: {targets}\n\n"
                f"For EACH target, you MUST:\n"
                f"  1. compile_cuda with the CUDA source for that target\n"
                f"  2. execute_binary to run the compiled binary\n"
                f"  3. Record the measured value from stdout\n\n"
                f"Do NOT skip any target. The pipeline will FAIL if any target is missing.\n"
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

        target = target_spec.get("target", "unknown")
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
        target = target_spec.get("target", "unknown")
        targets = target_spec.get("targets", [])
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
