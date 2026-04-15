"""Pipeline orchestrator — domain layer.

Coordinates the sequential execution of sub-agents with P7 enforcement,
retry support, persistence hooks, and harness engineering guardrails.

Each stage runs inside an AgentLoop, enabling LLM-driven iteration:
model calls tool → sees result → retries/refines → calls next tool → ...
until the model signals completion or max turns reached.

Harness engineering extensions:
- Handoff validation at every stage boundary
- Circuit breaker for progressive degradation detection
- Per-stage timing for audit reporting
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.domain.subagent import (
    AgentRole,
    BaseSubAgent,
    CollaborationMessage,
    PipelineStage,
    P7ViolationError,
    SubAgentResult,
    SubAgentStatus,
)
from src.infrastructure.state_persist import StatePersister


@dataclass
class PipelineStep:
    """A single stage in the pipeline."""
    stage: PipelineStage
    agent: BaseSubAgent
    retry_on_failure: int = 0


class Pipeline:
    """Orchestrates the multi-agent collaboration flow.

    Flow: target_spec → PLAN → CODE_GEN → METRIC_ANALYSIS → VERIFICATION

    Each stage runs inside an AgentLoop, allowing the LLM to iterate:
    - Call tools (compile_cuda, execute_binary, run_ncu, etc.)
    - See results and adapt
    - Retry/refine until task is complete or max turns reached

    P7 enforcement:
    - VerificationAgent must have an empty ContextManager before execution.
    - The pipeline checks this explicitly and raises P7ViolationError.

    Harness engineering extensions:
    - handoff_validator: validates stage outputs against contracts
    - circuit_breaker: detects progressive degradation and trips circuit
    - audit_report: collects execution metrics for post-run analysis
    """

    def __init__(
        self,
        stages: list[PipelineStep],
        state_dir: str,
        sandbox=None,
        tool_handlers: dict | None = None,
        max_turns_per_stage: int = 15,
        handoff_validator=None,  # HandoffValidator | None
        circuit_breaker=None,    # CircuitBreaker | None
    ) -> None:
        self._stages = stages
        self._state_dir = state_dir
        self._persister = StatePersister(log_dir=state_dir, filename="pipeline_log.jsonl")
        self._sandbox = sandbox
        self._tool_handlers = tool_handlers or {}
        self._max_turns_per_stage = max_turns_per_stage
        self._handoff_validator = handoff_validator
        self._circuit_breaker = circuit_breaker

    def run(self, target_spec: dict[str, Any]) -> SubAgentResult:
        """Execute the full pipeline.

        Args:
            target_spec: The target specification (from target_spec.json).

        Returns:
            The final SubAgentResult from the VERIFICATION stage,
            or a FAILED result if any stage fails permanently.
        """
        print(f"[Pipeline] Starting pipeline with targets: {target_spec.get('targets', [])}")
        self._persister.log_entry("pipeline_start", details={"target_spec": target_spec})

        prev_result: SubAgentResult | None = None
        code_gen_data: dict | None = None  # Preserve CodeGen measurements for final output
        prev_stage: PipelineStage | None = None

        for step in self._stages:
            stage_start = time.monotonic()
            print(f"[Pipeline] Executing stage: {step.stage.value}")

            # P7 gate
            self._check_p7(step.stage, prev_result)

            # Harness: validate handoff from previous stage
            if prev_stage is not None and self._handoff_validator is not None and prev_result is not None:
                handoff = self._handoff_validator.validate(
                    prev_stage, step.stage, prev_result
                )
                print(f"[Pipeline] Handoff validation: from {handoff.from_stage} to {handoff.to_stage} - valid={handoff.is_valid}")
                self._persister.log_entry(
                    "handoff_validation",
                    details={
                        "from": handoff.from_stage,
                        "to": handoff.to_stage,
                        "is_valid": handoff.is_valid,
                        "errors": len(handoff.errors),
                        "warnings": len(handoff.warnings),
                    },
                )
                if not handoff.is_valid:
                    # Log errors but allow pipeline to continue (stage may still work)
                    print(f"[Pipeline] Handoff validation failed with {len(handoff.errors)} errors")
                    for v in handoff.errors:
                        self._persister.log_entry(
                            "handoff_error",
                            details={"stage": v.stage, "field": v.field, "message": v.message},
                        )

            # Harness: check circuit breaker before executing stage
            if self._circuit_breaker is not None and self._circuit_breaker.is_open:
                error_msg = f"Circuit breaker open: {self._circuit_breaker._state.trip_reason}"
                print(f"[Pipeline] {error_msg}")
                return SubAgentResult(
                    agent_role=step.agent.role,
                    status=SubAgentStatus.FAILED,
                    error=error_msg,
                )

            # Execute with retries
            print(f"[Pipeline] Running stage {step.stage.value} with {step.retry_on_failure} retries")
            result = self._execute_stage(step, prev_result, target_spec)

            stage_duration = time.monotonic() - stage_start
            print(f"[Pipeline] Stage {step.stage.value} completed in {round(stage_duration, 2)}s with status: {result.status.value}")

            if result.is_failed():
                self._persister.log_entry(
                    "pipeline_stage_failed",
                    details={
                        "stage": step.stage.value,
                        "error": result.error,
                        "duration_seconds": round(stage_duration, 2),
                    },
                )
                print(f"[Pipeline] Stage {step.stage.value} failed: {result.error}")
                # Don't return immediately for non-critical stages
                # Allow pipeline to continue with partial results ONLY if
                # the stage produced some useful data (e.g., compiled at least
                # one binary). If the stage produced nothing, abort.
                if step.stage in (PipelineStage.CODE_GEN, PipelineStage.METRIC_ANALYSIS):
                    has_useful_data = bool(result.data and result.data.get("measurements"))
                    has_compiled_binary = any(
                        isinstance(tr, dict) and tr.get("binary_path")
                        for tr in result.data.get("tool_results", [])
                    )
                    if has_useful_data or has_compiled_binary:
                        self._persister.log_entry(
                            "pipeline_stage_partial",
                            details={
                                "stage": step.stage.value,
                                "message": "Continuing with partial results",
                            },
                        )
                        print(f"[Pipeline] Continuing with partial results for {step.stage.value}")
                        partial_result = SubAgentResult(
                            agent_role=step.agent.role,
                            status=SubAgentStatus.SUCCESS,
                            data=result.data,
                            error=f"Partial execution: {result.error}",
                        )
                        prev_result = partial_result
                        prev_stage = step.stage
                        continue
                return result

            # Harness: score stage quality on circuit breaker
            if self._circuit_breaker is not None and self._handoff_validator is not None:
                handoff = self._handoff_validator.validate(
                    step.stage, step.stage, result
                )
                self._circuit_breaker.score_stage(
                    stage=step.stage.value,
                    handoff_errors=len(handoff.errors),
                    handoff_warnings=len(handoff.warnings),
                    had_output=bool(result.data.get("final_output")),
                    tool_calls_made=result.data.get("num_tool_calls", 0),
                )

            # Preserve CodeGen measurements for final result assembly
            if step.stage == PipelineStage.CODE_GEN and result.is_success():
                code_gen_data = dict(result.data)
                print(f"[Pipeline] Preserved CodeGen measurements: {list(code_gen_data.get('measurements', {}).keys())}")

            # Bubble up CodeGen data into MetricAnalysis result so Verification
            # can see the full chain (CodeGen measurements + MetricAnalysis output)
            if step.stage == PipelineStage.METRIC_ANALYSIS and code_gen_data and result.is_success():
                if "measurements" not in result.data and "measurements" in code_gen_data:
                    result.data["measurements"] = code_gen_data["measurements"]
                if "code_gen_output" not in result.data and "code_gen_output" in code_gen_data:
                    result.data["code_gen_output"] = code_gen_data["code_gen_output"]
                if "tool_results" not in result.data and "tool_results" in code_gen_data:
                    result.data["tool_results"] = code_gen_data["tool_results"]
                if "code_gen_final_output" not in result.data and "final_output" in code_gen_data:
                    result.data["code_gen_final_output"] = code_gen_data["final_output"]

            prev_result = result
            prev_stage = step.stage

        # Bubble up CodeGen measurements and tool results to final result
        if code_gen_data and prev_result and prev_result.is_success():
            if "measurements" in code_gen_data:
                prev_result.data["measurements"] = code_gen_data["measurements"]
            if "analysis_method" in code_gen_data:
                prev_result.data["analysis_method"] = code_gen_data["analysis_method"]
            if "code_gen_output" in code_gen_data:
                prev_result.data["code_gen_output"] = code_gen_data["code_gen_output"]
            if "tool_results" in code_gen_data:
                prev_result.data["tool_results"] = code_gen_data["tool_results"]
            if "binary_path" in code_gen_data:
                prev_result.data["binary_path"] = code_gen_data["binary_path"]

        # Persist final result
        if prev_result:
            self._persister.log_entry("pipeline_complete", details=prev_result.to_dict())
            print(f"[Pipeline] Pipeline completed with status: {prev_result.status.value}")
        else:
            print("[Pipeline] Pipeline produced no result")

        return prev_result or SubAgentResult(
            agent_role=AgentRole.PLANNER,
            status=SubAgentStatus.FAILED,
            error="Pipeline produced no result",
        )

    def _execute_stage(
        self, step: PipelineStep, prev_result: SubAgentResult | None,
        target_spec: dict[str, Any] | None = None,
    ) -> SubAgentResult:
        """Execute a single pipeline stage with AgentLoop iteration.

        Instead of calling step.agent.run() once, this creates an AgentLoop
        that allows the LLM to iterate: call tools, see results, adapt.
        """
        self._persister.log_entry(
            "pipeline_stage_start",
            details={"stage": step.stage.value, "retry_limit": step.retry_on_failure},
        )
        print(f"[Pipeline] Starting stage {step.stage.value} with {step.retry_on_failure} retries")
        last_result: SubAgentResult | None = None

        for attempt in range(1 + step.retry_on_failure):
            if attempt > 0:
                self._persister.log_entry(
                    "pipeline_retry",
                    details={"stage": step.stage.value, "attempt": attempt},
                )
                print(f"[Pipeline] Retry {attempt} for stage {step.stage.value}")

            # Build the collaboration message
            # CRITICAL: target_spec must be available to ALL stages, not just Planner
            payload: dict[str, Any] = {}
            if target_spec:
                payload["target_spec"] = target_spec
            if prev_result is not None:
                payload["prev_result"] = prev_result.to_dict()
                payload["prev_fingerprint"] = prev_result.context_fingerprint

            message = CollaborationMessage(
                sender=prev_result.agent_role if prev_result else AgentRole.PLANNER,
                receiver=step.agent.role,
                message_type="task_dispatch",
                payload=payload,
            )

            # Use AgentLoop for iteration within this stage
            print(f"[Pipeline] Running AgentLoop for stage {step.stage.value}")
            last_result = self._run_with_agent_loop(step, message)

            print(f"[Pipeline] AgentLoop completed with status: {last_result.status.value}")
            if last_result.is_success():
                print(f"[Pipeline] Stage {step.stage.value} succeeded")
                break

            if last_result.status == SubAgentStatus.REJECTED:
                print(f"[Pipeline] Stage {step.stage.value} rejected")
                break

            self._persister.log_entry(
                "pipeline_attempt_failed",
                details={
                    "stage": step.stage.value,
                    "attempt": attempt + 1,
                    "error": last_result.error,
                },
            )
            print(f"[Pipeline] Attempt {attempt + 1} failed: {last_result.error}")

        if last_result:
            print(f"[Pipeline] Stage {step.stage.value} finished with status: {last_result.status.value}")
        else:
            print(f"[Pipeline] Stage {step.stage.value} produced no result after all retries")

        return last_result or SubAgentResult(
            agent_role=step.agent.role,
            status=SubAgentStatus.FAILED,
            error="Stage produced no result after all retries",
        )

    def _check_p7(
        self, current_stage: PipelineStage, prev_result: SubAgentResult | None
    ) -> None:
        """P7 gate: verify that VerificationAgent has clean context."""
        if current_stage != PipelineStage.VERIFICATION:
            return

        verify_agent = self._get_stage_agent(PipelineStage.VERIFICATION)
        if verify_agent is None:
            return

        tokens = verify_agent.context_manager.total_tokens
        if tokens > 0:
            raise P7ViolationError(
                f"P7 violation: VerificationAgent has non-empty context "
                f"({tokens} tokens). Verification must not inherit generation context."
            )

        # Log the audit trail
        self._persister.log_entry(
            "p7_audit",
            details={
                "generation_fingerprint": prev_result.context_fingerprint if prev_result else None,
                "verification_context_tokens": 0,
                "status": "clean",
            },
        )

    def _build_stage_system_prompt(self, agent: BaseSubAgent, stage_name: str) -> str:
        """Build system prompt for a pipeline stage's AgentLoop."""
        import json as _json
        available_tools = agent.tool_registry.list_tools()
        tool_list = _json.dumps(available_tools, indent=2) if available_tools else "(no tools registered)"

        # Stage-specific tool usage instructions
        tool_guidance = ""
        if stage_name == PipelineStage.CODE_GEN.value:
            tool_guidance = (
                f"\n\nYOUR TOOLS: compile_cuda, execute_binary, write_file, read_file\n"
                f"YOUR JOB: Write CUDA code → compile → execute → report values\n"
                f"DO NOT: run ncu (that's MetricAnalysis's job)\n"
                f"DO NOT: verify results (that's Verification's job)\n\n"
                f"IMPORTANT: You MUST call tools as JSON objects, one at a time:\n"
                f'{{"tool": "write_file", "args": {{"file_path": ".sandbox/benchmark.cu", "content": "...code..."}}}}\n'
                f'{{"tool": "compile_cuda", "args": {{"source": "...full .cu source code...", "flags": ["-O3"]}}}}\n'
                f'{{"tool": "execute_binary", "args": {{"binary_path": "<path from compile_cuda>"}}}}\n\n'
                f"After each tool call, you will see the result. Then call the next tool.\n"
                f"DO NOT just describe what you would do — actually call the tools.\n\n"
                f"CRITICAL: You MUST use the .sandbox directory for all file operations.\n"
                f"Example: .sandbox/benchmark.cu (correct)\n"
                f"Incorrect: /kaggle/working/gpu_profiling_system/benchmark.cu (will be blocked)\n"
                f"Incorrect: benchmark.cu (will be blocked)\n\n"
                f"MANDATORY WORKFLOW — process EACH target separately:\n"
                f"For target 1:\n"
                f'  1. compile_cuda(source="...full .cu source for target 1...", flags=["-O3"])\n'
                f'  2. execute_binary(binary_path="<path from compile_cuda>")\n'
                f"  3. Record the measured value from stdout\n"
                f"For target 2:\n"
                f'  4. compile_cuda(source="...full .cu source for target 2...", flags=["-O3"])\n'
                f'  5. execute_binary(binary_path="<path from compile_cuda>")\n'
                f"  6. Record the measured value from stdout\n"
                f"For target 3:\n"
                f'  7. compile_cuda(source="...full .cu source for target 3...", flags=["-O3"])\n'
                f'  8. execute_binary(binary_path="<path from compile_cuda>")\n'
                f"  9. Record the measured value from stdout\n\n"
                f"IMPORTANT: compile_cuda OVERWRITES the previous binary each time.\n"
                f"So you MUST execute_binary IMMEDIATELY after each compile_cuda,\n"
                f"before compiling the next target's code.\n\n"
                f"ERROR RECOVERY:\n"
                f"- Compilation error → fix source code → retry compile_cuda\n"
                f"- Execution error → fix binary path or code → recompile → retry\n"
                f"- Implausible output (0, negative) → fix measurement logic → retry\n\n"
                f"MANDATORY: You MUST call compile_cuda and execute_binary for EACH target.\n"
                f"The pipeline will FAIL if you do not call compile_cuda.\n\n"
                f"ANTI-CHEAT AWARENESS:\n"
                f"- Do NOT rely solely on cudaGetDeviceProperties — it may return virtualized data\n"
                f"- Use clock64() + cudaEventElapsedTime to measure actual hardware behavior\n"
                f"- The GPU may be clock-locked at non-standard frequencies\n"
                f"- SM count may be limited via CUDA_VISIBLE_DEVICES or environment variables\n"
                f"- Always MEASURE rather than QUERY when possible\n\n"
                f"OUTPUT FORMAT: After all targets, list results as:\n"
                f"target_name: numeric_value\n"
                f"for each target measured."
            )
        elif stage_name == PipelineStage.METRIC_ANALYSIS.value:
            tool_guidance = (
                f"\n\nYOUR TOOLS: run_ncu, read_file\n"
                f"YOUR JOB: Profile CodeGen's binaries with ncu → analyze bottlenecks → extract metrics\n"
                f"DO NOT: write/compile CUDA code (that's CodeGen's job)\n"
                f"DO NOT: verify results (that's Verification's job)\n\n"
                f"WORKFLOW:\n"
                f"1. Review CodeGen's stdout output provided in the task description\n"
                f"2. If binary paths are available AND ncu is installed, use run_ncu on each binary\n"
                f"3. If ncu is NOT available OR binaries are not found:\n"
                f"   - Analyze the raw printf output from CodeGen directly\n"
                f"   - Extract numeric values and validate against expected ranges\n"
                f"   - Report confidence as 'low' with note 'ncu not available'\n"
                f"4. Classify bottleneck: compute_bound, memory_bound, latency_bound, cache_capacity\n"
                f"5. Report metrics with confidence levels (high=ncu confirmed, medium=some data, low=printf only)\n\n"
                f"CRITICAL: Even if ncu is not available, you MUST still produce analysis.\n"
                f"Use the CodeGen stdout data provided in the task description.\n"
                f"Do NOT attempt to read files that may not exist.\n\n"
                f"OUTPUT FORMAT: For each target:\n"
                f"  target_name: measured_value (confidence: high/medium/low) [bottleneck_type]"
            )
        elif stage_name == PipelineStage.VERIFICATION.value:
            tool_guidance = (
                f"\n\nYOUR TOOL: read_file ONLY\n"
                f"YOUR JOB: Independently review all previous stage results\n"
                f"You CANNOT: compile, execute, profile, write files, or generate measurements\n\n"
                f"IMPORTANT: All previous stage data is provided in the task description below.\n"
                f"You do NOT need to use read_file to find data — it is already given to you.\n"
                f"Only use read_file if you need to check a specific evidence file.\n\n"
                f"VERIFICATION CHECKS (perform in order):\n"
                f"1. Data completeness — are ALL targets measured?\n"
                f"2. Numeric sanity — are values in plausible GPU hardware ranges?\n"
                f"3. Latency hierarchy — L1 < L2 < DRAM (when both measured)\n"
                f"4. Cross-validation — do CodeGen and MetricAnalysis agree?\n"
                f"5. Methodology soundness — were correct techniques used?\n\n"
                f"State your verdict as: Verdict: ACCEPT or Verdict: REJECT"
            )

        return (
            f"You are the {stage_name} stage in a GPU hardware profiling pipeline.\n"
            f"{agent._build_system_prompt()}\n\n"
            f"Available tools: {tool_list}\n\n"
            f"Tool call format: {{\"tool\": \"tool_name\", \"args\": {{\"key\": \"value\"}}}}\n"
            f"After each tool call result, you may call more tools or give your final answer.\n"
            f"When done, give your final answer as plain text (not JSON)."
            f"{tool_guidance}"
        )

    def _build_task_prompt(
        self, task: dict, prev_result: dict, stage_name: str
    ) -> str:
        """Build task-specific prompt for CODE_GEN, METRIC_ANALYSIS, VERIFICATION."""
        import json as _json

        if stage_name == PipelineStage.CODE_GEN.value:
            # CodeGen receives ALL tasks from Planner as a list
            tasks_list = task.get("tasks", [])

            if tasks_list:
                # Build design guide for ALL targets
                design_guides = []
                for t in tasks_list:
                    target = t.get("target", "unknown")
                    guide = _get_design_principle(target)
                    design_guides.append(f"--- Task: {target} ---\n{guide}")
                all_design = "\n\n".join(design_guides)

                task_summary = "\n".join(
                    f"- {t.get('target', 'unknown')}: {t.get('category', 'unknown')} "
                    f"(method: {t.get('method', 'custom')})"
                    for t in tasks_list
                )

                return (
                    f"You must generate and execute CUDA micro-benchmarks for "
                    f"{len(tasks_list)} targets:\n\n{task_summary}\n\n"
                    f"PROCESS EACH TARGET ONE BY ONE:\n"
                    f"1. Pick the next target from the list\n"
                    f"2. Read its design guide below\n"
                    f"3. Write complete CUDA C++ source code\n"
                    f"   - Use clock64() for cycle timing (NOT clock() — returns 0 on Pascal+)\n"
                    f"   - Prevent dead code elimination: write results to volatile output pointer\n"
                    f"   - Use cudaDeviceSynchronize() before reading device-side results\n"
                    f"   - Run 1 warm-up iteration before timing\n"
                    f"4. Use compile_cuda to compile\n"
                    f"5. If compile fails: FIX the code and retry (max 3 retries per target)\n"
                    f"6. Use execute_binary to run\n"
                    f"7. Parse the numeric output for 'target_name: numeric_value'\n"
                    f"8. If output is 0, negative, or implausible: fix code and retry\n"
                    f"9. Move to the next target\n\n"
                    f"CODING RULES:\n"
                    f"- Use clock64() for cycle timing (NOT clock() — returns 0 on Pascal+)\n"
                    f"- Use cudaEventElapsedTime for wall-clock timing (bandwidth, frequency)\n"
                    f"- Prevent compiler dead code elimination (volatile output or asm volatile)\n"
                    f"- Synchronize before reading: cudaDeviceSynchronize() or cudaEventSynchronize()\n"
                    f"- Each binary outputs: target_name: numeric_value\n"
                    f"- Include: cuda_runtime.h, main(), cudaMalloc, cudaMemcpy, kernel launch\n"
                    f"- Detect GPU arch: compile with -arch=sm_XX (from nvidia-smi, e.g. sm_60)\n"
                    f"- Run 3 trials, report median\n\n"
                    f"ANTI-CHEAT RULES (CRITICAL):\n"
                    f"- Do NOT rely solely on cudaGetDeviceProperties or cudaDeviceGetAttribute\n"
                    f"  These APIs may return virtualized/misleading data in evaluation\n"
                    f"- ALWAYS measure actual hardware behavior with micro-benchmarks\n"
                    f"- For clock frequency: measure with clock64()/cudaEventElapsedTime ratio\n"
                    f"- For SM count: verify API results with occupancy-based probing\n"
                    f"- For cache sizes: use working-set sweep (not API queries)\n"
                    f"- The GPU may be clock-locked at non-standard frequencies\n\n"
                    f"ERROR RECOVERY:\n"
                    f"- If a target fails after 3 retries: report the failure and continue\n"
                    f"- Do NOT let one failed target block the rest\n\n"
                    f"SELF-CORRECTION CHECKLIST (before final answer):\n"
                    f"- Every target has a numeric value (no zeros, no negatives for latency/bandwidth)\n"
                    f"- Values are plausible (DRAM latency 300-1000, SM count 8-256, etc.)\n"
                    f"- Used clock64() NOT clock()? Prevented dead code elimination? Synchronized?\n"
                    f"- If implausible: re-examine the code and retry\n\n"
                    f"DESIGN GUIDES (one per target):\n\n{all_design}"
                )
            else:
                # Fallback: single task format
                target = task.get("target", "unknown")
                category = task.get("category", "unknown")
                method = task.get("method", "custom micro-benchmark")
                design_guide = _get_design_principle(target)

                return (
                    f"Generate and execute a CUDA micro-benchmark to measure '{target}'.\n"
                    f"Category: {category}\n"
                    f"Method: {method}\n\n"
                    f"{design_guide}\n\n"
                    f"Steps:\n"
                    f"1. Write complete CUDA C++ source code implementing the above design\n"
                    f"   - Use clock64() for cycle timing (NOT clock() — returns 0 on Pascal+)\n"
                    f"   - Use cudaEventElapsedTime for wall-clock timing when needed\n"
                    f"   - Output parseable key: value pairs via printf\n"
                    f"   - Include: cuda_runtime.h, main(), cudaMalloc, cudaMemcpy, kernel launch\n"
                    f"2. Use compile_cuda tool to compile the CUDA source\n"
                    f"3. Use execute_binary tool to run the compiled benchmark\n"
                    f"4. Parse the numeric output from printf results\n"
                    f"5. If results are invalid (0, negative, implausible), revise the code and retry\n\n"
                    f"Output the measured value with: {target}: <numeric_value>\n"
                    f"Also report: confidence (0.0-1.0), method_used, num_trials."
                )

        elif stage_name == PipelineStage.METRIC_ANALYSIS.value:
            # MetricAnalysis profiles CodeGen's binaries with ncu
            code_gen_data = prev_result.get("data", {})
            raw_outputs = {}
            binary_paths = []
            measurements = {}
            stdout_lines = []
            for tr in code_gen_data.get("tool_results", []):
                if not isinstance(tr, dict):
                    continue
                if tr.get("binary_path"):
                    binary_paths.append(tr["binary_path"])
                if tr.get("stdout"):
                    stdout_text = tr["stdout"]
                    raw_outputs[f"stdout_{len(raw_outputs)}"] = stdout_text[:2000]
                    stdout_lines.append(stdout_text)
                if tr.get("output"):
                    raw_outputs[f"compile_{len(raw_outputs)}"] = tr["output"][:500]
            for key, val in code_gen_data.items():
                if isinstance(val, dict) and key == "measurements":
                    measurements = val

            binary_info = f"Compiled binaries found: {len(binary_paths)}"
            if binary_paths:
                binary_info += "\nPaths:\n" + "\n".join(f"  - {p}" for p in binary_paths)
            else:
                binary_info += "\nNo compiled binaries found — will analyze CodeGen stdout only."

            measurements_info = ""
            if measurements:
                measurements_info = (
                    f"\n\nCodeGen reported measurements:\n"
                    + "\n".join(f"  - {k}: {v}" for k, v in sorted(measurements.items()))
                )

            stdout_summary = ""
            if stdout_lines:
                stdout_summary = (
                    f"\n\nCodeGen execute_binary stdout (RAW DATA — use this for analysis):\n"
                    + "\n---\n".join(stdout_lines)
                )

            final_output = code_gen_data.get("final_output", "")
            final_output_section = ""
            if final_output:
                final_output_section = f"\n\nCodeGen final text output:\n{final_output[:1500]}"

            return (
                f"You are the Metric Analysis Agent.\n\n"
                f"YOUR JOB: Profile compiled binaries with Nsight Compute (ncu) and analyze results.\n"
                f"If ncu or binaries are unavailable, analyze CodeGen's stdout data directly.\n\n"
                f"{binary_info}{measurements_info}{stdout_summary}{final_output_section}\n\n"
                f"YOUR TASK:\n"
                f"1. If binary paths exist AND ncu is available, use run_ncu tool to profile each binary\n"
                f"2. If ncu is NOT available OR binaries not found:\n"
                f"   - Analyze the CodeGen stdout data provided above directly\n"
                f"   - Extract numeric values and validate against expected ranges\n"
                f"   - Report confidence = 'low' with note 'ncu not available'\n"
                f"3. Key metrics to collect (if ncu available):\n"
                f"   a) SM Throughput: sm__throughput.avg.pct_of_peak_sustained_elapsed\n"
                f"   b) Memory Throughput: dram__throughput.avg.pct_of_peak_sustained_elapsed\n"
                f"   c) L2 Hit Rate: l2__hit_rate.pct\n"
                f"   d) Achieved Occupancy: sm__warps_active.avg.pct_of_peak_sustained_active\n\n"
                f"4. BOTTLENECK CLASSIFICATION (roofline analysis):\n"
                f"   - dram__throughput% > sm__throughput% → memory_bound\n"
                f"   - sm__throughput% > dram__throughput% → compute_bound\n"
                f"   - both low (<30%) AND sm__warps_active% <20% → latency_bound\n\n"
                f"5. CROSS-VALIDATE against CodeGen measurements:\n"
                f"   - Compare CodeGen's printf values with ncu hardware counter data\n"
                f"   - If they disagree: report both values and explain the discrepancy\n\n"
                f"OUTPUT FORMAT:\n"
                f"For each measurement:\n"
                f"  target_name: measured_value (confidence: high/medium/low) [bottleneck_type]\n\n"
                f"Confidence levels:\n"
                f"- high: ncu profiling confirms the value + metrics are internally consistent\n"
                f"- medium: ncu profiling available but partial, OR printf consistent across trials\n"
                f"- low: only CodeGen printf output, no ncu confirmation\n\n"
                f"EXPECTED RANGES (for reference):\n"
                f"- L1 latency: 50-300 cycles, L2: 100-500, DRAM: 300-1000\n"
                f"- L2 cache: 2-100 MB (power of 2 typically)\n"
                f"- DRAM bandwidth: 100-900 GB/s\n"
                f"- SM clock: 1000-2500 MHz\n"
                f"- SM count: 8-256\n\n"
                f"TRUST MODEL: Do NOT blindly trust CodeGen's conclusions.\n"
                f"Verify against raw ncu output. If ncu is not available, analyze printf output\n"
                f"and report confidence as 'low'. Report which ncu metrics support each classification."
            )

        elif stage_name == PipelineStage.VERIFICATION.value:
            # Verification receives the FULL chain: Planner targets + CodeGen output + MetricAnalysis
            code_gen_data = prev_result.get("data", {})
            code_gen_measurements = code_gen_data.get("measurements", {})
            code_gen_final = code_gen_data.get("code_gen_final_output", "") or code_gen_data.get("final_output", "")
            code_gen_tool_results = code_gen_data.get("tool_results", [])

            stdout_data = []
            for tr in code_gen_tool_results:
                if isinstance(tr, dict) and tr.get("stdout"):
                    stdout_data.append(tr["stdout"][:1500])

            code_gen_section = ""
            if code_gen_measurements:
                code_gen_section = (
                    f"\n\n=== CODEGEN MEASUREMENTS ===\n"
                    + "\n".join(f"  {k}: {v}" for k, v in sorted(code_gen_measurements.items()))
                )
            if stdout_data:
                code_gen_section += (
                    f"\n\n=== CODEGEN RAW STDOUT ===\n"
                    + "\n---\n".join(stdout_data)
                )
            if code_gen_final:
                code_gen_section += f"\n\n=== CODEGEN FINAL OUTPUT ===\n{code_gen_final[:1500]}"

            metric_analysis_section = ""
            analysis_output = code_gen_data.get("analysis_output", "")
            if analysis_output:
                metric_analysis_section = f"\n\n=== METRIC ANALYSIS OUTPUT ===\n{analysis_output[:1500]}"

            return (
                f"You are the independent Verification Agent for GPU hardware profiling.\n\n"
                f"You do NOT trust any previous stage's conclusions.\n"
                f"Review from first principles using GPU hardware knowledge.\n\n"
                f"All data from previous stages is provided below — you do NOT need read_file.\n"
                f"{code_gen_section}{metric_analysis_section}\n\n"
                f"VERIFICATION CHECKS (perform in order):\n\n"
                f"1. DATA COMPLETENESS: Are ALL requested targets measured? Any missing?\n"
                f"2. NUMERIC SANITY: Are values within plausible GPU hardware ranges?\n"
                f"   - L1 latency: 50-300 cycles, L2: 100-500, DRAM: 300-1000\n"
                f"   - DRAM bandwidth: 100-900 GB/s, SM count: 8-256\n"
                f"3. LATENCY HIERARCHY: DRAM latency > L2 latency > L1 latency? (when available)\n"
                f"4. L2 CAPACITY: Is it a power of 2 (2, 4, 8, 40, 50, 60, 72 MB)?\n"
                f"5. CROSS-VALIDATION: Do CodeGen measurements agree with MetricAnalysis?\n"
                f"   - If confidence is 'low': note why\n"
                f"   - If values disagree: flag the discrepancy\n"
                f"6. METHODOLOGY: Were correct techniques used?\n"
                f"   - Latency: pointer-chasing with random permutation\n"
                f"   - Bandwidth: STREAM copy\n"
                f"   - SM count: cudaDeviceGetAttribute\n"
                f"   - SHMEM capacity: cudaOccupancyMaxActiveBlocksPerMultiprocessor\n\n"
                f"OUTPUT FORMAT:\n"
                f"VERIFICATION REPORT\n"
                f"==================\n"
                f"1. Completeness: PASS/FAIL\n"
                f"2. Numeric Sanity: PASS/FAIL\n"
                f"3. Latency Hierarchy: PASS/FAIL/N/A\n"
                f"4. Cross-validation: PASS/FAIL/PARTIAL\n"
                f"5. Methodology: PASS/FAIL\n\n"
                f"Verdict: ACCEPT — or —\n"
                f"Verdict: REJECT\n"
                f"  Reason: <specific reasons>\n\n"
                f"ACCEPT if: all targets measured + values plausible + hierarchy holds\n"
                f"REJECT if: missing targets + impossible values + hierarchy violation"
            )

        else:
            return f"Execute your {stage_name} stage task."

    def _get_stage_agent(self, stage: PipelineStage) -> BaseSubAgent | None:
        """Find the agent for a given stage."""
        for step in self._stages:
            if step.stage == stage:
                return step.agent
        return None

    def _tool_handler_tools(self, handlers: dict, tool_registry=None) -> list[dict]:
        """Build OpenAI function-calling tool definitions from ToolRegistry.

        Uses full ToolContract definitions (name + description + parameters)
        so the LLM knows what each tool expects. Without parameters, the model
        cannot invoke tools correctly.
        """
        tools = []
        if tool_registry:
            for name in tool_registry.list_tools():
                try:
                    contract = tool_registry.get(name)
                    # Build OpenAI-compatible tool definition
                    properties = {}
                    for param_name, param_type in contract.input_schema.items():
                        if isinstance(param_type, list):
                            # Array type
                            item_type = param_type[0] if param_type else "string"
                            properties[param_name] = {
                                "type": "array",
                                "items": {"type": item_type},
                            }
                        else:
                            properties[param_name] = {"type": param_type}
                    tools.append({
                        "type": "function",
                        "function": {
                            "name": contract.name,
                            "description": contract.description,
                            "parameters": {
                                "type": "object",
                                "properties": properties,
                                "required": list(contract.input_schema.keys()),
                            },
                        },
                    })
                except KeyError:
                    # Tool not in registry — skip
                    pass
        else:
            # Fallback: name-only (legacy, may not work well)
            for name in handlers.keys():
                tools.append({"type": "function", "function": {"name": name}})
        return tools

    def _run_with_agent_loop(
        self, step: PipelineStep, message: CollaborationMessage
    ) -> SubAgentResult:
        """Run a pipeline stage inside an AgentLoop for iteration.

        Creates an AgentLoop with the subagent's context and tools,
        lets the LLM iterate through tool calls, then extracts result.
        """
        import json as _json
        import uuid
        from src.application.agent_loop import AgentLoop
        from src.application.context import Role
        from src.application.control_plane import ControlPlane
        from src.application.session import SessionState
        from src.domain.permission import PermissionMode

        agent = step.agent
        stage_name = step.stage.value

        # Extract payload
        payload = message.payload
        task = payload.get("task", {})
        prev_result = payload.get("prev_result", {})
        target_spec = payload.get("target_spec", {})

        # Fix: Planner outputs "tasks" (list) in data, not "task" (singular)
        # Extract tasks list if this is CodeGen stage
        if not task and stage_name == PipelineStage.CODE_GEN.value:
            tasks_list = prev_result.get("data", {}).get("tasks", [])
            if tasks_list:
                task = {"tasks": tasks_list}  # Pass ALL tasks to CodeGen

        # System prompt for the AgentLoop stage
        system_prompt = self._build_stage_system_prompt(agent, stage_name)

        # Build user task description — CRITICAL: route by stage, not by payload
        if stage_name == PipelineStage.PLAN.value:
            # First stage: Planner receives targets
            user_task = (
                f"You are the PLANNER stage. Analyze these GPU profiling targets "
                f"and decompose them into actionable tasks.\n\n"
                f"Targets: {_json.dumps(target_spec, indent=2)}\n\n"
                f"Return a JSON array of task objects with: "
                f'"target", "category" (latency_measurement, capacity_measurement, '
                f'clock_measurement, bandwidth_measurement, or unknown), '
                f'"method" (detailed description of the measurement approach).'
            )
        elif task:
            # Subsequent stages: receive a specific task
            user_task = self._build_task_prompt(task, prev_result, stage_name)
        elif prev_result:
            # No task but has prev_result (MetricAnalysis, Verification)
            user_task = self._build_task_prompt(task, prev_result, stage_name)
        else:
            user_task = f"You are the {stage_name} stage. Execute your task."

        # Create session and control plane
        session_id = f"pipeline_{stage_name}_{uuid.uuid4().hex[:6]}"
        session = SessionState(session_id=session_id, goal=f"Pipeline stage: {stage_name}")
        control_plane = ControlPlane(rule_dir=None)

        # Create AgentLoop for this stage
        loop = AgentLoop(
            session=session,
            context_manager=agent.context_manager,
            control_plane=control_plane,
            tool_registry=agent.tool_registry,
            max_turns=self._max_turns_per_stage,
            state_dir=self._state_dir,
            permission_mode=agent.permission_mode,
        )

        # Wire model caller
        if agent._model_caller is not None:
            loop.set_model_caller(agent._model_caller)
            print(f"[Pipeline] Wired model caller to {stage_name} stage")
        else:
            print(f"[Pipeline] WARNING: No model caller for {stage_name} stage!")

        # Wire tool executor if sandbox available
        if self._sandbox and self._tool_handlers:
            handlers = dict(self._tool_handlers)
            loop.set_tool_executor(
                lambda tool_name, args: handlers[tool_name](args)
            )
            loop.set_available_tools(self._tool_handler_tools(handlers, agent.tool_registry))
            print(f"[Pipeline] Wired {len(handlers)} tool handlers to {stage_name} stage")
        else:
            print(f"[Pipeline] WARNING: No tool handlers for {stage_name} stage!")

        # Auto-approve tool calls in HIGH_AUTONOMY mode (no human in loop)
        if agent.permission_mode == PermissionMode.HIGH_AUTONOMY:
            loop.set_approval_callback(lambda request: True)

        # Set up context and run
        agent.context_manager.add_entry(
            Role.SYSTEM, system_prompt, token_count=50
        )
        agent.context_manager.add_entry(
            Role.USER, user_task, token_count=30
        )

        # Run the loop — iterates until model stops calling tools or max turns
        try:
            print(f"[Pipeline] Starting AgentLoop for {stage_name} (max_turns={self._max_turns_per_stage})")
            loop.start()
            print(f"[Pipeline] AgentLoop finished for {stage_name} (turns={loop.loop_state.turn_count})")
        except Exception as e:
            print(f"[Pipeline] AgentLoop CRASHED in {stage_name}: {e}")
            return SubAgentResult(
                agent_role=agent.role,
                status=SubAgentStatus.FAILED,
                error=f"AgentLoop failed in {stage_name}: {e}",
            )

        # Extract result from context
        return self._extract_stage_result(agent, step.stage, loop)

    def _extract_stage_result(
        self,
        agent: BaseSubAgent,
        stage: PipelineStage,
        loop: "AgentLoop",
    ) -> SubAgentResult:
        """Extract a SubAgentResult from the AgentLoop's final context."""
        import json as _json

        entries = agent.context_manager.get_entries()

        # Collect tool call results and model's text outputs
        tool_results = []
        assistant_outputs = []
        model_prompt_prefix = ("You are the", "Available tools", "Instructions:")

        for entry in entries:
            if entry.role.value != "assistant":
                continue

            # Try to parse as JSON
            try:
                data = _json.loads(entry.content)
                if isinstance(data, dict) and ("status" in data or "tool" in data):
                    tool_results.append(data)
                elif isinstance(data, (dict, list)):
                    # JSON output from model (not a tool result)
                    assistant_outputs.append(entry.content)
            except (_json.JSONDecodeError, TypeError):
                # Plain text from model
                content = entry.content.strip()
                if content and not content.startswith(model_prompt_prefix):
                    assistant_outputs.append(content)

        # Final output = last non-empty assistant text
        final_text = assistant_outputs[-1] if assistant_outputs else ""

        # Build structured result
        data = {
            "tool_results": tool_results,
            "final_output": final_text[:3000],
            "num_tool_calls": len(tool_results),
        }

        # Stage-specific extraction and status determination
        # AgentLoop's placeholder for empty model output — treat as empty
        _empty_placeholders = {
            "[Empty model output - will retry next turn]",
            "[Empty model output]",
        }
        effective_final_text = final_text if final_text not in _empty_placeholders else ""

        if stage == PipelineStage.PLAN:
            data["plan_text"] = effective_final_text[:2000]
            status = SubAgentStatus.SUCCESS if effective_final_text else SubAgentStatus.FAILED
            if not effective_final_text and not tool_results:
                data["error_detail"] = "Planner produced no output"

        elif stage == PipelineStage.CODE_GEN:
            data["code_gen_output"] = final_text[:2000]
            # CodeGen MUST call tools — talking about code is not enough.
            # compile_cuda returns {"success": True, "output": "...", "binary_path": "..."}
            # execute_binary returns {"stdout": "...", "return_code": 0, ...}
            has_compile = any(
                r.get("tool") == "compile_cuda" or r.get("binary_path")
                for r in tool_results
            )
            has_execute = any(
                r.get("tool") == "execute_binary" or r.get("return_code") is not None
                for r in tool_results
            )
            tool_succeeded = any(
                r.get("status") in ("success", True) or
                r.get("success") is True
                for r in tool_results
            )
            has_binary = any(r.get("binary_path") for r in tool_results)
            has_output = any(r.get("stdout") for r in tool_results)
            exec_succeeded = any(
                r.get("return_code", -1) == 0 for r in tool_results
                if "return_code" in r or "stdout" in r
            )
            # CodeGen requires at least one tool call — text-only output is a failure
            if tool_results and (tool_succeeded or has_binary or has_output or exec_succeeded):
                status = SubAgentStatus.SUCCESS
            else:
                status = SubAgentStatus.FAILED
                if not tool_results and not final_text:
                    data["error_detail"] = "CodeGen produced no output and made no tool calls"
                elif not tool_results:
                    data["error_detail"] = (
                        f"CodeGen output was text-only (no tool calls). "
                        f"Model must call compile_cuda to generate benchmarks. "
                        f"Output preview: {final_text[:200]}"
                    )
                elif not tool_succeeded and not has_binary:
                    data["error_detail"] = "CodeGen compilation failed — check tool call results for errors"

            # Extract structured measurements from CodeGen tool results
            # This feeds into _assemble_final_results in main.py
            measurements = {}
            methodology_parts = []
            for r in tool_results:
                if "stdout" in r:
                    stdout = r.get("stdout", "")
                    for line in stdout.splitlines():
                        line = line.strip()
                        if ":" in line and not line.startswith("//"):
                            parts = line.split(":", 1)
                            if len(parts) == 2:
                                key = parts[0].strip()
                                val_str = parts[1].strip()
                                try:
                                    measurements[key] = float(val_str)
                                    continue
                                except ValueError:
                                    pass
                if "output" in r and r.get("output"):
                    out = r["output"]
                    if len(out) < 500:
                        methodology_parts.append(out[:200])
            if measurements:
                data["measurements"] = measurements
            if final_text:
                methodology_parts.append(final_text[:1000])
            if methodology_parts:
                data["analysis_method"] = "\n---\n".join(methodology_parts[:5])

        elif stage == PipelineStage.METRIC_ANALYSIS:
            data["analysis_output"] = final_text[:2000]
            status = SubAgentStatus.SUCCESS if final_text else SubAgentStatus.FAILED

        elif stage == PipelineStage.VERIFICATION:
            data["review_text"] = final_text[:2000]
            lower = final_text.lower()
            # Use word-boundary matching to avoid substring collisions
            # (e.g., "acceptable" contains "accept", "unacceptable" contains "accept")
            import re as _re
            has_accept_word = bool(_re.search(r'\baccept(?:ed|ing)?\b', lower))
            has_reject_word = bool(_re.search(r'\breject(?:ed|ing|ion)?\b', lower))
            has_not_valid = "not valid" in lower or "is not valid" in lower
            has_cannot_accept = "cannot accept" in lower or "do not accept" in lower or "don't accept" in lower
            # Check for explicit verdict statements (highest priority)
            verdict_accept = bool(_re.search(r'\bverdict\s*:\s*accept\b', lower))
            verdict_reject = bool(_re.search(r'\bverdict\s*:\s*reject\b', lower))
            if verdict_reject or has_reject_word or has_not_valid or has_cannot_accept:
                status = SubAgentStatus.REJECTED
            elif verdict_accept or has_accept_word:
                status = SubAgentStatus.SUCCESS
            else:
                # No clear verdict — default to SUCCESS (pipeline continues)
                status = SubAgentStatus.SUCCESS

        else:
            status = SubAgentStatus.SUCCESS

        # Propagate error_detail to result.error for UI visibility
        error_msg = ""
        if status == SubAgentStatus.FAILED:
            error_msg = data.get("error_detail", "")
            if not error_msg and not final_text and not tool_results:
                error_msg = f"Stage {stage.value} produced no output"
            elif not error_msg and final_text:
                # Show a snippet of what the model actually output
                error_msg = f"Stage {stage.value} failed. Output: {final_text[:500]}"
            else:
                error_msg = error_msg or f"Stage {stage.value} failed"

        elif status == SubAgentStatus.REJECTED:
            review = data.get("review", [])
            concerns = data.get("concerns", [])
            review_text = data.get("review_text", "")[:300]
            if concerns:
                error_msg = f"Verification rejected: {'; '.join(concerns)}"
            elif review:
                error_msg = f"Verification rejected: {'; '.join(str(r) for r in review[:5])}"
            elif review_text:
                error_msg = f"Verification rejected: {review_text[:300]}"
            else:
                error_msg = "Verification rejected (no reason provided)"

        result = SubAgentResult(
            agent_role=agent.role,
            status=status,
            data=data,
            artifacts=[],
            error=error_msg or None,
        )

        result.context_fingerprint = result.compute_fingerprint(agent.context_manager)
        self._persister.log_entry(
            "stage_result",
            details={
                "stage": stage.value,
                "status": status.value,
                "tool_calls": len(tool_results),
                "output_length": len(final_text),
            },
        )

        return result

    @classmethod
    def build_default(
        cls,
        planner: BaseSubAgent,
        code_gen: BaseSubAgent,
        metric_analysis: BaseSubAgent,
        verification: BaseSubAgent,
        state_dir: str,
        sandbox=None,
        tool_handlers: dict | None = None,
        max_turns_per_stage: int = 15,
        handoff_validator=None,  # HandoffValidator | None
        circuit_breaker=None,    # CircuitBreaker | None
    ) -> Pipeline:
        """Build a standard pipeline with all 4 agents.

        Each stage runs inside an AgentLoop for LLM-driven iteration.

        Harness engineering:
        - handoff_validator: validates stage outputs against contracts
        - circuit_breaker: detects progressive degradation
        """
        return cls(
            stages=[
                PipelineStep(stage=PipelineStage.PLAN, agent=planner),
                PipelineStep(stage=PipelineStage.CODE_GEN, agent=code_gen),
                PipelineStep(stage=PipelineStage.METRIC_ANALYSIS, agent=metric_analysis),
                PipelineStep(
                    stage=PipelineStage.VERIFICATION,
                    agent=verification,
                    retry_on_failure=0,  # Verification failures are final
                ),
            ],
            state_dir=state_dir,
            sandbox=sandbox,
            tool_handlers=tool_handlers,
            max_turns_per_stage=max_turns_per_stage,
            handoff_validator=handoff_validator,
            circuit_breaker=circuit_breaker,
        )


def _get_design_principle(target: str) -> str:
    """Return design principle for a specific profiling target.

    This is the design thinking injected into the CodeGen agent's prompt.
    The agent generates CUDA code based on these principles — no hardcoded source.
    """
    principles: dict[str, str] = {
        "dram_latency_cycles": (
            "Design: Pointer-chasing with random permutation chains (LCG-seeded Knuth shuffle).\n"
            "- Allocate 32M uint32_t indices (=128 MB, >> any L2), fill with random permutation\n"
            "- Single thread follows chain: idx = next[idx] for 10M iterations\n"
            "- Use clock64() to measure total cycles; latency = cycles / iterations\n"
            "- Why random: hardware prefetchers detect strided patterns; random permutation defeats all prefetching\n"
            "- Run 3 trials, report median cycles/access"
        ),
        "l2_latency_cycles": (
            "Design: Pointer-chasing with 2 MB working set (512K uint32_t).\n"
            "- Working set fits in L2 cache but exceeds L1 on all GPUs\n"
            "- Same random permutation chain approach as DRAM latency\n"
            "- Use clock64(); latency = total_cycles / iterations\n"
            "- Expected: 100-500 cycles"
        ),
        "l1_latency_cycles": (
            "Design: Pointer-chasing with 8 KB working set (2K uint32_t).\n"
            "- Working set fits in all GPU L1 data caches\n"
            "- Use clock64(); latency = total_cycles / iterations\n"
            "- Expected: 50-300 cycles"
        ),
        "l2_cache_size_mb": (
            "Design: Working-set sweep with pointer-chasing at 14 sizes.\n"
            "- Sizes: 1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128 MB\n"
            "- At each size: run pointer-chasing, measure cycles/access with clock64()\n"
            "- Detect 'cliff': the size where latency jumps >3x (L2 miss → DRAM)\n"
            "- L2 size = last size before cliff (typically power of 2: 4, 8, 40, 50, 60, 72 MB)"
        ),
        "actual_boost_clock_mhz": (
            "Design: SM clock cycles / wall-clock time.\n"
            "- Kernel: 10M iterations of random permutation, measure total clock64() cycles\n"
            "- Host timing: cudaEventElapsedTime (GPU-side wall-clock, NOT host clock)\n"
            "- Record events before/after kernel, elapsed_us = cudaEventElapsedTime(stop, start)\n"
            "- freq_MHz = total_cycles / elapsed_us\n"
            "- Run 3 trials, report median frequency\n"
            "- Expected: 1000-2500 MHz"
        ),
        "dram_bandwidth_gbps": (
            "Design: STREAM copy — sequential memory write saturation.\n"
            "- Kernel: dst[i] = src[i] for 32M floats (128 MB)\n"
            "- Launch 65535 blocks to fully saturate memory bandwidth\n"
            "- Use cudaEventElapsedTime for timing (GPU-side, more precise than host)\n"
            "- BW = bytes_copied / elapsed_ns * 1e9 / 1e9 = bytes / ns = GB/s"
        ),
        "max_shmem_per_block_kb": (
            "Design: CUDA occupancy API sweep — no timing needed.\n"
            "- For each shmem size (1K, 2K, 4K, 8K, 16K, 32K, 48K, 64K, 96K, 100K):\n"
            "  Call cudaOccupancyMaxActiveBlocksPerMultiprocessor with dummy kernel\n"
            "- Max shmem where blocks_per_sm > 0 = per-block capacity\n"
            "- This is a direct CUDA API query — most reliable probe in the suite"
        ),
        "bank_conflict_penalty_ratio": (
            "Design: Shared memory bank conflict via strided vs sequential access.\n"
            "- Use cudaEventElapsedTime (NOT clock64()) — bank conflicts are too fast for clock64()\n"
            "- Run TWO separate kernel calls in same program:\n"
            "  (a) Strided: thread t accesses shared_mem[t * 32] — all threads hit bank 0\n"
            "  (b) Sequential: thread t accesses shared_mem[(t + offset) % 256] — one thread per bank\n"
            "- 1 block, 256 threads, __shared__ uint32_t[256]\n"
            "- ratio = strided_ms / sequential_ms (>1.0 means bank conflicts)\n"
            "- Run 3 trials, report minimum ratio (eliminates noise)"
        ),
        "shmem_bandwidth_gbps": (
            "Design: Cooperative shared memory read/write within single block.\n"
            "- 1 block, 256 threads, __shared__ uint32_t[256]\n"
            "- Each iteration: all threads cooperatively read + write shared memory\n"
            "- Use cudaEventElapsedTime for timing\n"
            "- BW = (iterations * 256 threads * 2 ops * 4 bytes) / elapsed_ns GB/s\n"
            "- Note: this measures per-SM bandwidth (shared memory is per-SM resource)"
        ),
        "sm_count": (
            "Design: Multi-strategy SM count detection.\n"
            "- Strategy 1: cudaDeviceGetAttribute for MultiProcessorCount (may be virtualized)\n"
            "- Strategy 2: Launch 1-block kernel that writes blockIdx.x to unique array position,\n"
            "  then count unique block IDs to determine actual SM count\n"
            "- Strategy 3: Use cudaOccupancyMaxActiveBlocksPerMultiprocessor with a tiny kernel\n"
            "  to verify SM count independently\n"
            "- Cross-validate all strategies — if they disagree, report all values\n"
            "- Print all values for cross-validation"
        ),
        "shmem_bank_conflict_penalty_ns": (
            "Design: Measure the latency penalty of shared memory bank conflicts.\n"
            "- Strategy 1: Two-kernel comparison (strided vs sequential access)\n"
            "  (a) Strided kernel: thread t accesses shared_mem[t * 32] — all hit bank 0\n"
            "  (b) Sequential kernel: thread t accesses shared_mem[(t + offset) % 256] — one per bank\n"
            "  penalty_ns = (strided_latency - sequential_latency)\n"
            "- Strategy 2: Vary the number of conflicting threads (2, 4, 8, 16, 32)\n"
            "  to measure how penalty scales with conflict degree\n"
            "- Use cudaEventElapsedTime for timing (bank conflicts are too fast for clock64)\n"
            "- 1 block, 256 threads, __shared__ uint32_t[256]\n"
            "- Run 3 trials, report median penalty in nanoseconds\n"
            "- Cross-validate: penalty should be positive and < 100ns typically"
        ),
        "max_shmem_per_block_kb": (
            "Design: Multi-strategy shared memory capacity detection.\n"
            "- Strategy 1: cudaDeviceGetAttribute for cudaDevAttrMaxSharedMemoryPerBlock (may be virtualized)\n"
            "- Strategy 2: Binary search — try allocating increasing __shared__ memory sizes\n"
            "  until kernel launch fails, then report the largest successful allocation\n"
            "- Strategy 3: Query cudaFuncAttributes with cudaFuncGetAttribute\n"
            "- Cross-validate all strategies\n"
            "- Report in KB"
        ),
        "l1_cache_size_kb": (
            "Design: Working-set sweep to detect L1 cache capacity.\n"
            "- Use pointer-chase kernel with varying working-set sizes (1KB to 256KB)\n"
            "- Plot latency vs size — look for the 'cliff' where latency jumps\n"
            "- The cliff indicates L1 cache capacity\n"
            "- Use clock64() for cycle-level timing\n"
            "- Ensure each access is dependent (pointer chase) to defeat prefetcher\n"
            "- Run 3 trials per size, report median\n"
            "- Cross-validate with cudaDeviceGetAttribute if available"
        ),
    }
    return principles.get(target, (
        f"Design: Custom micro-benchmark for '{target}'.\n"
        "- Write a complete CUDA .cu file with proper includes and main()\n"
        "- Use clock64() for cycle timing, cudaEventElapsedTime for wall-clock timing\n"
        "- Output must be parseable: printf(\"key: value\\n\")\n"
        "- Run multiple trials, report median\n"
        "- ANTI-CHEAT: Do NOT rely solely on cudaGetDeviceProperties or cudaDeviceGetAttribute\n"
        "  These may return virtualized data. Always MEASURE actual hardware behavior.\n"
        "- Use at least 2 measurement strategies and cross-validate results\n"
        "- Report confidence level and methodology used"
    ))
