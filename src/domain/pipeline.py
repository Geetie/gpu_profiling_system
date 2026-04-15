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
                f"\n\n🛠️ YOUR TOOLS: compile_cuda, execute_binary, write_file, read_file\n"
                f"🎯 YOUR JOB: Write CUDA code → compile → execute → report values\n"
                f"❌ DO NOT: run ncu (that's MetricAnalysis's job)\n"
                f"❌ DO NOT: verify results (that's Verification's job)\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⚠️  CRITICAL: FILE PATH FORMAT (READ THIS FIRST)\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"You MUST use the .sandbox directory for ALL file operations.\n\n"
                f"✅ CORRECT: .sandbox/benchmark.cu\n"
                f"✅ CORRECT: .sandbox/test1.cu\n"
                f"❌ WRONG: /kaggle/working/gpu_profiling_system/benchmark.cu (path escape)\n"
                f"❌ WRONG: benchmark.cu (missing .sandbox prefix)\n"
                f"❌ WRONG: './sandbox/benchmark.cu' (incorrect format)\n\n"
                f"⚠️  PATH FORMAT RULES:\n"
                f"1. Use RELATIVE paths starting with .sandbox/\n"
                f"2. Do NOT use absolute paths (/kaggle/...)\n"
                f"3. Do NOT wrap paths in quotes in your code (file_path: .sandbox/benchmark.cu)\n"
                f"4. The system will handle quotes automatically if needed\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📋 MANDATORY WORKFLOW — Process EACH Target Separately\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"For target 1:\n"
                f'  1. compile_cuda(source="...full .cu source for target 1...", flags=["-O3"])\n'
                f'  2. execute_binary(binary_path="<path from compile_cuda>")\n'
                f"  3. Record the measured value from stdout\n\n"
                f"For target 2:\n"
                f'  4. compile_cuda(source="...full .cu source for target 2...", flags=["-O3"])\n'
                f'  5. execute_binary(binary_path="<path from compile_cuda>")\n'
                f"  6. Record the measured value from stdout\n\n"
                f"For target 3:\n"
                f'  7. compile_cuda(source="...full .cu source for target 3...", flags=["-O3"])\n'
                f'  8. execute_binary(binary_path="<path from compile_cuda>")\n'
                f"  9. Record the measured value from stdout\n\n"
                f"⚠️  CRITICAL: compile_cuda OVERWRITES the previous binary each time.\n"
                f"So you MUST execute_binary IMMEDIATELY after each compile_cuda,\n"
                f"before compiling the next target's code.\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🔧 TOOL CALL FORMAT\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"You MUST call tools as JSON objects, one at a time:\n\n"
                f'{{"tool": "write_file", "args": {{"file_path": ".sandbox/benchmark.cu", "content": "...code..."}}}}\n'
                f'{{"tool": "compile_cuda", "args": {{"source": "...full .cu source code...", "flags": ["-O3"]}}}}\n'
                f'{{"tool": "execute_binary", "args": {{"binary_path": "<path from compile_cuda>"}}}}\n\n'
                f"After each tool call, you will see the result. Then call the next tool.\n"
                f"❌ DO NOT just describe what you would do — ACTUALLY CALL the tools.\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🔄 ERROR RECOVERY\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"- Compilation error → fix source code → retry compile_cuda\n"
                f"- Execution error → fix binary path or code → recompile → retry\n"
                f"- Implausible output (0, negative, NaN) → fix measurement logic → retry\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"✅ MANDATORY REQUIREMENT\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"You MUST call compile_cuda and execute_binary for EACH target.\n"
                f"The pipeline will FAIL if you do not call compile_cuda.\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🛡️  ANTI-CHEAT AWARENESS\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"- ❌ Do NOT rely solely on cudaGetDeviceProperties — may return virtualized data\n"
                f"- ✅ Use clock64() + cudaEventElapsedTime to measure actual hardware behavior\n"
                f"- ⚠️  The GPU may be clock-locked at non-standard frequencies\n"
                f"- ⚠️  SM count may be limited via CUDA_VISIBLE_DEVICES or environment variables\n"
                f"- ✅ Always MEASURE rather than QUERY when possible\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 OUTPUT FORMAT\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"After all targets, list results as:\n"
                f"target_name: numeric_value\n"
                f"for each target measured.\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💡 CUDA CODE DESIGN PRINCIPLES\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"For each target, you will receive detailed design principles below.\n"
                f"These principles include:\n"
                f"- 📐 ARCHITECTURAL INSIGHT: Why the measurement works\n"
                f"- 🔬 MEASUREMENT STRATEGY: Step-by-step implementation guide\n"
                f"- ⚠️  ANTI-CHEAT: What to avoid (prefetchers, virtualization, etc.)\n"
                f"- 📊 EXPECTED RANGE: Typical values for validation\n"
                f"- 🔍 VALIDATION: How to check if your measurement is correct\n\n"
                f"STUDY the design principles carefully and implement the CUDA code accordingly.\n"
                f"The design principles are based on proven micro-benchmarking techniques.\n"
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
            "🎯 DESIGN THINKING: DRAM Latency Measurement via Pointer-Chasing\n\n"
            "📐 ARCHITECTURAL INSIGHT:\n"
            "- DRAM latency is exposed when cache hierarchy is completely bypassed\n"
            "- Hardware prefetchers can hide latency by pre-loading data before it's needed\n"
            "- To measure TRUE latency: must defeat ALL prefetchers (streaming + stride)\n\n"
            "🔬 MEASUREMENT STRATEGY:\n"
            "1. Working Set: Allocate 32M uint32_t indices (=128 MB)\n"
            "   - Rationale: >> L2 cache size on ALL GPUs (typically 4-72 MB)\n"
            "   - Ensures every access misses L1, L2, and goes to DRAM\n\n"
            "2. Access Pattern: Random permutation chain (Knuth shuffle with LCG seed)\n"
            "   - CRITICAL: Random access defeats hardware prefetchers\n"
            "   - Strided patterns (arr[i], arr[i+stride]) are detected and prefetched\n"
            "   - Random chains force serial dependency: next = array[current]\n\n"
            "3. Timing Method: clock64() for cycle-accurate measurement\n"
            "   - Single thread: idx = next[idx] for 10M iterations\n"
            "   - Measure total_cycles = clock64(end) - clock64(start)\n"
            "   - latency_cycles = total_cycles / iterations\n\n"
            "4. Statistical Rigor: Run 3 trials, report MEDIAN\n"
            "   - Median eliminates outliers from context switches, thermal throttling\n\n"
            "⚠️ ANTI-CHEAT AWARENESS:\n"
            "- Do NOT use sequential access (prefetcher hides true latency)\n"
            "- Do NOT use cudaGetDeviceProperties — may return virtualized data\n"
            "- Do NOT measure with host timers (PCIe transfer overhead contaminates)\n"
            "- MUST use clock64() — measures GPU core cycles directly\n\n"
            "📊 EXPECTED RANGE: 400-800 cycles (varies by GPU generation, DDR speed)\n"
            "🔍 VALIDATION: If result < 200 cycles → prefetcher succeeded → REWRITE code"
        ),
        "l2_latency_cycles": (
            "🎯 DESIGN THINKING: L2 Cache Latency Measurement via Pointer-Chasing\n\n"
            "📐 ARCHITECTURAL INSIGHT:\n"
            "- L2 latency is exposed when working set fits in L2 but exceeds L1\n"
            "- L2 is typically 512 KB - 72 MB on modern GPUs\n"
            "- L1 is typically 32-128 KB per SM\n\n"
            "🔬 MEASUREMENT STRATEGY:\n"
            "1. Working Set: 2 MB (512K uint32_t indices)\n"
            "   - Rationale: >> L1 cache, << typical L2 cache\n"
            "   - Ensures L1 miss, L2 hit on every access\n\n"
            "2. Access Pattern: Random permutation chain (same as DRAM)\n"
            "   - Defeats L1 hardware prefetchers\n"
            "   - Forces serial dependency chain\n\n"
            "3. Timing: clock64() for cycle accuracy\n"
            "   - Single thread, 10M iterations\n"
            "   - latency_cycles = total_cycles / iterations\n\n"
            "4. Run 3 trials, report MEDIAN\n\n"
            "⚠️ ANTI-CHEAT:\n"
            "- Do NOT use sequential access\n"
            "- Do NOT use working set < 64 KB (may fit in L1)\n"
            "- Do NOT use cudaGetDeviceProperties\n\n"
            "📊 EXPECTED RANGE: 100-500 cycles\n"
            "🔍 VALIDATION: If result < 50 cycles → L1 hit → INCREASE working set"
        ),
        "l1_latency_cycles": (
            "🎯 DESIGN THINKING: L1 Cache Latency Measurement via Pointer-Chasing\n\n"
            "📐 ARCHITECTURAL INSIGHT:\n"
            "- L1 cache is smallest, fastest cache (typically 32-128 KB per SM)\n"
            "- L1 latency is lowest in hierarchy (excluding registers/shared)\n"
            "- Must ensure working set fits ENTIRELY in L1\n\n"
            "🔬 MEASUREMENT STRATEGY:\n"
            "1. Working Set: 8 KB (2K uint32_t indices)\n"
            "   - Rationale: << L1 cache on ALL GPUs\n"
            "   - Ensures every access hits L1\n\n"
            "2. Access Pattern: Random permutation chain\n"
            "   - Same methodology as DRAM/L2 for consistency\n"
            "   - Defeats any streaming prefetchers\n\n"
            "3. Timing: clock64() for cycle accuracy\n"
            "   - Single thread, 10M iterations\n"
            "   - latency_cycles = total_cycles / iterations\n\n"
            "4. Run 3 trials, report MEDIAN\n\n"
            "⚠️ ANTI-CHEAT:\n"
            "- Do NOT use working set > 64 KB (may miss L1)\n"
            "- Do NOT use sequential access\n"
            "- Do NOT use cudaGetDeviceProperties\n\n"
            "📊 EXPECTED RANGE: 50-300 cycles\n"
            "🔍 VALIDATION: If result > 400 cycles → L1 miss → DECREASE working set"
        ),
        "l2_cache_size_mb": (
            "🎯 DESIGN THINKING: L2 Cache Capacity Detection via Working-Set Sweep\n\n"
            "📐 ARCHITECTURAL INSIGHT:\n"
            "- Cache capacity is exposed by measuring latency vs working-set size\n"
            "- At cache capacity, latency jumps dramatically (cache cliff)\n"
            "- L2 miss → DRAM access causes 3-10x latency increase\n\n"
            "🔬 MEASUREMENT STRATEGY:\n"
            "1. Binary Search / Sweep across 14 sizes:\n"
            "   Sizes: 1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128 MB\n"
            "   - Fine granularity near expected L2 sizes (power of 2)\n\n"
            "2. At Each Size:\n"
            "   - Run pointer-chasing (random permutation)\n"
            "   - Measure cycles/access with clock64()\n"
            "   - Run 3 trials, compute median\n\n"
            "3. Detect 'Cache Cliff':\n"
            "   - Find size where latency jumps >3x\n"
            "   - L2 size = last size BEFORE the cliff\n"
            "   - Example: 16MB→200 cycles, 32MB→800 cycles → L2 = 16 MB\n\n"
            "4. Validation:\n"
            "   - L2 size is typically power of 2 or close (4, 8, 16, 24, 32, 40, 48, 64, 72 MB)\n"
            "   - If cliff is between sizes, interpolate\n\n"
            "⚠️ ANTI-CHEAT:\n"
            "- Do NOT use sequential access (prefetcher hides cache misses)\n"
            "- Do NOT use cudaGetDeviceProperties\n"
            "- MUST use random permutation to defeat prefetchers\n\n"
            "📊 EXPECTED RANGE: 4-72 MB (varies by GPU: consumer < datacenter)\n"
            "🔍 VALIDATION: If no clear cliff → INCREASE number of sweep points"
        ),
        "actual_boost_clock_mhz": (
            "🎯 DESIGN THINKING: GPU Boost Clock Measurement via Cycle Counter\n\n"
            "📐 ARCHITECTURAL INSIGHT:\n"
            "- GPU boost clock varies based on thermal headroom, power limits, workload\n"
            "- Reported clock (cudaGetDeviceProperties) may not reflect actual runtime frequency\n"
            "- True frequency = GPU cycles / wall-clock time\n\n"
            "🔬 MEASUREMENT STRATEGY:\n"
            "1. Compute-Intensive Kernel:\n"
            "   - 10M iterations of random permutation (no memory access)\n"
            "   - Ensures GPU is compute-bound, not memory-bound\n"
            "   - Measures core clock, not memory clock\n\n"
            "2. Dual Timing:\n"
            "   - GPU cycles: clock64() — counts GPU core cycles\n"
            "   - Wall time: cudaEventElapsedTime(start, stop) — GPU-side wall clock\n"
            "   - CRITICAL: Use GPU events, NOT host timers (PCIe overhead)\n\n"
            "3. Frequency Calculation:\n"
            "   - freq_MHz = total_cycles / elapsed_microseconds\n"
            "   - Example: 10B cycles / 5000 µs = 2000 MHz\n\n"
            "4. Run 3 trials, report MEDIAN\n\n"
            "⚠️ ANTI-CHEAT:\n"
            "- Do NOT use cudaGetDeviceProperties — returns base/max clock, not actual\n"
            "- Do NOT use host clock (GetTickCount, chrono) — PCIe transfer overhead\n"
            "- MUST use cudaEventElapsedTime for wall-clock timing\n"
            "- Ensure kernel is compute-bound (no global memory access)\n\n"
            "📊 EXPECTED RANGE: 1000-2500 MHz (varies by GPU, thermal state)\n"
            "🔍 VALIDATION: If result < 500 MHz → thermal throttling or measurement error"
        ),
        "dram_bandwidth_gbps": (
            "🎯 DESIGN THINKING: DRAM Bandwidth Measurement via STREAM Copy\n\n"
            "📐 ARCHITECTURAL INSIGHT:\n"
            "- DRAM bandwidth is exposed by saturating memory bus with sequential access\n"
            "- Unlike latency, bandwidth benefits from spatial locality and prefetching\n"
            "- Must launch enough threads to fully utilize memory controller\n\n"
            "🔬 MEASUREMENT STRATEGY:\n"
            "1. Workload: STREAM Copy — dst[i] = src[i]\n"
            "   - Sequential read + sequential write\n"
            "   - 32M floats (128 MB) — >> L2, ensures DRAM traffic\n"
            "   - Bytes transferred: 128 MB read + 128 MB write = 256 MB\n\n"
            "2. Parallelism: Launch 65535 blocks\n"
            "   - Maximizes memory-level parallelism\n"
            "   - Saturates all memory controllers\n\n"
            "3. Timing: cudaEventElapsedTime(start, stop)\n"
            "   - GPU-side timing (avoids PCIe transfer overhead)\n"
            "   - elapsed_ns = elapsed_us * 1000\n"
            "   - BW_GBps = bytes_transferred / elapsed_ns * 1e9\n\n"
            "4. Run 3 trials, report MAXIMUM (bandwidth is capacity, not latency)\n\n"
            "⚠️ ANTI-CHEAT:\n"
            "- Do NOT use host timers — PCIe overhead contaminates measurement\n"
            "- Do NOT use small working set (< 64 MB) — L2 cache may satisfy\n"
            "- Do NOT use random access — reduces effective bandwidth\n"
            "- MUST use cudaEventElapsedTime, NOT clock64()\n\n"
            "📊 EXPECTED RANGE: 200-1000+ GB/s (varies by GPU: consumer < datacenter)\n"
            "🔍 VALIDATION: If result < 100 GB/s → insufficient parallelism or L2 cache hit"
        ),
        "max_shmem_per_block_kb": (
            "🎯 DESIGN THINKING: Shared Memory Capacity via Occupancy API Sweep\n\n"
            "📐 ARCHITECTURAL INSIGHT:\n"
            "- Shared memory is a per-SM resource, partitioned among blocks\n"
            "- Maximum per-block capacity limits how much shared memory one block can use\n"
            "- CUDA occupancy API directly queries this hardware limit\n\n"
            "🔬 MEASUREMENT STRATEGY:\n"
            "1. Binary Search / Sweep across sizes:\n"
            "   Sizes: 1K, 2K, 4K, 8K, 16K, 32K, 48K, 64K, 96K, 100K, 120K, 150K, 180K, 200K bytes\n"
            "   - Fine granularity near common limits (48K, 64K, 96K)\n\n"
            "2. At Each Size:\n"
            "   - Call cudaOccupancyMaxActiveBlocksPerMultiprocessor\n"
            "   - Pass dummy kernel with specified shared memory size\n"
            "   - Record max_blocks_per_sm\n\n"
            "3. Detect Capacity:\n"
            "   - Max shmem where blocks_per_sm > 0 = per-block capacity\n"
            "   - Example: 64K→2 blocks, 65K→0 blocks → max = 64 KB\n\n"
            "4. Cross-Validation (Optional):\n"
            "   - cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlock)\n"
            "   - Compare with occupancy API result\n\n"
            "⚠️ ANTI-CHEAT:\n"
            "- Do NOT rely solely on cudaGetDeviceProperties — may be virtualized\n"
            "- Occupancy API is more reliable (queries actual hardware capability)\n"
            "- MUST test incrementally — do not assume standard values\n\n"
            "📊 EXPECTED RANGE: 48-232 KB (varies by GPU architecture)\n"
            "🔍 VALIDATION: If result < 16 KB → likely error (all modern GPUs support ≥48 KB)"
        ),
        "bank_conflict_penalty_ratio": (
            "🎯 DESIGN THINKING: Shared Memory Bank Conflict Penalty\n\n"
            "📐 ARCHITECTURAL INSIGHT:\n"
            "- Shared memory is divided into 32 banks (typically)\n"
            "- Concurrent accesses to same bank are serialized\n"
            "- Bank conflicts cause significant performance degradation\n\n"
            "🔬 MEASUREMENT STRATEGY:\n"
            "1. Two-Kernel Comparison:\n"
            "   Kernel A (Strided — Bank Conflict):\n"
            "   - thread t accesses shared_mem[t * 32]\n"
            "   - All 256 threads hit bank 0 → fully serialized\n\n"
            "   Kernel B (Sequential — No Conflict):\n"
            "   - thread t accesses shared_mem[(t + offset) % 256]\n"
            "   - One thread per bank → fully parallel\n\n"
            "2. Configuration:\n"
            "   - 1 block, 256 threads\n"
            "   - __shared__ uint32_t[256]\n"
            "   - Each thread: 10K iterations of load + store\n\n"
            "3. Timing: cudaEventElapsedTime (NOT clock64())\n"
            "   - Bank conflicts are too fast for clock64() resolution\n"
            "   - Use GPU events for wall-clock timing\n\n"
            "4. Penalty Calculation:\n"
            "   - ratio = strided_ms / sequential_ms\n"
            "   - ratio > 1.0 indicates bank conflicts\n"
            "   - Typical: 2x-32x slowdown\n\n"
            "5. Run 3 trials, report MINIMUM ratio (eliminates noise)\n\n"
            "⚠️ ANTI-CHEAT:\n"
            "- Do NOT use clock64() — insufficient resolution for shared memory\n"
            "- Do NOT use multiple blocks — shared memory is per-SM\n"
            "- MUST use cudaEventElapsedTime\n\n"
            "📊 EXPECTED RANGE: 2x-32x (varies by conflict degree)\n"
            "🔍 VALIDATION: If ratio < 1.5 → measurement error or incorrect access pattern"
        ),
        "shmem_bandwidth_gbps": (
            "🎯 DESIGN THINKING: Shared Memory Bandwidth Measurement\n\n"
            "📐 ARCHITECTURAL INSIGHT:\n"
            "- Shared memory bandwidth is MUCH higher than DRAM (on-chip)\n"
            "- Shared memory is per-SM resource — measure per-SM bandwidth\n"
            "- Cooperative access within block maximizes utilization\n\n"
            "🔬 MEASUREMENT STRATEGY:\n"
            "1. Configuration:\n"
            "   - 1 block, 256 threads\n"
            "   - __shared__ uint32_t[256]\n"
            "   - Each thread: load + store in each iteration\n\n"
            "2. Workload:\n"
            "   - 10M iterations per thread\n"
            "   - Cooperative read + write: temp = shmem[tid]; shmem[tid] = temp + 1\n"
            "   - Total bytes: iterations * 256 threads * 2 ops * 4 bytes\n\n"
            "3. Timing: cudaEventElapsedTime(start, stop)\n"
            "   - GPU-side wall-clock timing\n"
            "   - BW_GBps = total_bytes / elapsed_ns * 1e9\n\n"
            "4. Run 3 trials, report MAXIMUM\n\n"
            "⚠️ ANTI-CHEAT:\n"
            "- Do NOT use multiple blocks — shared memory is per-SM\n"
            "- Do NOT use DRAM access — contaminates measurement\n"
            "- MUST use cudaEventElapsedTime, NOT clock64()\n\n"
            "📊 EXPECTED RANGE: 5000-20000+ GB/s (on-chip, much higher than DRAM)\n"
            "🔍 VALIDATION: If result < 1000 GB/s → likely using DRAM instead of shared"
        ),
        "sm_count": (
            "🎯 DESIGN THINKING: Multi-Strategy SM Count Detection\n\n"
            "📐 ARCHITECTURAL INSIGHT:\n"
            "- SM (Streaming Multiprocessor) count is fundamental GPU topology\n"
            "- Virtualization (CUDA_VISIBLE_DEVICES, MIG) can hide SMs\n"
            "- cudaGetDeviceProperties may return virtualized count\n"
            "- Must cross-validate with multiple strategies\n\n"
            "🔬 MEASUREMENT STRATEGY:\n"
            "Strategy 1: CUDA API Query\n"
            "   - cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount)\n"
            "   - Fast, but may be virtualized\n\n"
            "Strategy 2: Block ID Sweep (Empirical)\n"
            "   - Launch kernel with 1024 blocks\n"
            "   - Each block writes blockIdx.x to unique array position\n"
            "   - Count unique block IDs executed simultaneously\n"
            "   - Reveals ACTUAL hardware SM count\n\n"
            "Strategy 3: Occupancy API Cross-Validation\n"
            "   - cudaOccupancyMaxActiveBlocksPerMultiprocessor with tiny kernel\n"
            "   - Verify SM count independently\n\n"
            "4. Cross-Validation:\n"
            "   - Compare all 3 strategies\n"
            "   - If they disagree, report ALL values with note\n"
            "   - Empirical (Strategy 2) is most trustworthy\n\n"
            "⚠️ ANTI-CHEAT:\n"
            "- Do NOT rely solely on cudaGetDeviceProperties — may be virtualized\n"
            "- MUST use empirical measurement (Strategy 2)\n"
            "- MUST cross-validate — if strategies disagree, report discrepancy\n\n"
            "📊 EXPECTED RANGE: 8-132 SMs (varies by GPU: consumer < datacenter)\n"
            "🔍 VALIDATION: If empirical < API → virtualization detected → report both"
        ),
        "shmem_bank_conflict_penalty_ns": (
            "🎯 DESIGN THINKING: Shared Memory Bank Conflict Absolute Penalty\n\n"
            "📐 ARCHITECTURAL INSIGHT:\n"
            "- Bank conflicts cause absolute latency increase (nanoseconds)\n"
            "- Penalty is measurable with cudaEventElapsedTime\n"
            "- Penalty scales with number of conflicting threads\n\n"
            "🔬 MEASUREMENT STRATEGY:\n"
            "1. Two-Kernel Comparison:\n"
            "   Kernel A (Strided — Full Conflict):\n"
            "   - thread t accesses shared_mem[t * 32]\n"
            "   - All threads hit bank 0 → fully serialized\n\n"
            "   Kernel B (Sequential — No Conflict):\n"
            "   - thread t accesses shared_mem[(t + offset) % 256]\n"
            "   - One thread per bank → fully parallel\n\n"
            "2. Configuration:\n"
            "   - 1 block, 256 threads\n"
            "   - __shared__ uint32_t[256]\n"
            "   - Each thread: 10K iterations\n\n"
            "3. Timing: cudaEventElapsedTime (NOT clock64())\n"
            "   - strided_latency_ns, sequential_latency_ns\n"
            "   - penalty_ns = strided_latency_ns - sequential_latency_ns\n\n"
            "4. Run 3 trials, report MEDIAN penalty\n\n"
            "5. Optional: Vary Conflict Degree\n"
            "   - Test with 2, 4, 8, 16, 32 threads per bank\n"
            "   - Measure how penalty scales\n\n"
            "⚠️ ANTI-CHEAT:\n"
            "- Do NOT use clock64() — insufficient resolution\n"
            "- Do NOT use multiple blocks\n"
            "- MUST use cudaEventElapsedTime\n\n"
            "📊 EXPECTED RANGE: 10-500 ns (varies by GPU, conflict degree)\n"
            "🔍 VALIDATION: If penalty < 0 ns → measurement error"
        ),
        "l1_cache_size_kb": (
            "🎯 DESIGN THINKING: L1 Cache Capacity Detection via Working-Set Sweep\n\n"
            "📐 ARCHITECTURAL INSIGHT:\n"
            "- L1 cache capacity is exposed by latency vs working-set size\n"
            "- L1 is typically 32-128 KB per SM\n"
            "- At L1 capacity, latency jumps (L1 miss → L2 access)\n\n"
            "🔬 MEASUREMENT STRATEGY:\n"
            "1. Binary Search / Sweep across sizes:\n"
            "   Sizes: 1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256 KB\n"
            "   - Fine granularity near common L1 sizes (32, 48, 64, 96, 128 KB)\n\n"
            "2. At Each Size:\n"
            "   - Run pointer-chasing (random permutation)\n"
            "   - Measure cycles/access with clock64()\n"
            "   - Run 3 trials, compute median\n\n"
            "3. Detect 'L1 Cliff':\n"
            "   - Find size where latency jumps >2x\n"
            "   - L1 size = last size BEFORE the cliff\n"
            "   - Example: 32KB→100 cycles, 64KB→300 cycles → L1 = 32 KB\n\n"
            "4. Cross-Validation:\n"
            "   - cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlock)\n"
            "   - Note: L1 and shared memory may be partitioned on some GPUs\n\n"
            "⚠️ ANTI-CHEAT:\n"
            "- Do NOT use sequential access (prefetcher hides cache misses)\n"
            "- Do NOT use cudaGetDeviceProperties as sole source\n"
            "- MUST use random permutation to defeat prefetchers\n\n"
            "📊 EXPECTED RANGE: 32-128 KB (varies by GPU architecture)\n"
            "🔍 VALIDATION: If no clear cliff → INCREASE sweep points or check L1/shared partition"
        ),
    }
    return principles.get(target, (
        f"🎯 DESIGN THINKING: Custom Micro-Benchmark for '{target}'\n\n"
        "📐 ARCHITECTURAL INSIGHT:\n"
        "- Analyze the hardware resource being measured\n"
        "- Identify potential interference (cache, prefetchers, virtualization)\n"
        "- Design measurement to isolate target behavior\n\n"
        "🔬 MEASUREMENT STRATEGY:\n"
        "1. Write a complete CUDA .cu file with proper includes and main()\n"
        "2. Choose appropriate timing method:\n"
        "   - clock64() for cycle-accurate timing (cache latency, compute)\n"
        "   - cudaEventElapsedTime for wall-clock timing (bandwidth, kernel execution)\n"
        "3. Use multiple trials (minimum 3) and report MEDIAN for latency, MAXIMUM for bandwidth\n"
        "4. Output must be parseable: printf(\"key: value\\n\")\n\n"
        "⚠️ ANTI-CHEAT REQUIREMENTS:\n"
        "- Do NOT rely solely on cudaGetDeviceProperties or cudaDeviceGetAttribute\n"
        "  These may return virtualized data in cloud/containerized environments\n"
        "- ALWAYS measure actual hardware behavior using:\n"
        "  - clock64() for GPU core cycles\n"
        "  - cudaEventElapsedTime for GPU wall-clock time\n"
        "  - Empirical measurement (block ID sweep, pointer-chasing)\n"
        "- Use at least 2 measurement strategies and cross-validate results\n"
        "- Report confidence level and methodology used\n\n"
        "📊 EXPECTED RANGE: Research typical values for target hardware\n"
        "🔍 VALIDATION: If result is outside expected range, REWRITE measurement code"
    ))
