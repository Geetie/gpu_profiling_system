"""Pipeline orchestrator — domain layer (refactored).

Thin mediator that coordinates StageExecutor, StageTransitionGuard,
and PipelineContext.  All heavy logic has been extracted into
dedicated components:

- StageExecutor: retry loop, AgentLoop wiring, result extraction
- StageTransitionGuard: P7, handoff validation, circuit breaker
- PipelineContext: mutable inter-stage state
- StagePromptBuilder: prompt construction (in prompt_builder.py)
- Design principles: in design_principles.py

The Pipeline class now only orchestrates the high-level flow:
  for each stage → guard.check → executor.execute → context.update
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.domain.pipeline_context import PipelineContext
from src.domain.stage_executor import StageExecutor
from src.domain.stage_transition_guard import StageTransitionGuard
from src.domain.subagent import (
    AgentRole,
    BaseSubAgent,
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

    This class acts as a mediator — it delegates to:
    - StageTransitionGuard for boundary checks
    - StageExecutor for actual stage execution
    - PipelineContext for inter-stage data flow
    """

    def __init__(
        self,
        stages: list[PipelineStep],
        state_dir: str,
        sandbox=None,
        tool_handlers: dict | None = None,
        max_turns_per_stage: int = 15,
        handoff_validator=None,
        circuit_breaker=None,
    ) -> None:
        self._stages = stages
        self._state_dir = state_dir
        self._persister = StatePersister(log_dir=state_dir, filename="pipeline_log.jsonl")
        self._sandbox = sandbox
        self._tool_handlers = tool_handlers or {}
        self._max_turns_per_stage = max_turns_per_stage
        self._handoff_validator = handoff_validator
        self._circuit_breaker = circuit_breaker

        self._guard = StageTransitionGuard(
            handoff_validator=handoff_validator,
            circuit_breaker=circuit_breaker,
            persister=self._persister,
        )
        self._executor = StageExecutor(
            state_dir=state_dir,
            sandbox=sandbox,
            tool_handlers=tool_handlers,
            max_turns_per_stage=max_turns_per_stage,
            persister=self._persister,
        )

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

        ctx = PipelineContext(target_spec=target_spec)

        stage_idx = 0
        while stage_idx < len(self._stages):
            step = self._stages[stage_idx]
            stage_start = time.monotonic()
            print(f"[Pipeline] Executing stage: {step.stage.value}")

            guard_decision = self._guard.check_before_stage(step.stage, _GuardContext(ctx, self._stages))
            if not guard_decision.allowed:
                if guard_decision.result is not None:
                    return guard_decision.result
                raise P7ViolationError(guard_decision.reason or "Transition guard blocked stage")

            result = self._executor.execute(step, ctx)

            stage_duration = time.monotonic() - stage_start
            print(f"[Pipeline] Stage {step.stage.value} completed in {round(stage_duration, 2)}s with status: {result.status.value}")

            ctx.record_stage_duration(step.stage.value, stage_duration, ctx.iteration_count)

            if step.stage == PipelineStage.CODE_GEN and ctx.is_optimization_round:
                ctx.clear_optimization_targets()
                print(f"[Pipeline] CodeGen optimization round completed — cleared optimization targets")

            if result.is_failed() and result.status != SubAgentStatus.REJECTED:
                result = self._handle_failure(step, result, ctx, stage_duration)
                if result.is_failed() and not self._can_continue_with_partial(step, result):
                    if step.stage == PipelineStage.CODE_GEN:
                        measurements = result.data.get("measurements", {})
                        if isinstance(measurements, dict) and len(measurements) > 0:
                            print(f"[Pipeline] CodeGen FAILED but has {len(measurements)} measurements — converting to PARTIAL and continuing")
                            result.status = SubAgentStatus.PARTIAL
                            result.data["completion_rate"] = len(measurements) / max(len(ctx.target_spec.get("targets", [])), 1)
                            ctx.update(step.stage, result)
                            stage_idx += 1
                            continue
                    result = ctx.assemble_final_result(result)
                    print(f"[Pipeline] Final result assembled with {len(result.data.get('measurements', {}))} measurements (after failure)")
                    return result

            self._guard.check_after_stage(step.stage, result)

            if result.status == SubAgentStatus.REJECTED and step.stage == PipelineStage.VERIFICATION:
                concerns = result.data.get("concerns", [])
                suggested_fixes = result.data.get("suggested_fixes", [])

                if not concerns:
                    review_text = result.data.get("review_text", "")
                    if review_text:
                        concerns = [line.strip() for line in review_text.splitlines()
                                    if line.strip() and any(kw in line.lower()
                                        for kw in ("missing", "zero", "invalid", "incorrect",
                                                   "failed", "error", "concern", "problem",
                                                   "not valid", "cannot accept", "reject"))]

                if not concerns:
                    error_detail = result.data.get("error_detail", "")
                    if error_detail:
                        concerns = [error_detail]

                if not suggested_fixes:
                    suggested_fixes = result.data.get("suggested_fixes", [])
                    if not suggested_fixes:
                        findings = result.data.get("findings", [])
                        if findings:
                            suggested_fixes = [str(f) for f in findings]

                ctx.add_rejection(step.stage.value, concerns, suggested_fixes)

                print(
                    f"[Pipeline] DEBUG: Verification REJECTED — "
                    f"iteration_count={ctx.iteration_count}, max_iterations={ctx.max_iterations}, "
                    f"can_retry={ctx.can_retry()}, concerns={len(concerns)}, "
                    f"suggested_fixes={len(suggested_fixes)}"
                )

                if ctx.can_retry():
                    ctx.increment_iteration()
                    print(
                        f"[Pipeline] Verification REJECTED (iteration {ctx.iteration_count}/{ctx.max_iterations}). "
                        f"Retrying from CodeGen with feedback."
                    )
                    self._persister.log_entry("pipeline_retry_iteration", details={
                        "iteration": ctx.iteration_count,
                        "concerns": concerns,
                    })
                    code_gen_idx = self._find_stage_index(PipelineStage.CODE_GEN)
                    if code_gen_idx is not None:
                        stage_idx = code_gen_idx
                        # FIX: Set prev_result and prev_stage to Verification for proper handoff validation
                        ctx.prev_result = result
                        ctx.prev_stage = PipelineStage.VERIFICATION
                        continue
                else:
                    print(
                        f"[Pipeline] Verification REJECTED and max iterations ({ctx.max_iterations}) reached. "
                        f"Pipeline terminating."
                    )
                    self._persister.log_entry("pipeline_max_iterations", details={
                        "iteration": ctx.iteration_count,
                        "concerns": concerns,
                    })
                    result = ctx.assemble_final_result(result)
                    print(f"[Pipeline] Final result assembled with {len(result.data.get('measurements', {}))} measurements (after rejection)")
                    return result

            ctx.update(step.stage, result)
            if step.stage == PipelineStage.METRIC_ANALYSIS:
                ctx.bubble_codegen_data(result)

                metric_data = result.data if result.data else {}
                if not metric_data and hasattr(result, 'payload') and result.payload:
                    metric_data = result.payload

                if metric_data:
                    suggested_fixes = metric_data.get("suggested_fixes", [])
                    bottleneck_type = metric_data.get("bottleneck_type", "")
                    bottleneck_sub_type = metric_data.get("bottleneck_sub_type", "")
                    recommendations = metric_data.get("recommendations", [])

                    if suggested_fixes or recommendations:
                        ctx.add_metric_feedback(
                            suggested_fixes=suggested_fixes,
                            bottleneck_type=bottleneck_type,
                            bottleneck_sub_type=bottleneck_sub_type,
                            recommendations=recommendations,
                        )
                        self._persister.log_entry("metric_feedback_collected", details={
                            "bottleneck_type": bottleneck_type,
                            "bottleneck_sub_type": bottleneck_sub_type,
                            "fixes_count": len(suggested_fixes),
                            "stage_success": result.is_success(),
                        })

                # ── CodeGen ↔ MetricAnalysis iterative optimization loop ──
                # When MetricAnalysis identifies bottlenecks or optimization
                # opportunities, loop back to CodeGen for another round of
                # code generation with the feedback injected.
                metric_feedback = ctx.get_feedback_for_codegen()
                opt_targets = ctx.get_optimization_targets() if hasattr(ctx, 'get_optimization_targets') else []
                if metric_feedback and ctx.can_retry():
                    bottleneck_type = metric_feedback.get("bottleneck_type", "")
                    is_balanced = bottleneck_type == "balanced"
                    has_actionable_feedback = bool(
                        (metric_feedback.get("metric_suggested_fixes")
                         or metric_feedback.get("metric_recommendations"))
                        and not is_balanced
                    )
                    metric_opt_count = sum(1 for mf in ctx.metric_feedback if mf.get("source") == "metric_analysis")
                    MAX_METRIC_OPTIMIZATION_ROUNDS = 3  # Allow up to 3 optimization rounds
                    if metric_opt_count >= MAX_METRIC_OPTIMIZATION_ROUNDS:
                        print(
                            f"[Pipeline] MetricAnalysis optimization loop limit reached "
                            f"({metric_opt_count}/{MAX_METRIC_OPTIMIZATION_ROUNDS} rounds). Proceeding to Verification."
                        )
                        self._persister.log_entry("metric_optimization_limit", details={
                            "bottleneck_type": bottleneck_type,
                            "metric_opt_count": metric_opt_count,
                        })
                    elif has_actionable_feedback and ctx.is_optimization_converged():
                        cg_durs = {k: v for k, v in ctx._stage_durations.items() if "code_gen" in k}
                        print(
                            f"[Pipeline] Optimization converged — same bottleneck "
                            f"('{bottleneck_type}') with no concrete fixes for 2 consecutive rounds, "
                            f"or diminishing returns detected. CodeGen durations: {cg_durs}"
                            f"Proceeding to Verification."
                        )
                        self._persister.log_entry("optimization_converged", details={
                            "bottleneck_type": bottleneck_type,
                            "iteration": ctx.iteration_count,
                        })
                    elif has_actionable_feedback:
                        ctx.increment_iteration()
                        ctx.is_optimization_round = True
                        opt_target_names = [t.get("target", "?") for t in opt_targets] if opt_targets else []
                        print(
                            f"[Pipeline] MetricAnalysis found optimization opportunities "
                            f"(iteration {ctx.iteration_count}/{ctx.max_iterations}). "
                            f"Looping back to CodeGen for optimization."
                            + (f" Optimization targets: {opt_target_names}" if opt_target_names else "")
                        )
                        self._persister.log_entry("metric_optimization_loop", details={
                            "iteration": ctx.iteration_count,
                            "bottleneck_type": metric_feedback.get("bottleneck_type", ""),
                            "fixes": metric_feedback.get("metric_suggested_fixes", []),
                            "optimization_targets": opt_target_names,
                        })
                        code_gen_idx = self._find_stage_index(PipelineStage.CODE_GEN)
                        if code_gen_idx is not None:
                            guard_decision = self._guard.check_before_stage(
                                PipelineStage.CODE_GEN, _GuardContext(ctx, self._stages)
                            )
                            if not guard_decision.allowed:
                                print(
                                    f"[Pipeline] Stage guard blocked CODE_GEN re-entry: "
                                    f"{guard_decision.reason or 'unknown reason'}. "
                                    f"Proceeding to Verification."
                                )
                            else:
                                cg_step = self._stages[code_gen_idx]
                                if cg_step.retry_on_failure < 1:
                                    cg_step.retry_on_failure = 1
                                stage_idx = code_gen_idx
                                # FIX: Set prev_result and prev_stage to MetricAnalysis for proper handoff validation
                                ctx.prev_result = result
                                ctx.prev_stage = PipelineStage.METRIC_ANALYSIS
                                continue
                    else:
                        if is_balanced:
                            print("[Pipeline] MetricAnalysis reports code is well-balanced — no optimization needed")
                        else:
                            print("[Pipeline] MetricAnalysis completed — no actionable optimization feedback")
                elif metric_feedback and not ctx.can_retry():
                    print(
                        f"[Pipeline] MetricAnalysis has feedback but max iterations "
                        f"({ctx.max_iterations}) reached. Proceeding to Verification."
                    )
            elif step.stage == PipelineStage.VERIFICATION:
                ctx.bubble_codegen_data(result)

            stage_idx += 1

        final_result = ctx.prev_result
        # CRITICAL FIX: Always assemble final result, regardless of status
        # This ensures measurements from CodeGen are included even if downstream stages fail
        if final_result is not None:
            # VERSION CONTROL: Rollback to best version if current is degraded
            if hasattr(ctx, 'measurement_versions') and ctx.measurement_versions:
                current_idx = ctx.current_version_idx
                if 0 <= current_idx < len(ctx.measurement_versions):
                    current_ver = ctx.measurement_versions[current_idx]
                    if not current_ver.quality_ok:
                        print(
                            f"[Pipeline] Current measurement version #{current_idx} has quality_ok=False "
                            f"(quality_score={current_ver.quality_score:.1f}, "
                            f"improvement_score={current_ver.improvement_score:.1f}). "
                            f"Rolling back to best version."
                        )
                        best = ctx.rollback_to_best_version()
                        ctx.key_measurements = best
                        print(f"[Pipeline] Rolled back key_measurements to best version")

            final_result = ctx.assemble_final_result(final_result)
            print(f"[Pipeline] Final result assembled with {len(final_result.data.get('measurements', {}))} measurements")

        if final_result:
            self._persister.log_entry("pipeline_complete", details=final_result.to_dict())
            print(f"[Pipeline] Pipeline completed with status: {final_result.status.value}")
        else:
            print("[Pipeline] Pipeline produced no result")

        return final_result or SubAgentResult(
            agent_role=AgentRole.PLANNER,
            status=SubAgentStatus.FAILED,
            error="Pipeline produced no result",
        )

    def _handle_failure(
        self,
        step: PipelineStep,
        result: SubAgentResult,
        ctx: PipelineContext,
        stage_duration: float,
    ) -> SubAgentResult:
        """Handle a failed stage — log and optionally convert to partial."""
        self._persister.log_entry(
            "pipeline_stage_failed",
            details={
                "stage": step.stage.value,
                "error": result.error,
                "duration_seconds": round(stage_duration, 2),
            },
        )
        print(f"[Pipeline] Stage {step.stage.value} failed: {result.error}")
        return result

    @staticmethod
    def _can_continue_with_partial(step: PipelineStep, result: SubAgentResult) -> bool:
        """Check if pipeline can continue with partial results from a failed stage."""
        if step.stage not in (PipelineStage.CODE_GEN, PipelineStage.METRIC_ANALYSIS):
            return False

        has_useful_data = bool(result.data and result.data.get("measurements"))
        has_compiled_binary = any(
            isinstance(tr, dict) and tr.get("binary_path")
            for tr in result.data.get("tool_results", [])
        )
        # CRITICAL FIX: MetricAnalysis may fail status but still have useful NCU data
        # Check for tool_results with NCU output or analysis_output
        has_analysis_output = bool(result.data and result.data.get("analysis_output"))
        has_ncu_results = bool(result.data and result.data.get("tool_results"))
        return has_useful_data or has_compiled_binary or has_analysis_output or has_ncu_results

    def _find_stage_index(self, stage: PipelineStage) -> int | None:
        """Find the index of a stage in the pipeline steps list."""
        for i, step in enumerate(self._stages):
            if step.stage == stage:
                return i
        return None

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
        handoff_validator=None,
        circuit_breaker=None,
    ) -> Pipeline:
        """Build a standard pipeline with all 4 agents."""
        return cls(
            stages=[
                PipelineStep(stage=PipelineStage.PLAN, agent=planner),
                PipelineStep(stage=PipelineStage.CODE_GEN, agent=code_gen),
                PipelineStep(stage=PipelineStage.METRIC_ANALYSIS, agent=metric_analysis),
                PipelineStep(
                    stage=PipelineStage.VERIFICATION,
                    agent=verification,
                    retry_on_failure=0,
                ),
            ],
            state_dir=state_dir,
            sandbox=sandbox,
            tool_handlers=tool_handlers,
            max_turns_per_stage=max_turns_per_stage,
            handoff_validator=handoff_validator,
            circuit_breaker=circuit_breaker,
        )


class _GuardContext:
    """Adapter that provides the guard with access to pipeline state.

    The guard needs prev_result, prev_stage, and the step list
    to find agents.  This adapter avoids exposing the full Pipeline.
    """

    def __init__(self, ctx: PipelineContext, steps: list[PipelineStep]) -> None:
        self.prev_result = ctx.prev_result
        self.prev_stage = ctx.prev_stage
        self.steps = steps
