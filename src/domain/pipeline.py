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

            if result.is_failed():
                result = self._handle_failure(step, result, ctx, stage_duration)
                if result.is_failed() and not self._can_continue_with_partial(step, result):
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
                        ctx.prev_result = None
                        ctx.prev_stage = None
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
                    return result

            ctx.update(step.stage, result)
            if step.stage == PipelineStage.METRIC_ANALYSIS:
                ctx.bubble_codegen_data(result)

                if result.is_success() and result.data:
                    suggested_fixes = result.data.get("suggested_fixes", [])
                    bottleneck_type = result.data.get("bottleneck_type", "")
                    bottleneck_sub_type = result.data.get("bottleneck_sub_type", "")
                    recommendations = result.data.get("recommendations", [])

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
                            "recommendations_count": len(recommendations),
                        })

            stage_idx += 1

        final_result = ctx.prev_result
        if final_result is not None and final_result.is_success():
            final_result = ctx.assemble_final_result(final_result)

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
        return has_useful_data or has_compiled_binary

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
