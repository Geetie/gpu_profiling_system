"""Pipeline orchestrator — domain layer.

Coordinates the sequential execution of sub-agents with P7 enforcement,
retry support, and persistence hooks.
"""
from __future__ import annotations

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

    P7 enforcement:
    - VerificationAgent must have an empty ContextManager before execution.
    - The pipeline checks this explicitly and raises P7ViolationError.
    """

    def __init__(
        self,
        stages: list[PipelineStep],
        state_dir: str,
    ) -> None:
        self._stages = stages
        self._state_dir = state_dir
        self._persister = StatePersister(log_dir=state_dir, filename="pipeline_log.jsonl")

    def run(self, target_spec: dict[str, Any]) -> SubAgentResult:
        """Execute the full pipeline.

        Args:
            target_spec: The target specification (from target_spec.json).

        Returns:
            The final SubAgentResult from the VERIFICATION stage,
            or a FAILED result if any stage fails permanently.
        """
        self._persister.log_entry("pipeline_start", details={"target_spec": target_spec})

        prev_result: SubAgentResult | None = None

        for step in self._stages:
            # P7 gate
            self._check_p7(step.stage, prev_result)

            # Execute with retries
            result = self._execute_stage(step, prev_result, target_spec)

            if result.is_failed():
                self._persister.log_entry(
                    "pipeline_stage_failed",
                    details={
                        "stage": step.stage.value,
                        "error": result.error,
                    },
                )
                return result

            prev_result = result

        # Persist final result
        if prev_result:
            self._persister.log_entry("pipeline_complete", details=prev_result.to_dict())

        return prev_result or SubAgentResult(
            agent_role=AgentRole.PLANNER,
            status=SubAgentStatus.FAILED,
            error="Pipeline produced no result",
        )

    def _execute_stage(
        self, step: PipelineStep, prev_result: SubAgentResult | None,
        target_spec: dict[str, Any] | None = None,
    ) -> SubAgentResult:
        """Execute a single pipeline stage with retry support."""
        self._persister.log_entry(
            "pipeline_stage_start",
            details={"stage": step.stage.value, "retry_limit": step.retry_on_failure},
        )
        last_result: SubAgentResult | None = None

        for attempt in range(1 + step.retry_on_failure):
            if attempt > 0:
                self._persister.log_entry(
                    "pipeline_retry",
                    details={"stage": step.stage.value, "attempt": attempt},
                )

            # Build the collaboration message
            payload: dict[str, Any] = {}
            if prev_result is not None:
                payload["prev_result"] = prev_result.to_dict()
                # Pass only the fingerprint, NOT the full context (P7)
                payload["prev_fingerprint"] = prev_result.context_fingerprint
            else:
                # First stage: pass the target_spec for the planner
                payload["target_spec"] = target_spec

            message = CollaborationMessage(
                sender=prev_result.agent_role if prev_result else AgentRole.PLANNER,
                receiver=step.agent.role,
                message_type="task_dispatch",
                payload=payload,
            )

            last_result = step.agent.run(message)

            if last_result.is_success():
                break

            # REJECTED is a semantic rejection — do not retry it
            if last_result.status == SubAgentStatus.REJECTED:
                break

            # Record failure for audit
            self._persister.log_entry(
                "pipeline_attempt_failed",
                details={
                    "stage": step.stage.value,
                    "attempt": attempt + 1,
                    "error": last_result.error,
                },
            )

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

    def _get_stage_agent(self, stage: PipelineStage) -> BaseSubAgent | None:
        """Find the agent for a given stage."""
        for step in self._stages:
            if step.stage == stage:
                return step.agent
        return None

    @classmethod
    def build_default(
        cls,
        planner: BaseSubAgent,
        code_gen: BaseSubAgent,
        metric_analysis: BaseSubAgent,
        verification: BaseSubAgent,
        state_dir: str,
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
                    retry_on_failure=0,  # Verification failures are final
                ),
            ],
            state_dir=state_dir,
        )
