"""Stage transition guards — boundary validation between pipeline stages.

Encapsulates P7 verification, handoff validation, and circuit breaker
checks into a single guard object.  The Pipeline queries this guard
before and after each stage transition.

Design pattern: Chain of Responsibility — each guard check is independent
and can short-circuit the transition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.domain.subagent import PipelineStage, P7ViolationError, SubAgentResult, SubAgentStatus


@dataclass
class GuardDecision:
    """Result of a transition guard check."""

    allowed: bool
    reason: str | None = None
    result: SubAgentResult | None = None


class StageTransitionGuard:
    """Validates pipeline stage transitions.

    Three independent checks (in order):
    1. P7 gate: VerificationAgent must have empty context
    2. Handoff validation: stage output meets next stage's contract
    3. Circuit breaker: pipeline not in degraded state

    Each check can veto the transition independently.
    """

    def __init__(
        self,
        handoff_validator: Any | None = None,
        circuit_breaker: Any | None = None,
        persister: Any | None = None,
    ) -> None:
        self._handoff_validator = handoff_validator
        self._circuit_breaker = circuit_breaker
        self._persister = persister

    def check_before_stage(
        self,
        current_stage: PipelineStage,
        ctx: Any,
    ) -> GuardDecision:
        """Run all pre-stage guard checks.

        Returns GuardDecision(allowed=False) if any check vetoes.
        """
        p7 = self._check_p7(current_stage, ctx)
        if not p7.allowed:
            return p7

        handoff = self._check_handoff(current_stage, ctx)
        if not handoff.allowed:
            return handoff

        breaker = self._check_circuit_breaker(current_stage)
        if not breaker.allowed:
            return breaker

        return GuardDecision(allowed=True)

    def check_after_stage(
        self,
        stage: PipelineStage,
        result: SubAgentResult,
    ) -> None:
        """Score stage quality on circuit breaker after execution."""
        if self._circuit_breaker is None or self._handoff_validator is None:
            return

        handoff = self._handoff_validator.validate(stage, stage, result)
        self._circuit_breaker.score_stage(
            stage=stage.value,
            handoff_errors=len(handoff.errors),
            handoff_warnings=len(handoff.warnings),
            had_output=bool(result.data.get("final_output")),
            tool_calls_made=result.data.get("num_tool_calls", 0),
        )

    def _check_p7(self, current_stage: PipelineStage, ctx: Any) -> GuardDecision:
        """P7 gate: VerificationAgent must have clean context."""
        if current_stage != PipelineStage.VERIFICATION:
            return GuardDecision(allowed=True)

        verify_agent = self._find_stage_agent(ctx, PipelineStage.VERIFICATION)
        if verify_agent is None:
            return GuardDecision(allowed=True)

        tokens = verify_agent.context_manager.total_tokens
        if tokens > 0:
            return GuardDecision(
                allowed=False,
                reason=(
                    f"P7 violation: VerificationAgent has non-empty context "
                    f"({tokens} tokens). Verification must not inherit generation context."
                ),
            )

        if self._persister:
            self._persister.log_entry(
                "p7_audit",
                details={
                    "generation_fingerprint": (
                        ctx.prev_result.context_fingerprint if ctx.prev_result else None
                    ),
                    "verification_context_tokens": 0,
                    "status": "clean",
                },
            )

        return GuardDecision(allowed=True)

    def _check_handoff(self, current_stage: PipelineStage, ctx: Any) -> GuardDecision:
        """Validate handoff from previous stage."""
        if ctx.prev_stage is None or self._handoff_validator is None or ctx.prev_result is None:
            return GuardDecision(allowed=True)

        handoff = self._handoff_validator.validate(
            ctx.prev_stage, current_stage, ctx.prev_result
        )

        if self._persister:
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
            error_messages = []
            for v in handoff.errors:
                if self._persister:
                    self._persister.log_entry(
                        "handoff_error",
                        details={
                            "stage": v.stage,
                            "field": v.field,
                            "message": v.message,
                            "severity": v.severity,
                        },
                    )
                error_messages.append(f"  - [{v.severity}] {v.field}: {v.message}")
            print(
                f"[Pipeline] Handoff validation BLOCKED transition "
                f"{handoff.from_stage}→{handoff.to_stage}: "
                f"{len(handoff.errors)} error(s), {len(handoff.warnings)} warning(s)"
            )
            for msg in error_messages:
                print(msg)
            return GuardDecision(
                allowed=False,
                reason=(
                    f"Handoff validation failed ({len(handoff.errors)} errors): "
                    f"; ".join(v.message for v in handoff.errors[:3])
                ),
            )

        if handoff.warnings and self._persister:
            for v in handoff.warnings:
                self._persister.log_entry(
                    "handoff_warning",
                    details={
                        "stage": v.stage,
                        "field": v.field,
                        "message": v.message,
                    },
                )

        return GuardDecision(allowed=True)

    def _check_circuit_breaker(self, current_stage: PipelineStage) -> GuardDecision:
        """Check if circuit breaker has tripped."""
        if self._circuit_breaker is None or not self._circuit_breaker.is_open:
            return GuardDecision(allowed=True)

        reason = self._circuit_breaker._state.trip_reason
        return GuardDecision(
            allowed=False,
            reason=f"Circuit breaker open: {reason}",
            result=SubAgentResult(
                agent_role=current_stage.to_agent_role() if hasattr(current_stage, 'to_agent_role') else None,
                status=SubAgentStatus.FAILED,
                error=f"Circuit breaker open: {reason}",
            ),
        )

    @staticmethod
    def _find_stage_agent(ctx: Any, stage: PipelineStage) -> Any | None:
        """Find the agent for a given stage in the pipeline steps."""
        for step in ctx.steps:
            if step.stage == stage:
                return step.agent
        return None
