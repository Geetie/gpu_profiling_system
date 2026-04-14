"""Harness Engineering: Circuit Breaker for Pipeline Degradation.

M4 (防死循环) covers repeated identical failures. The circuit breaker
complements it by detecting *progressive* degradation: stages that
succeed but produce increasingly poor-quality output.

Harness principle: "Don't keep running a broken pipeline — stop fast,
diagnose the root cause."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    HALF_OPEN = "half_open"  # Testing if recovery is possible
    OPEN = "open"           # Circuit tripped — stop pipeline


@dataclass
class StageQuality:
    """Quality score for a single pipeline stage execution."""
    stage: str
    score: float            # 0.0 (worst) → 1.0 (best)
    reasons: list[str] = field(default_factory=list)


@dataclass
class CircuitBreakerState:
    """Mutable state of the circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    quality_history: list[StageQuality] = field(default_factory=list)
    consecutive_degraded: int = 0
    degradation_threshold: int = 3  # Trip after N consecutive degraded stages
    min_quality_threshold: float = 0.3  # Below this = "degraded"
    trip_reason: str | None = None


class CircuitBreaker:
    """Monitors pipeline quality and trips when degradation is detected.

    Quality scoring is based on harness signals:
    - Stage produced empty output: 0.0
    - Stage had handoff warnings: 0.5
    - Stage had handoff errors: 0.0
    - Stage succeeded cleanly: 1.0
    """

    def __init__(
        self,
        degradation_threshold: int = 3,
        min_quality_threshold: float = 0.3,
    ) -> None:
        self._state = CircuitBreakerState(
            degradation_threshold=degradation_threshold,
            min_quality_threshold=min_quality_threshold,
        )

    @property
    def state(self) -> CircuitState:
        return self._state.state

    @property
    def is_open(self) -> bool:
        return self._state.state == CircuitState.OPEN

    def score_stage(
        self,
        stage: str,
        handoff_errors: int = 0,
        handoff_warnings: int = 0,
        had_output: bool = True,
        tool_calls_made: int = 0,
    ) -> StageQuality:
        """Score a stage's quality and update circuit breaker state."""
        score = 1.0
        reasons: list[str] = []

        if not had_output:
            score = 0.0
            reasons.append("No output produced")
        elif handoff_errors > 0:
            score = max(0.0, 1.0 - (handoff_errors * 0.4))
            reasons.append(f"{handoff_errors} handoff errors")
        elif handoff_warnings > 0:
            score = max(0.3, 1.0 - (handoff_warnings * 0.15))
            reasons.append(f"{handoff_warnings} handoff warnings")

        if tool_calls_made == 0 and stage in ("CODE_GEN", "METRIC_ANALYSIS"):
            score *= 0.5
            reasons.append("No tool calls made — stage likely skipped work")

        quality = StageQuality(
            stage=stage,
            score=round(score, 2),
            reasons=reasons,
        )
        self._state.quality_history.append(quality)

        if quality.score < self._state.min_quality_threshold:
            self._state.consecutive_degraded += 1
        else:
            self._state.consecutive_degraded = 0

        # Check if circuit should trip
        if self._state.consecutive_degraded >= self._state.degradation_threshold:
            self._state.state = CircuitState.OPEN
            self._state.trip_reason = (
                f"Circuit tripped: {self._state.consecutive_degraded} "
                f"consecutive degraded stages "
                f"(threshold={self._state.degradation_threshold})"
            )

        return quality

    def reset(self) -> None:
        """Reset circuit breaker (e.g., after user intervention)."""
        self._state = CircuitBreakerState(
            degradation_threshold=self._state.degradation_threshold,
            min_quality_threshold=self._state.min_quality_threshold,
        )

    def summary(self) -> dict[str, Any]:
        """Return circuit breaker summary for audit report."""
        return {
            "state": self._state.state.value,
            "total_stages_evaluated": len(self._state.quality_history),
            "consecutive_degraded": self._state.consecutive_degraded,
            "trip_reason": self._state.trip_reason,
            "quality_scores": [
                {"stage": q.stage, "score": q.score, "reasons": q.reasons}
                for q in self._state.quality_history
            ],
        }
