"""Pipeline shared context — mutable state passed between stages.

Encapsulates the data that flows through the pipeline:
- previous result from the last completed stage
- CodeGen data preserved for final result assembly
- the previous stage identifier
- conversation history for cross-Stage context inheritance
- iteration tracking for REJECT feedback loops

This is a pure data holder with no business logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.domain.subagent import PipelineStage, SubAgentResult, SubAgentStatus


@dataclass
class PipelineContext:
    """Mutable accumulator for pipeline execution state.

    Each stage reads from and writes to this context.
    The context is the single source of truth for inter-stage data flow.
    """

    prev_result: SubAgentResult | None = None
    prev_stage: PipelineStage | None = None
    code_gen_data: dict[str, Any] | None = None
    target_spec: dict[str, Any] = field(default_factory=dict)
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    iteration_count: int = 0
    max_iterations: int = 3
    rejection_history: list[dict[str, Any]] = field(default_factory=list)
    metric_feedback: list[dict[str, Any]] = field(default_factory=list)

    def update(self, stage: PipelineStage, result: SubAgentResult) -> None:
        """Advance the context after a stage completes."""
        if stage == PipelineStage.CODE_GEN and result.is_success():
            self.code_gen_data = dict(result.data)

        self.prev_result = result
        self.prev_stage = stage

    def append_history(self, role: str, content: str) -> None:
        """Append a conversation entry for cross-Stage context inheritance."""
        self.conversation_history.append({"role": role, "content": content})

    def get_history(self, limit: int = 20) -> list[dict[str, str]]:
        """Return the most recent conversation history entries."""
        return self.conversation_history[-limit:]

    def add_rejection(self, stage: str, concerns: list[str], suggested_fixes: list[str] | None = None) -> None:
        """Record a rejection event for iteration tracking."""
        self.rejection_history.append({
            "stage": stage,
            "concerns": concerns,
            "suggested_fixes": suggested_fixes or [],
            "iteration": self.iteration_count,
        })

    def add_metric_feedback(self, suggested_fixes: list[str], bottleneck_type: str = "",
                            bottleneck_sub_type: str = "", recommendations: list[str] | None = None) -> None:
        """Record MetricAnalysis feedback for CodeGen optimization.

        This enables the MetricAnalysis → CodeGen feedback loop:
        MetricAnalysis identifies bottlenecks and generates recommendations,
        which are then injected into CodeGen's task prompt for optimization.
        """
        self.metric_feedback.append({
            "stage": "metric_analysis",
            "suggested_fixes": suggested_fixes,
            "bottleneck_type": bottleneck_type,
            "bottleneck_sub_type": bottleneck_sub_type,
            "recommendations": recommendations or [],
            "iteration": self.iteration_count,
        })

    def can_retry(self) -> bool:
        """Check if another iteration is allowed."""
        return self.iteration_count < self.max_iterations

    def increment_iteration(self) -> None:
        """Increment the iteration counter."""
        self.iteration_count += 1

    def get_feedback_for_codegen(self) -> dict[str, Any] | None:
        """Extract combined feedback from Verification and MetricAnalysis for CodeGen retry.

        Merges:
        - Verification rejection concerns and suggested fixes
        - MetricAnalysis bottleneck identification and optimization recommendations

        Returns None if no feedback is available.
        """
        feedback: dict[str, Any] = {}

        if self.rejection_history:
            last = self.rejection_history[-1]
            feedback["concerns"] = last.get("concerns", [])
            feedback["suggested_fixes"] = last.get("suggested_fixes", [])
            feedback["iteration"] = last.get("iteration", 0)

        if self.metric_feedback:
            last_metric = self.metric_feedback[-1]
            feedback["metric_recommendations"] = last_metric.get("recommendations", [])
            feedback["metric_suggested_fixes"] = last_metric.get("suggested_fixes", [])
            feedback["bottleneck_type"] = last_metric.get("bottleneck_type", "")
            feedback["bottleneck_sub_type"] = last_metric.get("bottleneck_sub_type", "")

        return feedback if feedback else None

    def bubble_codegen_data(self, result: SubAgentResult) -> SubAgentResult:
        """Propagate CodeGen measurements into a downstream result.

        Ensures MetricAnalysis and Verification can see the full chain
        of CodeGen measurements even if they don't produce their own.
        """
        if not self.code_gen_data or not result.is_success():
            return result

        carry_keys = [
            "measurements", "code_gen_output", "tool_results",
            "code_gen_final_output",
        ]
        for key in carry_keys:
            src_key = "final_output" if key == "code_gen_final_output" else key
            if key not in result.data and src_key in self.code_gen_data:
                result.data[key] = self.code_gen_data[src_key]

        return result

    def assemble_final_result(self, result: SubAgentResult) -> SubAgentResult:
        """Merge CodeGen data into the final pipeline result."""
        if not self.code_gen_data or not result.is_success():
            return result

        merge_keys = [
            "measurements", "analysis_method", "code_gen_output",
            "tool_results", "binary_path",
        ]
        for key in merge_keys:
            if key in self.code_gen_data:
                result.data[key] = self.code_gen_data[key]

        return result
