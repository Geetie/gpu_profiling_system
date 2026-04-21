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

    Memory Architecture (4 layers):
    - L0 (Permanent): Architecture info, target spec — never compressed
    - L1 (High): CodeGen measurements, binary paths — preserved across stages
    - L2 (Medium): MetricAnalysis results, error patterns — compressed on budget pressure
    - L3 (Low): Conversation history, Control Plane snapshots — aggressively compressed
    """

    prev_result: SubAgentResult | None = None
    prev_stage: PipelineStage | None = None
    code_gen_data: dict[str, Any] | None = None
    target_spec: dict[str, Any] = field(default_factory=dict)
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    iteration_count: int = 0
    max_iterations: int = 2 # Reduced from 3 to prevent long retry loops (FIX for 32-min timeout)
    rejection_history: list[dict[str, Any]] = field(default_factory=list)
    metric_feedback: list[dict[str, Any]] = field(default_factory=list)
    _stage_results: dict[str, SubAgentResult] = field(default_factory=dict)

    # L0: Permanent memory — never compressed
    architecture_info: dict[str, Any] = field(default_factory=dict)
    # L1: High-priority memory — preserved across stages
    key_measurements: dict[str, Any] = field(default_factory=dict)
    binary_paths: list[str] = field(default_factory=list)
    # L2: Medium-priority memory — compressed on budget pressure
    stage_summaries: dict[str, str] = field(default_factory=dict)
    error_patterns: list[str] = field(default_factory=list)

    def update(self, stage: PipelineStage, result: SubAgentResult) -> None:
        """Advance the context after a stage completes.
        
        Populates the layered memory architecture:
        - L0: Architecture info from any stage that detects it
        - L1: Key measurements and binary paths from CodeGen
        - L2: Stage summaries and error patterns from all stages
        """
        if stage == PipelineStage.CODE_GEN:
            self.code_gen_data = dict(result.data)
            # CRITICAL FIX: Always save measurements to key_measurements, regardless of status
            # This ensures measurements are available even if CodeGen returns PARTIAL or FAILED
            # but still produced some valid measurements
            if "measurements" in result.data and isinstance(result.data["measurements"], dict):
                for k, v in result.data["measurements"].items():
                    self.key_measurements[k] = v
                logger.info("[PipelineContext] Saved %d measurements to key_measurements", 
                           len(result.data["measurements"]))
            if "binary_path" in result.data:
                bp = result.data["binary_path"]
                if isinstance(bp, str) and bp not in self.binary_paths:
                    self.binary_paths.append(bp)
            if "tool_results" in result.data and isinstance(result.data["tool_results"], list):
                for tr in result.data["tool_results"]:
                    if isinstance(tr, dict) and "binary_path" in tr:
                        bp = tr["binary_path"]
                        if isinstance(bp, str) and bp not in self.binary_paths:
                            self.binary_paths.append(bp)

        # L2: Record stage summary
        if result.is_success():
            summary = result.data.get("final_output", "")[:500] if result.data.get("final_output") else ""
            self.stage_summaries[stage.value] = summary
        else:
            error_msg = result.data.get("errors", result.data.get("error", "unknown"))
            self.error_patterns.append(f"{stage.value}: {str(error_msg)[:200]}")

        # L0: Extract architecture info if present
        if "arch" in result.data:
            self.architecture_info["gpu_arch"] = result.data["arch"]
        if "architecture" in result.data:
            self.architecture_info["gpu_architecture"] = result.data["architecture"]

        self.prev_result = result
        self.prev_stage = stage
        self._stage_results[stage.value] = result

    def append_history(self, role: str, content: str) -> None:
        """Append a conversation entry for cross-Stage context inheritance."""
        self.conversation_history.append({"role": role, "content": content})

    def get_stage_result(self, stage: PipelineStage) -> SubAgentResult | None:
        """Retrieve the result from a specific pipeline stage.

        Returns None if the stage has not yet completed.
        """
        return self._stage_results.get(stage.value)

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
        
        CRITICAL FIX: Always propagate measurements, even if downstream stage failed.
        This ensures measurements are available for final results even when
        MetricAnalysis or Verification fails.
        """
        if not self.code_gen_data:
            return result

        # Always propagate measurements, regardless of result status
        # This is critical for ensuring measurements appear in final output
        if "measurements" in self.code_gen_data:
            existing = result.data.get("measurements", {})
            if isinstance(existing, dict):
                for k, v in self.code_gen_data["measurements"].items():
                    if k not in existing:
                        existing[k] = v
                result.data["measurements"] = existing
            else:
                result.data["measurements"] = dict(self.code_gen_data["measurements"])

        # Only propagate other data if result is successful
        if result.is_success():
            carry_keys = [
                "code_gen_output", "tool_results",
                "code_gen_final_output",
            ]
            for key in carry_keys:
                src_key = "final_output" if key == "code_gen_final_output" else key
                if key not in result.data and src_key in self.code_gen_data:
                    result.data[key] = self.code_gen_data[src_key]

        return result

    def assemble_final_result(self, result: SubAgentResult) -> SubAgentResult:
        """Merge CodeGen data and layered memory into the final pipeline result.
        
        Uses the 4-layer memory architecture:
        - L0: Architecture info always included
        - L1: Key measurements and binary paths always included
        - L2: Stage summaries and error patterns included for debugging
        - L3: Conversation history excluded from final result (too verbose)
        """
        if not self.code_gen_data:
            # Even without CodeGen data, include L0 and L1 memory
            if self.architecture_info:
                result.data["architecture_info"] = dict(self.architecture_info)
            if self.key_measurements:
                existing = result.data.get("measurements", {})
                if isinstance(existing, dict):
                    for k, v in self.key_measurements.items():
                        if k not in existing:
                            existing[k] = v
                    result.data["measurements"] = existing
                else:
                    result.data["measurements"] = dict(self.key_measurements)
            if self.binary_paths:
                result.data["binary_paths"] = list(self.binary_paths)
            return result

        merge_keys = [
            "analysis_method", "code_gen_output",
            "tool_results", "binary_path",
        ]
        for key in merge_keys:
            if key in self.code_gen_data and key not in result.data:
                result.data[key] = self.code_gen_data[key]

        # Always merge measurements - CodeGen's are the primary source
        if "measurements" in self.code_gen_data:
            existing = result.data.get("measurements", {})
            if isinstance(existing, dict):
                for k, v in self.code_gen_data["measurements"].items():
                    if k not in existing:
                        existing[k] = v
                result.data["measurements"] = existing
            else:
                result.data["measurements"] = self.code_gen_data["measurements"]

        # L0: Architecture info
        if self.architecture_info:
            result.data["architecture_info"] = dict(self.architecture_info)

        # L1: Key measurements (supplementary)
        if self.key_measurements:
            existing = result.data.get("measurements", {})
            if isinstance(existing, dict):
                for k, v in self.key_measurements.items():
                    if k not in existing:
                        existing[k] = v
                result.data["measurements"] = existing
            # Also include key_measurements separately for downstream access
            result.data["key_measurements"] = dict(self.key_measurements)

        # L1: Binary paths
        if self.binary_paths:
            result.data["binary_paths"] = list(self.binary_paths)

        # L2: Stage summaries
        if self.stage_summaries:
            result.data["stage_summaries"] = dict(self.stage_summaries)

        # L2: Error patterns
        if self.error_patterns:
            result.data["error_patterns"] = list(self.error_patterns)

        return result
