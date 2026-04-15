"""Pipeline shared context — mutable state passed between stages.

Encapsulates the data that flows through the pipeline:
- previous result from the last completed stage
- CodeGen data preserved for final result assembly
- the previous stage identifier

This is a pure data holder with no business logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.domain.subagent import PipelineStage, SubAgentResult


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

    def update(self, stage: PipelineStage, result: SubAgentResult) -> None:
        """Advance the context after a stage completes."""
        if stage == PipelineStage.CODE_GEN and result.is_success():
            self.code_gen_data = dict(result.data)

        self.prev_result = result
        self.prev_stage = stage

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
