"""Harness Engineering: Inter-Agent Handoff Validation.

Enforces structured contracts at every stage boundary.
Each transition (Planner→CodeGen→MetricAnalysis→Verification)
validates that the output meets the minimum requirements for
the next stage to operate correctly.

Harness principle: "If you can't validate it, you can't pass it."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.domain.subagent import PipelineStage, SubAgentResult, SubAgentStatus


@dataclass
class HandoffViolation:
    """A single handoff validation failure."""
    stage: str
    field: str
    expected: str
    actual: str
    severity: str  # "error" (blocks handoff) | "warning" (logged, passes)
    message: str


@dataclass
class HandoffReport:
    """Complete handoff validation result for a stage transition."""
    from_stage: str
    to_stage: str
    violations: list[HandoffViolation] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(v.severity == "error" for v in self.violations)

    @property
    def warnings(self) -> list[HandoffViolation]:
        return [v for v in self.violations if v.severity == "warning"]

    @property
    def errors(self) -> list[HandoffViolation]:
        return [v for v in self.violations if v.severity == "error"]

    def summary(self) -> str:
        if self.is_valid:
            tag = "PASS" if not self.warnings else "PASS(warnings)"
            return f"Handoff {self.from_stage}→{self.to_stage}: {tag}"
        return (
            f"Handoff {self.from_stage}→{self.to_stage}: FAIL "
            f"({len(self.errors)} errors, {len(self.warnings)} warnings)"
        )


class HandoffValidator:
    """Validates SubAgentResult outputs at pipeline stage boundaries.

    Each stage transition has a minimum contract that the output must satisfy.
    If the contract is violated, the handoff is rejected.
    """

    # Minimum contracts for each stage transition
    PLANNER_CONTRACT = {
        "required_data_keys": {"tasks"},
        "tasks_must_be_list": True,
        "task_fields": {"target", "category", "method"},
        "min_tasks": 1,
    }

    CODEGEN_CONTRACT = {
        "required_data_keys": {"measurements"},  # Must have at least some measurements
        "measurements_must_have_values": True,
    }

    METRIC_ANALYSIS_CONTRACT = {
        "required_data_keys": {"bottleneck_type"},
    }

    VERIFICATION_CONTRACT = {
        "required_data_keys": {"review", "accepted"},
    }

    def validate(
        self,
        from_stage: PipelineStage,
        to_stage: PipelineStage,
        result: SubAgentResult | None,
    ) -> HandoffReport:
        """Validate the handoff between two pipeline stages."""
        report = HandoffReport(
            from_stage=from_stage.value,
            to_stage=to_stage.value,
        )

        if result is None:
            report.violations.append(HandoffViolation(
                stage=from_stage.value,
                field="result",
                expected="SubAgentResult",
                actual="None",
                severity="error",
                message=f"Stage {from_stage.value} produced no result to hand off",
            ))
            return report

        if result.status == SubAgentStatus.FAILED:
            report.violations.append(HandoffViolation(
                stage=from_stage.value,
                field="status",
                expected="SUCCESS or REJECTED",
                actual=result.status.value,
                severity="error",
                message=f"Stage {from_stage.value} failed: {result.error}",
            ))
            return report

        # Route to stage-specific validation
        if to_stage == PipelineStage.CODE_GEN:
            self._validate_planner_output(result, report)
        elif to_stage == PipelineStage.METRIC_ANALYSIS:
            self._validate_codegen_output(result, report)
        elif to_stage == PipelineStage.VERIFICATION:
            self._validate_metric_analysis_output(result, report)

        return report

    def _validate_planner_output(
        self, result: SubAgentResult, report: HandoffReport
    ) -> None:
        """Validate Planner→CodeGen handoff."""
        data = result.data

        # Required: tasks key
        if "tasks" not in data:
            report.violations.append(HandoffViolation(
                stage="PLAN", field="data.tasks",
                expected="list of task dicts", actual="key missing",
                severity="error",
                message="Planner did not produce 'tasks' key in data",
            ))
            return

        tasks = data["tasks"]
        if not isinstance(tasks, list):
            report.violations.append(HandoffViolation(
                stage="PLAN", field="data.tasks",
                expected="list", actual=type(tasks).__name__,
                severity="error",
                message="'tasks' is not a list",
            ))
            return

        if len(tasks) == 0:
            report.violations.append(HandoffViolation(
                stage="PLAN", field="data.tasks",
                expected="non-empty list", actual="empty list",
                severity="error",
                message="Planner produced zero tasks",
            ))
            return

        # Validate each task has required fields
        for i, task in enumerate(tasks):
            if not isinstance(task, dict):
                report.violations.append(HandoffViolation(
                    stage="PLAN", field=f"data.tasks[{i}]",
                    expected="dict", actual=type(task).__name__,
                    severity="error",
                    message=f"Task {i} is not a dict",
                ))
                continue

            missing = self.PLANNER_CONTRACT["task_fields"] - set(task.keys())
            if missing:
                report.violations.append(HandoffViolation(
                    stage="PLAN", field=f"data.tasks[{i}]",
                    expected=f"keys: {self.PLANNER_CONTRACT['task_fields']}",
                    actual=f"missing: {missing}",
                    severity="error",
                    message=f"Task '{task.get('target', '?')}' missing fields: {missing}",
                ))

        # Warning if many targets (potential over-scope)
        if len(tasks) > 20:
            report.violations.append(HandoffViolation(
                stage="PLAN", field="data.tasks",
                expected="reasonable task count",
                actual=f"{len(tasks)} tasks",
                severity="warning",
                message="Unusually large number of tasks — may exceed turn budget",
            ))

    def _validate_codegen_output(
        self, result: SubAgentResult, report: HandoffReport
    ) -> None:
        """Validate CodeGen→MetricAnalysis handoff."""
        data = result.data

        # Check for measurements (may be nested)
        measurements = data.get("measurements", {})
        if not isinstance(measurements, dict) or len(measurements) == 0:
            # Check if tool_results has any execute_binary output
            tool_results = data.get("tool_results", [])
            has_exec_output = any(
                tr.get("stdout") for tr in tool_results
                if isinstance(tr, dict)
            )
            has_binary = any(
                tr.get("binary_path") for tr in tool_results
                if isinstance(tr, dict)
            )

            if has_binary and not has_exec_output:
                report.violations.append(HandoffViolation(
                    stage="CODE_GEN", field="measurements",
                    expected="numeric measurements from executed binaries",
                    actual="binaries compiled but no execution output",
                    severity="warning",
                    message="CodeGen compiled binaries but produced no measurement output",
                ))
            elif not has_binary:
                report.violations.append(HandoffViolation(
                    stage="CODE_GEN", field="measurements",
                    expected="at least some measurements or compiled binaries",
                    actual="no measurements, no binaries",
                    severity="error",
                    message="CodeGen produced no measurements or compiled binaries",
                ))
            else:
                # Has exec output but measurements not extracted — warn
                report.violations.append(HandoffViolation(
                    stage="CODE_GEN", field="measurements",
                    expected="structured measurements dict",
                    actual="stdout exists but not parsed to measurements",
                    severity="warning",
                    message="CodeGen produced output but measurements dict is empty",
                ))

        # Warning if no artifacts (source files) referenced
        if not result.artifacts and not any(
            tr.get("binary_path") for tr in data.get("tool_results", [])
            if isinstance(tr, dict)
        ):
            report.violations.append(HandoffViolation(
                stage="CODE_GEN", field="artifacts",
                expected="at least one compiled binary or source file",
                actual="none",
                severity="warning",
                message="No artifacts referenced — may indicate compilation failures",
            ))

    def _validate_metric_analysis_output(
        self, result: SubAgentResult, report: HandoffReport
    ) -> None:
        """Validate MetricAnalysis→Verification handoff."""
        data = result.data

        if "bottleneck_type" not in data and "parsed_metrics" not in data:
            report.violations.append(HandoffViolation(
                stage="METRIC_ANALYSIS", field="data",
                expected="bottleneck_type or parsed_metrics",
                actual="neither present",
                severity="warning",
                message="MetricAnalysis produced no bottleneck classification or metrics",
            ))

        # If there's an analysis_output or review_text, it's free-form — warn but allow
        if not data.get("bottleneck_type") and not data.get("parsed_metrics"):
            report.violations.append(HandoffViolation(
                stage="METRIC_ANALYSIS", field="structured_output",
                expected="structured metrics or bottleneck type",
                actual=f"keys: {list(data.keys())}",
                severity="warning",
                message="MetricAnalysis output is unstructured — Verification may struggle",
            ))
