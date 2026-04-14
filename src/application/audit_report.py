"""Harness Engineering: Pipeline Execution Audit Report.

Generates a comprehensive, human-readable audit report covering:
- Stage-by-stage execution timeline
- Handoff validation results
- Circuit breaker state
- Tool execution summary
- Output quality assessment
- P7 compliance verification

The report is written to the output folder for review.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from src.application.circuit_breaker import CircuitBreaker
from src.application.handoff_validation import HandoffReport
from src.domain.subagent import SubAgentResult, SubAgentStatus


class PipelineAuditReport:
    """Comprehensive audit report for a pipeline execution."""

    def __init__(self) -> None:
        self._start_time: str | None = None
        self._end_time: str | None = None
        self._stage_results: list[dict[str, Any]] = []
        self._handoff_reports: list[dict[str, Any]] = []
        self._circuit_breaker_summary: dict[str, Any] | None = None
        self._p7_audit: dict[str, Any] = {}
        self._final_result: SubAgentResult | None = None
        self._tool_executions: list[dict[str, Any]] = []
        self._errors: list[str] = []

    def record_start(self) -> None:
        self._start_time = datetime.now(timezone.utc).isoformat()

    def record_end(self) -> None:
        self._end_time = datetime.now(timezone.utc).isoformat()

    def record_stage(
        self,
        stage_name: str,
        result: SubAgentResult,
        duration_seconds: float | None = None,
    ) -> None:
        self._stage_results.append({
            "stage": stage_name,
            "status": result.status.value,
            "context_fingerprint": result.context_fingerprint,
            "tool_calls": result.data.get("num_tool_calls", 0),
            "output_length": len(result.data.get("final_output", "")),
            "measurements_count": len(result.data.get("measurements", {})),
            "error": result.error,
            "duration_seconds": duration_seconds,
        })

    def record_handoff(self, report: HandoffReport) -> None:
        self._handoff_reports.append({
            "from_stage": report.from_stage,
            "to_stage": report.to_stage,
            "is_valid": report.is_valid,
            "errors": [
                {"field": v.field, "message": v.message}
                for v in report.errors
            ],
            "warnings": [
                {"field": v.field, "message": v.message}
                for v in report.warnings
            ],
        })

    def record_circuit_breaker(self, cb: CircuitBreaker) -> None:
        self._circuit_breaker_summary = cb.summary()

    def record_p7_audit(
        self, generation_fingerprint: str | None,
        verification_context_tokens: int,
        status: str,
    ) -> None:
        self._p7_audit = {
            "generation_fingerprint": generation_fingerprint,
            "verification_context_tokens": verification_context_tokens,
            "status": status,
        }

    def record_tool_execution(self, tool_name: str, status: str) -> None:
        self._tool_executions.append({
            "tool": tool_name,
            "status": status,
        })

    def record_error(self, error: str) -> None:
        self._errors.append(error)

    def set_final_result(self, result: SubAgentResult) -> None:
        self._final_result = result

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_type": "pipeline_audit",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "start_time": self._start_time,
            "end_time": self._end_time,
            "stages": self._stage_results,
            "handoffs": self._handoff_reports,
            "circuit_breaker": self._circuit_breaker_summary,
            "p7_audit": self._p7_audit,
            "tool_executions": self._tool_executions,
            "errors": self._errors,
            "final_status": self._final_result.status.value if self._final_result else "unknown",
        }

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append("# Pipeline Execution Audit Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        if self._start_time:
            lines.append(f"**Start:** {self._start_time}")
        if self._end_time:
            lines.append(f"**End:** {self._end_time}")
        lines.append("")

        # Stage timeline
        lines.append("## Stage Timeline")
        lines.append("")
        for i, stage in enumerate(self._stage_results):
            status_icon = {
                "success": "[PASS]",
                "failed": "[FAIL]",
                "rejected": "[REJECT]",
                "pending": "[PEND]",
            }.get(stage["status"], "[????]")
            duration = f" ({stage['duration_seconds']:.1f}s)" if stage.get("duration_seconds") else ""
            lines.append(
                f"{i+1}. **{stage['stage']}** {status_icon} "
                f"— tools={stage['tool_calls']}, "
                f"output={stage['output_length']} chars, "
                f"measurements={stage['measurements_count']}"
                f"{duration}"
            )
            if stage.get("error"):
                lines.append(f"   - Error: {stage['error']}")
        lines.append("")

        # Handoff validation
        lines.append("## Handoff Validation")
        lines.append("")
        for h in self._handoff_reports:
            icon = "PASS" if h["is_valid"] else "FAIL"
            lines.append(
                f"- **{h['from_stage']}→{h['to_stage']}**: {icon}"
            )
            for err in h["errors"]:
                lines.append(f"  - [ERROR] {err['field']}: {err['message']}")
            for warn in h["warnings"]:
                lines.append(f"  - [WARN] {warn['field']}: {warn['message']}")
        lines.append("")

        # Circuit breaker
        if self._circuit_breaker_summary:
            lines.append("## Circuit Breaker")
            lines.append("")
            cb = self._circuit_breaker_summary
            lines.append(f"- State: **{cb['state']}**")
            lines.append(f"- Stages evaluated: {cb['total_stages_evaluated']}")
            lines.append(f"- Consecutive degraded: {cb['consecutive_degraded']}")
            if cb.get("trip_reason"):
                lines.append(f"- Trip reason: {cb['trip_reason']}")
            lines.append("")

        # P7 audit
        if self._p7_audit:
            lines.append("## P7 Compliance (Generation/Verification Separation)")
            lines.append("")
            lines.append(f"- Status: **{self._p7_audit['status']}**")
            lines.append(f"- Verification context tokens: {self._p7_audit['verification_context_tokens']}")
            if self._p7_audit.get("generation_fingerprint"):
                lines.append(f"- Generation fingerprint: {self._p7_audit['generation_fingerprint']}")
            lines.append("")

        # Tool execution summary
        if self._tool_executions:
            lines.append("## Tool Execution Summary")
            lines.append("")
            tool_counts: dict[str, dict[str, int]] = {}
            for t in self._tool_executions:
                name = t["tool"]
                if name not in tool_counts:
                    tool_counts[name] = {"success": 0, "failed": 0}
                if "success" in t["status"].lower():
                    tool_counts[name]["success"] += 1
                else:
                    tool_counts[name]["failed"] += 1
            for name, counts in sorted(tool_counts.items()):
                lines.append(f"- **{name}**: {counts['success']} OK, {counts['failed']} failed")
            lines.append("")

        # Final status
        lines.append("## Final Status")
        lines.append("")
        lines.append(f"- Pipeline result: **{self._final_result.status.value if self._final_result else 'unknown'}**")
        if self._final_result and self._final_result.data:
            measurements = self._final_result.data.get("measurements", {})
            if measurements:
                lines.append(f"- Measurements: {len(measurements)} targets profiled")
                for k, v in sorted(measurements.items()):
                    lines.append(f"  - `{k}`: {v}")
        if self._errors:
            lines.append(f"- Errors ({len(self._errors)}):")
            for err in self._errors:
                lines.append(f"  - {err}")
        lines.append("")

        return "\n".join(lines)

    def save(self, output_dir: str) -> tuple[str, str]:
        """Save audit report as both JSON and Markdown.

        Returns (json_path, markdown_path).
        """
        os.makedirs(output_dir, exist_ok=True)

        # JSON report
        json_path = os.path.join(output_dir, "audit_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        # Markdown report
        md_path = os.path.join(output_dir, "audit_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown())

        return json_path, md_path
