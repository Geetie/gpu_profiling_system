"""Metric Analysis Agent — parses Nsight Compute output.

Identifies performance bottleneck types from NCU profiling data:
compute-bound, memory-bound, latency-bound, or cache-capacity cliffs.
"""
from __future__ import annotations

import re
from typing import Any

from src.application.context import ContextManager, Role
from src.domain.permission import PermissionMode
from src.domain.subagent import (
    AgentRole,
    BaseSubAgent,
    CollaborationMessage,
    SubAgentResult,
    SubAgentStatus,
)
from src.domain.tool_contract import ToolRegistry


class MetricAnalysisAgent(BaseSubAgent):
    """Parses NCU output and identifies performance bottlenecks."""

    def __init__(
        self,
        context_manager: ContextManager | None = None,
        tool_registry: ToolRegistry | None = None,
        state_dir: str = ".state",
        permission_mode: PermissionMode = PermissionMode.DEFAULT,
        max_tokens: int = 8000,
    ) -> None:
        super().__init__(
            role=AgentRole.METRIC_ANALYSIS,
            context_manager=context_manager or ContextManager(max_tokens=max_tokens),
            tool_registry=tool_registry or ToolRegistry(),
            state_dir=state_dir,
            permission_mode=permission_mode,
            max_tokens=max_tokens,
        )

    def _process(self, message: CollaborationMessage) -> SubAgentResult:
        """Analyze metrics from the previous stage's output."""
        prev_result = message.payload.get("prev_result", {})
        raw_output = prev_result.get("data", {}).get("raw_output", "")

        if not raw_output:
            return SubAgentResult(
                agent_role=self.role,
                status=SubAgentStatus.FAILED,
                error="No raw output to analyze",
            )

        # Try LLM-based analysis first, fall back to rule-based
        if self._model_caller is not None:
            parsed_metrics, bottleneck = self._llm_analyze(raw_output)
        else:
            # Parse the output
            parsed_metrics = self._parse_output(raw_output)
            bottleneck = self.identify_bottleneck(parsed_metrics)

        result = SubAgentResult(
            agent_role=self.role,
            status=SubAgentStatus.SUCCESS,
            data={
                "bottleneck_type": bottleneck,
                "parsed_metrics": parsed_metrics,
                "confidence": self._assess_confidence(parsed_metrics),
            },
            metadata={"analysis_method": "llm" if self._model_caller else "pattern_matching"},
        )

        return result

    def _llm_analyze(self, raw_output: str) -> tuple[dict[str, Any], str]:
        """Use LLM to analyze raw profiling output."""
        import json as _json

        user_msg = (
            f"Analyze this GPU profiling output and identify the bottleneck. "
            f"Extract all numeric metrics and key-value pairs.\n\n"
            f"Raw output:\n{raw_output}\n\n"
            f'Return a JSON object with "metrics" (dict of key->value) and '
            f'"bottleneck_type" (one of: compute_bound, memory_bound, '
            f'latency_bound, cache_capacity, unknown).'
        )
        messages = self.context_manager.to_messages()
        messages.append({"role": "user", "content": user_msg})

        try:
            response = self._model_caller(messages)
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = _json.loads(response[start:end])
                metrics = parsed.get("metrics", {})
                bottleneck = parsed.get("bottleneck_type", "unknown")
                return metrics, bottleneck
        except Exception:
            pass

        # Fallback to rule-based
        metrics = self._parse_output(raw_output)
        bottleneck = self.identify_bottleneck(metrics)
        return metrics, bottleneck

    def _parse_output(self, raw: str) -> dict[str, Any]:
        """Parse raw output into structured metrics.

        Handles both ncu CSV output and simple numeric results.
        """
        metrics: dict[str, Any] = {}

        # Try to extract key-value pairs (ncu-like output)
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("---"):
                continue

            # Pattern: "metric_name: value"
            match = re.match(r"^([^:]+):\s*(.+)$", line)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                try:
                    metrics[key] = float(value)
                except ValueError:
                    metrics[key] = value

            # Pattern: just a number
            elif re.match(r"^\d+(\.\d+)?$", line):
                metrics.setdefault("result", []).append(float(line))

        # If no structured metrics found, store raw output
        if not metrics:
            metrics["raw"] = raw[:2000]  # truncate for safety

        return metrics

    def identify_bottleneck(self, metrics: dict[str, Any]) -> str:
        """Identify the bottleneck type from parsed metrics.

        Returns one of:
        - "compute_bound": GPU compute units saturated
        - "memory_bound": memory bandwidth saturated
        - "latency_bound": memory latency dominates
        - "cache_capacity": cache size cliff detected
        - "unknown": insufficient data
        """
        # Check for latency patterns
        if any("latency" in k.lower() or "cycle" in k.lower() for k in metrics):
            return "latency_bound"

        # Check for bandwidth patterns
        if any("bandwidth" in k.lower() or "throughput" in k.lower() for k in metrics):
            return "memory_bound"

        # Check for compute patterns
        if any("ipc" in k.lower() or "flop" in k.lower() for k in metrics):
            return "compute_bound"

        # Check for cache capacity patterns
        if any("cache" in k.lower() or "miss" in k.lower() for k in metrics):
            return "cache_capacity"

        # Check numeric results for cliff detection
        values = [v for v in metrics.values() if isinstance(v, (int, float))]
        if len(values) >= 3:
            # Simple cliff detection: large jump between consecutive values
            for i in range(1, len(values)):
                if values[i] > values[i-1] * 2:
                    return "cache_capacity"

        return "unknown"

    def _assess_confidence(self, metrics: dict[str, Any]) -> float:
        """Assess confidence in the bottleneck identification."""
        if not metrics:
            return 0.0

        numeric_count = sum(1 for v in metrics.values() if isinstance(v, (int, float)))
        # More numeric metrics → higher confidence
        return min(1.0, numeric_count / 5.0)
