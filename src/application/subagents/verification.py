"""Verification Agent — independent reviewer.

P7 enforcement: this agent NEVER inherits a generator's context.
It creates a fresh ContextManager on construction and reviews results
from first principles.
"""
from __future__ import annotations

from typing import Any

from src.application.context import ContextManager, Role
from src.domain.permission import PermissionMode
from src.domain.subagent import (
    AgentRole,
    BaseSubAgent,
    CollaborationMessage,
    P7ViolationError,
    SubAgentResult,
    SubAgentStatus,
)
from src.domain.tool_contract import ToolRegistry


class VerificationAgent(BaseSubAgent):
    """Independent reviewer that validates experimental results.

    CRITICAL: The constructor ALWAYS creates a fresh ContextManager.
    It never accepts an external one. This is the core P7 guarantee.
    """

    def __init__(
        self,
        *,
        tool_registry: ToolRegistry | None = None,
        state_dir: str = ".state",
        permission_mode: PermissionMode = PermissionMode.DEFAULT,
        max_tokens: int = 8000,
    ) -> None:
        # P7: ALWAYS create a fresh ContextManager — never accept one externally
        super().__init__(
            role=AgentRole.VERIFICATION,
            context_manager=ContextManager(max_tokens=max_tokens),  # always fresh
            tool_registry=tool_registry or ToolRegistry(),
            state_dir=state_dir,
            permission_mode=permission_mode,
            max_tokens=max_tokens,
        )

    def _process(self, message: CollaborationMessage) -> SubAgentResult:
        """Review the previous stage's result independently.

        Receives only the structured SubAgentResult (data + artifacts +
        context fingerprint) — NOT the generation process or context.
        """
        prev_result = message.payload.get("prev_result", {})
        prev_fingerprint = message.payload.get("prev_fingerprint", "none")
        target_spec = message.payload.get("target_spec", {})  # For completeness check

        # Extract data from previous result
        data = prev_result.get("data", {})
        artifacts = prev_result.get("artifacts", [])
        prev_status = prev_result.get("status", "unknown")
        prev_role = prev_result.get("agent_role", "unknown")

        # Independent review — pass target_spec for completeness validation
        review = self._review(data, artifacts, prev_status, prev_role, target_spec)

        result = SubAgentResult(
            agent_role=self.role,
            status=review["status"],
            data={
                "review": review["findings"],
                "accepted": review["accepted"],
                "concerns": review["concerns"],
                "generation_fingerprint": prev_fingerprint,
            },
            metadata={
                "reviewed_agent": prev_role,
                "review_method": "independent_analysis",
            },
        )

        return result

    def _review(
        self,
        data: dict[str, Any],
        artifacts: list[str],
        prev_status: str,
        prev_role: str,
        target_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Perform independent review of the previous stage's result.

        Uses LLM when available for deeper analysis, falls back to
        rule-based checks (completeness, numeric sanity, methodology).
        """
        if self._model_caller is not None:
            return self._llm_review(data, artifacts, prev_status, prev_role, target_spec)

        # Rule-based fallback (original logic)
        return self._rule_review(data, artifacts, prev_status, prev_role, target_spec)

    def _llm_review(
        self,
        data: dict[str, Any],
        artifacts: list[str],
        prev_status: str,
        prev_role: str,
        target_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Use LLM for independent review of the previous stage's result."""
        import json as _json

        review_payload = {
            "data": data,
            "artifacts": artifacts,
            "prev_status": prev_status,
            "prev_role": prev_role,
            "requested_targets": target_spec.get("targets", []) if target_spec else None,
        }

        user_msg = (
            f"Independently review this GPU profiling result. "
            f"Check: (1) data completeness (all requested targets measured?), "
            f"(2) numeric sanity, (3) methodology soundness, (4) artifact existence.\n\n"
            f"IMPORTANT: You are reviewing structured data, not directory contents. "
            f"The working directory is NOT empty - it contains the complete project code.\n\n"
            f"Review data:\n{_json.dumps(review_payload, indent=2)}\n\n"
            f'Return a JSON object with "findings" (list of strings), '
            f'"concerns" (list of strings), "accepted" (boolean), '
            f'and "status" ("success" or "rejected").'
        )
        messages = self.context_manager.to_messages()
        messages.append({"role": "user", "content": user_msg})

        try:
            response = self._model_caller(messages)
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = _json.loads(response[start:end])
                return {
                    "status": SubAgentStatus.SUCCESS
                    if parsed.get("status") == "success"
                    else SubAgentStatus.REJECTED,
                    "findings": parsed.get("findings", []),
                    "concerns": parsed.get("concerns", []),
                    "accepted": parsed.get("accepted", True),
                }
        except Exception:
            pass

        # Fallback to rule-based review
        return self._rule_review(data, artifacts, prev_status, prev_role, target_spec)

    def _rule_review(
        self,
        data: dict[str, Any],
        artifacts: list[str],
        prev_status: str,
        prev_role: str,
        target_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Rule-based independent review.

        Checks:
        1. Data completeness — all requested targets measured?
        2. Numeric sanity — values within plausible GPU ranges?
        3. Methodology — valid bottleneck type?
        4. Artifacts — evidence files referenced?
        """
        findings: list[str] = []
        concerns: list[str] = []
        accepted = True

        # Check 1: Data completeness against target_spec
        requested_targets = set()
        if target_spec:
            requested_targets = set(target_spec.get("targets", []))
            if requested_targets:
                # Measurements may be at top level OR nested in parsed_metrics
                measured_keys = set(data.keys())
                if "parsed_metrics" in data and isinstance(data["parsed_metrics"], dict):
                    measured_keys |= set(data["parsed_metrics"].keys())
                if "measurements" in data and isinstance(data["measurements"], dict):
                    measured_keys |= set(data["measurements"].keys())
                missing = requested_targets - measured_keys
                if missing:
                    concerns.append(f"Missing targets: {', '.join(sorted(missing))}")
                    accepted = False
                else:
                    findings.append(f"All {len(requested_targets)} requested targets measured")
                findings.append(f"Data contains {len(data)} fields")
            else:
                if not data:
                    concerns.append("No data provided for review")
                    accepted = False
                else:
                    findings.append(f"Data contains {len(data)} fields")
        else:
            if not data:
                concerns.append("No data provided for review")
                accepted = False
            else:
                findings.append(f"Data contains {len(data)} fields (no target_spec for completeness check)")

        # Check 2: Numeric sanity
        for key, value in data.items():
            if isinstance(value, (int, float)):
                if value < 0:
                    concerns.append(f"Negative value for '{key}': {value}")
                    accepted = False
                elif value > 1e12:
                    concerns.append(f"Suspiciously large value for '{key}': {value}")
                    accepted = False
                else:
                    findings.append(f"'{key}' = {value} (within plausible range)")

        # Check 3: Methodology
        if "bottleneck_type" in data:
            valid_types = {
                "compute_bound", "memory_bound", "latency_bound",
                "cache_capacity", "unknown",
            }
            if data["bottleneck_type"] in valid_types:
                findings.append(
                    f"Bottleneck type '{data['bottleneck_type']}' is valid"
                )
            else:
                concerns.append(
                    f"Unknown bottleneck type: '{data['bottleneck_type']}'"
                )
                accepted = False

        # Check 4: Artifacts
        if artifacts:
            findings.append(f"{len(artifacts)} artifact(s) referenced")
        else:
            concerns.append("No artifacts referenced")

        # Final determination
        status = SubAgentStatus.SUCCESS if accepted else SubAgentStatus.REJECTED

        return {
            "status": status,
            "findings": findings,
            "concerns": concerns,
            "accepted": accepted,
        }
