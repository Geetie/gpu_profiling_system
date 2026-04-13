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

    def run(self, message: CollaborationMessage) -> SubAgentResult:
        """Review the previous stage's result independently.

        Receives only the structured SubAgentResult (data + artifacts +
        context fingerprint) — NOT the generation process or context.
        """
        # P7 guard: verify context is empty at start
        if self.context_manager.total_tokens > 0:
            raise P7ViolationError(
                "VerificationAgent context must be empty at start of run(). "
                f"Found {self.context_manager.total_tokens} tokens."
            )

        self.context_manager.add_entry(
            Role.SYSTEM, self._build_system_prompt(), token_count=30
        )

        prev_result = message.payload.get("prev_result", {})
        prev_fingerprint = message.payload.get("prev_fingerprint", "none")

        # Extract data from previous result
        data = prev_result.get("data", {})
        artifacts = prev_result.get("artifacts", [])
        prev_status = prev_result.get("status", "unknown")
        prev_role = prev_result.get("agent_role", "unknown")

        # Independent review
        review = self._review(data, artifacts, prev_status, prev_role)

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

        result.context_fingerprint = result.compute_fingerprint(self.context_manager)
        self._persist_result(result)
        return result

    def _review(
        self,
        data: dict[str, Any],
        artifacts: list[str],
        prev_status: str,
        prev_role: str,
    ) -> dict[str, Any]:
        """Perform independent review of the previous stage's result.

        Uses LLM when available for deeper analysis, falls back to
        rule-based checks (completeness, numeric sanity, methodology).
        """
        if self._model_caller is not None:
            return self._llm_review(data, artifacts, prev_status, prev_role)

        # Rule-based fallback (original logic)
        findings: list[str] = []
        concerns: list[str] = []
        accepted = True

        # Check 1: Data completeness
        if not data:
            concerns.append("No data provided for review")
            accepted = False
        else:
            findings.append(f"Data contains {len(data)} fields")

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

    def _llm_review(
        self,
        data: dict[str, Any],
        artifacts: list[str],
        prev_status: str,
        prev_role: str,
    ) -> dict[str, Any]:
        """Use LLM for independent review of the previous stage's result."""
        import json as _json

        review_payload = {
            "data": data,
            "artifacts": artifacts,
            "prev_status": prev_status,
            "prev_role": prev_role,
        }

        user_msg = (
            f"Independently review this GPU profiling result. "
            f"Check: (1) data completeness, (2) numeric sanity, "
            f"(3) methodology soundness, (4) artifact existence.\n\n"
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
        return self._rule_review(data, artifacts, prev_status, prev_role)

    def _rule_review(
        self,
        data: dict[str, Any],
        artifacts: list[str],
        prev_status: str,
        prev_role: str,
    ) -> dict[str, Any]:
        """Rule-based independent review (original logic, extracted)."""
        findings: list[str] = []
        concerns: list[str] = []
        accepted = True

        # Check 1: Data completeness
        if not data:
            concerns.append("No data provided for review")
            accepted = False
        else:
            findings.append(f"Data contains {len(data)} fields")

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
