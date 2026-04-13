"""Sub-agent domain model — abstract base, roles, results, and messages.

P7 (生成与评估分离): This module defines the structural guarantees that
a VerificationAgent can never inherit a generator's context.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.application.context import ContextManager
from src.domain.permission import PermissionMode
from src.domain.tool_contract import ToolRegistry
from src.infrastructure.state_persist import StatePersister


# ── Enums ────────────────────────────────────────────────────────────


class AgentRole(Enum):
    """Roles in the multi-agent team."""
    PLANNER = "planner"
    CODE_GEN = "code_gen"
    METRIC_ANALYSIS = "metric_analysis"
    VERIFICATION = "verification"


class SubAgentStatus(Enum):
    """Lifecycle status of a sub-agent result."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    REJECTED = "rejected"   # Verification rejected the result


class PipelineStage(Enum):
    """Sequential stages in the collaboration pipeline."""
    PLAN = "plan"
    CODE_GEN = "code_gen"
    METRIC_ANALYSIS = "metric_analysis"
    VERIFICATION = "verification"


# ── Result & Message ─────────────────────────────────────────────────


@dataclass
class SubAgentResult:
    """Structured output from a sub-agent execution.

    Agents communicate ONLY through these objects — no shared mutable state.
    """
    agent_role: AgentRole
    status: SubAgentStatus = SubAgentStatus.PENDING
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None
    context_fingerprint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.context_fingerprint is None:
            self.context_fingerprint = "none"

    # ── Serialization ──────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_role": self.agent_role.value,
            "status": self.status.value,
            "data": self.data,
            "artifacts": self.artifacts,
            "error": self.error,
            "context_fingerprint": self.context_fingerprint,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubAgentResult:
        return cls(
            agent_role=AgentRole(data["agent_role"]),
            status=SubAgentStatus(data["status"]),
            data=data.get("data", {}),
            artifacts=data.get("artifacts", []),
            error=data.get("error"),
            context_fingerprint=data.get("context_fingerprint", "none"),
            metadata=data.get("metadata", {}),
        )

    def compute_fingerprint(self, context_manager: ContextManager) -> str:
        """SHA-256 hash of the agent's context for P7 audit trail."""
        entries = context_manager.get_entries()
        content = "|".join(f"{e.role.value}:{e.content}" for e in entries)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def is_success(self) -> bool:
        return self.status == SubAgentStatus.SUCCESS

    def is_failed(self) -> bool:
        return self.status in (SubAgentStatus.FAILED, SubAgentStatus.REJECTED)


@dataclass
class CollaborationMessage:
    """Message passed between agents during collaboration."""
    sender: AgentRole
    receiver: AgentRole
    message_type: str       # "task_dispatch", "result", "error", "retry"
    payload: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender": self.sender.value,
            "receiver": self.receiver.value,
            "message_type": self.message_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }


# ── P7 Violation ─────────────────────────────────────────────────────


class P7ViolationError(Exception):
    """Raised when generation and verification contexts are improperly shared."""
    pass


# ── Base Sub-Agent ───────────────────────────────────────────────────

# Signature: (messages) -> str
ModelCaller = Any  # Callable[[list[dict[str, Any]]], str]


class BaseSubAgent(ABC):
    """Abstract base for all sub-agents.

    Each sub-agent owns its own ContextManager (context isolation).
    Sub-agents communicate through SubAgentResult objects only.
    """

    def __init__(
        self,
        role: AgentRole,
        context_manager: ContextManager,
        tool_registry: ToolRegistry,
        state_dir: str,
        permission_mode: PermissionMode = PermissionMode.DEFAULT,
        max_tokens: int = 8000,
    ) -> None:
        self.role = role
        self.context_manager = context_manager
        self.tool_registry = tool_registry
        self.state_dir = state_dir
        self.permission_mode = permission_mode
        self._persister = StatePersister(log_dir=state_dir, filename=f"agent_{role.value}_log.jsonl")
        self._model_caller: ModelCaller | None = None

    def set_model_caller(self, caller: ModelCaller) -> None:
        self._model_caller = caller

    @abstractmethod
    def run(self, message: CollaborationMessage) -> SubAgentResult:
        """Execute the sub-agent's task and return a structured result."""
        ...

    def _build_system_prompt(self) -> str:
        """Build a role-specific system prompt prefix."""
        role_prompts: dict[AgentRole, str] = {
            AgentRole.PLANNER: (
                "You are the Planner Agent. You receive GPU profiling targets, "
                "decompose them into sub-tasks, dispatch to specialists, and "
                "integrate final results."
            ),
            AgentRole.CODE_GEN: (
                "You are the Code Generation Agent. You write CUDA micro-benchmark "
                "kernels (pointer-chasing, bandwidth tests, etc.) and compile them. "
                "All code must be correct, measurable, and self-contained."
            ),
            AgentRole.METRIC_ANALYSIS: (
                "You are the Metric Analysis Agent. You parse Nsight Compute "
                "(ncu) output to identify performance bottlenecks: compute-bound, "
                "memory-bound, latency-bound, or cache-capacity cliffs."
            ),
            AgentRole.VERIFICATION: (
                "You are the Verification Agent. You independently review "
                "experimental results and methodology. You do NOT trust "
                "the generator's reasoning — you verify from first principles."
            ),
        }
        return role_prompts.get(self.role, f"You are the {self.role.value} agent.")

    def _persist_result(self, result: SubAgentResult) -> None:
        """P6: persist result to agent-specific log."""
        self._persister.log_entry(
            action="subagent_result",
            details={
                "role": result.agent_role.value,
                "status": result.status.value,
                "fingerprint": result.context_fingerprint,
            },
            result_data=result.to_dict(),
        )
