"""Sub-agent domain model — abstract base, roles, results, and messages.

P7 (生成与评估分离): This module defines the structural guarantees that
a VerificationAgent can never inherit a generator's context.

Refactored with Template Method pattern:
- BaseSubAgent.execute() defines the invariant skeleton
- Subclasses override _process() for stage-specific logic
- System prompts are delegated to agent_prompts module
- Enumerations extracted to enums.py to break circular imports
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.application.context import ContextManager, Role
from src.domain.agent_prompts import get_system_prompt
from src.domain.enums import AgentRole, PipelineStage, SubAgentStatus
from src.domain.permission import PermissionMode
from src.domain.tool_contract import ToolRegistry
from src.infrastructure.state_persist import StatePersister


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
    message_type: str
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


class P7ViolationError(Exception):
    """Raised when generation and verification contexts are improperly shared."""
    pass


ModelCaller = Any


class BaseSubAgent(ABC):
    """Abstract base for all sub-agents.

    Template Method pattern:
    - execute() is the invariant skeleton (context setup → process → fingerprint → persist)
    - _process() is the hook that subclasses override for stage-specific logic
    - _build_system_prompt() delegates to agent_prompts module

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

    def execute(self, message: CollaborationMessage) -> SubAgentResult:
        """Template method — invariant execution skeleton.

        1. P7 guard: verification agent context must be empty before execution
        2. Add system prompt to context
        3. Delegate to _process() for stage-specific logic
        4. Compute context fingerprint
        5. Persist result (P6)

        Subclasses should NOT override this method.
        Override _process() instead.
        """
        if self.role == AgentRole.VERIFICATION:
            existing = self.context_manager.get_entries()
            if len(existing) > 0:
                raise P7ViolationError(
                    f"Verification agent context must be empty before execution, "
                    f"but found {len(existing)} entries — possible context leak"
                )

        self.context_manager.add_entry(
            Role.SYSTEM, self._build_system_prompt(), token_count=30
        )

        result = self._process(message)

        result.context_fingerprint = result.compute_fingerprint(self.context_manager)
        self._persist_result(result)
        return result

    @abstractmethod
    def _process(self, message: CollaborationMessage) -> SubAgentResult:
        """Hook: stage-specific processing logic.

        Subclasses MUST override this instead of run().
        The template method execute() handles context setup,
        fingerprinting, and persistence.
        """
        ...

    def run(self, message: CollaborationMessage) -> SubAgentResult:
        """Public entry point — delegates to template method execute().

        Kept for backward compatibility with existing callers.
        """
        return self.execute(message)

    def _build_system_prompt(self) -> str:
        """Build a role-specific system prompt.

        Delegates to agent_prompts module for separation of concerns.
        Subclasses can override for custom prompt construction.
        """
        return get_system_prompt(self.role)

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
