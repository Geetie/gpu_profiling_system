"""Approval queue for tool execution — application layer.

Manages human-in-the-loop approval for operations that require it
under the current permission mode. Decisions are persisted (P6).
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.domain.permission import PermissionMode
from src.infrastructure.state_persist import StatePersister


class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_REJECTED = "auto_rejected"


@dataclass
class ApprovalRequest:
    """A pending tool execution approval request."""
    id: str
    tool_name: str
    arguments: dict[str, Any]
    permissions: list[str]
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    responded_at: str | None = None
    reason: str | None = None
    _event: threading.Event = field(default_factory=threading.Event, repr=False)


class ApprovalQueue:
    """Thread-safe queue for tool execution approval requests.

    All decisions are persisted via StatePersister (P6).
    """

    def __init__(self, state_dir: str, persister: StatePersister | None = None) -> None:
        self._requests: dict[str, ApprovalRequest] = {}
        self._lock = threading.Lock()
        self._state_dir = state_dir
        self._persister = persister or StatePersister(log_dir=state_dir, filename="approval_log.jsonl")

    def submit(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        permissions: list[str],
        mode: PermissionMode,
    ) -> ApprovalRequest:
        """Submit a new approval request.

        Auto-rejects in CONSERVATIVE mode.
        """
        request_id = f"{tool_name}_{uuid.uuid4().hex[:12]}"

        # CONSERVATIVE mode: auto-reject all modifications
        if mode == PermissionMode.CONSERVATIVE:
            request = ApprovalRequest(
                id=request_id,
                tool_name=tool_name,
                arguments=arguments,
                permissions=permissions,
                status=ApprovalStatus.AUTO_REJECTED,
                reason="Auto-rejected in CONSERVATIVE mode",
                responded_at=datetime.now(timezone.utc).isoformat(),
            )
            with self._lock:
                self._requests[request_id] = request
            self._persist_decision(request)
            return request

        request = ApprovalRequest(
            id=request_id,
            tool_name=tool_name,
            arguments=arguments,
            permissions=permissions,
        )
        with self._lock:
            self._requests[request_id] = request

        self._persister.log_entry(
            "approval_request",
            details={
                "id": request_id,
                "tool_name": tool_name,
                "permissions": permissions,
                "mode": mode.value,
            },
        )
        return request

    def respond(self, request_id: str, approved: bool, reason: str | None = None) -> None:
        """Respond to an approval request.

        Raises:
            KeyError: if request_id not found
            ValueError: if request already responded to
        """
        with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                raise KeyError(f"Approval request '{request_id}' not found")
            if request.status != ApprovalStatus.PENDING:
                raise ValueError(
                    f"Request '{request_id}' already responded: {request.status.value}"
                )

            request.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
            request.responded_at = datetime.now(timezone.utc).isoformat()
            request.reason = reason
            request._event.set()

        self._persist_decision(request)

    def wait_for_decision(self, request: ApprovalRequest, timeout: float = 60.0) -> ApprovalStatus:
        """Block until the request is decided or timeout.

        Returns:
            The final status (APPROVED, REJECTED, or AUTO_REJECTED on timeout).
        """
        if request.status != ApprovalStatus.PENDING:
            return request.status

        decided = request._event.wait(timeout=timeout)
        if not decided:
            # Timeout — treat as rejection
            with self._lock:
                if request.status == ApprovalStatus.PENDING:
                    request.status = ApprovalStatus.REJECTED
                    request.responded_at = datetime.now(timezone.utc).isoformat()
                    request.reason = f"Timed out after {timeout}s"
                    request._event.set()
            self._persist_decision(request)

        return request.status

    def get_pending(self) -> list[ApprovalRequest]:
        """Return all pending (undecided) requests."""
        with self._lock:
            return [r for r in self._requests.values() if r.status == ApprovalStatus.PENDING]

    def get_request(self, request_id: str) -> ApprovalRequest | None:
        """Get a specific request by ID."""
        with self._lock:
            return self._requests.get(request_id)

    def _persist_decision(self, request: ApprovalRequest) -> None:
        """Persist the approval decision (P6)."""
        self._persister.log_entry(
            "approval_decision",
            details={
                "id": request.id,
                "tool_name": request.tool_name,
                "status": request.status.value,
                "reason": request.reason,
                "responded_at": request.responded_at,
            },
        )
