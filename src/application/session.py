"""Session State & Resume — application layer.

P6: All session state must be persisted to disk.
--resume recovers from the last saved checkpoint.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class SessionState:
    """Persistent state for a single agent session.

    This is the authoritative record of "what we are doing" and
    "how far we have progressed." If it is not on disk, it does not exist.
    """

    session_id: str
    goal: str
    step_count: int = 0
    is_complete: bool = False
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    context_snapshot: dict[str, Any] = field(default_factory=dict)

    def increment_step(self) -> None:
        self.step_count += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def mark_complete(self) -> None:
        self.is_complete = True
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def mark_error(self, error: str) -> None:
        self.error = error
        self.updated_at = datetime.now(timezone.utc).isoformat()

    # ── Serialization ──────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "goal": self.goal,
            "step_count": self.step_count,
            "is_complete": self.is_complete,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "context_snapshot": self.context_snapshot,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionState:
        return cls(
            session_id=data["session_id"],
            goal=data["goal"],
            step_count=data.get("step_count", 0),
            is_complete=data.get("is_complete", False),
            error=data.get("error"),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
            context_snapshot=data.get("context_snapshot", {}),
        )


class SessionManager:
    """Manages session lifecycle: create, save, load, resume, delete."""

    def __init__(self, state_dir: str) -> None:
        self._state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)

    def _session_path(self, session_id: str) -> str:
        return os.path.join(self._state_dir, f"session_{session_id}.json")

    def create_session(self, session_id: str, goal: str) -> SessionState:
        return SessionState(session_id=session_id, goal=goal)

    def save_session(self, session: SessionState) -> None:
        path = self._session_path(session.session_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)

    def load_session(self, session_id: str) -> SessionState | None:
        path = self._session_path(session_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SessionState.from_dict(data)

    def resume(self, session_id: str, new_goal: str | None = None) -> SessionState:
        """Resume an existing session or create a new one.

        Args:
            session_id: The session to resume.
            new_goal: If provided, overrides the original goal.
        """
        existing = self.load_session(session_id)
        if existing is not None:
            if new_goal is not None:
                existing.goal = new_goal
            return existing
        # No existing session — create a fresh one
        return self.create_session(session_id, goal=new_goal or "")

    def delete_session(self, session_id: str) -> None:
        path = self._session_path(session_id)
        if os.path.exists(path):
            os.remove(path)

    def list_sessions(self) -> list[str]:
        names: list[str] = []
        if not os.path.isdir(self._state_dir):
            return names
        for fname in os.listdir(self._state_dir):
            if fname.startswith("session_") and fname.endswith(".json"):
                name = fname[len("session_"):-len(".json")]
                names.append(name)
        return names
