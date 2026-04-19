"""State persistence — infrastructure layer.

Enforces P6 (状态必须落盘): every significant action is appended as a
JSON line to `session_log.jsonl`.  If a state change is not persisted
to disk, it is treated as if it never happened.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any


class StatePersister:
    """Append-only JSONL session logger with structured entry helpers."""

    def __init__(self, log_dir: str, filename: str = "session_log.jsonl") -> None:
        self._log_path = os.path.join(log_dir, filename)
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self._log_path)), exist_ok=True)

    @property
    def log_path(self) -> str:
        return self._log_path

    # ── Core ───────────────────────────────────────────────────────

    def _append(self, entry: dict[str, Any]) -> None:
        """Atomically append one JSON line to the log file."""
        entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        line = json.dumps(entry, ensure_ascii=False)
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ── Structured helpers ─────────────────────────────────────────

    def log_entry(self, action: str, details: dict[str, Any] | None = None, **extra: Any) -> None:
        """Log a generic action."""
        payload: dict[str, Any] = {"action": action}
        if details is not None:
            payload["details"] = details
        payload.update(extra)
        self._append(payload)

    def log_tool_execution(
        self,
        tool_name: str,
        inputs: dict[str, Any] | None = None,
        status: str = "success",
        output: Any = None,
    ) -> None:
        self._append({
            "action": "tool_execution",
            "tool_name": tool_name,
            "inputs": inputs or {},
            "status": status,
            "output": output,
        })

    def get_last_tool_execution(self, tool_name: str) -> dict[str, Any] | None:
        """Get the last execution log for a specific tool.

        Args:
            tool_name: Name of the tool to search for

        Returns:
            dict | None: The last log entry for this tool, or None if not found
        """
        if not os.path.exists(self._log_path):
            return None

        last_entry = None
        with open(self._log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("tool_name") == tool_name:
                        last_entry = entry
                except (json.JSONDecodeError, TypeError):
                    pass

        return last_entry

    def log_permission_decision(
        self,
        permission: str,
        mode: str,
        decision: str,
        reason: str | None = None,
    ) -> None:
        self._append({
            "action": "permission_decision",
            "permission": permission,
            "mode": mode,
            "decision": decision,
            "reason": reason,
        })

    def log_error(
        self,
        error_type: str,
        context: str,
        message: str,
    ) -> None:
        self._append({
            "action": "error",
            "error_type": error_type,
            "context": context,
            "message": message,
        })

    def log_invariant_violation(
        self,
        invariant: str,
        detail: str,
    ) -> None:
        """Record a mechanical invariant violation (M1–M4)."""
        self._append({
            "action": "invariant_violation",
            "invariant": invariant,
            "detail": detail,
        })

    # ── Read-back ──────────────────────────────────────────────────

    def load_history(self) -> list[dict[str, Any]]:
        """Load all log entries (returns [] if the file does not exist yet)."""
        if not os.path.exists(self._log_path):
            return []
        entries: list[dict[str, Any]] = []
        with open(self._log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
