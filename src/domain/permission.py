"""Permission checking and Mechanical Invariants — domain layer.

Enforces P2 (故障关闭): any unauthorised operation is denied by default.
Enforces M1 (先读后写) and M4 (防死循环) via InvariantTracker.
"""
from __future__ import annotations

from enum import Enum
from collections import defaultdict


# ── Permission Mode ────────────────────────────────────────────────


class PermissionMode(Enum):
    """分级自主机制 — spec.md §4.3."""

    CONSERVATIVE = "conservative"   # read-only
    DEFAULT = "default"             # modification needs approval
    RELAXED = "relaxed"             # edits auto-approve, shell still needs approval
    HIGH_AUTONOMY = "high_autonomy" # strategy-driven but hard boundaries remain

    @classmethod
    def from_string(cls, value: str) -> PermissionMode:
        for mode in cls:
            if mode.value == value:
                return mode
        raise ValueError(f"Unknown permission mode: {value!r}")


# ── Permission Checker ─────────────────────────────────────────────

# Map each mode to the set of permissions it allows.
_MODE_PERMISSIONS: dict[PermissionMode, frozenset[str]] = {
    PermissionMode.CONSERVATIVE: frozenset({"file:read"}),
    PermissionMode.DEFAULT: frozenset({"file:read", "file:write", "process:exec"}),
    PermissionMode.RELAXED: frozenset({"file:read", "file:write", "process:exec"}),
    PermissionMode.HIGH_AUTONOMY: frozenset({"file:read", "file:write", "process:exec"}),
}

# Permissions that always require approval regardless of mode.
_ALWAYS_REQUIRES_APPROVAL: dict[PermissionMode, frozenset[str]] = {
    PermissionMode.CONSERVATIVE: frozenset({"file:write", "process:exec"}),
    PermissionMode.DEFAULT: frozenset({"file:write", "process:exec"}),
    PermissionMode.RELAXED: frozenset({"process:exec"}),
    PermissionMode.HIGH_AUTONOMY: frozenset({"process:exec"}),
}


class PermissionChecker:
    """Checks whether a permission is granted under the current mode.

    P2: anything not explicitly allowed is denied.
    """

    def __init__(self, mode: PermissionMode = PermissionMode.DEFAULT) -> None:
        self._mode = mode

    @property
    def mode(self) -> PermissionMode:
        return self._mode

    def set_mode(self, mode: PermissionMode) -> None:
        self._mode = mode

    def is_allowed(self, permission: str) -> bool:
        """P2: unknown permissions are denied (fail-closed)."""
        return permission in _MODE_PERMISSIONS.get(self._mode, frozenset())

    def requires_approval(self, permission: str) -> bool:
        """Returns True if the permission requires human approval."""
        if not self.is_allowed(permission):
            return True  # denied → would need approval to proceed
        return permission in _ALWAYS_REQUIRES_APPROVAL.get(self._mode, frozenset())

    def list_allowed_permissions(self) -> list[str]:
        return sorted(_MODE_PERMISSIONS[self._mode])


# ── Mechanical Invariants ──────────────────────────────────────────


class InvariantTracker:
    """Tracks mechanical invariants that the architecture enforces.

    M1 (先读后写): file writes must be preceded by a successful read.
    M4 (防死循环): same failure pattern repeated 3 times forces termination.
    """

    def __init__(self) -> None:
        self._read_ledger: set[str] = set()
        self._created_files: set[str] = set()
        self._failure_counts: dict[str, int] = defaultdict(int)

    # ── M1: Read-before-Write ──────────────────────────────────────

    def record_read(self, file_path: str) -> None:
        """Record that a file has been successfully read."""
        self._read_ledger.add(file_path)

    def has_read(self, file_path: str) -> bool:
        return file_path in self._read_ledger

    def can_write(self, file_path: str) -> bool:
        """M1: writing is only allowed after the file has been read."""
        return file_path in self._read_ledger

    def clear_read(self, file_path: str) -> None:
        """Remove a read entry (e.g. after a successful write)."""
        self._read_ledger.discard(file_path)

    def get_read_ledger(self) -> set[str]:
        return set(self._read_ledger)

    # ── File Creation Tracking ─────────────────────────────────────

    def record_created(self, file_path: str) -> None:
        """Record that a file was created (bypassing M1 for generation tools)."""
        self._created_files.add(file_path)

    def was_created(self, file_path: str) -> bool:
        return file_path in self._created_files

    def get_created_files(self) -> set[str]:
        return set(self._created_files)

    # ── M4: Anti-loop ──────────────────────────────────────────────

    def record_failure(self, pattern: str) -> None:
        """Increment the failure counter for a given pattern."""
        self._failure_counts[pattern] += 1

    def reset_failure(self, pattern: str) -> None:
        """Reset the failure counter (e.g. after a successful retry)."""
        self._failure_counts[pattern] = 0

    def get_failure_count(self, pattern: str) -> int:
        return self._failure_counts.get(pattern, 0)

    def should_terminate(self, pattern: str) -> bool:
        """M4: terminate if the same failure pattern repeats 3 times."""
        return self._failure_counts.get(pattern, 0) >= 3
