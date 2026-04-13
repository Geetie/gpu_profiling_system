"""Context Manager — application layer.

P3: Context engineering over prompt engineering. Context is dynamically
assembled, compressed, and managed — not a static prompt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ContextEntry:
    """A single entry in the conversation context."""

    role: Role
    content: str
    token_count: int = 0

    def __post_init__(self) -> None:
        if self.token_count <= 0:
            # Rough estimate: ~4 chars per token for English text
            self.token_count = max(1, len(self.content) // 4 + 1)


class ContextManager:
    """Manages the dynamic assembly and compression of model context.

    - Entries are appended as the conversation progresses.
    - When token budget is exceeded, oldest non-system entries are removed.
    - System entries are protected from compression.
    """

    # Compression targets ~80% of max_tokens to leave headroom
    COMPRESSION_RATIO = 0.8

    def __init__(self, max_tokens: int = 8000) -> None:
        self._entries: list[ContextEntry] = []
        self._max_tokens = max_tokens
        self._total_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def add_entry(
        self,
        role: Role,
        content: str,
        token_count: int = 0,
    ) -> None:
        entry = ContextEntry(role=role, content=content, token_count=token_count)
        self._entries.append(entry)
        self._total_tokens += entry.token_count

    def update_system_entry(self, content: str, token_count: int = 0) -> None:
        """Replace the existing SYSTEM entry or add a new one if none exists.

        Prevents duplicate system context injection across turns (BUG-1 fix).
        """
        existing_idx = None
        for i, e in enumerate(self._entries):
            if e.role == Role.SYSTEM:
                existing_idx = i
                break

        new_entry = ContextEntry(role=Role.SYSTEM, content=content, token_count=token_count)

        if existing_idx is not None:
            old = self._entries[existing_idx]
            self._total_tokens -= old.token_count
            self._entries[existing_idx] = new_entry
            self._total_tokens += new_entry.token_count
        else:
            self._entries.append(new_entry)
            self._total_tokens += new_entry.token_count

    def get_entries(self) -> list[ContextEntry]:
        """Return a copy to prevent external mutation."""
        return list(self._entries)

    def is_over_budget(self) -> bool:
        return self._total_tokens > self._max_tokens

    def compress(self) -> int:
        """Remove oldest non-system entries until under budget.

        Returns the number of entries removed.
        If system entries alone exceed the target, all non-system entries
        are dropped (cannot fit anything).
        Atomic: builds new list without intermediate None markers.
        """
        target = int(self._max_tokens * self.COMPRESSION_RATIO)

        # Identify entries to keep: all system entries + newest N non-system entries
        system_entries = [e for e in self._entries if e.role == Role.SYSTEM]
        non_system = [e for e in self._entries if e.role != Role.SYSTEM]

        system_tokens = sum(e.token_count for e in system_entries)
        remaining_budget = target - system_tokens

        # Edge case: system entries alone exceed target — drop all non-system entries
        if remaining_budget <= 0:
            removed = len(non_system)
            self._entries = system_entries
            self._total_tokens = system_tokens
            return removed

        kept_non_system: list[ContextEntry] = []
        current_tokens = 0
        # Iterate from newest to oldest (reverse) to keep the most recent entries
        for entry in reversed(non_system):
            if current_tokens + entry.token_count <= remaining_budget:
                kept_non_system.append(entry)
                current_tokens += entry.token_count
            # Always keep at least the newest non-system entry
            if not kept_non_system:
                kept_non_system.append(entry)
                current_tokens += entry.token_count
                break

        # Atomic rebuild (no None markers)
        # Keep all system entries first, then the kept non-system entries (in original order)
        kept_non_system.reverse()  # restore original order
        self._entries = system_entries + kept_non_system
        self._total_tokens = system_tokens + current_tokens

        return len(non_system) - len(kept_non_system)

    def clear(self) -> None:
        self._entries.clear()
        self._total_tokens = 0

    def to_messages(self) -> list[dict[str, Any]]:
        """Convert entries to the standard message format for LLM APIs."""
        return [
            {"role": e.role.value, "content": e.content}
            for e in self._entries
        ]
