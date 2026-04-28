"""Context Manager — application layer.

P3: Context engineering over prompt engineering. Context is dynamically
assembled, compressed, and managed — not a static prompt.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("context_manager")

class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Priority(Enum):
    PERMANENT = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    DISPOSABLE = 4


@dataclass
class ContextEntry:
    """A single entry in the conversation context."""

    role: Role
    content: str
    token_count: int = 0
    priority: Priority = Priority.MEDIUM

    def __post_init__(self) -> None:
        if self.token_count <= 0:
            self.token_count = max(1, len(self.content) // 3 + 1)


def _classify_priority(role: Role, content: str) -> Priority:
    """Classify entry priority based on role and content.

    Priority levels:
    - PERMANENT: Architecture info, user instructions, P7 constraints
    - HIGH: Successful tool outputs (compile/execute results), measurements
    - MEDIUM: Error messages, LLM natural language responses
    - LOW: Control Plane snapshots, repetitive guidance, design principles
    - DISPOSABLE: Old anti-loop guidance, duplicate entries, short responses
    """
    if role == Role.SYSTEM:
        # Architecture detection info — always preserve
        if "Detected GPU architecture" in content or "arch=sm_" in content:
            return Priority.PERMANENT
        # Control Plane — can be replaced each turn
        if "[ControlPlane]" in content:
            return Priority.LOW
        # Design principle injections — DISPOSABLE (repetitive, can be regenerated)
        if "DESIGN PRINCIPLES" in content or "design principle" in content.lower():
            return Priority.DISPOSABLE
        # Next target guidance — HIGH (critical for workflow)
        if "NEXT TARGET" in content or "NEXT: Write CUDA code" in content:
            return Priority.HIGH
        # Compilation success + execute hint — HIGH
        if "Compilation #" in content and "IMMEDIATELY call execute_binary" in content:
            return Priority.HIGH
        # Measurement recording — HIGH
        if "MEASUREMENTS RECORDED" in content:
            return Priority.HIGH
        # Critical range errors — HIGH (must be addressed)
        if "CRITICAL RANGE ERROR" in content or "CRITICAL MEASUREMENT RANGE ERROR" in content:
            return Priority.HIGH
        # Code-fix bridge — HIGH (actionable fixes)
        if "CODE-FIX BRIDGE" in content:
            return Priority.HIGH
        # Reference value hints — HIGH (guides LLM when CodeGen fails)
        if "REFERENCE VALUE" in content:
            return Priority.HIGH
        # Repeated compilation errors — DISPOSABLE (noise after first occurrence)
        if "COMPILATION FAILED" in content or "compile_cuda REQUIRES" in content:
            return Priority.DISPOSABLE
        # Error guidance — low importance (noise if repeated)
        if "⚠️" in content or "ERROR" in content:
            return Priority.LOW
        # Tool usage instructions — medium (only needed once)
        if "TOOL USAGE" in content or "MANDATORY" in content:
            return Priority.MEDIUM
        # Other system messages — medium
        return Priority.MEDIUM

    if role == Role.USER:
        return Priority.HIGH

    if role == Role.ASSISTANT:
        # Tool call results — classify by success/failure
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                status = data.get("status", "")
                if status in ("success", "success_with_warning"):
                    return Priority.HIGH
                if status == "error":
                    return Priority.MEDIUM
                if "binary_path" in data and data.get("binary_path"):
                    return Priority.HIGH
                if "stdout" in data and data.get("stdout"):
                    return Priority.HIGH
        except (json.JSONDecodeError, TypeError):
            pass
        # Natural language responses — low priority
        if len(content) < 100:
            return Priority.DISPOSABLE
        return Priority.LOW

    return Priority.MEDIUM


def _estimate_tokens(content: str) -> int:
    """Estimate token count based on content type.
    
    Code/JSON: ~2.5 chars/token (denser, more special chars)
    Natural language: ~4 chars/token (sparser)
    Mixed: ~3 chars/token (default)
    """
    if not content:
        return 0
    
    special_char_count = sum(1 for c in content if c in "{}[]();=<>\"':,")
    total_chars = len(content)
    
    if total_chars == 0:
        return 0
    
    special_ratio = special_char_count / total_chars
    
    if special_ratio > 0.15:
        estimated = int(total_chars / 2.5)
    elif special_ratio < 0.05:
        estimated = int(total_chars / 4.0)
    else:
        estimated = int(total_chars / 3.0)
    
    return max(10, estimated)


def _summarize_entry(entry: ContextEntry) -> ContextEntry:
    """Summarize a context entry to reduce token usage.

    Preserves key information while reducing verbosity:
    - Tool results: keep status, binary_path, key measurements from stdout
    - Error messages: keep error type and hint, truncate stderr
    - Natural language: truncate to first sentence
    - Design principles: keep first 200 chars only
    """
    if entry.role == Role.ASSISTANT:
        try:
            data = json.loads(entry.content)
            if isinstance(data, dict):
                summary_parts = []
                if "status" in data:
                    summary_parts.append(f"status={data['status']}")
                if "success" in data:
                    summary_parts.append(f"success={data['success']}")
                if "binary_path" in data and data["binary_path"]:
                    summary_parts.append(f"binary_path={data['binary_path']}")
                if "tool" in data:
                    summary_parts.append(f"tool={data['tool']}")
                if "stdout" in data and data["stdout"]:
                    stdout = str(data["stdout"])
                    measurement_lines = [l for l in stdout.splitlines()
                                         if l.strip() and ":" in l and not l.strip().startswith("//")]
                    if measurement_lines:
                        summary_parts.append(f"measurements=[{'; '.join(measurement_lines[:10])}]")
                    else:
                        summary_parts.append(f"stdout={stdout[:150]}")
                if "output" in data and data["output"]:
                    output = str(data["output"])[:150]
                    summary_parts.append(f"output={output}")
                if "errors" in data and data["errors"]:
                    errors = str(data["errors"])
                    error_lines = errors.splitlines()
                    if len(error_lines) > 3:
                        summary_parts.append(f"errors=[{'; '.join(error_lines[:3])}; ...{len(error_lines)-3} more]")
                    else:
                        summary_parts.append(f"errors={errors[:200]}")
                if "stderr" in data and data["stderr"]:
                    stderr = str(data["stderr"])[:200]
                    summary_parts.append(f"stderr={stderr}")
                if "next_action" in data:
                    summary_parts.append(f"next_action={data['next_action']}")
                if "parsed_metrics" in data and data["parsed_metrics"]:
                    metrics = str(data["parsed_metrics"])[:200]
                    summary_parts.append(f"metrics={metrics}")
                if "has_warning" in data:
                    summary_parts.append(f"has_warning={data['has_warning']}")
                if "return_code" in data:
                    summary_parts.append(f"return_code={data['return_code']}")
                if summary_parts:
                    summary = "[SUMMARY] " + ", ".join(summary_parts)
                    return ContextEntry(
                        role=entry.role,
                        content=summary,
                        priority=entry.priority,
                    )
        except (json.JSONDecodeError, TypeError):
            pass
        # Truncate long natural language
        if len(entry.content) > 200:
            return ContextEntry(
                role=entry.role,
                content=entry.content[:200] + "...[truncated]",
                priority=entry.priority,
            )

    if entry.role == Role.SYSTEM:
        # Truncate long design principle injections
        if "DESIGN PRINCIPLES" in entry.content or "design principle" in entry.content.lower():
            if len(entry.content) > 500:
                return ContextEntry(
                    role=entry.role,
                    content=entry.content[:500] + "\n...[principle truncated]",
                    priority=entry.priority,
                )
        # Truncate long error guidance
        if "⚠️" in entry.content and len(entry.content) > 400:
            return ContextEntry(
                role=entry.role,
                content=entry.content[:400] + "\n...[guidance truncated]",
                priority=entry.priority,
            )

    return entry


class ContextManager:
    """Manages the dynamic assembly and compression of model context.

    Features:
    - Priority-based entry classification (PERMANENT > HIGH > MEDIUM > LOW > DISPOSABLE)
    - Smart compression: summarize before deleting, preserve important info
    - System entries protected from compression
    - Control Plane entries tracked separately to avoid overwriting
    """

    COMPRESSION_RATIO = 0.8
    MAX_DUPLICATE_ENTRIES = 2

    def __init__(self, max_tokens: int = 16000) -> None:
        self._entries: list[ContextEntry] = []
        self._max_tokens = max_tokens
        self._total_tokens = 0
        self._content_fingerprint_counts: dict[str, int] = {}

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @staticmethod
    def _fingerprint(content: str) -> str:
        """Create a short fingerprint for duplicate detection."""
        normalized = re.sub(r'\d+', 'N', content[:200])
        return hashlib.md5(normalized.encode()).hexdigest()[:12]

    def add_entry(
        self,
        role: Role,
        content: str,
        token_count: int = 0,
    ) -> None:
        priority = _classify_priority(role, content)
        if token_count <= 0:
            token_count = _estimate_tokens(content)

        fp = self._fingerprint(content)
        count = self._content_fingerprint_counts.get(fp, 0)
        if count >= self.MAX_DUPLICATE_ENTRIES and priority not in (Priority.PERMANENT, Priority.HIGH):
            logger.info(
                "[CONTEXT] DEDUP: Skipping duplicate entry (count=%d) preview=%s",
                count, content[:80].replace('\n', '\\n')
            )
            return

        self._content_fingerprint_counts[fp] = count + 1
        entry = ContextEntry(role=role, content=content, token_count=token_count, priority=priority)
        self._entries.append(entry)
        self._total_tokens += entry.token_count
        logger.info(
            "[CONTEXT] ADD: role=%s priority=%s tokens=%d total_tokens=%d preview=%s",
            role.value, priority.name, token_count, self._total_tokens,
            content[:100].replace('\n', '\\n')
        )

    def update_system_entry(self, content: str, token_count: int = 0) -> None:
        """Replace the Control Plane SYSTEM entry or add a new one if none exists.
        
        NEVER replaces PERMANENT priority entries (architecture detection, etc.).
        Only replaces entries that are LOW priority (Control Plane snapshots)
        or entries that contain [ControlPlane] markers.
        """
        cp_marker = "[ControlPlane]"
        existing_idx = None
        for i, e in enumerate(self._entries):
            if e.role == Role.SYSTEM and cp_marker in e.content:
                existing_idx = i
                break

        if existing_idx is None:
            for i, e in enumerate(self._entries):
                if e.role == Role.SYSTEM and e.priority == Priority.LOW and ("Turn " in e.content or "Progress:" in e.content):
                    existing_idx = i
                    break

        if existing_idx is not None and self._entries[existing_idx].priority == Priority.PERMANENT:
            existing_idx = None

        priority = _classify_priority(Role.SYSTEM, content)
        new_entry = ContextEntry(role=Role.SYSTEM, content=content, token_count=token_count, priority=priority)

        if existing_idx is not None:
            old = self._entries[existing_idx]
            self._total_tokens -= old.token_count
            self._entries[existing_idx] = new_entry
            self._total_tokens += new_entry.token_count
            logger.info(
                "[CONTEXT] UPDATE_SYSTEM: replaced idx=%d old_tokens=%d new_tokens=%d total_tokens=%d",
                existing_idx, old.token_count, new_entry.token_count, self._total_tokens
            )
        else:
            self._entries.append(new_entry)
            self._total_tokens += new_entry.token_count
            logger.info(
                "[CONTEXT] ADD_SYSTEM: new_tokens=%d total_tokens=%d preview=%s",
                new_entry.token_count, self._total_tokens,
                content[:100].replace('\n', '\\n')
            )

    def get_entries(self) -> list[ContextEntry]:
        """Return a copy to prevent external mutation."""
        return list(self._entries)

    def is_over_budget(self) -> bool:
        return self._total_tokens > self._max_tokens

    def compress(self) -> int:
        """Smart compression: summarize before deleting, respect priorities.

        Compression strategy (in order):
        1. Remove DISPOSABLE entries (old guidance, short responses, design principles)
        2. Remove LOW priority entries (Control Plane snapshots, error guidance)
        3. Summarize MEDIUM priority entries (tool usage instructions)
        4. Remove oldest MEDIUM entries if still over budget
        5. Never remove PERMANENT or HIGH priority entries

        Returns the number of entries removed.
        """
        target = int(self._max_tokens * self.COMPRESSION_RATIO)
        if self._total_tokens <= target:
            return 0

        initial_tokens = self._total_tokens
        initial_entries = len(self._entries)
        removed_count = 0

        logger.info(
            "[CONTEXT] COMPRESSION START: total_tokens=%d target=%d entries=%d",
            self._total_tokens, target, initial_entries
        )

        # Phase 1: Remove DISPOSABLE entries
        new_entries = []
        phase1_removed = 0
        for e in self._entries:
            if e.priority == Priority.DISPOSABLE and self._total_tokens > target:
                self._total_tokens -= e.token_count
                removed_count += 1
                phase1_removed += 1
            else:
                new_entries.append(e)
        self._entries = new_entries
        if phase1_removed > 0:
            logger.info("[CONTEXT] COMPRESSION Phase1: removed %d DISPOSABLE entries, tokens=%d", phase1_removed, self._total_tokens)

        if self._total_tokens <= target:
            return removed_count

        # Phase 2: Remove LOW priority entries (not just summarize)
        new_entries = []
        phase2_removed = 0
        for e in self._entries:
            if e.priority == Priority.LOW and self._total_tokens > target:
                self._total_tokens -= e.token_count
                removed_count += 1
                phase2_removed += 1
            else:
                new_entries.append(e)
        self._entries = new_entries
        if phase2_removed > 0:
            logger.info("[CONTEXT] COMPRESSION Phase2: removed %d LOW entries, tokens=%d", phase2_removed, self._total_tokens)

        if self._total_tokens <= target:
            return removed_count

        # Phase 3: Summarize MEDIUM priority entries (oldest first)
        medium_entries = [(i, e) for i, e in enumerate(self._entries) if e.priority == Priority.MEDIUM]
        phase3_summarized = 0
        for idx, entry in medium_entries:
            if self._total_tokens <= target:
                break
            summarized = _summarize_entry(entry)
            if summarized.content != entry.content:
                saved = entry.token_count - summarized.token_count
                self._total_tokens -= saved
                self._entries[idx] = summarized
                phase3_summarized += 1
        if phase3_summarized > 0:
            logger.info("[CONTEXT] COMPRESSION Phase3: summarized %d MEDIUM entries, tokens=%d", phase3_summarized, self._total_tokens)

        if self._total_tokens <= target:
            return removed_count

        # Phase 4: Remove oldest MEDIUM entries if still over budget
        new_entries = []
        phase4_removed = 0
        for e in self._entries:
            if e.priority == Priority.MEDIUM and self._total_tokens > target:
                self._total_tokens -= e.token_count
                removed_count += 1
                phase4_removed += 1
            else:
                new_entries.append(e)
        self._entries = new_entries
        if phase4_removed > 0:
            logger.info("[CONTEXT] COMPRESSION Phase4: removed %d MEDIUM entries, tokens=%d", phase4_removed, self._total_tokens)

        logger.info(
            "[CONTEXT] COMPRESSION END: removed=%d tokens_before=%d tokens_after=%d entries_before=%d entries_after=%d",
            removed_count, initial_tokens, self._total_tokens, initial_entries, len(self._entries)
        )

        return removed_count

    def clear(self) -> None:
        self._entries.clear()
        self._total_tokens = 0

    def to_messages(self) -> list[dict[str, Any]]:
        """Convert entries to the standard message format for LLM APIs."""
        return [
            {"role": e.role.value, "content": e.content if e.content is not None else ""}
            for e in self._entries
        ]
