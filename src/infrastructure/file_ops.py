"""File operations with read-before-write enforcement — infrastructure layer.

Wraps raw file I/O with invariant checks and sandbox boundaries.
M1 (先读后写) is enforced: no file can be modified without a prior
successful read through this module.
M2 (唯一锚点编辑) is enforced: anchored edits require exact line or hash match.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

from src.domain.permission import InvariantTracker


class FileOperations:
    """File I/O with mandatory read-before-write and sandbox confinement."""

    def __init__(self, sandbox_root: str, prior_reads: set[str] | None = None) -> None:
        self._sandbox = os.path.abspath(sandbox_root)
        self._tracker = InvariantTracker()
        # M1: restore read ledger from prior state (e.g. after restart)
        for path in (prior_reads or set()):
            self._tracker.record_read(path)

    # ── Path Safety ────────────────────────────────────────────────

    def _resolve(self, path: str) -> str:
        """Resolve and validate that *path* is inside the sandbox."""
        resolved = os.path.abspath(os.path.normpath(path))
        sandbox = self._sandbox.rstrip(os.sep) + os.sep
        if not (resolved.startswith(sandbox) or resolved == self._sandbox.rstrip(os.sep)):
            raise PermissionError(
                f"Path escape blocked: {path!r} resolves outside sandbox {self._sandbox}"
            )
        return resolved

    # ── Read ───────────────────────────────────────────────────────

    def read(self, file_path: str) -> str:
        """Read a file and record it in the M1 ledger.

        Raises:
            FileNotFoundError: if the file does not exist.
            PermissionError: if the path escapes the sandbox.
        """
        resolved = self._resolve(file_path)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"File not found: {resolved}")
        with open(resolved, "r", encoding="utf-8") as f:
            content = f.read()
        self._tracker.record_read(resolved)
        return content

    # ── Write ──────────────────────────────────────────────────────

    def write(self, file_path: str, content: str) -> int:
        """Write content to a file.

        M1 enforcement:
        - Modifying an existing file requires a prior read through this module.
        - Creating a NEW file (that doesn't exist on disk) is always allowed,
          since there is nothing to read. The file is tracked as created.

        Raises:
            PermissionError: if the file was not previously read (M1 violation)
                             or if the path escapes the sandbox.
        """
        resolved = self._resolve(file_path)

        # New file creation: allow since there's nothing to read (M1 applies to edits)
        if not os.path.exists(resolved):
            with open(resolved, "w", encoding="utf-8") as f:
                f.write(content)
            bytes_written = len(content.encode("utf-8"))
            self._tracker.record_created(resolved)
            return bytes_written

        # Existing file: require prior read (M1)
        if not self._tracker.can_write(resolved):
            raise PermissionError(
                f"M1 violation: cannot write '{resolved}' without a prior read. "
                "Call read() first."
            )
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(content)
        bytes_written = len(content.encode("utf-8"))
        # Consume the read ledger entry after successful write
        self._tracker.clear_read(resolved)
        return bytes_written

    # ── Create (bypasses M1 for generation tools) ──────────────────

    def create(self, file_path: str, content: str) -> int:
        """Create a new file, bypassing M1 read-before-write.

        This is intended for tools that generate new content (e.g.
        generate_microbenchmark) where no prior read is possible.
        The creation is tracked in the tracker for audit purposes.

        Raises:
            PermissionError: if the path escapes the sandbox.
        """
        resolved = self._resolve(file_path)
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(content)
        bytes_written = len(content.encode("utf-8"))
        self._tracker.record_created(resolved)
        return bytes_written

    # ── Ledger Access (for persistence layer) ──────────────────────

    @property
    def tracker(self) -> InvariantTracker:
        return self._tracker

    # ── M2: Anchored Edit ──────────────────────────────────────────

    def anchored_write(
        self,
        file_path: str,
        content: str,
        *,
        line_range: tuple[int, int] | None = None,
        expected_hash: str | None = None,
    ) -> int:
        """Write content with M2 anchor verification.

        Args:
            file_path: Target file path.
            content: New content (full file or replacement for line_range).
            line_range: (start, end) 1-indexed line range to replace.
                        If None, replaces entire file.
            expected_hash: SHA-256 hash of current file content.
                          If provided, verifies file hasn't changed since read.

        Raises:
            PermissionError: M1 violation (no prior read) or sandbox escape.
            ValueError: Hash mismatch — file content changed since read.
            IndexError: line_range is out of bounds.
        """
        resolved = self._resolve(file_path)
        if not self._tracker.can_write(resolved):
            raise PermissionError(
                f"M1 violation: cannot write '{resolved}' without a prior read."
            )

        # Read current content for anchor verification
        with open(resolved, "r", encoding="utf-8") as f:
            current = f.read()

        # Verify hash anchor if provided
        if expected_hash is not None:
            actual_hash = hashlib.sha256(current.encode("utf-8")).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(
                    f"M2 anchor violation: file '{resolved}' content changed "
                    f"since read. Expected hash {expected_hash[:16]}..., "
                    f"got {actual_hash[:16]}..."
                )

        # Line-range anchored replacement
        if line_range is not None:
            start, end = line_range
            lines = current.splitlines(True)  # keep line endings
            if start < 1 or end > len(lines) or start > end:
                raise IndexError(
                    f"M2: line_range ({start}, {end}) out of bounds "
                    f"for file with {len(lines)} lines"
                )
            # Replace lines (convert 1-indexed to 0-indexed)
            new_lines = lines[:start - 1] + [content] + lines[end:]
            content = "".join(new_lines)

        # Write the result
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(content)
        bytes_written = len(content.encode("utf-8"))
        self._tracker.clear_read(resolved)
        return bytes_written
