"""Control Plane — application layer.

Automatically injects system state before each inference cycle.
All injected data is READ-ONLY: the model cannot modify it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SystemSnapshot:
    """Immutable snapshot of the current system environment."""

    working_dir: str
    env_vars: dict[str, str]
    rule_files: dict[str, str]
    memory_summary: list[str]

    @classmethod
    def capture(
        cls,
        rule_dir: str | None = None,
        cuda_env_keys: tuple[str, ...] = (
            "CUDA_VISIBLE_DEVICES",
            "CUDA_HOME",
            "NVCC_PREV_FLAGS",
        ),
        max_rule_file_bytes: int = 64 * 1024,  # 64 KB cap per file
    ) -> SystemSnapshot:
        working_dir = os.getcwd()

        # Capture CUDA-relevant env vars
        env_vars: dict[str, str] = {}
        for key in cuda_env_keys:
            value = os.environ.get(key)
            if value is not None:
                env_vars[key] = value

        # Load rule files if directory exists (size-limited to prevent OOM)
        rule_files: dict[str, str] = {}
        if rule_dir and os.path.isdir(rule_dir):
            for fname in os.listdir(rule_dir):
                fpath = os.path.join(rule_dir, fname)
                if os.path.isfile(fpath):
                    try:
                        file_size = os.path.getsize(fpath)
                        if file_size > max_rule_file_bytes:
                            with open(fpath, "r", encoding="utf-8") as f:
                                rule_files[fname] = f.read(max_rule_file_bytes)
                                rule_files[fname] += f"\n... (truncated, {file_size} bytes total)"
                        else:
                            with open(fpath, "r", encoding="utf-8") as f:
                                rule_files[fname] = f.read()
                    except (OSError, UnicodeDecodeError):
                        pass

        return cls(
            working_dir=working_dir,
            env_vars=env_vars,
            rule_files=rule_files,
            memory_summary=[],
        )


@dataclass
class InjectedContext:
    """Read-only context injected into the model's prompt.

    This is the assembled system state that the model sees before
    making any inference. It is strictly immutable from the model's
    perspective — any attempt to modify it should be rejected.
    """

    working_dir: str
    env_vars: dict[str, str]
    rules: dict[str, str]
    memory_summary: list[str]

    def render(self) -> str:
        """Render as a system prompt section."""
        lines: list[str] = []
        lines.append("[ControlPlane]")
        lines.append("=" * 60)
        lines.append("SYSTEM CONTEXT (READ-ONLY — do not attempt to modify)")
        lines.append("=" * 60)
        lines.append("")

        lines.append(f"WORKING DIRECTORY: {self.working_dir}")
        lines.append("")

        if self.env_vars:
            lines.append("CUDA ENVIRONMENT:")
            for k, v in self.env_vars.items():
                lines.append(f"  {k}={v}")
            lines.append("")

        if self.rules:
            lines.append("RULE FILES:")
            for name, content in self.rules.items():
                lines.append(f"--- {name} ---")
                lines.append(content[:500])  # truncate to avoid bloat
                lines.append("")

        if self.memory_summary:
            lines.append("MEMORY SUMMARY:")
            for item in self.memory_summary:
                lines.append(f"  - {item}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


class ControlPlane:
    """Manages system state injection for the model's world model.

    Before each inference, the control plane:
    1. Takes a fresh snapshot of the system
    2. Assembles read-only context (env, rules, memory)
    3. Injects it as a system-level prompt prefix

    The model has zero write access to any injected data.
    """

    def __init__(
        self,
        rule_dir: str | None = None,
        memory_entries: list[str] | None = None,
    ) -> None:
        self._rule_dir = rule_dir
        self._memory_entries = memory_entries or []

    def take_snapshot(self) -> SystemSnapshot:
        """Capture a fresh, independent system snapshot."""
        return SystemSnapshot.capture(rule_dir=self._rule_dir)

    def inject(self) -> InjectedContext:
        """Build a read-only injected context from the current snapshot."""
        snapshot = self.take_snapshot()
        return InjectedContext(
            working_dir=snapshot.working_dir,
            env_vars=snapshot.env_vars,
            rules=snapshot.rule_files,
            memory_summary=list(self._memory_entries),
        )

    def build_system_prompt(self) -> str:
        """Build the full system prompt with injected context."""
        ctx = self.inject()
        header = ctx.render()
        return header

    def add_memory(self, entry: str) -> None:
        """Add a memory summary entry (e.g. past experimental findings)."""
        self._memory_entries.append(entry)

    def clear_memories(self) -> None:
        self._memory_entries.clear()
