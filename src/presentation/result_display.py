"""Tool result display — pretty-printing of execution results.
"""
from __future__ import annotations

import re
import sys
from typing import Any, TextIO


# Regex to strip ANSI escape sequences from tool output
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return _ANSI_ESCAPE_RE.sub("", text)


class ToolResultDisplay:
    """Pretty-prints tool execution results in the terminal."""

    _GREEN = "\033[32m"
    _RED = "\033[31m"
    _CYAN = "\033[36m"
    _YELLOW = "\033[33m"
    _RST = "\033[0m"
    _MAX_STRING = 2000

    def __init__(self, output: TextIO | None = None) -> None:
        self._output = output or sys.stdout

    def show(self, tool_name: str, result: dict[str, Any]) -> None:
        """Display tool result with formatting."""
        g = self._GREEN
        r = self._RED
        c = self._CYAN
        y = self._YELLOW
        rst = self._RST

        status = "success" if not result.get("error") else "completed_with_errors"
        color = g if status == "success" else y

        self._output.write(f"\n{color}[{tool_name}]{rst} ({status})\n")

        for key, value in result.items():
            val_str = self._format_value(value)
            self._output.write(f"  {c}{key}{rst}: {val_str}\n")
        self._output.write("\n")
        self._output.flush()

    def show_error(self, tool_name: str, error: str) -> None:
        """Display a tool execution error."""
        r = self._RED
        rst = self._RST
        c = self._CYAN

        self._output.write(f"\n{r}[{tool_name}] ERROR{rst}\n")
        self._output.write(f"  {c}message{rst}: {error[:500]}\n\n")
        self._output.flush()

    def _format_value(self, value: Any) -> str:
        """Format a value for display, truncating long strings and stripping ANSI."""
        if isinstance(value, str):
            clean = _strip_ansi(value)
            if len(clean) > self._MAX_STRING:
                return clean[:self._MAX_STRING - 3] + "..."
            return clean
        if isinstance(value, dict):
            parts = []
            for k, v in value.items():
                v_str = _strip_ansi(str(v))
                if len(v_str) > 100:
                    v_str = v_str[:97] + "..."
                parts.append(f"{k}={v_str}")
            return "{" + ", ".join(parts) + "}"
        if isinstance(value, list):
            if len(value) > 10:
                return f"[{', '.join(_strip_ansi(str(x)) for x in value[:10])}, ... ({len(value)} items)]"
            return str(value)
        return str(value)
