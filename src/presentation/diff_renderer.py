"""Diff renderer — unified diff display.

Uses stdlib difflib to generate colored diff output.
"""
from __future__ import annotations

import difflib
import sys
from typing import TextIO


class DiffRenderer:
    """Renders file diffs in the terminal."""

    # ANSI color codes
    _ADD = "\033[32m"      # green for additions
    _DEL = "\033[31m"      # red for deletions
    _HDR = "\033[36m"      # cyan for headers
    _RST = "\033[0m"       # reset
    _DIM = "\033[2m"       # dim for context

    def __init__(self, output: TextIO | None = None) -> None:
        self._output = output or sys.stdout

    def render(
        self,
        file_path: str,
        old_content: str,
        new_content: str,
        context_lines: int = 3,
    ) -> None:
        """Render a unified diff between old and new content."""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            n=context_lines,
        )

        self._output.write(f"\n--- diff: {file_path} ---\n")
        for line in diff:
            if line.startswith("+") and not line.startswith("+++"):
                self._output.write(f"{self._ADD}{line}{self._RST}")
            elif line.startswith("-") and not line.startswith("---"):
                self._output.write(f"{self._DEL}{line}{self._RST}")
            elif line.startswith("@@"):
                self._output.write(f"{self._HDR}{line}{self._RST}")
            elif line.startswith(" "):
                self._output.write(f"{self._DIM}{line}{self._RST}")
            else:
                self._output.write(line)
        self._output.write("--- end diff ---\n\n")

    @staticmethod
    def generate_diff(
        file_path: str,
        old_content: str,
        new_content: str,
    ) -> str:
        """Generate a plain-text diff string (no ANSI codes)."""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        return "".join(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
            )
        )
