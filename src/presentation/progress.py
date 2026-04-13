"""Progress indicator — spinner-based progress for tool execution.

Uses ASCII spinner with elapsed time display. No external dependencies.
"""
from __future__ import annotations

import sys
import time
from typing import TextIO


class ProgressBar:
    """Indeterminate progress spinner with elapsed time."""

    _SPINNERS = ["|", "/", "-", "\\"]
    _CLEAR = "\r"

    def __init__(self, output: TextIO | None = None) -> None:
        self._output = output or sys.stdout
        self._message = ""
        self._start_time: float = 0
        self._done = False
        self._frame = 0

    def start(self, message: str) -> None:
        """Start the progress spinner."""
        self._message = message
        self._start_time = time.monotonic()
        self._done = False
        self._frame = 0
        self._render()

    def update(self) -> None:
        """Advance the spinner frame."""
        if self._done:
            return
        self._frame += 1
        self._render()

    def complete(self) -> None:
        """Mark progress as complete."""
        elapsed = time.monotonic() - self._start_time
        self._done = True
        self._output.write(
            f"\r\033[32m[OK]\033[0m {self._message} ({elapsed:.1f}s)\n"
        )
        self._output.flush()

    def fail(self) -> None:
        """Mark progress as failed."""
        elapsed = time.monotonic() - self._start_time
        self._done = True
        self._output.write(
            f"\r\033[31m[FAIL]\033[0m {self._message} ({elapsed:.1f}s)\n"
        )
        self._output.flush()

    def _render(self) -> None:
        spinner = self._SPINNERS[self._frame % len(self._SPINNERS)]
        elapsed = time.monotonic() - self._start_time
        self._output.write(f"\r{spinner} {self._message} ({elapsed:.1f}s)")
        self._output.flush()
