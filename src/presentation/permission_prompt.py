"""Permission prompt — interactive approval request display.

Shows tool name, arguments, permissions and reads user decision.
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from src.application.approval_queue import ApprovalRequest


class PermissionPrompt:
    """Interactive prompt for tool execution approval."""

    def __init__(self, input_fn=None, output: TextIO | None = None) -> None:
        self._input = input_fn or input
        self._output = output or sys.stdout
        self._GREEN = "\033[32m"
        self._RED = "\033[31m"
        self._YELLOW = "\033[33m"
        self._CYAN = "\033[36m"
        self._RST = "\033[0m"
        self._BOLD = "\033[1m"

    def prompt(self, request: "ApprovalRequest") -> bool:
        """Display an approval request and wait for user input.

        Returns:
            True if approved, False if rejected or quit.
        """
        g = self._GREEN
        r = self._RED
        y = self._YELLOW
        c = self._CYAN
        rst = self._RST
        bold = self._BOLD

        self._output.write(f"\n{bold}{y}Approval Required{rst}\n")
        self._output.write(f"  Tool:        {c}{request.tool_name}{rst}\n")
        self._output.write(f"  Permissions: {', '.join(request.permissions)}\n")

        if request.arguments:
            self._output.write("  Arguments:\n")
            for key, value in request.arguments.items():
                val_str = str(value)
                if len(val_str) > 80:
                    val_str = val_str[:77] + "..."
                self._output.write(f"    {key}: {val_str}\n")

        self._output.write(f"\n  [{g}a{rst}]pprove  [{r}r{rst}]eject  [q]uit: ")
        self._output.flush()

        try:
            choice = self._input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = "r"

        if choice == "a":
            self._output.write(f"  {g}Approved{rst}\n")
            return True
        elif choice == "q":
            self._output.write(f"  {r}Quit -- operation cancelled{rst}\n")
            return False
        else:
            self._output.write(f"  {r}Rejected{rst}\n")
            return False
