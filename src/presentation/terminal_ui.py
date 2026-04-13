"""TerminalUI facade — main entry point for presentation layer.

Wires together diff rendering, progress, permission prompts, and result display.
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, TextIO

from src.presentation.diff_renderer import DiffRenderer
from src.presentation.progress import ProgressBar
from src.presentation.permission_prompt import PermissionPrompt
from src.presentation.result_display import ToolResultDisplay

if TYPE_CHECKING:
    from src.application.approval_queue import ApprovalRequest


class TerminalUI:
    """Main facade for terminal-based UI.

    Usage:
        ui = TerminalUI()
        ui.show_tool_start("read_file", {"file_path": "test.cu"})
        ui.show_tool_complete("read_file", {"content": "...", "lines": 42})
        ui.show_file_diff("test.cu", old_code, new_code)
        approved = ui.request_approval(approval_request)
    """

    def __init__(
        self,
        output: TextIO | None = None,
        input_fn=None,
    ) -> None:
        self._output = output or sys.stdout
        self._input_fn = input_fn
        self._diff = DiffRenderer(output=output)
        self._progress = ProgressBar(output=output)
        self._prompt = PermissionPrompt(input_fn=input_fn, output=output)
        self._result = ToolResultDisplay(output=output)
        self._GREEN = "\033[32m"
        self._RST = "\033[0m"

    def show_tool_start(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Show that a tool is starting execution."""
        self._progress.start(f"Running {tool_name}")

    def show_tool_complete(self, tool_name: str, result: dict[str, Any]) -> None:
        """Show tool completion with result."""
        self._progress.complete()
        self._result.show(tool_name, result)

    def show_tool_error(self, tool_name: str, error: str) -> None:
        """Show tool execution error."""
        self._progress.fail()
        self._result.show_error(tool_name, error)

    def show_file_diff(
        self, file_path: str, old_content: str, new_content: str
    ) -> None:
        """Render a unified diff between old and new file content."""
        self._diff.render(file_path, old_content, new_content)

    def request_approval(self, request: "ApprovalRequest") -> bool:
        """Show approval prompt and return user decision."""
        return self._prompt.prompt(request)

    def show_message(self, message: str) -> None:
        """Show a general message."""
        self._output.write(f"{self._GREEN}{message}{self._RST}\n")
        self._output.flush()

    @staticmethod
    def generate_diff_string(
        file_path: str, old_content: str, new_content: str
    ) -> str:
        """Generate a plain-text diff string (no ANSI codes, no output)."""
        return DiffRenderer.generate_diff(file_path, old_content, new_content)
