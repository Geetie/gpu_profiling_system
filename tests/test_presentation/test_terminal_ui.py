"""Tests for TerminalUI and ToolResultDisplay (presentation layer)."""
import io
from src.presentation.terminal_ui import TerminalUI
from src.presentation.result_display import ToolResultDisplay


# ── TerminalUI Tests ─────────────────────────────────────────────────


class TestTerminalUI:
    def test_show_tool_start_and_complete(self):
        output = io.StringIO()
        ui = TerminalUI(output=output)
        ui.show_tool_start("read_file", {"file_path": "test.cu"})
        ui.show_tool_complete("read_file", {"content": "data", "lines": 1})
        result = output.getvalue()
        assert "read_file" in result
        assert "[OK]" in result

    def test_show_tool_error(self):
        output = io.StringIO()
        ui = TerminalUI(output=output)
        ui.show_tool_error("run_ncu", "Executable not found")
        result = output.getvalue()
        assert "ERROR" in result
        assert "run_ncu" in result

    def test_show_file_diff(self):
        output = io.StringIO()
        ui = TerminalUI(output=output)
        ui.show_file_diff("test.txt", "old\n", "new\n")
        result = output.getvalue()
        assert "diff:" in result

    def test_show_message(self):
        output = io.StringIO()
        ui = TerminalUI(output=output)
        ui.show_message("Hello world")
        result = output.getvalue()
        assert "Hello world" in result

    def test_request_approval(self):
        from src.application.approval_queue import ApprovalRequest
        output = io.StringIO()
        ui = TerminalUI(output=output, input_fn=lambda: "a")
        request = ApprovalRequest(
            id="test_1",
            tool_name="compile_cuda",
            arguments={"source": "test"},
            permissions=["file:write"],
        )
        result = ui.request_approval(request)
        assert result is True

    def test_request_approval_reject(self):
        from src.application.approval_queue import ApprovalRequest
        output = io.StringIO()
        ui = TerminalUI(output=output, input_fn=lambda: "r")
        request = ApprovalRequest(
            id="test_2",
            tool_name="compile_cuda",
            arguments={},
            permissions=["file:write"],
        )
        result = ui.request_approval(request)
        assert result is False

    def test_generate_diff_string(self):
        diff = TerminalUI.generate_diff_string("test.txt", "old\n", "new\n")
        assert "-old" in diff
        assert "+new" in diff


# ── ToolResultDisplay Tests ──────────────────────────────────────────


class TestToolResultDisplay:
    def test_show_success_result(self):
        output = io.StringIO()
        display = ToolResultDisplay(output=output)
        display.show("echo_tool", {"response": "hello"})
        result = output.getvalue()
        assert "echo_tool" in result
        assert "success" in result
        assert "response" in result
        assert "hello" in result

    def test_show_result_with_error(self):
        output = io.StringIO()
        display = ToolResultDisplay(output=output)
        display.show("run_ncu", {"error": "not found", "raw_output": ""})
        result = output.getvalue()
        assert "completed_with_errors" in result

    def test_show_error(self):
        output = io.StringIO()
        display = ToolResultDisplay(output=output)
        display.show_error("compile_cuda", "nvcc not found")
        result = output.getvalue()
        assert "ERROR" in result
        assert "nvcc not found" in result

    def test_truncate_long_string(self):
        output = io.StringIO()
        display = ToolResultDisplay(output=output)
        long_str = "x" * 3000
        display.show("test", {"data": long_str})
        result = output.getvalue()
        # Should be truncated
        assert len(result) < len(long_str)
        assert "..." in result

    def test_truncate_long_dict_value(self):
        output = io.StringIO()
        display = ToolResultDisplay(output=output)
        long_val = {"key": "y" * 200}
        display.show("test", {"nested": long_val})
        result = output.getvalue()
        assert "test" in result

    def test_format_list_truncation(self):
        output = io.StringIO()
        display = ToolResultDisplay(output=output)
        display.show("test", {"items": list(range(50))})
        result = output.getvalue()
        assert "50 items" in result

    def test_strips_ansi_escape_sequences(self):
        """ANSI escape codes in tool output values should be stripped from display."""
        output = io.StringIO()
        display = ToolResultDisplay(output=output)
        # String with ANSI escape codes in the VALUE
        ansi_str = "\033[32mgreen text\033[0m and \033[31mred text\033[0m"
        display.show("test", {"raw": ansi_str})
        result = output.getvalue()
        # The VALUE should have ANSI stripped: "green text and red text"
        assert "green text and red text" in result
        # Original escape sequences from the value should NOT appear
        assert "\033[31mred" not in result
