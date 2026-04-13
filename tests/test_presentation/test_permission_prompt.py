"""Tests for PermissionPrompt (presentation/permission_prompt.py)."""
import io
from src.application.approval_queue import ApprovalRequest
from src.presentation.permission_prompt import PermissionPrompt


class TestPermissionPrompt:
    def test_approve(self):
        output = io.StringIO()
        prompt = PermissionPrompt(input_fn=lambda: "a", output=output)
        request = ApprovalRequest(
            id="test_1",
            tool_name="compile_cuda",
            arguments={"source": "test"},
            permissions=["file:write", "process:exec"],
        )
        result = prompt.prompt(request)
        assert result is True
        assert "Approved" in output.getvalue()

    def test_reject(self):
        output = io.StringIO()
        prompt = PermissionPrompt(input_fn=lambda: "r", output=output)
        request = ApprovalRequest(
            id="test_2",
            tool_name="write_file",
            arguments={},
            permissions=["file:write"],
        )
        result = prompt.prompt(request)
        assert result is False
        assert "Rejected" in output.getvalue()

    def test_quit(self):
        output = io.StringIO()
        prompt = PermissionPrompt(input_fn=lambda: "q", output=output)
        request = ApprovalRequest(
            id="test_3",
            tool_name="execute_binary",
            arguments={"binary_path": "/bin/test"},
            permissions=["process:exec"],
        )
        result = prompt.prompt(request)
        assert result is False
        assert "cancelled" in output.getvalue()

    def test_displays_tool_name_and_permissions(self):
        output = io.StringIO()
        prompt = PermissionPrompt(input_fn=lambda: "a", output=output)
        request = ApprovalRequest(
            id="test_4",
            tool_name="my_tool",
            arguments={"key": "value"},
            permissions=["file:write"],
        )
        prompt.prompt(request)
        result = output.getvalue()
        assert "my_tool" in result
        assert "file:write" in result
        assert "key" in result
        assert "value" in result

    def test_eof_returns_reject(self):
        output = io.StringIO()

        def raise_eof():
            raise EOFError()

        prompt = PermissionPrompt(input_fn=raise_eof, output=output)
        request = ApprovalRequest(id="test_5", tool_name="t", arguments={}, permissions=[])
        result = prompt.prompt(request)
        assert result is False

    def test_keyboard_interrupt_returns_reject(self):
        output = io.StringIO()

        def raise_kbi():
            raise KeyboardInterrupt()

        prompt = PermissionPrompt(input_fn=raise_kbi, output=output)
        request = ApprovalRequest(id="test_6", tool_name="t", arguments={}, permissions=[])
        result = prompt.prompt(request)
        assert result is False

    def test_unknown_input_rejects(self):
        output = io.StringIO()
        prompt = PermissionPrompt(input_fn=lambda: "maybe", output=output)
        request = ApprovalRequest(id="test_7", tool_name="t", arguments={}, permissions=[])
        result = prompt.prompt(request)
        assert result is False
