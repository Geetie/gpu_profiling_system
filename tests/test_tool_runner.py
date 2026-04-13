"""Tests for Tool Runner (application/tool_runner.py)."""
import pytest
from src.application.approval_queue import ApprovalQueue, ApprovalStatus
from src.application.tool_runner import ToolRunner, ApprovalRequiredError
from src.domain.permission import PermissionChecker, PermissionMode
from src.domain.schema_validator import SchemaValidator, SchemaValidationError
from src.domain.tool_contract import ToolContract, ToolRegistry
from src.infrastructure.state_persist import StatePersister


class TestToolRunner:
    @pytest.fixture
    def registry(self):
        reg = ToolRegistry()
        reg.register(ToolContract(
            name="echo_tool",
            description="Echoes input",
            input_schema={"message": "string"},
            output_schema={"response": "string"},
            permissions=["file:read"],
            requires_approval=False,
        ))
        reg.register(ToolContract(
            name="needs_approval",
            description="Requires approval",
            input_schema={"data": "string"},
            output_schema={"result": "string"},
            permissions=["file:write"],
            requires_approval=True,
        ))
        return reg

    @pytest.fixture
    def runner(self, registry, tmp_path):
        persister = StatePersister(log_dir=str(tmp_path))
        approval_queue = ApprovalQueue(state_dir=str(tmp_path), persister=persister)
        permission_checker = PermissionChecker(mode=PermissionMode.DEFAULT)
        return ToolRunner(
            registry=registry,
            tool_handlers={"echo_tool": lambda args: {"response": args["message"]}},
            approval_queue=approval_queue,
            permission_checker=permission_checker,
            persister=persister,
            validator=SchemaValidator(),
        )

    def test_full_pipeline_valid(self, runner):
        result = runner.execute("echo_tool", {"message": "hello"})
        assert result == {"response": "hello"}

    def test_unregistered_tool_raises(self, runner):
        with pytest.raises(KeyError, match="not registered"):
            runner.execute("nonexistent_tool", {})

    def test_invalid_input_raises(self, runner):
        with pytest.raises(SchemaValidationError):
            runner.execute("echo_tool", {"message": 123})  # should be string

    def test_invalid_output_raises(self, runner):
        # Handler returns wrong type
        runner._handlers["echo_tool"] = lambda args: {"response": 123}  # should be string
        with pytest.raises(SchemaValidationError):
            runner.execute("echo_tool", {"message": "hello"})

    def test_missing_input_field_raises(self, runner):
        with pytest.raises(SchemaValidationError):
            runner.execute("echo_tool", {})

    def test_approval_needed_raises_approval_required_error(self, tmp_path):
        reg = ToolRegistry()
        reg.register(ToolContract(
            name="needs_approval",
            description="Requires approval",
            input_schema={"data": "string"},
            output_schema={"result": "string"},
            permissions=["file:write"],
            requires_approval=True,
        ))
        persister = StatePersister(log_dir=str(tmp_path))
        approval_queue = ApprovalQueue(state_dir=str(tmp_path), persister=persister)
        # DEFAULT mode requires approval for file:write
        permission_checker = PermissionChecker(mode=PermissionMode.DEFAULT)
        runner = ToolRunner(
            registry=reg,
            tool_handlers={"needs_approval": lambda args: {"result": "done"}},
            approval_queue=approval_queue,
            permission_checker=permission_checker,
            persister=persister,
        )
        with pytest.raises(ApprovalRequiredError) as exc:
            runner.execute("needs_approval", {"data": "test"})
        assert exc.value.request.tool_name == "needs_approval"

    def test_persistence_on_success(self, runner, tmp_path):
        runner.execute("echo_tool", {"message": "persist test"})
        log_path = str(tmp_path / "session_log.jsonl")
        # Session log should exist
        import os
        assert os.path.exists(log_path)


class TestToolRunnerConservativeMode:
    def test_conservative_auto_rejects(self, tmp_path):
        reg = ToolRegistry()
        reg.register(ToolContract(
            name="write_something",
            description="Writes data",
            input_schema={"data": "string"},
            output_schema={"result": "string"},
            permissions=["file:write"],
            requires_approval=True,
        ))
        persister = StatePersister(log_dir=str(tmp_path))
        approval_queue = ApprovalQueue(state_dir=str(tmp_path), persister=persister)
        permission_checker = PermissionChecker(mode=PermissionMode.CONSERVATIVE)
        runner = ToolRunner(
            registry=reg,
            tool_handlers={"write_something": lambda args: {"result": "done"}},
            approval_queue=approval_queue,
            permission_checker=permission_checker,
            persister=persister,
        )
        # CONSERVATIVE mode auto-rejects write operations
        with pytest.raises(PermissionError):
            runner.execute("write_something", {"data": "test"})
