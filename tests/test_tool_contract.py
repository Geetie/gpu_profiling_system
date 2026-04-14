"""Tests for ToolContract and ToolRegistry (domain/tool_contract.py)"""
import pytest
from src.domain.tool_contract import ToolContract, ToolRegistry


# ── ToolContract Tests ──────────────────────────────────────────────


class TestToolContract:
    """Test ToolContract creation and validation."""

    def test_create_valid_contract(self):
        contract = ToolContract(
            name="read_file",
            description="Read a file from disk",
            input_schema={"file_path": "string"},
            output_schema={"content": "string", "lines": "integer"},
            permissions=["file:read"],
            requires_approval=False,
            is_blocking=True,
        )
        assert contract.name == "read_file"
        assert contract.permissions == ["file:read"]
        assert contract.requires_approval is False
        assert contract.is_blocking is True

    def test_name_must_be_unique_identifier(self):
        contract = ToolContract(
            name="run_ncu",
            description="Run Nsight Compute",
            input_schema={"target": "string"},
            output_schema={"output": "string"},
            permissions=["file:read", "process:exec"],
            requires_approval=False,
            is_blocking=True,
        )
        assert contract.name == "run_ncu"

    def test_name_must_not_be_empty(self):
        with pytest.raises(ValueError):
            ToolContract(
                name="",
                description="empty name",
                input_schema={},
                output_schema={},
                permissions=[],
            )

    def test_permissions_must_be_non_empty(self):
        with pytest.raises(ValueError):
            ToolContract(
                name="test_tool",
                description="no permissions",
                input_schema={},
                output_schema={},
                permissions=[],
            )

    def test_to_dict_roundtrip(self):
        contract = ToolContract(
            name="write_file",
            description="Write content to file",
            input_schema={"file_path": "string", "content": "string"},
            output_schema={"bytes_written": "integer"},
            permissions=["file:write"],
            requires_approval=True,
            is_blocking=True,
        )
        d = contract.to_dict()
        assert d["name"] == "write_file"
        assert d["requires_approval"] is True
        assert d["permissions"] == ["file:write"]

        restored = ToolContract.from_dict(d)
        assert restored.name == contract.name
        assert restored.requires_approval == contract.requires_approval

    def test_read_only_tool(self):
        """Read-only tools should not require approval by design."""
        contract = ToolContract(
            name="read_file",
            description="Read file",
            input_schema={"file_path": "string"},
            output_schema={"content": "string"},
            permissions=["file:read"],
            requires_approval=False,
            is_blocking=True,
        )
        assert "file:write" not in contract.permissions
        assert contract.requires_approval is False


# ── ToolRegistry Tests ──────────────────────────────────────────────


class TestToolRegistry:
    """Test ToolRegistry: registration, lookup, and P2 fail-closed."""

    def test_register_and_get(self):
        registry = ToolRegistry()
        contract = ToolContract(
            name="read_file",
            description="Read file",
            input_schema={"file_path": "string"},
            output_schema={"content": "string"},
            permissions=["file:read"],
            requires_approval=False,
            is_blocking=True,
        )
        registry.register(contract)
        assert registry.get("read_file") is contract

    def test_get_unregistered_raises(self):
        """P2: unregistered tools are rejected (fail-closed)."""
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent_tool")

    def test_duplicate_name_raises(self):
        registry = ToolRegistry()
        contract = ToolContract(
            name="read_file",
            description="Read file",
            input_schema={"file_path": "string"},
            output_schema={"content": "string"},
            permissions=["file:read"],
        )
        registry.register(contract)
        with pytest.raises(ValueError):
            registry.register(contract)

    def test_has_tool(self):
        registry = ToolRegistry()
        contract = ToolContract(
            name="read_file",
            description="Read file",
            input_schema={"file_path": "string"},
            output_schema={"content": "string"},
            permissions=["file:read"],
        )
        registry.register(contract)
        assert registry.has_tool("read_file") is True
        assert registry.has_tool("missing") is False

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register(ToolContract(
            name="read_file", description="Read",
            input_schema={}, output_schema={}, permissions=["file:read"],
        ))
        registry.register(ToolContract(
            name="write_file", description="Write",
            input_schema={}, output_schema={}, permissions=["file:write"],
            requires_approval=True,
        ))
        names = registry.list_tools()
        assert set(names) == {"read_file", "write_file"}

    def test_register_bulk(self):
        registry = ToolRegistry()
        contracts = [
            ToolContract(name=f"tool_{i}", description=f"Tool {i}",
                         input_schema={}, output_schema={}, permissions=["exec"])
            for i in range(3)
        ]
        registry.register_bulk(contracts)
        assert registry.list_tools() == ["tool_0", "tool_1", "tool_2"]

    def test_prebuilt_contracts(self):
        """Verify the standard tool set from spec matches expectations."""
        from src.domain.tool_contract import build_standard_registry
        registry = build_standard_registry()
        expected = {
            "run_ncu", "compile_cuda", "execute_binary",
            "write_file", "read_file", "generate_microbenchmark",
            "kaggle_push",
        }
        assert set(registry.list_tools()) == expected
        # read_file should NOT require approval
        assert registry.get("read_file").requires_approval is False
        # write_file SHOULD require approval
        assert registry.get("write_file").requires_approval is True
