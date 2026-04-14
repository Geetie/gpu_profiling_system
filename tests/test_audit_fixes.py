"""Tests for audit fixes — new and updated tests for the fixes."""
import json
import os
import pytest

from src.application.agent_loop import AgentLoop, LoopState
from src.application.control_plane import ControlPlane, SystemSnapshot
from src.application.context import ContextManager, Role
from src.application.session import SessionState
from src.domain.tool_contract import ToolContract, ToolRegistry, build_standard_registry
from src.domain.permission import PermissionMode
from src.infrastructure.file_ops import FileOperations


# ── P5: Build-time tool elimination ─────────────────────────────────


class TestP5BuildTimeElimination:
    def test_disabled_tool_not_registered(self):
        """P5: disabled tools are removed at build time."""
        registry = ToolRegistry(disabled_tools={"compile_cuda"})
        contracts = [
            ToolContract(name="read_file", description="Read",
                         input_schema={}, output_schema={}, permissions=["file:read"]),
            ToolContract(name="compile_cuda", description="Compile",
                         input_schema={}, output_schema={}, permissions=["process:exec"]),
        ]
        registry.register_bulk(contracts)
        assert registry.has_tool("read_file") is True
        assert registry.has_tool("compile_cuda") is False

    def test_disabled_tools_property(self):
        registry = ToolRegistry(disabled_tools={"tool_a", "tool_b"})
        assert registry.disabled_tools == frozenset({"tool_a", "tool_b"})

    def test_no_disabled_by_default(self):
        registry = ToolRegistry()
        assert registry.disabled_tools == frozenset()

    def test_empty_disabled_set(self):
        registry = ToolRegistry(disabled_tools=set())
        contracts = [
            ToolContract(name="x", description="X",
                         input_schema={}, output_schema={}, permissions=["x"]),
        ]
        registry.register_bulk(contracts)
        assert registry.has_tool("x") is True


# ── VULN-1: Permission checking in execution path ───────────────────


class TestVuln1PermissionWiring:
    @pytest.fixture()
    def loop_conservative(self, tmp_path):
        os.chdir(str(tmp_path))
        control = ControlPlane(rule_dir=str(tmp_path))
        ctx = ContextManager(max_tokens=4000)
        session = SessionState(session_id="vuln1", goal="test")
        registry = ToolRegistry()
        registry.register(ToolContract(
            name="exec", description="Execute",
            input_schema={}, output_schema={}, permissions=["process:exec"],
        ))
        registry.register(ToolContract(
            name="reader", description="Read",
            input_schema={}, output_schema={}, permissions=["file:read"],
        ))
        return AgentLoop(
            session=session,
            context_manager=ctx,
            control_plane=control,
            tool_registry=registry,
            max_turns=10,
            state_dir=str(tmp_path),
            permission_mode=PermissionMode.CONSERVATIVE,
        )

    def test_conservative_blocks_exec(self, loop_conservative):
        """In CONSERVATIVE mode, process:exec should be denied."""
        with pytest.raises(PermissionError, match="P2 fail-closed"):
            loop_conservative._execute_tool_call(
                type("ToolCall", (), {"name": "exec", "arguments": {}})()
            )

    def test_conservative_allows_read(self, loop_conservative):
        """In CONSERVATIVE mode, file:read should be allowed."""
        result = loop_conservative._execute_tool_call(
            type("ToolCall", (), {"name": "reader", "arguments": {}})()
        )
        # No executor → returns contract info
        assert result["status"] == "no_executor_installed"

    def test_set_permission_mode(self, loop_conservative):
        """After switching to RELAXED, is_allowed('process:exec') returns True.

        INT-2 fix: AgentLoop only checks is_allowed() (fail-closed).
        Approval checking is delegated to ToolRunner.
        In RELAXED mode, process:exec is allowed (but still requires approval
        at the ToolRunner level, not at the AgentLoop level).
        """
        loop_conservative.set_permission_mode(PermissionMode.RELAXED)
        # process:exec is now allowed by is_allowed() in RELAXED mode
        result = loop_conservative._execute_tool_call(
            type("ToolCall", (), {"name": "exec", "arguments": {}})()
        )
        # No executor → returns contract info (not denied)
        assert result["status"] == "no_executor_installed"

    def test_conservative_denies_exec(self, tmp_path):
        """In CONSERVATIVE mode, process:exec should be denied by is_allowed."""
        os.chdir(str(tmp_path))
        control = ControlPlane(rule_dir=str(tmp_path))
        ctx = ContextManager(max_tokens=4000)
        session = SessionState(session_id="vuln1b", goal="test")
        registry = ToolRegistry()
        registry.register(ToolContract(
            name="exec", description="Execute",
            input_schema={}, output_schema={}, permissions=["process:exec"],
        ))
        loop = AgentLoop(
            session=session,
            context_manager=ctx,
            control_plane=control,
            tool_registry=registry,
            max_turns=10,
            state_dir=str(tmp_path),
            permission_mode=PermissionMode.CONSERVATIVE,
        )
        # CONSERVATIVE mode should deny process:exec
        with pytest.raises(PermissionError, match="P2 fail-closed"):
            loop._execute_tool_call(
                type("ToolCall", (), {"name": "exec", "arguments": {}})()
            )


# ── BUG-1: Context deduplication ────────────────────────────────────


class TestBug1ContextDedup:
    def test_update_system_entry_replaces(self):
        cm = ContextManager(max_tokens=4000)
        cm.update_system_entry("System v1", token_count=10)
        cm.update_system_entry("System v2", token_count=10)
        entries = cm.get_entries()
        assert len(entries) == 1
        assert entries[0].content == "System v2"

    def test_update_system_entry_adds_if_missing(self):
        cm = ContextManager(max_tokens=4000)
        cm.update_system_entry("First system", token_count=10)
        assert len(cm.get_entries()) == 1

    def test_no_duplicate_system_after_multiple_injections(self):
        """Multiple calls to update_system_entry should not create duplicates."""
        cm = ContextManager(max_tokens=4000)
        for i in range(10):
            cm.update_system_entry(f"System {i}", token_count=10)
        system_entries = [e for e in cm.get_entries() if e.role == Role.SYSTEM]
        assert len(system_entries) == 1
        assert system_entries[0].content == "System 9"


# ── VULN-2: Sandbox path bypass ─────────────────────────────────────


class TestVuln2SandboxBypass:
    def test_sibling_path_blocked(self, tmp_path):
        """A path like /tmp/sandbox_evil should NOT pass /tmp/sandbox check."""
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        evil = tmp_path / "sandbox_evil" / "file.txt"
        evil.parent.mkdir()
        evil.write_text("evil")

        ops = FileOperations(sandbox_root=str(sandbox))
        with pytest.raises(PermissionError, match="Path escape blocked"):
            ops.read(str(evil))

    def test_child_path_allowed(self, tmp_path):
        """A path inside the sandbox should still work."""
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        child = sandbox / "child.txt"
        child.write_text("child")

        ops = FileOperations(sandbox_root=str(sandbox))
        content = ops.read(str(child))
        assert content == "child"


# ── VULN-3: Rule file size limit ────────────────────────────────────


class TestVuln3RuleFileSizeLimit:
    def test_large_file_truncated(self, tmp_path):
        """Files larger than 64KB should be truncated on read."""
        rule_dir = tmp_path / "rules"
        rule_dir.mkdir()
        big_file = rule_dir / "CLAUDE.md"
        # Create a 100KB file
        big_file.write_text("x" * 100_000)

        snapshot = SystemSnapshot.capture(rule_dir=str(rule_dir))
        content = snapshot.rule_files["CLAUDE.md"]
        assert len(content.encode("utf-8")) < 100_000
        assert "truncated" in content.lower()

    def test_small_file_read_in_full(self, tmp_path):
        rule_dir = tmp_path / "rules"
        rule_dir.mkdir()
        small = rule_dir / "CLAUDE.md"
        small.write_text("small rule")
        snapshot = SystemSnapshot.capture(rule_dir=str(rule_dir))
        assert snapshot.rule_files["CLAUDE.md"] == "small rule"


# ── M2: Anchored edit ───────────────────────────────────────────────


class TestM2AnchoredEdit:
    def test_hash_anchor_mismatch(self, tmp_path):
        """Writing with wrong hash should fail (file changed)."""
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        f = sandbox / "data.txt"
        f.write_text("original")

        ops = FileOperations(sandbox_root=str(sandbox))
        ops.read(str(f))
        with pytest.raises(ValueError, match="M2 anchor violation"):
            ops.anchored_write(
                str(f), "modified",
                expected_hash="wrong_hash_0000000000000000000000000000000000000000000000000000000000000000"
            )

    def test_hash_anchor_match(self, tmp_path):
        """Writing with correct hash should succeed."""
        import hashlib
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        f = sandbox / "data.txt"
        f.write_text("original")
        content_hash = hashlib.sha256("original".encode("utf-8")).hexdigest()

        ops = FileOperations(sandbox_root=str(sandbox))
        ops.read(str(f))
        ops.anchored_write(str(f), "modified", expected_hash=content_hash)
        assert f.read_text() == "modified"

    def test_line_range_replacement(self, tmp_path):
        """Replace specific lines with anchored_write."""
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        f = sandbox / "code.txt"
        f.write_text("line1\nline2\nline3\nline4\n")

        ops = FileOperations(sandbox_root=str(sandbox))
        ops.read(str(f))
        ops.anchored_write(str(f), "REPLACED\n", line_range=(2, 3))
        result = f.read_text()
        assert "line1" in result
        assert "REPLACED" in result
        assert "line2" not in result
        assert "line3" not in result
        assert "line4" in result

    def test_line_range_out_of_bounds(self, tmp_path):
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        f = sandbox / "code.txt"
        f.write_text("line1\nline2\n")

        ops = FileOperations(sandbox_root=str(sandbox))
        ops.read(str(f))
        with pytest.raises(IndexError, match="M2.*out of bounds"):
            ops.anchored_write(str(f), "X", line_range=(1, 100))


# ── BUG-2: Error logging ────────────────────────────────────────────


class TestBug2ErrorLogging:
    def test_error_logged_to_persister(self, tmp_path):
        """When a tool execution fails, error should be in the log."""
        os.chdir(str(tmp_path))
        control = ControlPlane(rule_dir=str(tmp_path))
        ctx = ContextManager(max_tokens=4000)
        session = SessionState(session_id="err", goal="test")
        registry = ToolRegistry()
        registry.register(ToolContract(
            name="fail_tool", description="Fails",
            input_schema={}, output_schema={}, permissions=["file:read"],
        ))
        loop = AgentLoop(
            session=session,
            context_manager=ctx,
            control_plane=control,
            tool_registry=registry,
            max_turns=10,
            state_dir=str(tmp_path),
        )
        # Install an executor that always fails
        loop.set_tool_executor(lambda name, args: (_ for _ in ()).throw(RuntimeError("boom")))
        loop._model_output = json.dumps({"tool": "fail_tool", "args": {}})
        loop.loop_state.is_running = True

        loop._inner_loop_step()

        history = loop._persister.load_history()
        error_entries = [h for h in history if h.get("action") == "error"]
        assert len(error_entries) >= 1
        assert any("boom" in str(e) for e in error_entries)


# ── Missing coverage: build_standard_registry with disabled_tools ────


class TestBuildStandardRegistryDisabled:
    def test_disabled_via_build_standard_registry(self):
        """build_standard_registry(disabled_tools=...) should exclude tools at build time."""
        registry = build_standard_registry(disabled_tools={"compile_cuda", "execute_binary"})
        assert registry.has_tool("compile_cuda") is False
        assert registry.has_tool("execute_binary") is False
        # Others should be present
        assert registry.has_tool("read_file") is True
        assert registry.has_tool("run_ncu") is True
        assert registry.has_tool("write_file") is True
        assert registry.has_tool("generate_microbenchmark") is True

    def test_default_has_all_tools(self):
        """Without disabled_tools, all 7 standard tools should be present."""
        registry = build_standard_registry()
        assert len(registry.list_tools()) == 7


# ── Missing coverage: FileOperations(prior_reads=...) ────────────────


class TestFileOpsPriorReadsRestore:
    def test_prior_reads_restored(self, tmp_path):
        """FileOperations should restore the read ledger from prior_reads."""
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        f = sandbox / "existing.txt"
        f.write_text("data")

        ops = FileOperations(sandbox_root=str(sandbox), prior_reads={str(f)})
        # Even without calling read(), the ledger should have the file
        assert ops.tracker.has_read(str(f)) is True
        # So write should succeed
        ops.write(str(f), "modified")
        assert f.read_text() == "modified"

    def test_no_prior_reads_blocks_write(self, tmp_path):
        """Without prior_reads, writing should fail (M1)."""
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        f = sandbox / "existing.txt"
        f.write_text("data")

        ops = FileOperations(sandbox_root=str(sandbox))
        with pytest.raises(PermissionError, match="M1 violation"):
            ops.write(str(f), "modified")


# ── Missing coverage: compress when system_tokens > target ───────────


class TestCompressSystemTokensExceedBudget:
    def test_drops_all_non_system_when_system_over_budget(self):
        """If system entries alone exceed target, all non-system entries should be dropped."""
        cm = ContextManager(max_tokens=100)
        cm.add_entry(Role.SYSTEM, "huge system context" * 10, token_count=90)
        cm.add_entry(Role.USER, "user msg", token_count=20)
        cm.add_entry(Role.ASSISTANT, "assistant reply", token_count=20)
        assert cm.total_tokens == 130

        removed = cm.compress()
        # All non-system entries should be removed
        assert removed == 2
        entries = cm.get_entries()
        assert len(entries) == 1
        assert entries[0].role == Role.SYSTEM
        assert cm.total_tokens == 90


# ── Missing coverage: update_system_entry preserves other entries ────


class TestUpdateSystemPreservesOthers:
    def test_update_system_preserves_user_entries(self):
        """update_system_entry should not remove user/assistant entries."""
        cm = ContextManager(max_tokens=4000)
        cm.add_entry(Role.USER, "question", token_count=5)
        cm.update_system_entry("System v1", token_count=10)
        cm.add_entry(Role.ASSISTANT, "answer", token_count=5)
        cm.update_system_entry("System v2", token_count=10)

        entries = cm.get_entries()
        # Should have USER, SYSTEM (replaced), ASSISTANT
        assert len(entries) == 3
        roles = [e.role for e in entries]
        assert Role.USER in roles
        assert Role.SYSTEM in roles
        assert Role.ASSISTANT in roles
        # System should be v2
        system = [e for e in entries if e.role == Role.SYSTEM][0]
        assert system.content == "System v2"
