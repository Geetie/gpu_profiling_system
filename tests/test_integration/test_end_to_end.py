"""阶段 2：端到端功能测试（正常环境）。

验证系统在标准环境下能否完成完整的探测流程。
"""
import json
import os
import pytest


# ── 测试用例 2.1：AgentLoop + ToolRunner 端到端 ─────────────────────


class TestAgentLoopToolRunnerIntegration:
    """验证 AgentLoop 能通过 ToolRunner 正确执行工具调用。"""

    def _setup_loop_with_tools(self, tmp_path):
        """Build a fully wired AgentLoop + ToolRunner with real handlers."""
        from src.application.agent_loop import AgentLoop, LoopState
        from src.application.approval_queue import ApprovalQueue
        from src.application.context import ContextManager
        from src.application.control_plane import ControlPlane
        from src.application.session import SessionState
        from src.application.tool_runner import ToolRunner
        from src.domain.permission import PermissionChecker, PermissionMode
        from src.domain.schema_validator import SchemaValidator
        from src.domain.tool_contract import build_standard_registry
        from src.infrastructure.file_ops import FileOperations
        from src.infrastructure.sandbox import LocalSandbox, SandboxConfig
        from src.infrastructure.state_persist import StatePersister
        from src.infrastructure.tools.file_tools import (
            make_read_file_handler,
            make_write_file_handler,
        )
        from src.infrastructure.tools.microbenchmark import generate_microbenchmark_handler

        os.chdir(str(tmp_path))

        sandbox = LocalSandbox(SandboxConfig(), sandbox_root=str(tmp_path / ".sandbox"))
        file_ops = FileOperations(sandbox_root=sandbox._sandbox_root)

        registry = build_standard_registry()
        persister = StatePersister(log_dir=str(tmp_path))
        approval_queue = ApprovalQueue(state_dir=str(tmp_path), persister=persister)
        permission_checker = PermissionChecker(mode=PermissionMode.RELAXED)
        validator = SchemaValidator()

        handlers = {
            "read_file": make_read_file_handler(file_ops),
            "write_file": make_write_file_handler(file_ops),
            "generate_microbenchmark": generate_microbenchmark_handler,
        }

        tool_runner = ToolRunner(
            registry=registry,
            tool_handlers=handlers,
            approval_queue=approval_queue,
            permission_checker=permission_checker,
            persister=persister,
            validator=validator,
        )

        session = SessionState(session_id="e2e_test", goal="test")
        control_plane = ControlPlane(rule_dir=str(tmp_path))
        context_manager = ContextManager(max_tokens=8000)

        loop = AgentLoop(
            session=session,
            context_manager=context_manager,
            control_plane=control_plane,
            tool_registry=registry,
            max_turns=5,
            state_dir=str(tmp_path),
            permission_mode=PermissionMode.RELAXED,
        )

        loop.set_tool_executor(tool_runner.execute)
        loop.set_approval_callback(lambda req: True)

        return loop, file_ops, sandbox

    def test_read_file_through_full_pipeline(self, tmp_path):
        """read_file 通过完整链路: AgentLoop → ToolRunner → FileOperations → 磁盘。"""
        loop, file_ops, sandbox = self._setup_loop_with_tools(tmp_path)

        # Create a test file in sandbox
        sandbox_root = sandbox._sandbox_root
        test_file = os.path.join(sandbox_root, "test.cu")
        with open(test_file, "w") as f:
            f.write("__global__ void test() {}")

        # Pre-read the file (M1 requirement for write, but read needs no prior)
        # Actually read_file should work without prior read
        # But FileOperations read doesn't need prior read — only write does

        # Inject tool call as model output
        loop._model_output = json.dumps({
            "tool": "read_file",
            "args": {"file_path": test_file},
        })
        loop.loop_state.is_running = True

        # Run one turn
        loop._inner_loop_step()

        # Verify tool executed
        history = loop._persister.load_history()
        tool_execs = [h for h in history if h.get("action") == "tool_execution"]
        assert len(tool_execs) >= 1
        assert tool_execs[0]["tool_name"] == "read_file"

    def test_write_file_m1_enforced_through_pipeline(self, tmp_path):
        """write_file 通过完整链路 — M1 必须被遵守。"""
        loop, file_ops, sandbox = self._setup_loop_with_tools(tmp_path)

        sandbox_root = sandbox._sandbox_root
        test_file = os.path.join(sandbox_root, "m1test.txt")
        with open(test_file, "w") as f:
            f.write("initial")

        # Without reading first, write should fail at FileOperations level
        loop._model_output = json.dumps({
            "tool": "write_file",
            "args": {"file_path": test_file, "content": "modified"},
        })
        loop.loop_state.is_running = True

        loop._inner_loop_step()

        # Should have error in loop state
        assert loop.loop_state.last_error is not None
        assert "M1 violation" in loop.loop_state.last_error

    def test_write_file_read_then_write_succeeds(self, tmp_path):
        """先 read 再 write — 完整链路通过。"""
        loop, file_ops, sandbox = self._setup_loop_with_tools(tmp_path)

        sandbox_root = sandbox._sandbox_root
        test_file = os.path.join(sandbox_root, "rwtest.txt")
        with open(test_file, "w") as f:
            f.write("original")

        # Step 1: Read the file
        loop._model_output = json.dumps({
            "tool": "read_file",
            "args": {"file_path": test_file},
        })
        loop.loop_state.is_running = True
        loop._inner_loop_step()
        assert loop.loop_state.last_error is None

        # Step 2: Write the file (now M1 is satisfied)
        loop._model_output = json.dumps({
            "tool": "write_file",
            "args": {"file_path": test_file, "content": "modified"},
        })
        loop._inner_loop_step()
        assert loop.loop_state.last_error is None

        # Verify file was actually modified
        assert open(test_file).read() == "modified"

    def test_generate_microbenchmark_through_pipeline(self, tmp_path):
        """generate_microbenchmark 通过完整链路。"""
        loop, file_ops, sandbox = self._setup_loop_with_tools(tmp_path)

        loop._model_output = json.dumps({
            "tool": "generate_microbenchmark",
            "args": {"benchmark_type": "pointer_chase", "parameters": {}},
        })
        loop.loop_state.is_running = True
        loop._inner_loop_step()

        assert loop.loop_state.last_error is None
        history = loop._persister.load_history()
        tool_execs = [h for h in history if h.get("action") == "tool_execution"]
        gen_execs = [h for h in tool_execs if h["tool_name"] == "generate_microbenchmark"]
        assert len(gen_execs) >= 1


# ── 测试用例 2.2：双层循环验证 — 会话持久化与恢复 ──────────────────


class TestSessionPersistenceAndResume:
    """验证会话状态落盘和 --resume 恢复机制。"""

    def test_session_saved_to_disk(self, tmp_path):
        """AgentLoop 每次 turn 都应保存 session 状态到磁盘。"""
        from src.application.agent_loop import AgentLoop
        from src.application.context import ContextManager
        from src.application.control_plane import ControlPlane
        from src.application.session import SessionState, SessionManager
        from src.domain.tool_contract import ToolRegistry

        os.chdir(str(tmp_path))
        session = SessionState(session_id="persist_test", goal="test goal")
        loop = AgentLoop(
            session=session,
            context_manager=ContextManager(max_tokens=4000),
            control_plane=ControlPlane(rule_dir=str(tmp_path)),
            tool_registry=ToolRegistry(),
            max_turns=5,
            state_dir=str(tmp_path),
        )

        # Run one turn
        loop._model_output = "no tool call"
        loop.loop_state.is_running = True
        loop._inner_loop_step()

        # Verify session saved
        mgr = SessionManager(state_dir=str(tmp_path))
        restored = mgr.load_session("persist_test")
        assert restored is not None
        assert restored.step_count == 1

    def test_resume_restores_session(self, tmp_path):
        """--resume 应能从磁盘恢复会话状态。"""
        from src.application.agent_loop import AgentLoop
        from src.application.context import ContextManager
        from src.application.control_plane import ControlPlane
        from src.application.session import SessionState, SessionManager
        from src.domain.tool_contract import ToolRegistry

        os.chdir(str(tmp_path))
        session = SessionState(session_id="resume_test", goal="original goal")
        session.step_count = 5
        session.mark_complete()

        mgr = SessionManager(state_dir=str(tmp_path))
        mgr.save_session(session)

        # Resume via AgentLoop.from_resume
        loop = AgentLoop.from_resume(
            session_id="resume_test",
            control_plane=ControlPlane(rule_dir=str(tmp_path)),
            context_manager=ContextManager(max_tokens=4000),
            tool_registry=ToolRegistry(),
            state_dir=str(tmp_path),
        )

        assert loop.session.session_id == "resume_test"
        assert loop.session.step_count == 5
        assert loop.session.is_complete is True

    def test_resume_with_new_goal(self, tmp_path):
        """--resume 可以覆盖原始 goal。"""
        from src.application.agent_loop import AgentLoop
        from src.application.context import ContextManager
        from src.application.control_plane import ControlPlane
        from src.application.session import SessionState, SessionManager
        from src.domain.tool_contract import ToolRegistry

        os.chdir(str(tmp_path))
        session = SessionState(session_id="resume_goal", goal="old goal")
        mgr = SessionManager(state_dir=str(tmp_path))
        mgr.save_session(session)

        loop = AgentLoop.from_resume(
            session_id="resume_goal",
            control_plane=ControlPlane(rule_dir=str(tmp_path)),
            context_manager=ContextManager(max_tokens=4000),
            tool_registry=ToolRegistry(),
            state_dir=str(tmp_path),
            new_goal="new goal for resumed session",
        )

        assert loop.session.goal == "new goal for resumed session"


# ── 测试用例 2.3：上下文压缩 + 控制平面注入集成 ─────────────────────


class TestContextCompressionIntegration:
    """验证上下文压缩在 AgentLoop 中正常工作。"""

    def test_compress_triggered_when_over_budget(self, tmp_path):
        """当 token 超预算时，AgentLoop 应触发压缩。"""
        from src.application.agent_loop import AgentLoop, EventKind
        from src.application.context import ContextManager, Role
        from src.application.control_plane import ControlPlane
        from src.application.session import SessionState
        from src.domain.tool_contract import ToolRegistry

        os.chdir(str(tmp_path))
        ctx = ContextManager(max_tokens=100)
        ctx.add_entry(Role.SYSTEM, "system", token_count=10)
        # Add enough entries to exceed budget
        for i in range(20):
            ctx.add_entry(Role.USER, f"user msg {i}", token_count=10)
        assert ctx.is_over_budget()

        events = []
        session = SessionState(session_id="compress", goal="test")
        loop = AgentLoop(
            session=session,
            context_manager=ctx,
            control_plane=ControlPlane(rule_dir=str(tmp_path)),
            tool_registry=ToolRegistry(),
            max_turns=3,
            state_dir=str(tmp_path),
        )

        def capture_event(ev):
            events.append(ev)
        loop.on_event(capture_event)

        loop._model_output = "done"
        loop.loop_state.is_running = True
        loop._inner_loop_step()

        # Compress event should have been emitted
        compress_events = [e for e in events if e.kind == EventKind.COMPRESS]
        assert len(compress_events) >= 1
        assert compress_events[0].payload["entries_removed"] > 0
