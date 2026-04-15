"""Tests for Dual-Layer Agent Loop (application/agent_loop.py)

Outer loop: session management, lifecycle, sub-agent coordination.
Inner loop: context assembly → model call → tool execution → repeat.
"""
import json
import os
import pytest
from src.application.agent_loop import (
    AgentLoop,
    LoopEvent,
    LoopState,
)
from src.application.tool_call_parser import ToolCall
from src.application.control_plane import ControlPlane
from src.application.context import ContextManager
from src.application.session import SessionState
from src.domain.tool_contract import ToolContract, ToolRegistry


# ── Helpers ──────────────────────────────────────────────────────────


def _make_registry():
    reg = ToolRegistry()
    reg.register(ToolContract(
        name="echo",
        description="Echo back input",
        input_schema={"text": "string"},
        output_schema={"echo": "string"},
        permissions=["process:exec"],
        requires_approval=False,
        is_blocking=True,
    ))
    reg.register(ToolContract(
        name="read_file",
        description="Read file",
        input_schema={"file_path": "string"},
        output_schema={"content": "string"},
        permissions=["file:read"],
        requires_approval=False,
        is_blocking=True,
    ))
    return reg


# ── ToolCall Tests ───────────────────────────────────────────────────


class TestToolCall:
    def test_create_tool_call(self):
        tc = ToolCall(name="echo", arguments={"text": "hello"})
        assert tc.name == "echo"
        assert tc.arguments == {"text": "hello"}

    def test_from_dict(self):
        tc = ToolCall.from_dict({"name": "read_file", "arguments": {"file_path": "a.txt"}})
        assert tc.name == "read_file"
        assert tc.arguments == {"file_path": "a.txt"}


# ── LoopState Tests ─────────────────────────────────────────────────


class TestLoopState:
    def test_initial_state(self):
        ls = LoopState(session_id="s1")
        assert ls.is_running is False
        assert ls.turn_count == 0
        assert ls.pending_tool_call is None

    def test_to_dict_roundtrip(self):
        ls = LoopState(session_id="s1")
        ls.turn_count = 5
        ls.is_running = True
        d = ls.to_dict()
        restored = LoopState.from_dict(d)
        assert restored.turn_count == 5
        assert restored.is_running is True
        assert restored.session_id == "s1"


# ── AgentLoop Tests ─────────────────────────────────────────────────


class TestAgentLoop:
    @pytest.fixture()
    def loop(self, tmp_path):
        os.chdir(str(tmp_path))
        control = ControlPlane(rule_dir=str(tmp_path))
        ctx = ContextManager(max_tokens=4000)
        session = SessionState(session_id="test", goal="test goal")
        registry = _make_registry()
        lp = AgentLoop(
            session=session,
            context_manager=ctx,
            control_plane=control,
            tool_registry=registry,
            max_turns=10,
            state_dir=str(tmp_path),
        )
        # Default: no tool call → single-turn then stop
        lp._model_output = "final answer"
        lp._model_tool_call = None
        lp.loop_state.is_running = True  # enable _inner_loop_step() to execute
        return lp

    def test_loop_starts_not_running(self, loop):
        """Fresh LoopState should default to not running."""
        fresh = LoopState(session_id="fresh")
        assert fresh.is_running is False

    def test_start_marks_running(self, loop):
        """is_running should be True during the loop execution."""
        running_during_start = []
        loop.on_event(lambda ev: running_during_start.append(loop.loop_state.is_running))
        loop.start()
        # During execution, is_running was True
        assert True in running_during_start

    def test_stop_marks_not_running(self, loop):
        loop.start()
        loop.stop()
        assert loop.loop_state.is_running is False

    def test_max_turns_enforced(self, loop):
        """Loop should terminate after max_turns."""
        loop._model_output = "text"  # no tool call → each turn ends
        loop._model_tool_call = None
        loop.start()
        # Run the internal loop logic manually for testing
        for _ in range(12):
            loop._inner_loop_step()
        # Should have stopped after max_turns (10)
        assert loop.loop_state.turn_count <= 10
        assert loop.loop_state.is_running is False

    def test_context_receives_system_injection(self, loop):
        """Each turn should inject control plane context."""
        loop._inner_loop_step()
        entries = loop.context_manager.get_entries()
        # At least the system context should be present
        assert len(entries) >= 1

    def test_tool_call_parsing(self, loop):
        """Parse a tool call from model output."""
        loop._model_output = json.dumps({"tool": "echo", "args": {"text": "hi"}})
        tc = loop._tool_call_parser.parse(loop._model_output, loop.tool_registry)
        assert tc is not None
        assert tc.name == "echo"
        assert tc.arguments == {"text": "hi"}

    def test_tool_call_none_when_no_json(self, loop):
        loop._model_output = "I think the L2 cache is 4MB."
        tc = loop._tool_call_parser.parse(loop._model_output, loop.tool_registry)
        assert tc is None

    def test_tool_call_for_unregistered_raises(self, loop):
        loop._model_output = json.dumps({"tool": "nonexistent", "args": {}})
        with pytest.raises(KeyError):
            loop._execute_tool_call(ToolCall(name="nonexistent", arguments={}))

    def test_events_emitted(self, loop):
        events = []
        loop.on_event(lambda ev: events.append(ev))
        loop.start()
        # Should have at least start and turn events
        assert any(ev.kind.value == "start" for ev in events)
        assert any(ev.kind.value in ("turn", "stop") for ev in events)

    def test_inner_loop_yields_events(self, loop):
        events = []
        loop.on_event(lambda ev: events.append(ev))
        loop._model_output = "final answer"
        loop._model_tool_call = None
        loop._inner_loop_step()
        # At least one event for the turn
        assert len(events) >= 1

    def test_persist_state_after_turn(self, loop):
        """P6: state must be persisted after each turn."""
        loop._model_output = "done"
        loop._model_tool_call = None
        loop._inner_loop_step()
        # State dir should have a session file
        state_dir = loop._state_dir
        files = os.listdir(state_dir)
        assert any(f.startswith("session_") for f in files)

    def test_resume_from_checkpoint(self, tmp_path):
        os.chdir(str(tmp_path))
        control = ControlPlane(rule_dir=str(tmp_path))
        ctx = ContextManager(max_tokens=4000)
        session = SessionState(session_id="chk", goal="goal")
        session.increment_step()
        registry = _make_registry()
        loop = AgentLoop(
            session=session,
            context_manager=ctx,
            control_plane=control,
            tool_registry=registry,
            max_turns=10,
            state_dir=str(tmp_path),
        )
        # Persist then resume
        loop._persist_state()
        loop2 = AgentLoop.from_resume(
            session_id="chk",
            control_plane=control,
            context_manager=ctx,
            tool_registry=registry,
            state_dir=str(tmp_path),
        )
        assert loop2.session.step_count == 1
        assert loop2.session.session_id == "chk"
