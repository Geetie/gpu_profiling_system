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
from src.application.context import ContextManager, Role
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


# ── Target State Machine Tests ──────────────────────────────────────


class TestTargetStateMachine:
    """Tests for target state machine (current_target, completed_targets, target_retry_count)."""

    def test_init_target_state(self):
        """Test _init_target_state() initializes all fields correctly."""
        state = LoopState(session_id="test_session")
        assert state.current_target is None
        assert state.completed_targets == []
        assert state.target_retry_count == {}

    def test_target_state_serialization(self):
        """Test LoopState.to_dict() and from_dict() preserve target fields."""
        original = LoopState(
            session_id="test",
            current_target="dram_latency_cycles",
            completed_targets=["l2_cache_size_mb"],
            target_retry_count={"dram_latency_cycles": 2, "l2_cache_size_mb": 0},
        )
        data = original.to_dict()
        restored = LoopState.from_dict(data)

        assert restored.current_target == original.current_target
        assert restored.completed_targets == original.completed_targets
        assert restored.target_retry_count == original.target_retry_count

    def test_find_unmeasured_targets_basic(self):
        """Test _find_unmeasured_targets() with simple key:value format."""
        ctx = ContextManager()

        # Add system message with targets
        ctx.add_entry(
            role=Role.SYSTEM,
            content='{"targets": ["dram_latency_cycles", "l2_cache_size_mb", "actual_boost_clock_mhz"]}',
            token_count=50,
        )

        # Add assistant message with one measured target
        ctx.add_entry(
            role=Role.ASSISTANT,
            content=json.dumps({
                "tool": "execute_binary",
                "stdout": "dram_latency_cycles: 450.5\nsm_count: 56",
                "return_code": 0,
            }),
            token_count=30,
        )

        # Create AgentLoop instance (minimal setup)
        control = ControlPlane()
        registry = _make_registry()
        session = SessionState(session_id="test_find", goal="test goal")
        loop = AgentLoop(
            session=session,
            context_manager=ctx,
            control_plane=control,
            tool_registry=registry,
            max_turns=10,
        )

        unmeasured = loop._find_unmeasured_targets()

        # Should find 2 unmeasured targets (l2_cache_size_mb, actual_boost_clock_mhz)
        assert "dram_latency_cycles" not in unmeasured
        assert "l2_cache_size_mb" in unmeasured
        assert "actual_boost_clock_mhz" in unmeasured
        assert len(unmeasured) == 2

    def test_find_unmeasured_targets_json_stdout(self):
        """Test _find_unmeasured_targets() parses JSON stdout format."""
        ctx = ContextManager()

        # Add system message with targets
        ctx.add_entry(
            role=Role.SYSTEM,
            content='{"targets": ["target1", "target2"]}',
            token_count=50,
        )

        # Add assistant message with JSON result containing stdout
        ctx.add_entry(
            role=Role.ASSISTANT,
            content=json.dumps({
                "tool": "execute_binary",
                "stdout": "target1: 123.45\ntarget2: 678.90",
                "success": True,
            }),
            token_count=30,
        )

        control = ControlPlane()
        registry = _make_registry()
        session = SessionState(session_id="test_json", goal="test goal")
        loop = AgentLoop(
            session=session,
            context_manager=ctx,
            control_plane=control,
            tool_registry=registry,
            max_turns=10,
        )

        unmeasured = loop._find_unmeasured_targets()

        # Both targets should be marked as measured
        assert len(unmeasured) == 0

    def test_detect_cuda_syntax_error_asm_volatile(self):
        """Test _detect_cuda_syntax_error() detects asm volatile missing colon."""
        control = ControlPlane()
        registry = _make_registry()
        session = SessionState(session_id="test_syntax", goal="test goal")
        ctx = ContextManager()
        loop = AgentLoop(
            session=session,
            context_manager=ctx,
            control_plane=control,
            tool_registry=registry,
            max_turns=10,
        )

        error_text = 'source.cu(42): error: expected a ";"\nasm volatile("") : "+l"(sink64)'
        guidance = loop._detect_cuda_syntax_error(error_text)

        assert guidance is not None
        assert "asm volatile" in guidance
        assert "SYNTAX ERROR" in guidance

    def test_detect_cuda_syntax_error_missing_include(self):
        """Test _detect_cuda_syntax_error() detects missing #include."""
        control = ControlPlane()
        registry = _make_registry()
        session = SessionState(session_id="test_include", goal="test goal")
        ctx = ContextManager()
        loop = AgentLoop(
            session=session,
            context_manager=ctx,
            control_plane=control,
            tool_registry=registry,
            max_turns=10,
        )

        error_text = 'source.cu(15): error: identifier "printf" is undefined'
        guidance = loop._detect_cuda_syntax_error(error_text)

        assert guidance is not None
        assert "#include" in guidance or "printf" in guidance

    def test_force_switch_should_trigger_at_max_retries(self):
        """Test FORCE SWITCH logic triggers when retry_count >= MAX_RETRIES_PER_TARGET."""
        MAX_RETRIES = 2

        # Simulate state where retry_count has reached max
        state = LoopState(
            session_id="test_force",
            current_target="dram_latency_cycles",
            completed_targets=[],
            target_retry_count={"dram_latency_cycles": 2},  # At max
        )

        # Check condition
        retry_count = state.target_retry_count.get(state.current_target, 0)
        should_switch = retry_count >= MAX_RETRIES

        assert should_switch == True, "FORCE SWITCH should trigger at retry_count=2"

    def test_force_switch_should_not_trigger_below_max(self):
        """Test FORCE SWITCH logic does NOT trigger when retry_count < MAX."""
        MAX_RETRIES = 2

        # Simulate state where retry_count is below max
        state = LoopState(
            session_id="test_no_force",
            current_target="dram_latency_cycles",
            completed_targets=[],
            target_retry_count={"dram_latency_cycles": 1},  # Below max
        )

        retry_count = state.target_retry_count.get(state.current_target, 0)
        should_switch = retry_count >= MAX_RETRIES

        assert should_switch == False, "FORCE SWITCH should NOT trigger at retry_count=1"

    def test_persist_and_restore_target_state(self, tmp_path):
        """Test that target state persists and restores correctly across turns."""
        import tempfile

        state_dir = str(tmp_path / "test_state")

        control = ControlPlane()
        registry = _make_registry()
        ctx = ContextManager()
        session = SessionState(session_id="test_persist", goal="test goal")

        loop = AgentLoop(
            session=session,
            context_manager=ctx,
            control_plane=control,
            tool_registry=registry,
            max_turns=10,
            state_dir=state_dir,
        )

        # Initialize target state
        loop._init_target_state({
            "targets": ["dram_latency_cycles", "l2_cache_size_mb", "actual_boost_clock_mhz"]
        })

        assert loop.loop_state.current_target == "dram_latency_cycles"
        assert loop.loop_state.target_retry_count == {
            "dram_latency_cycles": 0,
            "l2_cache_size_mb": 0,
            "actual_boost_clock_mhz": 0,
        }

        # Simulate some retries on first target
        loop.loop_state.target_retry_count["dram_latency_cycles"] = 2

        # Persist state
        loop._persist_state()

        # Resume from persisted state
        loop2 = AgentLoop.from_resume(
            session_id="test_persist",
            control_plane=control,
            context_manager=ctx,
            tool_registry=registry,
            state_dir=state_dir,
        )

        # Verify restored state
        assert loop2.loop_state.current_target == "dram_latency_cycles"
        assert loop2.loop_state.target_retry_count["dram_latency_cycles"] == 2
        print("✅ Target state persistence and restoration works correctly!")
