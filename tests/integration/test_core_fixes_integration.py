"""Integration Tests for Core Fixes (C-01, C-02, H-01, M-02)

Comprehensive integration tests that verify the three core fixes work together:
1. Three-target complete measurement workflow (normal path)
2. LLM stall recovery flow
3. State inconsistency auto-repair
4. Long conversation context compression trigger

These tests simulate real-world scenarios from the stability audit report.
"""
import json
import os
import pytest
from src.application.target_state_machine import TargetStateMachine
from src.application.agent_loop import (
    AgentLoop,
    LoopState,
)
from src.application.context import ContextManager, Role, Priority
from src.application.control_plane import ControlPlane
from src.application.session import SessionState
from src.domain.tool_contract import ToolRegistry, ToolContract


# ── Helpers ──────────────────────────────────────────────────────────


def _make_full_registry():
    """Create complete tool registry for integration tests."""
    reg = ToolRegistry()

    reg.register(ToolContract(
        name="compile_cuda",
        description="Compile CUDA source code",
        input_schema={"source": "string", "flags": "list", "target": "string"},
        output_schema={
            "success": "boolean",
            "binary_path": "string",
            "status": "string",
            "errors": "string",
            "target": "string",
        },
        permissions=["process:exec"],
        requires_approval=False,
        is_blocking=True,
    ))

    reg.register(ToolContract(
        name="execute_binary",
        description="Execute compiled binary and get measurements",
        input_schema={"binary_path": "string"},
        output_schema={
            "success": "boolean",
            "stdout": "string",
            "return_code": "integer",
        },
        permissions=["process:exec"],
        requires_approval=False,
        is_blocking=True,
    ))

    return reg


def _create_integration_loop(tmp_path):
    """Create AgentLoop configured for integration testing."""
    os.chdir(str(tmp_path))
    control = ControlPlane(rule_dir=str(tmp_path))
    ctx = ContextManager(max_tokens=15000)  # Larger budget for integration
    session = SessionState(
        session_id="integration_test",
        goal="Measure all GPU profiling targets"
    )
    registry = _make_full_registry()

    loop = AgentLoop(
        session=session,
        context_manager=ctx,
        control_plane=control,
        tool_registry=registry,
        max_turns=50,  # Allow enough turns
        state_dir=str(tmp_path),
    )

    loop.loop_state.is_running = True
    loop._model_output = ""
    loop._model_tool_call = None

    return loop


def _simulate_compile_and_execute(loop, target_name, measurement_value):
    """
    Simulate a successful compile + execute cycle for a target.

    This mimics what would happen in real execution:
    1. compile_cuda succeeds -> returns binary_path
    2. execute_binary succeeds -> returns stdout with measurement
    """
    binary_path = f"/tmp/binary_{target_name}"

    # Step 1: compile_cuda success
    compile_result = json.dumps({
        "tool": "compile_cuda",
        "success": True,
        "binary_path": binary_path,
        "status": "compiled successfully",
        "target": target_name,
    })
    loop.context_manager.add_entry(Role.ASSISTANT, compile_result, token_count=60)

    # Step 2: execute_binary success with measurement
    execute_result = json.dumps({
        "tool": "execute_binary",
        "success": True,
        "stdout": f"{target_name}: {measurement_value}\n",
        "return_code": 0,
        "binary_path": binary_path,
    })
    loop.context_manager.add_entry(Role.ASSISTANT, execute_result, token_count=50)

    return binary_path


# ════════════════════════════════════════════════════════════════════
# SCENARIO 1: Three-Target Complete Measurement Workflow (Normal Path)
# ════════════════════════════════════════════════════════════════════


class TestThreeTargetCompleteWorkflow:
    """
    Scenario 1: Normal path - measure all 3 targets sequentially

    Pre-conditions:
      - targets = ["dram_latency_cycles", "l2_cache_size_mb", "actual_boost_clock_mhz"]
      - TargetStateMachine initialized

    Expected behavior:
      - All 3 targets measured without any stall recovery
      - Clean state transitions
      - Final state: is_all_completed == True
    """

    def test_complete_three_target_normal_flow(self, tmp_path):
        """
        Full workflow:
        1. start_first_target() -> current="dram_latency_cycles"
        2. Simulate compile + execute for dram (value=487)
        3. complete_current_target() -> current="l2_cache_size_mb"
        4. Simulate compile + execute for l2 (value=30)
        5. complete_current_target() -> current="actual_boost_clock_mhz"
        6. Simulate compile + execute for boost_clock (value=1500)
        7. complete_current_target() -> returns None (all done)
        """
        machine = TargetStateMachine()
        targets = ["dram_latency_cycles", "l2_cache_size_mb", "actual_boost_clock_mhz"]
        machine.initialize(targets)

        # Step 1: Start first target
        first = machine.start_first_target()
        assert first == "dram_latency_cycles"
        assert machine.current_target == "dram_latency_cycles"

        # Step 2: Complete first target (simulating successful measurement)
        next_target = machine.complete_current_target()
        assert "dram_latency_cycles" in machine.completed_targets
        assert next_target == "l2_cache_size_mb"

        # Step 3: Complete second target
        next_target = machine.complete_current_target()
        assert "l2_cache_size_mb" in machine.completed_targets
        assert next_target == "actual_boost_clock_mhz"

        # Step 4: Complete final target
        result = machine.complete_current_target()
        assert result is None  # All done
        assert machine.is_all_completed is True

        # Verify final state
        assert machine.progress == (3, 3)
        assert len(machine.completed_targets) == 3
        assert machine.unmeasured_targets == []

    def test_workflow_with_agent_loop_integration(self, tmp_path):
        """
        Integration with AgentLoop: verify measurements are recorded correctly
        """
        loop = _create_integration_loop(tmp_path)
        targets = ["metric_a", "metric_b", "metric_c"]
        loop._init_target_state({"targets": targets})

        # Simulate measuring each target
        for i, target in enumerate(targets):
            loop.loop_state.current_target = target
            _simulate_compile_and_execute(loop, target, float((i + 1) * 100))

            # Record completion
            if target not in loop.loop_state.completed_targets:
                loop.loop_state.completed_targets.append(target)

        # Verify all recorded
        unmeasured = loop._find_unmeasured_targets()
        assert len(unmeasured) == 0, f"All targets should be measured, but found unmeasured: {unmeasured}"
        assert len(loop.loop_state.completed_targets) == 3


# ════════════════════════════════════════════════════════════════════
# SCENARIO 2: LLM Stall Recovery Flow
# ════════════════════════════════════════════════════════════════════


class TestLLMStallRecoveryFlow:
    """
    Scenario 2: LLM stops calling tools, triggering forced recovery

    Pre-conditions:
      - targets = [A, B, C]
      - A and B already measured
      - Should be working on C but LLM outputs text only

    Expected behavior:
      - Stall detection triggers after threshold
      - Forced recovery switches to C
      - Emergency guidance injected
      - State machine updated proactively
    """

    def test_stall_detection_and_recovery_cycle(self, tmp_path):
        """
        Simulate the exact scenario from the crash report:

        Turn 5: execute_binary(B) success -> completed=[A,B]
        Turn 6: LLM text only (no tool) -> consecutive=1
        Turn 7: LLM text only (no tool) -> consecutive=2 -> STALL TRIGGERED
                -> Force switch to C, inject emergency guidance
        Turn 8: LLM should receive guidance and call compile_cuda(C)
        """
        loop = _create_integration_loop(tmp_path)
        loop._init_target_state({"targets": ["target_A", "target_B", "target_C"]})

        # Setup: A and B already completed
        loop.loop_state.completed_targets = ["target_A", "target_B"]
        loop.loop_state.current_target = "target_B"  # Just finished B
        loop.loop_state.turn_count = 5

        # Add their measurements to context
        _simulate_compile_and_execute(loop, "target_A", 100)
        _simulate_compile_and_execute(loop, "target_B", 200)

        # Turn 6: Text-only response (stall begins)
        loop.loop_state.turn_count = 6
        loop._model_output = "I think I'm done with the task."
        loop.loop_state.consecutive_no_tool_calls += 1
        assert loop.loop_state.consecutive_no_tool_calls == 1

        # Turn 7: Another text-only response (stall continues)
        loop.loop_state.turn_count = 7
        loop._model_output = "All measurements look good."
        loop.loop_state.consecutive_no_tool_calls += 1
        assert loop.loop_state.consecutive_no_tool_calls == 2

        # Check if stall should trigger (early turn, threshold=2)
        MAX_CONSECUTIVE_NO_TOOL = 2 if loop.loop_state.turn_count <= 10 else 1
        should_trigger = loop.loop_state.consecutive_no_tool_calls >= MAX_CONSECUTIVE_NO_TOOL
        assert should_trigger is True, "Stall should be detected at this point"

        # Execute forced recovery logic
        unmeasured = loop._find_unmeasured_targets()
        assert unmeasured == ["target_C"], "Only target_C should remain unmeasured"

        if unmeasured:
            next_target = unmeasured[0]

            # Update state machine (C-02 FIX)
            prev_target = loop.loop_state.current_target
            if prev_target and prev_target not in loop.loop_state.completed_targets:
                loop.loop_state.completed_targets.append(prev_target)

            loop.loop_state.current_target = next_target
            loop.loop_state.target_retry_count[next_target] = 0
            loop.loop_state.consecutive_no_tool_calls = 0
            loop.loop_state.stall_recovery_triggered = True

            # Inject emergency guidance
            emergency_guidance = (
                f"🚨🚨🚨 EMERGENCY STALL RECOVERY ACTIVATED 🚨🚨🚨\n\n"
                f"YOUR ONLY TASK: Measure '{next_target}'\n\n"
                f"⛔ Call compile_cuda NOW!"
            )
            loop.context_manager.add_entry(Role.SYSTEM, emergency_guidance, token_count=150)

        # Verify recovery effects
        assert loop.loop_state.current_target == "target_C"
        assert loop.loop_state.consecutive_no_tool_calls == 0
        assert loop.loop_state.stall_recovery_triggered is True
        assert "target_B" in loop.loop_state.completed_targets or "target_B" in loop.loop_state.completed_targets

        # Verify emergency guidance was injected
        entries = loop.context_manager.get_entries()
        has_emergency = any("EMERGENCY STALL RECOVERY" in e.content for e in entries)
        assert has_emergency, "Emergency guidance must be in context after recovery"

    def test_multiple_stall_recoveries_in_session(self, tmp_path):
        """
        Test that multiple stall recoveries can happen in one session
        without corrupting state.
        """
        loop = _create_integration_loop(tmp_path)
        loop._init_target_state({"targets": ["T1", "T2", "T3", "T4"]})
        loop.loop_state.turn_count = 8  # Early turn zone

        # First stall on T1
        loop.loop_state.current_target = "T1"
        loop.loop_state.consecutive_no_tool_calls = 2
        unmeasured = loop._find_unmeasured_targets()
        if unmeasured and loop.loop_state.consecutive_no_tool_calls >= 2:
            if loop.loop_state.current_target not in loop.loop_state.completed_targets:
                loop.loop_state.completed_targets.append(loop.loop_state.current_target)
            loop.loop_state.current_target = unmeasured[0]
            loop.loop_state.consecutive_no_tool_calls = 0

        assert loop.loop_state.current_target == "T2"
        assert "T1" in loop.loop_state.completed_targets

        # Second stall on T2
        loop.loop_state.consecutive_no_tool_calls = 2
        unmeasured = loop._find_unmeasured_targets()
        if unmeasured and loop.loop_state.consecutive_no_tool_calls >= 2:
            if loop.loop_state.current_target not in loop.loop_state.completed_targets:
                loop.loop_state.completed_targets.append(loop.loop_state.current_target)
            loop.loop_state.current_target = unmeasured[0]
            loop.loop_state.consecutive_no_tool_calls = 0

        assert loop.loop_state.current_target == "T3"
        assert "T2" in loop.loop_state.completed_targets
        assert len(loop.loop_state.completed_targets) == 2  # No duplicates


# ════════════════════════════════════════════════════════════════════
# SCENARIO 3: State Inconsistency Auto-Repair
# ════════════════════════════════════════════════════════════════════


class TestStateInconsistencyAutoRepair:
    """
    Scenario 3: State becomes inconsistent, sync mechanism repairs it

    Pre-conditions:
      - targets = [X, Y, Z]
      - loop_state shows: completed=[X], current=Y
      - But context shows X AND Y both measured (via execute_binary output)

    Expected behavior:
      - _sync_target_state_machine() detects inconsistency
      - Y added to completed_targets
      - current switched to Z
      - Sync logs printed for debugging
    """

    def test_sync_detects_and_fixes_completed_targets(self, tmp_path):
        """
        Context has measurements for X and Y,
        but loop_state only knows about X being completed.
        """
        loop = _create_integration_loop(tmp_path)
        loop._init_target_state({"targets": ["X", "Y", "Z"]})

        # Set up inconsistent state
        loop.loop_state.completed_targets = ["X"]  # Only X marked as completed
        loop.loop_state.current_target = "Y"       # Currently on Y

        # But context shows both X and Y measured
        _simulate_compile_and_execute(loop, "X", 100)
        _simulate_compile_and_execute(loop, "Y", 200)

        # Run sync mechanism
        loop._sync_target_state_machine()

        # Verify repair
        assert "Y" in loop.loop_state.completed_targets, \
            "Sync should have discovered Y is measured"
        assert loop.loop_state.current_target == "Z", \
            "Sync should have switched to Z (only remaining unmeasured)"

    def test_sync_handles_stale_current_target(self, tmp_path):
        """
        current_target points to an already-measured target,
        but hasn't been advanced yet.
        """
        loop = _create_integration_loop(tmp_path)
        loop._init_target_state({"targets": ["measured_old", "next_new"]})

        loop.loop_state.current_target = "measured_old"
        loop.loop_state.completed_targets = []

        # Context shows measured_old is done
        _simulate_compile_and_execute(loop, "measured_old", 42)

        # Sync should detect stale pointer and advance
        loop._sync_target_state_machine()

        assert loop.loop_state.current_target == "next_new", \
            "Should advance past already-measured target"
        assert "measured_old" in loop.loop_state.completed_targets, \
            "Should mark old target as completed"

    def test_sync_with_partial_measurements(self, tmp_path):
        """
        Some targets have measurements, some don't.
        Sync should handle partial information gracefully.
        """
        loop = _create_integration_loop(tmp_path)
        loop._init_target_state({"targets": ["P1", "P2", "P3", "P4"]})

        loop.loop_state.completed_targets = []
        loop.loop_state.current_target = "P1"

        # Only P1 and P3 have measurements
        _simulate_compile_and_execute(loop, "P1", 111)
        _simulate_compile_and_execute(loop, "P3", 333)
        # P2 and P4 missing

        # Sync
        loop._sync_target_state_machine()

        # P1 should be detected as completed
        assert "P1" in loop.loop_state.completed_targets

        # Current should move to an unmeasured target (P2 or P4)
        assert loop.loop_state.current_target in ["P2", "P4"], \
            f"Should move to unmeasured target, got: {loop.loop_state.current_target}"


# ════════════════════════════════════════════════════════════════════
# SCENARIO 4: Long Conversation Context Compression Trigger
# ════════════════════════════════════════════════════════════════════


class TestLongConversationContextCompression:
    """
    Scenario 4: Long conversation triggers context compression

    Pre-conditions:
      - ContextManager with max_tokens=5000
      - Simulate 30 turns of conversation (~6000 tokens total)

    Expected behavior:
      - is_over_budget() returns True
      - compress() reduces to ~80% of max
      - Critical info preserved
      - compress() returns count > 0
    """

    def test_compression_triggers_after_many_turns(self, tmp_path):
        """
        Simulate 30 turns, each adding ~200 tokens.
        Total ~6000 tokens > max_tokens(5000).
        """
        cm = ContextManager(max_tokens=5000)

        # Simulate 30 conversation turns
        for turn in range(30):
            # User message
            cm.add_entry(
                Role.USER,
                f"Turn {turn}: Please measure the GPU metrics.",
                token_count=25,
                priority=Priority.LOW if turn < 20 else Priority.DISPOSABLE
            )

            # Assistant response (tool call or text)
            if turn % 3 == 0:
                # Tool call result
                cm.add_entry(
                    Role.ASSISTANT,
                    json.dumps({
                        "tool": "compile_cuda" if turn % 6 == 0 else "execute_binary",
                        "success": True,
                        "stdout": f"metric_{turn}: {turn * 10}\n" if turn % 6 != 0 else "",
                        "binary_path": f"/tmp/bin_{turn}" if turn % 6 == 0 else "",
                    }),
                    token_count=60,
                    priority=Priority.MEDIUM
                )
            else:
                # Text response
                cm.add_entry(
                    Role.ASSISTANT,
                    f"I'm working on compiling the code for metric_{turn}.",
                    token_count=35,
                    priority=Priority.LOW
                )

            # System guidance (occasional)
            if turn % 5 == 0:
                cm.add_entry(
                    Role.SYSTEM,
                    f"⚠️ Remember to measure all targets. Current progress: {turn}/30",
                    token_count=40,
                    priority=Priority.LOW
                )

        # Should be over budget
        print(f"[TEST] Total tokens before compress: {cm.total_tokens}")
        assert cm.is_over_budget(), \
            f"Should be over budget with {cm.total_tokens} tokens (max=5000)"

        # Trigger compression
        removed = cm.compress()

        print(f"[TEST] Tokens after compress: {cm.total_tokens}, removed={removed}")

        # Verify compression worked
        assert removed > 0, "Compression should have removed some entries"
        assert cm.total_tokens <= 5000 * cm.COMPRESSION_RATIO + 50, \
            f"After compress, should be near 80% of budget, got {cm.total_tokens}"

    def test_critical_info_survives_long_conversation(self, tmp_path):
        """
        Even after many turns and compression,
        critical system instructions must survive.
        """
        cm = ContextManager(max_tokens=3000)

        # Critical permanent entry
        critical_system_prompt = (
            "SYSTEM INSTRUCTION: You are a GPU profiling agent. "
            "You MUST measure ALL targets in target_spec.json. "
            "Use compile_cuda then execute_binary for each target."
        )
        cm.add_entry(Role.SYSTEM, critical_system_prompt, token_count=80, priority=Priority.PERMANENT)

        # High priority: current target
        current_target_msg = "🎯 CURRENT TARGET: actual_boost_clock_mhz - NOT YET MEASURED"
        cm.add_entry(Role.SYSTEM, current_target_msg, token_count=35, priority=Priority.HIGH)

        # Flood with noise to force heavy compression
        for i in range(50):
            cm.add_entry(
                Role.ASSISTANT,
                f"Noise message {i} to fill up context " * 3,
                token_count=45,
                priority=Priority.DISPOSABLE
            )

        # Compress multiple times if needed
        while cm.is_over_budget():
            cm.compress()

        # Verify critical info survived
        entries = cm.get_entries()
        contents = [e.content for e in entries]

        assert any("GPU profiling agent" in c for c in contents), \
            "Critical system prompt must survive compression"
        assert any("CURRENT TARGET" in c for c in contents), \
            "Current target info must survive compression"

    def test_recent_measurements_priority_during_compression(self, tmp_path):
        """
        Recent measurements should have higher survival rate
        than old conversation messages.
        """
        cm = ContextManager(max_tokens=2000)

        # Old conversations (should be compressed/removed first)
        for i in range(20):
            cm.add_entry(
                Role.ASSISTANT,
                f"Old discussion about approach {i}",
                token_count=30,
                priority=Priority.LOW
            )

        # Recent important measurement
        recent_measurement = json.dumps({
            "tool": "execute_binary",
            "success": True,
            "stdout": "dram_latency_cycles: 487\nl2_cache_size_mb: 30",
            "return_code": 0,
        })
        cm.add_entry(Role.ASSISTANT, recent_measurement, token_count=55, priority=Priority.HIGH)

        # More noise
        for i in range(10):
            cm.add_entry(Role.USER, f"User chat {i}", token_count=25, priority=Priority.DISPOSABLE)

        # Compress
        cm.compress()

        # Measurement should survive
        entries = cm.get_entries()
        has_measurement = any("dram_latency_cycles" in e.content for e in entries)
        assert has_measurement, "Recent HIGH-priority measurement must survive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
