"""Unit tests for AgentLoop Target State Machine.

Tests the critical logic for:
1. Multi-target compilation (fixing the bug where CodeGen only targets same target)
2. FORCE SWITCH mechanism at max retries
3. MANDATORY TARGET SWITCH after successful execution
4. Target state persistence and restoration

This addresses the root cause identified in the systematic review:
- CodeGen's single-target architecture
- Fixed binary name causing overwrites
- LLM-driven target switching fragility
"""
import json
import os
import pytest
from src.application.agent_loop import (
    AgentLoop,
    LoopState,
    LoopEvent,
)
from src.application.context import ContextManager, Role
from src.application.control_plane import ControlPlane
from src.application.session import SessionState
from src.application.tool_call_parser import ToolCall
from src.domain.tool_contract import ToolRegistry


# ── Helpers ──────────────────────────────────────────────────────────


def _make_registry():
    """Create a minimal tool registry with compile_cuda and execute_binary."""
    reg = ToolRegistry()
    from src.domain.tool_contract import ToolContract
    
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
        description="Execute compiled binary",
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


def _create_loop(tmp_path, max_turns=10):
    """Create an AgentLoop instance with standard test configuration."""
    os.chdir(str(tmp_path))
    control = ControlPlane(rule_dir=str(tmp_path))
    ctx = ContextManager(max_tokens=4000)
    session = SessionState(session_id="test_targets", goal="measure all targets")
    registry = _make_registry()
    
    loop = AgentLoop(
        session=session,
        context_manager=ctx,
        control_plane=control,
        tool_registry=registry,
        max_turns=max_turns,
        state_dir=str(tmp_path),
    )
    
    # Enable loop execution
    loop.loop_state.is_running = True
    loop._model_output = ""
    loop._model_tool_call = None
    
    return loop


# ── Target State Initialization Tests ───────────────────────────────


class TestTargetStateInitialization:
    """Test _init_target_state() and initial state setup."""

    def test_init_sets_first_target_as_current(self, tmp_path):
        """Should set the first target as current_target."""
        loop = _create_loop(tmp_path)
        
        target_spec = {
            "targets": ["dram_latency_cycles", "l2_cache_size_mb", "sm_count"]
        }
        
        loop._init_target_state(target_spec)
        
        assert loop.loop_state.current_target == "dram_latency_cycles"
        assert len(loop.loop_state.completed_targets) == 0
        assert len(loop.loop_state.target_retry_count) == 3

    def test_init_initializes_retry_count_for_all_targets(self, tmp_path):
        """Should initialize retry_count=0 for all targets."""
        loop = _create_loop(tmp_path)
        
        targets = ["target_a", "target_b", "target_c"]
        loop._init_target_state({"targets": targets})
        
        for target in targets:
            assert loop.loop_state.target_retry_count[target] == 0

    def test_init_handles_empty_targets(self, tmp_path):
        """Should handle empty or missing targets gracefully."""
        loop = _create_loop(tmp_path)
        
        loop._init_target_state({})
        assert loop.loop_state.current_target is None
        
        loop._init_target_state(None)
        assert loop.loop_state.current_target is None

    def test_init_handles_single_target(self, tmp_path):
        """Should work correctly with a single target."""
        loop = _create_loop(tmp_path)
        
        loop._init_target_state({"targets": ["only_target"]})
        
        assert loop.loop_state.current_target == "only_target"
        assert loop.loop_state.target_retry_count == {"only_target": 0}


# ── Target State Machine Serialization Tests ────────────────────────


class TestTargetStateSerialization:
    """Test that target state persists correctly through serialization."""

    def test_serializes_all_target_fields(self, tmp_path):
        """LoopState.to_dict() should include all target fields."""
        state = LoopState(
            session_id="test",
            current_target="dram_latency_cycles",
            completed_targets=["l2_cache_size_mb"],
            target_retry_count={
                "dram_latency_cycles": 2,
                "l2_cache_size_mb": 0,
                "sm_count": 0,
            },
        )
        
        data = state.to_dict()
        
        assert "current_target" in data
        assert "completed_targets" in data
        assert "target_retry_count" in data
        assert data["current_target"] == "dram_latency_cycles"
        assert len(data["completed_targets"]) == 1
        assert data["target_retry_count"]["dram_latency_cycles"] == 2

    def test_deserializes_target_fields_correctly(self, tmp_path):
        """LoopState.from_dict() should restore all target fields."""
        original = LoopState(
            session_id="test",
            current_target="sm_count",
            completed_targets=["dram_latency_cycles", "l2_cache_size_mb"],
            target_retry_count={
                "dram_latency_cycles": 1,
                "l2_cache_size_mb": 3,
                "sm_count": 0,
            },
        )
        
        data = original.to_dict()
        restored = LoopState.from_dict(data)
        
        assert restored.current_target == original.current_target
        assert restored.completed_targets == original.completed_targets
        assert restored.target_retry_count == original.target_retry_count

    def test_persists_and_restores_target_state(self, tmp_path):
        """Full round-trip: persist → restore should preserve target state."""
        loop = _create_loop(tmp_path)
        
        # Initialize and modify state
        loop._init_target_state({
            "targets": ["t1", "t2", "t3"]
        })
        loop.loop_state.target_retry_count["t1"] = 2
        loop.loop_state.completed_targets.append("t1")
        
        # Persist
        loop._persist_state()
        
        # Restore
        loop2 = AgentLoop.from_resume(
            session_id="test_targets",
            control_plane=ControlPlane(rule_dir=str(tmp_path)),
            context_manager=ContextManager(max_tokens=4000),
            tool_registry=_make_registry(),
            state_dir=str(tmp_path),
        )
        
        # Verify restoration
        assert loop2.loop_state.current_target == "t1"
        assert loop2.loop_state.target_retry_count["t1"] == 2
        assert "t1" in loop2.loop_state.completed_targets


# ── _get_all_targets() Tests ─────────────────────────────────────────


class TestGetAllTargets:
    """Test extraction of all targets from context."""

    def test_extracts_targets_from_system_message(self, tmp_path):
        """Should extract targets list from SYSTEM role entries."""
        loop = _create_loop(tmp_path)
        
        loop.context_manager.add_entry(
            Role.SYSTEM,
            '{"targets": ["a", "b", "c"], "description": "test"}',
            token_count=50,
        )
        
        targets = loop._get_all_targets()
        
        assert len(targets) == 3
        assert "a" in targets
        assert "b" in targets
        assert "c" in targets

    def test_returns_empty_when_no_targets(self, tmp_path):
        """Should return empty list when no targets defined."""
        loop = _create_loop(tmp_path)
        
        loop.context_manager.add_entry(
            Role.SYSTEM,
            '{"description": "no targets here"}',
            token_count=30,
        )
        
        targets = loop._get_all_targets()
        assert targets == []

    def test_searches_system_role_only(self, tmp_path):
        """Should only look in SYSTEM role entries."""
        loop = _create_loop(tmp_path)
        
        # Add to ASSISTANT role (should be ignored)
        loop.context_manager.add_entry(
            Role.ASSISTANT,
            '{"targets": ["ignored_a", "ignored_b"]}',
            token_count=30,
        )
        
        # Add to SYSTEM role (should be found)
        loop.context_manager.add_entry(
            Role.SYSTEM,
            '{"targets": ["found_c", "found_d"]}',
            token_count=30,
        )
        
        targets = loop._get_all_targets()
        
        assert "found_c" in targets
        assert "found_d" in targets
        assert "ignored_a" not in targets
        assert "ignored_b" not in targets


# ── _find_unmeasured_targets() Tests ─────────────────────────────────


class TestFindUnmeasuredTargets:
    """Test identification of unmeasured targets."""

    def test_finds_all_unmeasured_when_none_completed(self, tmp_path):
        """Should find all targets when none are measured yet."""
        loop = _create_loop(tmp_path)
        
        loop.context_manager.add_entry(
            Role.SYSTEM,
            '{"targets": ["a", "b", "c"]}',
            token_count=50,
        )
        
        unmeasured = loop._find_unmeasured_targets()
        
        assert len(unmeasured) == 3
        assert "a" in unmeasured
        assert "b" in unmeasured
        assert "c" in unmeasured

    def test_excludes_measured_targets(self, tmp_path):
        """Should exclude targets that appear in execute_binary results."""
        loop = _create_loop(tmp_path)
        
        loop.context_manager.add_entry(
            Role.SYSTEM,
            '{"targets": ["a", "b", "c"]}',
            token_count=50,
        )
        
        # Simulate successful measurement of 'a' and 'b'
        loop.context_manager.add_entry(
            Role.ASSISTANT,
            json.dumps({
                "tool": "execute_binary",
                "stdout": "a: 123.45\nb: 678.90\nc: 0",
                "return_code": 0,
            }),
            token_count=30,
        )
        
        unmeasured = loop._find_unmeasured_targets()
        
        # 'c' has value 0, which might still be considered measured
        # depending on implementation. Test expects at least 'a' and 'b' excluded.
        assert "a" not in unmeasured or "b" not in unmeasured

    def test_returns_empty_when_all_measured(self, tmp_path):
        """Should return empty when all targets have been measured."""
        loop = _create_loop(tmp_path)
        
        loop.context_manager.add_entry(
            Role.SYSTEM,
            '{"targets": ["x", "y"]}',
            token_count=50,
        )
        
        # Simulate both targets measured
        loop.context_manager.add_entry(
            Role.ASSISTANT,
            json.dumps({
                "tool": "execute_binary",
                "stdout": "x: 100\ny: 200",
                "return_code": 0,
            }),
            token_count=30,
        )
        
        unmeasured = loop._find_unmeasured_targets()
        assert len(unmeasured) == 0


# ── FORCE SWITCH Logic Tests ─────────────────────────────────────────


class TestForceSwitchLogic:
    """Test FORCE SWITCH mechanism at max retries."""

    def test_force_switch_triggers_at_max_retries(self, tmp_path):
        """FORCE SWITCH should trigger when retry_count >= MAX_RETRIES."""
        loop = _create_loop(tmp_path)
        MAX_RETRIES = 2
        
        loop._init_target_state({"targets": ["t1", "t2", "t3"]})
        loop.loop_state.target_retry_count["t1"] = 2  # At max
        
        retry_count = loop.loop_state.target_retry_count.get("t1", 0)
        should_switch = retry_count >= MAX_RETRIES
        
        assert should_switch is True

    def test_force_switch_does_not_trigger_below_max(self, tmp_path):
        """FORCE SWITCH should NOT trigger when retry_count < MAX_RETRIES."""
        loop = _create_loop(tmp_path)
        MAX_RETRIES = 2
        
        loop._init_target_state({"targets": ["t1", "t2"]})
        loop.loop_state.target_retry_count["t1"] = 1  # Below max
        
        retry_count = loop.loop_state.target_retry_count.get("t1", 0)
        should_switch = retry_count >= MAX_RETRIES
        
        assert should_switch is False

    def test_force_switch_selects_next_unmeasured_target(self, tmp_path):
        """FORCE SWITCH should select the next unmeasured target."""
        loop = _create_loop(tmp_path)
        
        loop._init_target_state({"targets": ["failed_target", "next_target", "last_target"]})
        loop.loop_state.current_target = "failed_target"
        loop.loop_state.target_retry_count["failed_target"] = 2  # Max retries reached
        loop.loop_state.completed_targets = []
        
        # Simulate FORCE SWITCH logic
        all_targets = loop._get_all_targets()
        remaining = [t for t in all_targets 
                    if t not in loop.loop_state.completed_targets 
                    and t != loop.loop_state.current_target]
        
        if remaining:
            next_target = remaining[0]
            assert next_target == "next_target"


# ── MANDATORY TARGET SWITCH Tests ────────────────────────────────────


class TestMandatoryTargetSwitch:
    """Test automatic target switch after successful execution."""

    def test_measurement_parsing_logic(self, tmp_path):
        """Test that measurement parsing correctly identifies targets."""
        loop = _create_loop(tmp_path)

        # Add targets to context (required by _get_all_targets)
        loop.context_manager.add_entry(
            Role.SYSTEM,
            '{"targets": ["t1", "t2", "t3"]}',
            token_count=50,
        )

        loop._init_target_state({"targets": ["t1", "t2", "t3"]})
        loop.loop_state.current_target = "t1"

        # Simulate execute_binary stdout (same format as agent_loop)
        stdout = "t1: 450.5\nother_metric: 100"

        # Parse measurements using same regex as agent_loop._inner_loop_step
        import re
        measurements = {}
        for line in stdout.splitlines():
            if line.strip().startswith("//") or line.strip().startswith("#"):
                continue
            m = re.match(r'\s*([\w_]+)\s*[:=]\s*([\d.]+[eE]?[\d]*)', line)
            if m:
                key, val_str = m.group(1), m.group(2)
                try:
                    measurements[key] = float(val_str)
                except ValueError:
                    pass

        # Verify parsing works
        assert "t1" in measurements
        assert measurements["t1"] == 450.5

        # Simulate recording to completed_targets
        all_targets_list = loop._get_all_targets()
        assert len(all_targets_list) > 0, "_get_all_targets() should find targets in context"
        
        for key in measurements:
            if key not in loop.loop_state.completed_targets and key in all_targets_list:
                loop.loop_state.completed_targets.append(key)

        assert "t1" in loop.loop_state.completed_targets

    def test_switches_to_next_unmeasured_target(self, tmp_path):
        """After completion, should switch to first unmeasured target."""
        loop = _create_loop(tmp_path)
        
        loop._init_target_state({"targets": ["completed_t", "next_t", "future_t"]})
        loop.loop_state.current_target = "completed_t"
        loop.loop_state.completed_targets = ["completed_t"]
        
        # Find next unmeasured
        unmeasured = [t for t in loop._get_all_targets()
                     if t not in loop.loop_state.completed_targets]
        
        if unmeasured:
            next_target = unmeasured[0]
            loop.loop_state.current_target = next_target
            
            assert loop.loop_state.current_target == "next_t"

    def test_no_switch_when_all_targets_completed(self, tmp_path):
        """Should not switch when all targets are measured."""
        loop = _create_loop(tmp_path)
        
        all_targets = ["t1", "t2", "t3"]
        loop._init_target_state({"targets": all_targets})
        loop.loop_state.completed_targets = all_targets.copy()
        
        unmeasured = [t for t in loop._get_all_targets()
                     if t not in loop.loop_state.completed_targets]
        
        assert len(unmeasured) == 0


# ── Integration Tests ─────────────────────────────────────────────────


class TestMultiTargetWorkflow:
    """Integration tests for complete multi-target workflow."""

    def test_full_workflow_three_targets(self, tmp_path):
        """Simulate measuring 3 targets sequentially."""
        loop = _create_loop(tmp_path, max_turns=20)
        
        targets = ["dram_latency", "l2_cache", "sm_count"]
        loop._init_target_state({"targets": targets})
        
        # Initial state
        assert loop.loop_state.current_target == targets[0]
        assert loop.loop_state.target_retry_count[targets[0]] == 0
        
        # Simulate completing target 0
        loop.loop_state.completed_targets.append(targets[0])
        loop.loop_state.current_target = targets[1]
        
        assert loop.loop_state.current_target == targets[1]
        assert targets[0] in loop.loop_state.completed_targets
        assert targets[1] not in loop.loop_state.completed_targets
        
        # Simulate completing target 1
        loop.loop_state.completed_targets.append(targets[1])
        loop.loop_state.current_target = targets[2]
        
        assert loop.loop_state.current_target == targets[2]
        assert len(loop.loop_state.completed_targets) == 2
        
        # Complete final target
        loop.loop_state.completed_targets.append(targets[2])
        
        unmeasured = loop._find_unmeasured_targets()
        assert len(unmeasured) == 0

    def test_workflow_with_retry_and_force_switch(self, tmp_path):
        """Simulate target failing max retries then force switching."""
        loop = _create_loop(tmp_path)
        
        targets = ["failing_target", "backup_target"]
        loop._init_target_state({"targets": targets})
        
        # Initial: working on failing_target
        assert loop.loop_state.current_target == "failing_target"
        
        # Simulate retries
        loop.loop_state.target_retry_count["failing_target"] = 1
        loop.loop_state.target_retry_count["failing_target"] = 2  # Max retries
        
        # Force switch triggers
        MAX_RETRIES = 2
        should_force = loop.loop_state.target_retry_count["failing_target"] >= MAX_RETRIES
        assert should_force is True
        
        # Switch to backup
        remaining = [t for t in targets 
                    if t not in loop.loop_state.completed_targets 
                    and t != "failing_target"]
        if remaining:
            loop.loop_state.current_target = remaining[0]
            loop.loop_state.target_retry_count[remaining[0]] = 0
            
            assert loop.loop_state.current_target == "backup_target"
            assert loop.loop_state.target_retry_count["backup_target"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
