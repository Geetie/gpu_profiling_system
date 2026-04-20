"""Unit tests for TargetStateMachine (H-01: Unified State Manager)

Comprehensive tests for the centralized target state machine that addresses:
- AgentLoop God Object anti-pattern
- State inconsistency issues
- Independent state management

Test Coverage:
1. Initialization and basic operations (4 tests)
2. State transitions (6 tests)
3. Serialization and persistence (4 tests)
4. Event notification mechanism (4 tests)
5. Edge cases and error handling (3 tests)

Total: 21 test cases
"""
import pytest
from src.application.target_state_machine import (
    TargetStateMachine,
    TargetState,
)
import logging


class TestTargetStateMachineInitialization:
    """Verify H-01: Basic initialization functionality"""

    def test_initialize_with_valid_targets(self):
        """Normal initialization with 3 targets"""
        machine = TargetStateMachine()
        targets = ["dram_latency_cycles", "l2_cache_size_mb", "actual_boost_clock_mhz"]

        machine.initialize(targets)

        assert machine.is_initialized is True
        assert machine.current_target is None  # Not started yet
        assert machine.unmeasured_targets == targets
        assert all(machine._target_retry_count[t] == 0 for t in targets)

    def test_initialize_with_empty_targets_raises_error(self):
        """Empty target list should raise ValueError"""
        machine = TargetStateMachine()

        with pytest.raises(ValueError, match="Targets list cannot be empty"):
            machine.initialize([])

        with pytest.raises(ValueError, match="Targets list cannot be empty"):
            machine.initialize([])

    def test_reinitialize_warning(self, caplog):
        """Re-initialization should emit warning but allow execution"""
        caplog.set_level(logging.WARNING)
        machine = TargetStateMachine()

        machine.initialize(["target_a", "target_b"])
        machine.initialize(["target_c", "target_d"])

        assert "already initialized" in caplog.text.lower()
        assert machine.unmeasured_targets == ["target_c", "target_d"]

    def test_start_first_target(self):
        """start_first_target() should set current_target to first target"""
        machine = TargetStateMachine()
        targets = ["first_target", "second_target", "third_target"]
        machine.initialize(targets)

        result = machine.start_first_target()

        assert result == "first_target"
        assert machine.current_target == "first_target"
        assert len(machine.completed_targets) == 0


class TestTargetStateTransitions:
    """Verify state transition logic"""

    def test_complete_current_target_advances_to_next(self):
        """
        Scenario: targets = [A, B, C], current=A
        Action: complete_current_target()
        Verify:
          - A in completed_targets
          - current_target == B
          - unmeasured_targets == [B, C]
        """
        machine = TargetStateMachine()
        machine.initialize(["A", "B", "C"])
        machine.start_first_target()  # current = A

        next_target = machine.complete_current_target()

        assert "A" in machine.completed_targets
        assert machine.current_target == "B"
        assert next_target == "B"
        assert machine.unmeasured_targets == ["B", "C"]

    def test_complete_all_targets_returns_none(self):
        """Completing last target should return None"""
        machine = TargetStateMachine()
        machine.initialize(["only_target"])
        machine.start_first_target()

        result = machine.complete_current_target()

        assert result is None
        assert machine.is_all_completed is True

    def test_fail_target_at_max_retries_marks_completed(self):
        """
        Scenario: target A reaches max_retries (3 times)
        Action: fail_current_target() called 3 times
        Verify:
          - A marked as completed (to avoid infinite loop)
          - current_target advances to next
        """
        machine = TargetStateMachine()
        machine._max_retries_per_target = 3
        machine.initialize(["failing_target", "next_target"])
        machine.start_first_target()  # current = failing_target

        # Fail 3 times to reach max retries
        for _ in range(3):
            result = machine.fail_current_target()

        # After max retries, should be marked completed and moved on
        assert "failing_target" in machine.completed_targets
        assert machine.current_target == "next_target"

    def test_fail_target_below_max_retries_increments_count(self):
        """Below retry limit should only increment count, not switch"""
        machine = TargetStateMachine()
        machine._max_retries_per_target = 3
        machine.initialize(["target_a", "target_b"])
        machine.start_first_target()

        # Fail once (below max of 3)
        machine.fail_current_target()

        assert machine.current_target == "target_a"  # Should not switch
        assert machine._target_retry_count["target_a"] == 1
        assert "target_a" not in machine.completed_targets

    def test_force_switch_to_specific_target(self):
        """
        BUG#2 FIX: Force switch to specified target
        Scenario: Currently on target B, force switch to target C
        Verify:
          - current_target == C
          - C's retry_count reset to 0
          - Returns True for success
        """
        machine = TargetStateMachine()
        machine.initialize(["A", "B", "C"])
        machine.start_first_target()
        machine.complete_current_target()  # Now on B

        success = machine.force_switch_to_target("C")

        assert success is True
        assert machine.current_target == "C"
        assert machine._target_retry_count["C"] == 0

    def test_force_switch_to_invalid_target_returns_false(self):
        """Switching to non-existent target should return False"""
        machine = TargetStateMachine()
        machine.initialize(["A", "B"])

        success = machine.force_switch_to_target("nonexistent")

        assert success is False
        assert machine.current_target is None  # Unchanged


class TestTargetStateSerialization:
    """Verify P6: State persistence capability"""

    def test_get_snapshot_returns_immutable_data(self):
        """get_snapshot() should return immutable TargetState object"""
        machine = TargetStateMachine()
        machine.initialize(["X", "Y"])
        machine.start_first_target()

        snapshot = machine.get_snapshot()

        assert isinstance(snapshot, TargetState)
        assert snapshot.all_targets == ("X", "Y")
        assert snapshot.current_target == "X"
        # Verify immutability (frozen dataclass)
        with pytest.raises(AttributeError):
            snapshot.current_target = "Y"  # type: ignore

    def test_to_dict_contains_all_fields(self):
        """to_dict() should contain all necessary fields for persistence"""
        machine = TargetStateMachine()
        machine.initialize(["t1", "t2", "t3"])
        machine.start_first_target()
        machine._target_retry_count["t1"] = 2

        data = machine.to_dict()

        assert "all_targets" in data
        assert "completed_targets" in data
        assert "current_target" in data
        assert "target_retry_count" in data
        assert "is_initialized" in data
        assert "progress" in data
        assert data["current_target"] == "t1"
        assert data["target_retry_count"]["t1"] == 2

    def test_from_dict_restores_exact_state(self):
        """from_dict() should fully restore original state"""
        original = TargetStateMachine()
        original.initialize(["a", "b", "c"])
        original.start_first_target()
        original.complete_current_target()
        original._target_retry_count["a"] = 1

        data = original.to_dict()
        restored = TargetStateMachine.from_dict(data)

        assert restored.is_initialized == original.is_initialized
        assert restored.current_target == original.current_target
        assert restored.completed_targets == original.completed_targets
        assert restored._target_retry_count == original._target_retry_count

    def test_roundtrip_preserves_progress(self):
        """to_dict -> from_dict roundtrip should preserve progress info"""
        machine = TargetStateMachine()
        machine.initialize(["x", "y", "z"])
        machine.start_first_target()
        machine.complete_current_target()
        machine.complete_current_target()  # Complete y too

        data = machine.to_dict()
        restored = TargetStateMachine.from_dict(data)

        assert restored.progress == machine.progress
        assert restored.is_all_completed == machine.is_all_completed


class TestStateChangeCallbacks:
    """Verify state change notification mechanism"""

    def test_callback_fired_on_completion(self):
        """Callback should fire when target completes"""
        machine = TargetStateMachine()
        machine.initialize(["A", "B"])

        callbacks_fired = []
        machine.on_state_change(lambda s: callbacks_fired.append(s))

        machine.start_first_target()
        initial_count = len(callbacks_fired)

        machine.complete_current_target()

        assert len(callbacks_fired) > initial_count

    def test_callback_receives_correct_snapshot(self):
        """Callback should receive latest state snapshot"""
        machine = TargetStateMachine()
        machine.initialize(["T1", "T2"])
        machine.start_first_target()

        received_snapshots = []
        machine.on_state_change(lambda s: received_snapshots.append(s))

        machine.complete_current_target()

        assert len(received_snapshots) >= 1
        latest = received_snapshots[-1]
        assert "T1" in latest.completed_targets
        assert latest.current_target == "T2"

    def test_multiple_callbacks_all_fired(self):
        """All registered callbacks should be invoked"""
        machine = TargetStateMachine()
        machine.initialize(["X"])

        call_counts = {"cb1": 0, "cb2": 0}

        def cb1(s):
            call_counts["cb1"] += 1

        def cb2(s):
            call_counts["cb2"] += 1

        machine.on_state_change(cb1)
        machine.on_state_change(cb2)
        machine.start_first_target()

        assert call_counts["cb1"] == 1
        assert call_counts["cb2"] == 1

    def test_callback_exception_does_not_crash_machine(self):
        """Callback exception should not crash the state machine"""
        machine = TargetStateMachine()
        machine.initialize(["safe_target"])

        def failing_callback(s):
            raise Exception("Intentional test error")

        def working_callback(s):
            pass  # This should still execute

        machine.on_state_change(failing_callback)
        machine.on_state_change(working_callback)

        # Should not raise exception
        machine.start_first_target()
        machine.complete_current_target()

        # Machine should still be in valid state
        assert machine.is_all_completed is True


class TestEdgeCasesAndErrorHandling:
    """Edge cases and boundary conditions"""

    def test_single_target_workflow(self):
        """Should work correctly with single target"""
        machine = TargetStateMachine()
        machine.initialize(["only_one"])

        machine.start_first_target()
        assert machine.current_target == "only_one"

        result = machine.complete_current_target()
        assert result is None
        assert machine.is_all_completed is True

    def test_duplicate_targets_in_list(self):
        """Should handle duplicate target names gracefully"""
        machine = TargetStateMachine()
        # Note: Current implementation doesn't deduplicate,
        # this test documents expected behavior
        machine.initialize(["dup", "dup", "unique"])

        unmeasured = machine.unmeasured_targets
        assert "dup" in unmeasured
        assert "unique" in unmeasured

    def test_progress_property_accuracy(self):
        """Progress property should accurately track completion"""
        machine = TargetStateMachine()
        machine.initialize(["p1", "p2", "p3", "p4"])
        machine.start_first_target()

        assert machine.progress == (0, 4)

        machine.complete_current_target()
        assert machine.progress == (1, 4)

        machine.complete_current_target()
        assert machine.progress == (2, 4)

        machine.complete_current_target()
        assert machine.progress == (3, 4)

        machine.complete_current_target()
        assert machine.progress == (4, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
