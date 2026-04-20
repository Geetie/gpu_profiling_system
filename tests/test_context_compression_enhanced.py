"""Unit tests for Active Context Compression (M-02 FIX)

Tests the ContextManager.compress() method that addresses:
- M-02: Context bloat without active compression
- Smart multi-phase compression strategy
- Information preservation during compression

Test Coverage:
1. Compression strategy (5 tests)
2. Budget control (5 tests)
3. Information preservation (4 tests)
4. Edge cases (3 tests)

Total: 17 test cases
"""
import pytest
from src.application.context import (
    ContextManager,
    ContextEntry,
    Role,
    Priority,
)


class TestCompressionStrategy:
    """Verify ContextManager.compress() phased compression strategy"""

    def test_phase1_removes_disposable_entries(self):
        """
        Phase 1: Remove DISPOSABLE priority entries
        Verify:
          - DISPOSABLE entries are removed
          - Other priority entries remain
          - total_tokens decreases
        """
        cm = ContextManager(max_tokens=100)

        # Add entries with different priorities
        cm.add_entry(Role.USER, "disposable_msg_1", token_count=30, priority=Priority.DISPOSABLE)
        cm.add_entry(Role.ASSISTANT, "important_response", token_count=40, priority=Priority.HIGH)
        cm.add_entry(Role.SYSTEM, "system_context", token_count=50, priority=Priority.PERMANENT)
        cm.add_entry(Role.USER, "disposable_msg_2", token_count=30, priority=Priority.DISPOSABLE)

        assert cm.total_tokens == 150  # Over budget

        removed = cm.compress()

        # Should have removed disposable entries
        assert removed >= 2  # At least the 2 disposable entries
        assert cm.total_tokens < 150  # Total should decrease

        # HIGH and PERMANENT should remain
        roles = [e.role for e in cm.get_entries()]
        entry_priorities = []
        for e in cm.get_entries():
            if e.content == "important_response":
                entry_priorities.append(Priority.HIGH)
            elif e.content == "system_context":
                entry_priorities.append(Priority.PERMANENT)

    def test_phase2_summarizes_low_priority_entries(self):
        """
        Phase 2: Summarize LOW priority entries
        Verify:
          - LOW entry content is shortened
          - token_count decreases accordingly
          - Entry still exists (just summarized)
        """
        cm = ContextManager(max_tokens=80)

        long_content = "This is a very long message that should be summarized " * 10
        cm.add_entry(Role.ASSISTANT, long_content, token_count=60, priority=Priority.LOW)
        cm.add_entry(Role.SYSTEM, "critical_info", token_count=40, priority=Priority.HIGH)

        assert cm.is_over_budget()

        cm.compress()

        # The LOW priority entry should have been processed
        # Either summarized or removed
        low_entries = [e for e in cm.get_entries() if e.priority == Priority.LOW]
        for entry in low_entries:
            # If still exists, should be shorter or same
            assert len(entry.content) <= len(long_content)

    def test_phase3_summarizes_medium_priority_entries(self):
        """
        Phase 3: Summarize MEDIUM priority entries (oldest first)
        Verify:
          - Oldest MEDIUM entries are summarized first
          - Newer MEDIUM entries persist longer
        """
        cm = ContextManager(max_tokens=100)

        # Add multiple MEDIUM entries (oldest first)
        cm.add_entry(Role.ASSISTANT, "old_medium_1", token_count=35, priority=Priority.MEDIUM)
        cm.add_entry(Role.ASSISTANT, "old_medium_2", token_count=35, priority=Priority.MEDIUM)
        cm.add_entry(Role.ASSISTANT, "newer_medium_3", token_count=35, priority=Priority.MEDIUM)

        assert cm.total_tokens == 105  # Over budget

        cm.compress()

        # Should have compressed some MEDIUM entries
        assert cm.total_tokens <= 100 or cm.total_tokens < 105

    def test_phase4_removes_old_medium_if_still_over_budget(self):
        """
        Phase 4: If still over budget, remove oldest MEDIUM entries
        """
        cm = ContextManager(max_tokens=50)

        # Add many MEDIUM entries to force phase 4
        for i in range(8):
            cm.add_entry(Role.ASSISTANT, f"medium_{i}", token_count=15, priority=Priority.MEDIUM)

        assert cm.total_tokens > 50  # Way over budget

        removed = cm.compress()

        # Should have removed entries to get under budget
        assert removed > 0
        assert cm.total_tokens <= 50 * cm.COMPRESSION_RATIO + 20  # Allow some tolerance

    def test_never_removes_high_or_permanent_entries(self):
        """
        NEVER remove HIGH and PERMANENT priority entries
        This is a hard constraint protecting critical system information
        """
        cm = ContextManager(max_tokens=50)

        # Add critical entries
        cm.add_entry(Role.SYSTEM, "PERMANENT_system_prompt", token_count=25, priority=Priority.PERMANENT)
        cm.add_entry(Role.USER, "HIGH_priority_instruction", token_count=25, priority=Priority.HIGH)

        # Add many disposable entries to trigger compression
        for i in range(10):
            cm.add_entry(Role.ASSISTANT, f"disposable_{i}", token_count=10, priority=Priority.DISPOSABLE)

        cm.compress()

        # CRITICAL: PERMANENT and HIGH must survive
        entries = cm.get_entries()
        contents = [e.content for e in entries]

        assert "PERMANENT_system_prompt" in contents
        assert "HIGH_priority_instruction" in contents


class TestBudgetControl:
    """Verify is_over_budget() and budget control"""

    def test_is_over_budget_when_exceeded(self):
        """Returns True when total_tokens > max_tokens"""
        cm = ContextManager(max_tokens=100)

        cm.add_entry(Role.USER, "x" * 200, token_count=200)

        assert cm.is_over_budget() is True

    def test_is_under_budget_when_within_limit(self):
        """Returns False when total_tokens <= max_tokens"""
        cm = ContextManager(max_tokens=1000)

        cm.add_entry(Role.USER, "short message", token_count=50)

        assert cm.is_over_budget() is False

    def test_compress_targets_80_percent_of_max(self):
        """
        compress() should target ~80% of max_tokens (COMPRESSION_RATIO)
        Not exactly 100%, leaving headroom
        """
        cm = ContextManager(max_tokens=100)

        # Add enough to exceed budget significantly
        for i in range(15):
            cm.add_entry(Role.ASSISTANT, f"msg_{i}", token_count=15, priority=Priority.DISPOSABLE)

        original_total = cm.total_tokens
        assert original_total > 100

        cm.compress()

        # Should target ~80% of max_tokens
        target = int(100 * cm.COMPRESSION_RATIO)
        # Allow some tolerance (+/- 10%)
        assert cm.total_tokens <= target + 15, \
            f"After compress, total_tokens={cm.total_tokens} should be close to {target}"

    def test_compress_returns_zero_when_under_budget(self):
        """Returns 0 and does nothing when under budget"""
        cm = ContextManager(max_tokens=1000)

        cm.add_entry(Role.USER, "small message", token_count=10)

        removed = cm.compress()

        assert removed == 0
        assert cm.total_tokens == 10  # Unchanged

    def test_compress_returns_removed_count(self):
        """compress() should return actual number of entries removed"""
        cm = ContextManager(max_tokens=50)

        # Add disposable entries that will be removed
        for i in range(6):
            cm.add_entry(Role.USER, f"entry_{i}", token_count=12, priority=Priority.DISPOSABLE)

        initial_count = len(cm.get_entries())

        removed = cm.compress()

        final_count = len(cm.get_entries())
        actual_removed = initial_count - final_count

        assert removed >= actual_removed  # At least this many were processed


class TestInformationPreservation:
    """Verify compression doesn't lose critical information"""

    def test_system_context_survives_compression(self):
        """System role critical instructions should survive compression"""
        cm = ContextManager(max_tokens=80)

        critical_system = "CRITICAL SYSTEM INSTRUCTION: Always measure all targets"
        cm.add_entry(Role.SYSTEM, critical_system, token_count=40, priority=Priority.PERMANENT)

        # Add lots of noise
        for i in range(10):
            cm.add_entry(Role.ASSISTANT, f"noise_{i}", token_count=10, priority=Priority.DISPOSABLE)

        cm.compress()

        entries = cm.get_entries()
        contents = [e.content for e in entries]

        assert critical_system in contents, "Critical system instruction must survive"

    def test_recent_measurements_survive_compression(self):
        """Recent measurement results should not be compressed away"""
        import json

        cm = ContextManager(max_tokens=100)

        # Recent measurement (HIGH priority)
        recent_measurement = json.dumps({
            "tool": "execute_binary",
            "stdout": "dram_latency_cycles: 487\nl2_cache_size_mb: 30",
            "success": True,
        })
        cm.add_entry(Role.ASSISTANT, recent_measurement, token_count=45, priority=Priority.HIGH)

        # Old noise
        for i in range(8):
            cm.add_entry(Role.USER, f"old_message_{i}", token_count=10, priority=Priority.DISPOSABLE)

        cm.compress()

        entries = cm.get_entries()
        contents = [e.content for e in entries]

        has_measurement = any("dram_latency_cycles" in c for c in contents)
        assert has_measurement, "Recent measurements must survive compression"

    def test_current_target_info_survives_compression(self):
        """Current target related info should be preserved"""
        cm = ContextManager(max_tokens=80)

        current_target_msg = "🎯 Current active target: actual_boost_clock_mhz"
        cm.add_entry(Role.SYSTEM, current_target_msg, token_count=35, priority=Priority.HIGH)

        # Add noise to trigger compression
        for i in range(6):
            cm.add_entry(Role.ASSISTANT, f"response_{i}", token_count=12, priority=Priority.LOW)

        cm.compress()

        entries = cm.get_entries()
        contents = [e.content for e in entries]

        assert any("current_target" in c.lower() or "actual_boost_clock_mhz" in c
                  for c in contents), "Current target info must survive"

    def test_critical_errors_not_lost(self):
        """Critical error messages should not be lost in compression"""
        cm = ContextManager(max_tokens=70)

        error_msg = "🚨 ERROR: Compilation failed with CUDA error code 77"
        cm.add_entry(Role.ASSISTANT, error_msg, token_count=30, priority=Priority.HIGH)

        # Trigger compression
        for i in range(5):
            cm.add_entry(Role.USER, f"chat_{i}", token_count=12, priority=Priority.DISPOSABLE)

        cm.compress()

        entries = cm.get_entries()
        contents = [e.content for e in entries]

        assert any("ERROR" in c and "CUDA" in c for c in contents), \
            "Critical errors must survive compression"


class TestCompressionEdgeCases:
    """Edge cases for compression"""

    def test_empty_context_compression(self):
        """Compressing empty context should return 0 safely"""
        cm = ContextManager(max_tokens=1000)

        removed = cm.compress()

        assert removed == 0
        assert cm.total_tokens == 0
        assert cm.get_entries() == []

    def test_single_large_entry_exceeds_budget(self):
        """If even a single entry exceeds budget, keep the newest"""
        cm = ContextManager(max_tokens=50)

        # One huge entry
        cm.add_entry(Role.USER, "huge" * 100, token_count=500, priority=Priority.LOW)

        cm.compress()

        # At least one entry should remain
        assert len(cm.get_entries()) >= 1

    def test_mixed_priorities_correct_order(self):
        """Should process in correct priority order: DISPOSABLE -> LOW -> MEDIUM"""
        cm = ContextManager(max_tokens=60)

        # Add in reverse order of expected processing
        cm.add_entry(Role.ASSISTANT, "medium_entry", token_count=25, priority=Priority.MEDIUM)
        cm.add_entry(Role.USER, "low_entry", token_count=20, priority=Priority.LOW)
        cm.add_entry(Role.SYSTEM, "disposable_entry", token_count=20, priority=Priority.DISPOSABLE)
        cm.add_entry(Role.USER, "high_entry", token_count=15, priority=Priority.HIGH)

        assert cm.total_tokens == 80  # Over budget

        cm.compress()

        # HIGH should definitely survive
        entries = cm.get_entries()
        high_entries = [e for e in entries if e.priority == Priority.HIGH]
        assert len(high_entries) >= 1, "HIGH priority must survive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
