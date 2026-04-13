"""Tests for InvariantTracker (domain/permission.py)"""
import pytest
from src.domain.permission import InvariantTracker


class TestInvariantTracker:
    def test_record_read(self):
        tracker = InvariantTracker()
        tracker.record_read("/path/to/file.txt")
        assert tracker.has_read("/path/to/file.txt") is True

    def test_write_denied_without_read(self):
        """M1: write before read is forbidden."""
        tracker = InvariantTracker()
        assert tracker.can_write("/path/to/file.txt") is False

    def test_write_allowed_after_read(self):
        tracker = InvariantTracker()
        tracker.record_read("/path/to/file.txt")
        assert tracker.can_write("/path/to/file.txt") is True

    def test_can_write_unrelated_file_denied(self):
        tracker = InvariantTracker()
        tracker.record_read("/path/a.txt")
        assert tracker.can_write("/path/b.txt") is False

    def test_clear_after_write(self):
        """After a successful write, the read ledger should clear that entry."""
        tracker = InvariantTracker()
        tracker.record_read("/path/to/file.txt")
        tracker.clear_read("/path/to/file.txt")
        assert tracker.can_write("/path/to/file.txt") is False

    def test_multiple_reads(self):
        tracker = InvariantTracker()
        tracker.record_read("a.txt")
        tracker.record_read("b.txt")
        assert tracker.can_write("a.txt") is True
        assert tracker.can_write("b.txt") is True
        assert tracker.can_write("c.txt") is False

    def test_get_read_ledger(self):
        tracker = InvariantTracker()
        tracker.record_read("x.txt")
        ledger = tracker.get_read_ledger()
        assert "x.txt" in ledger


# ── Failure Loop Tracker Tests ──────────────────────────────────────


class TestFailureLoopTracker:
    def test_no_failure_by_default(self):
        tracker = InvariantTracker()
        assert tracker.get_failure_count("some_task") == 0
        assert tracker.should_terminate("some_task") is False

    def test_increment_failure(self):
        tracker = InvariantTracker()
        tracker.record_failure("task_a")
        assert tracker.get_failure_count("task_a") == 1
        assert tracker.should_terminate("task_a") is False

    def test_terminate_after_three_failures(self):
        """M4: same failure pattern repeated 3 times forces termination."""
        tracker = InvariantTracker()
        tracker.record_failure("task_a")
        tracker.record_failure("task_a")
        tracker.record_failure("task_a")
        assert tracker.should_terminate("task_a") is True

    def test_fourth_failure_also_terminates(self):
        tracker = InvariantTracker()
        for _ in range(4):
            tracker.record_failure("task_a")
        assert tracker.should_terminate("task_a") is True

    def test_different_failures_independent(self):
        tracker = InvariantTracker()
        tracker.record_failure("task_a")
        tracker.record_failure("task_a")
        tracker.record_failure("task_b")
        assert tracker.should_terminate("task_a") is False
        assert tracker.should_terminate("task_b") is False

    def test_reset_failure(self):
        tracker = InvariantTracker()
        tracker.record_failure("task_a")
        tracker.record_failure("task_a")
        tracker.reset_failure("task_a")
        assert tracker.get_failure_count("task_a") == 0
        assert tracker.should_terminate("task_a") is False
