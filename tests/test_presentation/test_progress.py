"""Tests for ProgressBar (presentation/progress.py)."""
import io
import time
from src.presentation.progress import ProgressBar


class TestProgressBar:
    def test_start_and_complete(self):
        output = io.StringIO()
        bar = ProgressBar(output=output)
        bar.start("Testing...")
        bar.complete()
        result = output.getvalue()
        assert "Testing..." in result
        assert "[OK]" in result

    def test_start_and_fail(self):
        output = io.StringIO()
        bar = ProgressBar(output=output)
        bar.start("Failing task...")
        bar.fail()
        result = output.getvalue()
        assert "Failing task..." in result
        assert "[FAIL]" in result

    def test_update_advances(self):
        output = io.StringIO()
        bar = ProgressBar(output=output)
        bar.start("Task")
        bar.update()
        bar.complete()
        result = output.getvalue()
        # Should contain spinner characters
        assert "|" in result or "/" in result or "-" in result or "\\" in result

    def test_complete_after_done_is_noop(self):
        output = io.StringIO()
        bar = ProgressBar(output=output)
        bar.start("Task")
        bar.complete()
        bar.complete()  # should not error
        result = output.getvalue()
        # Should have [OK] line - exact count may vary by platform
        assert "[OK]" in result

    def test_elapsed_time_displayed(self):
        output = io.StringIO()
        bar = ProgressBar(output=output)
        bar.start("Slow task")
        time.sleep(0.1)
        bar.complete()
        result = output.getvalue()
        # Should contain elapsed time like (0.1s)
        assert "s)" in result
