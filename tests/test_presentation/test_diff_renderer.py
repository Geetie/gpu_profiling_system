"""Tests for DiffRenderer (presentation/diff_renderer.py)."""
import io
from src.presentation.diff_renderer import DiffRenderer


class TestDiffRenderer:
    def test_render_adds_lines(self):
        output = io.StringIO()
        renderer = DiffRenderer(output=output)
        renderer.render("test.txt", "line1\nline2\n", "line1\nline2\nline3\n")
        result = output.getvalue()
        assert "+line3" in result
        assert "--- diff:" in result
        assert "test.txt" in result

    def test_render_deletes_lines(self):
        output = io.StringIO()
        renderer = DiffRenderer(output=output)
        renderer.render("test.txt", "line1\nline2\nline3\n", "line1\nline2\n")
        result = output.getvalue()
        assert "-line3" in result

    def test_render_modifies_lines(self):
        output = io.StringIO()
        renderer = DiffRenderer(output=output)
        renderer.render("test.txt", "old_value\n", "new_value\n")
        result = output.getvalue()
        assert "-old_value" in result
        assert "+new_value" in result

    def test_no_changes_produces_minimal_output(self):
        output = io.StringIO()
        renderer = DiffRenderer(output=output)
        renderer.render("test.txt", "same\n", "same\n")
        result = output.getvalue()
        # unified_diff produces no hunks for identical content
        assert "--- diff:" in result

    def test_generate_diff_string(self):
        diff = DiffRenderer.generate_diff("test.txt", "a\nb\n", "a\nc\n")
        assert "-b" in diff
        assert "+c" in diff

    def test_generate_diff_string_plain_text(self):
        """generate_diff should not contain ANSI codes."""
        diff = DiffRenderer.generate_diff("test.txt", "old\n", "new\n")
        # Check for absence of ANSI escape sequences
        import re
        assert not re.search(r'\x1b\[\d+m', diff)
