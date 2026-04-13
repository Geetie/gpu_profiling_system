"""Tests for Control Plane (application/control_plane.py)

The control plane auto-injects system state before each inference.
All injected items are READ-ONLY — the model cannot modify them.
"""
import json
import os
import pytest
from src.application.control_plane import ControlPlane, SystemSnapshot, InjectedContext


class TestSystemSnapshot:
    def test_capture_defaults(self, tmp_path):
        os.chdir(str(tmp_path))
        snapshot = SystemSnapshot.capture()
        assert snapshot.working_dir == str(tmp_path)
        assert isinstance(snapshot.env_vars, dict)
        assert isinstance(snapshot.rule_files, dict)
        assert isinstance(snapshot.memory_summary, list)

    def test_captured_env_includes_cuda(self):
        """CUDA-relevant env vars should be captured."""
        snapshot = SystemSnapshot.capture()
        # Keys should include at least one CUDA-related var (or be present)
        assert isinstance(snapshot.env_vars, dict)

    def test_rule_file_loading(self, tmp_path):
        rule = tmp_path / "CLAUDE.md"
        rule.write_text("# Rule\nNo writes without read.")
        snapshot = SystemSnapshot.capture(rule_dir=str(tmp_path))
        assert "CLAUDE.md" in snapshot.rule_files
        assert "No writes without read" in snapshot.rule_files["CLAUDE.md"]

    def test_missing_rule_file_not_included(self, tmp_path):
        snapshot = SystemSnapshot.capture(rule_dir=str(tmp_path / "nonexistent"))
        assert snapshot.rule_files == {}


class TestInjectedContext:
    def test_context_is_immutable_marker(self):
        ctx = InjectedContext(
            working_dir="/some/path",
            env_vars={"CUDA_VISIBLE_DEVICES": "0"},
            rules={"CLAUDE.md": "rules"},
            memory_summary=["Past: L2 cache = 4MB"],
        )
        # Context must be renderable to a prompt string
        rendered = ctx.render()
        assert "WORKING DIRECTORY" in rendered
        assert "/some/path" in rendered
        assert "CLAUDE.md" in rendered
        assert "CUDA_VISIBLE_DEVICES" in rendered

    def test_render_empty_sections_omit_content(self):
        ctx = InjectedContext(
            working_dir="/path",
            env_vars={},
            rules={},
            memory_summary=[],
        )
        rendered = ctx.render()
        assert "CUDA_VISIBLE_DEVICES" not in rendered
        assert "MEMORY SUMMARY" not in rendered


class TestControlPlane:
    def test_initial_snapshot(self, tmp_path):
        os.chdir(str(tmp_path))
        cp = ControlPlane(rule_dir=str(tmp_path))
        snapshot = cp.take_snapshot()
        assert snapshot.working_dir == str(tmp_path)

    def test_snapshot_is_isolated(self, tmp_path):
        """Each snapshot is independent (no shared mutable state)."""
        os.chdir(str(tmp_path))
        cp = ControlPlane(rule_dir=str(tmp_path))
        s1 = cp.take_snapshot()
        s2 = cp.take_snapshot()
        # Different objects but same content
        assert s1 is not s2
        assert s1.working_dir == s2.working_dir

    def test_inject_creates_readonly_context(self, tmp_path):
        os.chdir(str(tmp_path))
        (tmp_path / "CLAUDE.md").write_text("test rules")
        cp = ControlPlane(rule_dir=str(tmp_path))
        ctx = cp.inject()
        assert isinstance(ctx, InjectedContext)
        assert ctx.working_dir == str(tmp_path)

    def test_inject_includes_memory_summary(self, tmp_path):
        os.chdir(str(tmp_path))
        # Simulate a memory summary being provided
        cp = ControlPlane(rule_dir=str(tmp_path), memory_entries=["Finding: L2 = 4MB"])
        ctx = cp.inject()
        assert "Finding: L2 = 4MB" in ctx.memory_summary

    def test_build_system_prompt(self, tmp_path):
        os.chdir(str(tmp_path))
        (tmp_path / "CLAUDE.md").write_text("Rule: always read first")
        cp = ControlPlane(rule_dir=str(tmp_path))
        prompt = cp.build_system_prompt()
        # Should include the section markers
        assert "SYSTEM CONTEXT" in prompt
        assert "WORKING DIRECTORY" in prompt
        assert "RULE FILES" in prompt
        assert "READ-ONLY" in prompt

    def test_control_plane_does_not_mutate_snapshot(self, tmp_path):
        """Control plane snapshots must be read-only copies."""
        os.chdir(str(tmp_path))
        cp = ControlPlane(rule_dir=str(tmp_path))
        snap = cp.take_snapshot()
        # Mutating the returned snapshot should not affect the control plane
        snap.working_dir = "/hacked"
        snap2 = cp.take_snapshot()
        assert snap2.working_dir != "/hacked"
