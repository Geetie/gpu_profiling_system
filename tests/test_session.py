"""Tests for Session State & Resume (application/session.py)

P6: All session state must be persisted to disk. --resume recovers from
the last checkpoint.
"""
import json
import os
import pytest
from src.application.session import SessionState, SessionManager


class TestSessionState:
    def test_initial_state(self):
        state = SessionState(session_id="test-001", goal="Find L2 cache size")
        assert state.session_id == "test-001"
        assert state.goal == "Find L2 cache size"
        assert state.step_count == 0
        assert state.is_complete is False
        assert state.error is None

    def test_to_dict_roundtrip(self):
        state = SessionState(session_id="s1", goal="probe DRAM latency")
        state.step_count = 5
        state.is_complete = True
        d = state.to_dict()
        restored = SessionState.from_dict(d)
        assert restored.session_id == "s1"
        assert restored.goal == "probe DRAM latency"
        assert restored.step_count == 5
        assert restored.is_complete is True

    def test_increment_step(self):
        state = SessionState(session_id="s1", goal="g")
        state.increment_step()
        assert state.step_count == 1

    def test_mark_complete(self):
        state = SessionState(session_id="s1", goal="g")
        state.mark_complete()
        assert state.is_complete is True


class TestSessionManager:
    def test_new_session(self, tmp_path):
        mgr = SessionManager(state_dir=str(tmp_path))
        session = mgr.create_session("test-s1", "Find DRAM latency")
        assert session.session_id == "test-s1"
        assert session.goal == "Find DRAM latency"

    def test_save_and_load_session(self, tmp_path):
        mgr = SessionManager(state_dir=str(tmp_path))
        session = mgr.create_session("s1", "goal")
        session.increment_step()
        mgr.save_session(session)
        loaded = mgr.load_session("s1")
        assert loaded is not None
        assert loaded.session_id == "s1"
        assert loaded.step_count == 1

    def test_load_nonexistent_returns_none(self, tmp_path):
        mgr = SessionManager(state_dir=str(tmp_path))
        assert mgr.load_session("ghost") is None

    def test_resume_existing_session(self, tmp_path):
        mgr = SessionManager(state_dir=str(tmp_path))
        session = mgr.create_session("resume-me", "continue work")
        session.increment_step()
        session.increment_step()
        mgr.save_session(session)

        resumed = mgr.resume("resume-me", new_goal=None)
        assert resumed is not None
        assert resumed.session_id == "resume-me"
        assert resumed.step_count == 2
        assert resumed.goal == "continue work"

    def test_resume_with_new_goal(self, tmp_path):
        mgr = SessionManager(state_dir=str(tmp_path))
        session = mgr.create_session("r1", "old goal")
        mgr.save_session(session)

        resumed = mgr.resume("r1", new_goal="new goal")
        assert resumed.goal == "new goal"

    def test_resume_nonexistent_creates_new(self, tmp_path):
        mgr = SessionManager(state_dir=str(tmp_path))
        session = mgr.resume("fresh", new_goal="start fresh")
        assert session.session_id == "fresh"
        assert session.step_count == 0

    def test_save_overwrites(self, tmp_path):
        mgr = SessionManager(state_dir=str(tmp_path))
        s = mgr.create_session("s1", "goal")
        mgr.save_session(s)
        s.increment_step()
        mgr.save_session(s)
        loaded = mgr.load_session("s1")
        assert loaded.step_count == 1

    def test_list_sessions(self, tmp_path):
        mgr = SessionManager(state_dir=str(tmp_path))
        sa = mgr.create_session("a", "ga")
        sb = mgr.create_session("b", "gb")
        mgr.save_session(sa)
        mgr.save_session(sb)
        names = mgr.list_sessions()
        assert set(names) == {"a", "b"}

    def test_delete_session(self, tmp_path):
        mgr = SessionManager(state_dir=str(tmp_path))
        s = mgr.create_session("del-me", "g")
        mgr.save_session(s)
        mgr.delete_session("del-me")
        assert mgr.load_session("del-me") is None

    def test_state_file_path(self, tmp_path):
        mgr = SessionManager(state_dir=str(tmp_path))
        session = mgr.create_session("s1", "g")
        mgr.save_session(session)
        expected = os.path.join(str(tmp_path), "session_s1.json")
        assert os.path.exists(expected)
