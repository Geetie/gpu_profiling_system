"""Tests for state persistence to session_log.jsonl (infrastructure/state_persist.py)"""
import json
import os
import pytest
from src.infrastructure.state_persist import StatePersister


@pytest.fixture()
def persister(tmp_path):
    return StatePersister(log_dir=str(tmp_path))


class TestStatePersister:
    def test_log_entry_persists(self, persister):
        persister.log_entry(action="test", details={"key": "value"})
        log_path = persister.log_path
        assert os.path.exists(log_path)
        with open(log_path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["action"] == "test"
        assert entry["details"] == {"key": "value"}

    def test_log_entry_has_timestamp(self, persister):
        persister.log_entry(action="timestamp_test")
        with open(persister.log_path) as f:
            entry = json.loads(f.readline())
        assert "timestamp" in entry

    def test_multiple_entries_append(self, persister):
        persister.log_entry(action="first")
        persister.log_entry(action="second")
        persister.log_entry(action="third")
        with open(persister.log_path) as f:
            lines = f.readlines()
        assert len(lines) == 3
        entries = [json.loads(l) for l in lines]
        assert [e["action"] for e in entries] == ["first", "second", "third"]

    def test_log_tool_execution(self, persister):
        persister.log_tool_execution(
            tool_name="read_file",
            inputs={"file_path": "test.txt"},
            status="success",
        )
        with open(persister.log_path) as f:
            entry = json.loads(f.readline())
        assert entry["action"] == "tool_execution"
        assert entry["tool_name"] == "read_file"
        assert entry["status"] == "success"

    def test_log_permission_decision(self, persister):
        persister.log_permission_decision(
            permission="file:write",
            mode="default",
            decision="denied",
        )
        with open(persister.log_path) as f:
            entry = json.loads(f.readline())
        assert entry["action"] == "permission_decision"
        assert entry["permission"] == "file:write"
        assert entry["decision"] == "denied"

    def test_log_error(self, persister):
        persister.log_error("FileNotFoundError", "/no/file", "File not found")
        with open(persister.log_path) as f:
            entry = json.loads(f.readline())
        assert entry["action"] == "error"
        assert entry["error_type"] == "FileNotFoundError"

    def test_load_history(self, persister):
        persister.log_entry(action="a")
        persister.log_entry(action="b")
        history = persister.load_history()
        assert len(history) == 2
        assert history[0]["action"] == "a"
        assert history[1]["action"] == "b"

    def test_load_history_empty(self, persister):
        history = persister.load_history()
        assert history == []

    def test_log_invariant_violation(self, persister):
        persister.log_invariant_violation(
            invariant="M1",
            detail="write without read on /tmp/x.txt",
        )
        with open(persister.log_path) as f:
            entry = json.loads(f.readline())
        assert entry["action"] == "invariant_violation"
        assert entry["invariant"] == "M1"

    def test_concurrent_safety_single_process(self, persister):
        """Verify entries are line-delimited (JSONL), not corrupted."""
        for i in range(100):
            persister.log_entry(action="batch", index=i)
        with open(persister.log_path) as f:
            lines = f.readlines()
        assert len(lines) == 100
        for line in lines:
            json.loads(line)  # should not raise
