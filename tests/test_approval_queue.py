"""Tests for approval queue (application/approval_queue.py)."""
import os
import pytest
import threading
from src.application.approval_queue import ApprovalQueue, ApprovalRequest, ApprovalStatus
from src.domain.permission import PermissionMode
from src.infrastructure.state_persist import StatePersister


class TestApprovalQueue:
    @pytest.fixture
    def queue(self, tmp_path):
        persister = StatePersister(log_dir=str(tmp_path), filename="approval_log.jsonl")
        return ApprovalQueue(state_dir=str(tmp_path), persister=persister)

    def test_default_mode_submits_pending(self, queue):
        request = queue.submit(
            tool_name="compile_cuda",
            arguments={"source": "test"},
            permissions=["file:write", "process:exec"],
            mode=PermissionMode.DEFAULT,
        )
        assert request.status == ApprovalStatus.PENDING

    def test_conservative_mode_auto_rejects(self, queue):
        request = queue.submit(
            tool_name="compile_cuda",
            arguments={"source": "test"},
            permissions=["file:write"],
            mode=PermissionMode.CONSERVATIVE,
        )
        assert request.status == ApprovalStatus.AUTO_REJECTED
        assert "CONSERVATIVE" in (request.reason or "")

    def test_approve_request(self, queue):
        request = queue.submit(
            tool_name="write_file",
            arguments={},
            permissions=["file:write"],
            mode=PermissionMode.DEFAULT,
        )
        queue.respond(request.id, approved=True, reason="looks good")
        assert request.status == ApprovalStatus.APPROVED
        assert request.reason == "looks good"
        assert request.responded_at is not None

    def test_reject_request(self, queue):
        request = queue.submit(
            tool_name="write_file",
            arguments={},
            permissions=["file:write"],
            mode=PermissionMode.DEFAULT,
        )
        queue.respond(request.id, approved=False, reason="not allowed")
        assert request.status == ApprovalStatus.REJECTED

    def test_double_respond_raises(self, queue):
        request = queue.submit(
            tool_name="write_file",
            arguments={},
            permissions=["file:write"],
            mode=PermissionMode.DEFAULT,
        )
        queue.respond(request.id, approved=True)
        with pytest.raises(ValueError, match="already responded"):
            queue.respond(request.id, approved=False)

    def test_nonexistent_id_raises(self, queue):
        with pytest.raises(KeyError, match="not found"):
            queue.respond("nonexistent_id", approved=True)

    def test_get_pending_filters_correctly(self, queue):
        r1 = queue.submit("tool_a", {}, ["file:write"], PermissionMode.DEFAULT)
        r2 = queue.submit("tool_b", {}, ["file:write"], PermissionMode.DEFAULT)
        queue.respond(r1.id, approved=True)
        pending = queue.get_pending()
        assert len(pending) == 1
        assert pending[0].id == r2.id

    def test_persistence_writes_log(self, queue, tmp_path):
        request = queue.submit(
            tool_name="compile_cuda",
            arguments={"source": "test"},
            permissions=["file:write"],
            mode=PermissionMode.DEFAULT,
        )
        queue.respond(request.id, approved=True)
        log_path = os.path.join(str(tmp_path), "approval_log.jsonl")
        assert os.path.exists(log_path)
        with open(log_path) as f:
            lines = f.readlines()
        # Should have at least: request + decision
        assert len(lines) >= 2

    def test_wait_for_decision_approved(self, queue):
        request = queue.submit(
            tool_name="tool_x",
            arguments={},
            permissions=["file:write"],
            mode=PermissionMode.DEFAULT,
        )

        def approve_after_delay():
            import time
            time.sleep(0.1)
            queue.respond(request.id, approved=True)

        t = threading.Thread(target=approve_after_delay)
        t.start()
        status = queue.wait_for_decision(request, timeout=5.0)
        t.join()
        assert status == ApprovalStatus.APPROVED

    def test_wait_for_decision_timeout(self, queue):
        request = queue.submit(
            tool_name="tool_y",
            arguments={},
            permissions=["file:write"],
            mode=PermissionMode.DEFAULT,
        )
        status = queue.wait_for_decision(request, timeout=0.2)
        assert status == ApprovalStatus.REJECTED
        assert "Timed out" in (request.reason or "")

    def test_wait_returns_immediately_for_auto_rejected(self, queue):
        request = queue.submit(
            tool_name="tool_z",
            arguments={},
            permissions=["file:write"],
            mode=PermissionMode.CONSERVATIVE,
        )
        status = queue.wait_for_decision(request, timeout=5.0)
        assert status == ApprovalStatus.AUTO_REJECTED

    def test_get_request_by_id(self, queue):
        request = queue.submit("tool_a", {}, ["file:write"], PermissionMode.DEFAULT)
        found = queue.get_request(request.id)
        assert found is request
        assert queue.get_request("nope") is None

    def test_request_has_required_fields(self, queue):
        request = queue.submit(
            tool_name="compile_cuda",
            arguments={"source": "test"},
            permissions=["file:write", "process:exec"],
            mode=PermissionMode.DEFAULT,
        )
        assert request.tool_name == "compile_cuda"
        assert request.arguments == {"source": "test"}
        assert request.permissions == ["file:write", "process:exec"]
        assert request.created_at is not None
