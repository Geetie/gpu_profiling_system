"""Tests for file operations with read-before-write enforcement (infrastructure/file_ops.py)"""
import os
import pytest
from src.infrastructure.file_ops import FileOperations


@pytest.fixture()
def file_ops(tmp_path):
    """Create a FileOperations instance with a temp directory as sandbox root."""
    return FileOperations(sandbox_root=str(tmp_path))


class TestFileOperationsRead:
    def test_read_existing_file(self, file_ops, tmp_path):
        test_file = tmp_path / "hello.txt"
        test_file.write_text("hello world")
        content = file_ops.read(str(test_file))
        assert content == "hello world"

    def test_read_nonexistent_raises(self, file_ops, tmp_path):
        with pytest.raises(FileNotFoundError):
            file_ops.read(str(tmp_path / "does_not_exist.txt"))

    def test_read_records_in_ledger(self, file_ops, tmp_path):
        test_file = tmp_path / "data.txt"
        test_file.write_text("data")
        file_ops.read(str(test_file))
        # Internal ledger should record this read
        assert file_ops._tracker.has_read(str(test_file))


class TestFileOperationsWrite:
    def test_write_after_read(self, file_ops, tmp_path):
        test_file = tmp_path / "edit.txt"
        test_file.write_text("original")
        file_ops.read(str(test_file))
        file_ops.write(str(test_file), "modified")
        assert test_file.read_text() == "modified"

    def test_write_without_read_raises(self, file_ops, tmp_path):
        """M1: writing without prior read must be rejected."""
        test_file = tmp_path / "new.txt"
        test_file.write_text("exists")  # file exists on disk
        with pytest.raises(PermissionError):
            # But the invariant tracker hasn't recorded a read
            file_ops.write(str(test_file), "hacked")

    def test_write_outside_sandbox_raises(self, file_ops):
        """Sandbox: writes outside sandbox_root must be rejected."""
        with pytest.raises(PermissionError):
            file_ops.write("/etc/passwd", "evil")

    def test_read_clears_ledger_entry(self, file_ops, tmp_path):
        """After a successful write, the read ledger entry should be cleared."""
        test_file = tmp_path / "cycle.txt"
        test_file.write_text("v1")
        file_ops.read(str(test_file))
        file_ops.write(str(test_file), "v2")
        # The read ledger entry should be consumed
        assert file_ops._tracker.has_read(str(test_file)) is False

    def test_write_new_file_not_allowed(self, file_ops, tmp_path):
        """Cannot create brand-new files without reading first (M1)."""
        new_file = tmp_path / "brand_new.txt"
        with pytest.raises(PermissionError):
            file_ops.write(str(new_file), "content")


class TestFileOperationsSandbox:
    def test_path_escape_attempt(self, file_ops):
        """Path traversal outside sandbox must be blocked."""
        with pytest.raises(PermissionError):
            file_ops.read("../../../etc/passwd")

    def test_path_escape_in_write(self, file_ops, tmp_path):
        test_file = tmp_path / "safe.txt"
        test_file.write_text("safe")
        file_ops.read(str(test_file))
        # Even after read, writing outside sandbox is forbidden
        with pytest.raises(PermissionError):
            file_ops.write("../../../tmp/escape.txt", "evil")


class TestFileOperationsCreate:
    def test_create_new_file(self, file_ops, tmp_path):
        """create() should create a new file without prior read."""
        new_file = tmp_path / "created.txt"
        bytes_written = file_ops.create(str(new_file), "new content")
        assert bytes_written > 0
        assert new_file.read_text() == "new content"

    def test_create_tracks_in_ledger(self, file_ops, tmp_path):
        """create() should record the file in the created files ledger."""
        new_file = tmp_path / "tracked.txt"
        file_ops.create(str(new_file), "data")
        assert file_ops._tracker.was_created(str(new_file))

    def test_create_sandbox_escape_blocked(self, file_ops):
        """create() should block path escape outside sandbox."""
        with pytest.raises(PermissionError):
            file_ops.create("../../../etc/evil.txt", "malicious")

    def test_create_overwrites_existing(self, file_ops, tmp_path):
        """create() should overwrite existing file content."""
        existing = tmp_path / "existing.txt"
        existing.write_text("old")
        file_ops.create(str(existing), "new")
        assert existing.read_text() == "new"
