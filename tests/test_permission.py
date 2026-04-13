"""Tests for Permission checking and Mechanical Invariants (domain/permission.py)"""
import pytest
from src.domain.permission import PermissionMode, PermissionChecker, InvariantTracker


# ── PermissionMode Tests ────────────────────────────────────────────


class TestPermissionMode:
    def test_mode_values(self):
        assert PermissionMode.CONSERVATIVE.value == "conservative"
        assert PermissionMode.DEFAULT.value == "default"
        assert PermissionMode.RELAXED.value == "relaxed"
        assert PermissionMode.HIGH_AUTONOMY.value == "high_autonomy"

    def test_from_string(self):
        from src.domain.permission import PermissionMode
        assert PermissionMode.from_string("conservative") == PermissionMode.CONSERVATIVE
        assert PermissionMode.from_string("default") == PermissionMode.DEFAULT

    def test_from_string_invalid_raises(self):
        with pytest.raises(ValueError):
            PermissionMode.from_string("invalid_mode")


# ── PermissionChecker Tests ─────────────────────────────────────────


class TestPermissionChecker:
    def test_conservative_allows_read_only(self):
        checker = PermissionChecker(PermissionMode.CONSERVATIVE)
        assert checker.is_allowed("file:read") is True
        assert checker.is_allowed("file:write") is False  # P2: fail-closed
        assert checker.is_allowed("process:exec") is False

    def test_default_requires_approval_for_write(self):
        checker = PermissionChecker(PermissionMode.DEFAULT)
        assert checker.is_allowed("file:read") is True
        assert checker.is_allowed("file:write") is True  # allowed but needs approval
        assert checker.requires_approval("file:read") is False
        assert checker.requires_approval("file:write") is True
        assert checker.requires_approval("process:exec") is True

    def test_relaxed_auto_approves_regular_edits(self):
        checker = PermissionChecker(PermissionMode.RELAXED)
        assert checker.is_allowed("file:read") is True
        assert checker.is_allowed("file:write") is True
        assert checker.is_allowed("process:exec") is True
        # Shell still requires approval even in relaxed mode
        assert checker.requires_approval("process:exec") is True

    def test_high_autonomy(self):
        checker = PermissionChecker(PermissionMode.HIGH_AUTONOMY)
        assert checker.is_allowed("file:read") is True
        assert checker.is_allowed("file:write") is True
        assert checker.is_allowed("process:exec") is True
        # Still has hard boundaries — process:exec needs approval
        assert checker.requires_approval("process:exec") is True

    def test_unknown_permission_denied(self):
        """P2: any unknown permission is denied."""
        checker = PermissionChecker(PermissionMode.HIGH_AUTONOMY)
        assert checker.is_allowed("unknown:perm") is False

    def test_all_permissions_in_conservative(self):
        checker = PermissionChecker(PermissionMode.CONSERVATIVE)
        allowed = checker.list_allowed_permissions()
        assert "file:read" in allowed
        assert "file:write" not in allowed
        assert "process:exec" not in allowed
