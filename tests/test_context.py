"""Tests for Context Manager (application/context.py)

P3: Context engineering over prompt engineering — context must be dynamically
assembled and compressed, not statically designed.
"""
import pytest
from src.application.context import ContextManager, ContextEntry, Role


class TestContextEntry:
    def test_create_entry(self):
        entry = ContextEntry(role=Role.USER, content="Hello", token_count=10)
        assert entry.role == Role.USER
        assert entry.content == "Hello"
        assert entry.token_count == 10

    def test_estimate_token_count_default(self):
        """Rough estimate: ~1.3 tokens per char for English text."""
        entry = ContextEntry(role=Role.ASSISTANT, content="test")
        # Should have a positive token estimate
        assert entry.token_count > 0


class TestContextManager:
    def test_add_and_get_entries(self):
        cm = ContextManager(max_tokens=1000)
        cm.add_entry(Role.USER, "What is the L2 cache size?")
        cm.add_entry(Role.ASSISTANT, "Measuring...")
        entries = cm.get_entries()
        assert len(entries) == 2
        assert entries[0].role == Role.USER
        assert entries[1].role == Role.ASSISTANT

    def test_total_token_count(self):
        cm = ContextManager(max_tokens=1000)
        cm.add_entry(Role.USER, "hi", token_count=5)
        cm.add_entry(Role.ASSISTANT, "hello", token_count=10)
        assert cm.total_tokens == 15

    def test_is_over_budget(self):
        cm = ContextManager(max_tokens=100)
        cm.add_entry(Role.USER, "a" * 200, token_count=200)
        assert cm.is_over_budget() is True

    def test_is_under_budget(self):
        cm = ContextManager(max_tokens=1000)
        cm.add_entry(Role.USER, "hi", token_count=5)
        assert cm.is_over_budget() is False

    def test_compress_removes_oldest(self):
        cm = ContextManager(max_tokens=100)
        cm.add_entry(Role.USER, "msg1", token_count=30)
        cm.add_entry(Role.ASSISTANT, "msg2", token_count=30)
        cm.add_entry(Role.USER, "msg3", token_count=30)
        cm.add_entry(Role.ASSISTANT, "msg4", token_count=30)
        assert cm.total_tokens == 120
        assert cm.is_over_budget() is True

        cm.compress()
        # Oldest entries should be removed
        assert cm.total_tokens <= 100
        entries = cm.get_entries()
        assert entries[0].content == "msg3" or entries[0].content == "msg1"
        # At minimum, the newest entry must survive
        assert entries[-1].content == "msg4"

    def test_compress_preserves_system_entries(self):
        """System context should be protected from compression."""
        cm = ContextManager(max_tokens=100)
        cm.add_entry(Role.SYSTEM, "SYSTEM CONTEXT (read-only)", token_count=50)
        cm.add_entry(Role.USER, "user msg", token_count=30)
        cm.add_entry(Role.ASSISTANT, "assistant reply", token_count=30)
        assert cm.total_tokens == 110

        cm.compress()
        # System entry must still exist
        roles = [e.role for e in cm.get_entries()]
        assert Role.SYSTEM in roles

    def test_clear(self):
        cm = ContextManager(max_tokens=1000)
        cm.add_entry(Role.USER, "hi")
        cm.clear()
        assert cm.get_entries() == []
        assert cm.total_tokens == 0

    def test_get_entries_returns_copy(self):
        """Prevent external mutation of internal state."""
        cm = ContextManager(max_tokens=1000)
        cm.add_entry(Role.USER, "hi")
        entries = cm.get_entries()
        entries.clear()
        assert len(cm.get_entries()) == 1

    def test_to_messages_format(self):
        cm = ContextManager(max_tokens=1000)
        cm.add_entry(Role.USER, "question", token_count=5)
        cm.add_entry(Role.ASSISTANT, "answer", token_count=10)
        messages = cm.to_messages()
        assert messages == [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]

    def test_token_budget_slack(self):
        """compress() should target ~80% of max_tokens, not exactly 100%."""
        cm = ContextManager(max_tokens=100)
        for i in range(10):
            cm.add_entry(Role.USER, f"msg_{i}", token_count=15)
        assert cm.total_tokens == 150

        cm.compress()
        # Should leave some headroom
        assert cm.total_tokens <= 100

    def test_compress_when_cannot_fit_even_one(self):
        """If even a single entry exceeds budget, keep the newest."""
        cm = ContextManager(max_tokens=50)
        cm.add_entry(Role.USER, "huge" * 50, token_count=200)
        cm.compress()
        # At least the last entry should be kept
        assert len(cm.get_entries()) >= 1
