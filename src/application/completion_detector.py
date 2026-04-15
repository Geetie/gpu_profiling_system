"""Completion signal detection for AgentLoop.

Extracted from agent_loop.py — encapsulates the heuristics for
detecting when the model's text output signals task completion.
"""
from __future__ import annotations

import re


class CompletionDetector:
    """Detects whether model output signals task completion.

    Instead of stopping on ANY non-tool output, we only stop when
    the model explicitly signals it's done. This prevents premature
    termination when the model explains before calling tools.
    """

    _COMPLETION_PHRASES = (
        "verdict: accept",
        "verdict: reject",
        "reject:",
        "rejected:",
        "all targets measured",
        "all targets have been measured",
        "measurement complete",
        "profiling complete",
        "task complete",
        "final answer:",
        "final results:",
        "summary of findings",
        "verification report",
    )

    _DONE_PHRASES = (
        "i have completed",
        "i am done",
        "i'm done",
        "here are the final",
        "here is the final",
        "these are the measured",
        "the measured values",
    )

    def is_completion(self, text: str) -> bool:
        lower = text.lower().strip()
        if len(lower) < 10:
            return False

        for phrase in self._COMPLETION_PHRASES:
            if phrase in lower:
                return True

        has_key_value_pairs = self._has_key_value_pairs(lower)
        has_done_statement = self._has_done_statement(lower)
        return has_key_value_pairs and has_done_statement

    def _has_key_value_pairs(self, lower: str) -> bool:
        kv_matches = re.findall(r'^[a-z_]+:\s*\d+\.?\d*', lower, re.MULTILINE)
        return len(kv_matches) >= 3

    def _has_done_statement(self, lower: str) -> bool:
        for phrase in self._DONE_PHRASES:
            if phrase in lower:
                return True
        return False
