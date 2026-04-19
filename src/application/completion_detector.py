"""Completion signal detection for AgentLoop.

Extracted from agent_loop.py — encapsulates the heuristics for
detecting when the model's text output signals task completion.
"""
from __future__ import annotations

import json
import re


class CompletionDetector:
    """Detects whether model output signals task completion.

    Instead of stopping on ANY non-tool output, we only stop when
    the model explicitly signals it's done. This prevents premature
    termination when the model explains before calling tools.

    Enhanced with stage-specific detection for faster convergence:
    - Plan stage: Detect valid JSON task list + content stabilization
    - CodeGen/MetricAnalysis: Original completion phrase detection
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

    _PLAN_COMPLETION_INDICATORS = (
        "[",
        '{"target"',
        '{"target":',
        '"category":',
        '"method":',
    )

    def __init__(self) -> None:
        self._recent_outputs: list[str] = []
        self._max_history: int = 3

    def is_completion(self, text: str) -> bool:
        lower = text.lower().strip()
        if len(lower) < 10:
            return False

        for phrase in self._COMPLETION_PHRASES:
            if phrase in lower:
                return True

        has_key_value_pairs = self._has_key_value_pairs(lower)
        has_done_statement = self._has_done_statement(lower)

        if has_key_value_pairs and has_done_statement:
            return True

        if self._is_plan_completion(lower):
            return True

        return False

    def _is_plan_completion(self, lower: str) -> bool:
        """Check if output looks like a completed plan (valid JSON + stable)."""
        has_plan_indicator = any(indicator in lower for indicator in self._PLAN_COMPLETION_INDICATORS)
        if not has_plan_indicator:
            return False

        try:
            start_idx = lower.index("[")
            json_str = lower[start_idx:]
            data = json.loads(json_str)
            
            if isinstance(data, list) and len(data) >= 1:
                first_item = data[0]
                if isinstance(first_item, dict):
                    has_target = "target" in first_item
                    has_category = "category" in first_item or "method" in first_item
                    
                    if has_target and (has_category or len(first_item) >= 2):
                        if self._is_content_stable(lower):
                            return True
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        return False

    def _is_content_stable(self, current_output: str) -> bool:
        """Check if recent outputs are similar (content stabilization)."""
        self._recent_outputs.append(current_output)
        if len(self._recent_outputs) > self._max_history:
            self._recent_outputs.pop(0)

        if len(self._recent_outputs) < 2:
            return False

        last_output = self._recent_outputs[-1]
        prev_output = self._recent_outputs[-2]

        similarity = self._calculate_similarity(last_output, prev_output)
        return similarity > 0.85

    @staticmethod
    def _calculate_similarity(s1: str, s2: str) -> float:
        """Calculate simple character-level similarity ratio."""
        if not s1 or not s2:
            return 0.0
        
        shorter, longer = (s1, s2) if len(s1) <= len(s2) else (s2, s1)
        
        if len(longer) == 0:
            return 1.0
            
        matching_chars = sum(1 for a, b in zip(shorter, longer) if a == b)
        return matching_chars / len(longer)

    def _has_key_value_pairs(self, lower: str) -> bool:
        kv_matches = re.findall(r'^[a-z_]+:\s*\d+\.?\d*', lower, re.MULTILINE)
        return len(kv_matches) >= 3

    def _has_done_statement(self, lower: str) -> bool:
        for phrase in self._DONE_PHRASES:
            if phrase in lower:
                return True
        return False
