"""Agent role and pipeline stage enumerations.

Extracted from subagent.py to break circular imports.
These enums are pure value objects with no dependencies,
safe to import from any module.
"""
from __future__ import annotations

from enum import Enum


class AgentRole(Enum):
    PLANNER = "planner"
    CODE_GEN = "code_gen"
    METRIC_ANALYSIS = "metric_analysis"
    VERIFICATION = "verification"


class SubAgentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    REJECTED = "rejected"
    PARTIAL = "partial"  # BUG#8 FIX: Partial success - some but not all targets measured (completion rate < 80%)


class PipelineStage(Enum):
    PLAN = "plan"
    CODE_GEN = "code_gen"
    METRIC_ANALYSIS = "metric_analysis"
    VERIFICATION = "verification"
