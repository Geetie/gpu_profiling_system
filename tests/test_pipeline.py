"""Tests for Pipeline orchestrator (domain/pipeline.py)."""
import json
import os
import pytest

from src.application.context import ContextManager, Role
from src.domain.permission import PermissionMode
from src.domain.pipeline import Pipeline, PipelineStep
from src.domain.subagent import (
    AgentRole,
    BaseSubAgent,
    CollaborationMessage,
    P7ViolationError,
    PipelineStage,
    SubAgentResult,
    SubAgentStatus,
)
from src.domain.tool_contract import ToolRegistry


# ── Mock Agent ───────────────────────────────────────────────────────


class MockAgent(BaseSubAgent):
    """Mock agent that returns a pre-configured result."""

    def __init__(
        self,
        role: AgentRole,
        result: SubAgentResult | None = None,
        state_dir: str = ".state",
    ) -> None:
        super().__init__(
            role=role,
            context_manager=ContextManager(max_tokens=4000),
            tool_registry=ToolRegistry(),
            state_dir=state_dir,
            permission_mode=PermissionMode.DEFAULT,
        )
        self._result = result or SubAgentResult(
            agent_role=role,
            status=SubAgentStatus.SUCCESS,
            data={"mock": True},
        )

    def run(self, message: CollaborationMessage) -> SubAgentResult:
        self.context_manager.add_entry(Role.USER, f"task for {self.role.value}")
        return self._result


# ── Pipeline Tests ───────────────────────────────────────────────────


class TestPipeline:
    def test_full_happy_path(self, tmp_path):
        """All 4 stages succeed in sequence."""
        stages = [
            PipelineStep(PipelineStage.PLAN, MockAgent(AgentRole.PLANNER, state_dir=str(tmp_path))),
            PipelineStep(PipelineStage.CODE_GEN, MockAgent(AgentRole.CODE_GEN, state_dir=str(tmp_path))),
            PipelineStep(PipelineStage.METRIC_ANALYSIS, MockAgent(AgentRole.METRIC_ANALYSIS, state_dir=str(tmp_path))),
            PipelineStep(PipelineStage.VERIFICATION, MockAgent(AgentRole.VERIFICATION, state_dir=str(tmp_path))),
        ]
        pipeline = Pipeline(stages=stages, state_dir=str(tmp_path))
        result = pipeline.run({"targets": ["dram_latency_cycles"]})
        assert result.is_success()
        assert result.agent_role == AgentRole.VERIFICATION

    def test_stage_failure_aborts(self, tmp_path):
        """If CodeGen fails, pipeline should abort."""
        stages = [
            PipelineStep(PipelineStage.PLAN, MockAgent(AgentRole.PLANNER, state_dir=str(tmp_path))),
            PipelineStep(PipelineStage.CODE_GEN, MockAgent(
                AgentRole.CODE_GEN,
                result=SubAgentResult(
                    agent_role=AgentRole.CODE_GEN,
                    status=SubAgentStatus.FAILED,
                    error="Compilation error",
                ),
                state_dir=str(tmp_path),
            )),
            PipelineStep(PipelineStage.METRIC_ANALYSIS, MockAgent(AgentRole.METRIC_ANALYSIS, state_dir=str(tmp_path))),
        ]
        pipeline = Pipeline(stages=stages, state_dir=str(tmp_path))
        result = pipeline.run({"targets": ["dram_latency_cycles"]})
        assert result.is_failed()
        assert result.agent_role == AgentRole.CODE_GEN
        assert "Compilation error" in (result.error or "")

    def test_input_chaining(self, tmp_path):
        """Each stage receives the previous stage's result."""
        received_payloads = []

        class CapturingAgent(MockAgent):
            def run(self, message: CollaborationMessage) -> SubAgentResult:
                received_payloads.append(message.payload)
                return super().run(message)

        stages = [
            PipelineStep(PipelineStage.PLAN, CapturingAgent(
                AgentRole.PLANNER,
                result=SubAgentResult(AgentRole.PLANNER, SubAgentStatus.SUCCESS, data={"plan": "A"}),
                state_dir=str(tmp_path),
            )),
            PipelineStep(PipelineStage.CODE_GEN, CapturingAgent(
                AgentRole.CODE_GEN,
                result=SubAgentResult(AgentRole.CODE_GEN, SubAgentStatus.SUCCESS, data={"code": "B"}),
                state_dir=str(tmp_path),
            )),
        ]
        pipeline = Pipeline(stages=stages, state_dir=str(tmp_path))
        pipeline.run({"targets": ["test"]})

        # Second stage should have received first stage's result
        assert len(received_payloads) >= 2
        assert received_payloads[1].get("prev_result", {}).get("data") == {"plan": "A"}

    def test_p7_violation_raised(self, tmp_path):
        """Pipeline should raise P7ViolationError if verification has non-empty context."""
        # Create a verification agent with pre-populated context
        cm = ContextManager(max_tokens=4000)
        cm.add_entry(Role.USER, "leaked context")  # non-empty!
        verify_agent = MockAgent(
            AgentRole.VERIFICATION,
            state_dir=str(tmp_path),
        )
        # Manually set the context to non-empty
        verify_agent.context_manager = cm

        stages = [
            PipelineStep(PipelineStage.PLAN, MockAgent(AgentRole.PLANNER, state_dir=str(tmp_path))),
            PipelineStep(PipelineStage.VERIFICATION, verify_agent),
        ]
        pipeline = Pipeline(stages=stages, state_dir=str(tmp_path))

        with pytest.raises(P7ViolationError, match="P7 violation"):
            pipeline.run({"targets": ["test"]})

    def test_p7_passes_with_fresh_context(self, tmp_path):
        """Verification with fresh context should pass P7 check."""
        stages = [
            PipelineStep(PipelineStage.PLAN, MockAgent(AgentRole.PLANNER, state_dir=str(tmp_path))),
            PipelineStep(PipelineStage.VERIFICATION, MockAgent(
                AgentRole.VERIFICATION,
                state_dir=str(tmp_path),
            )),
        ]
        pipeline = Pipeline(stages=stages, state_dir=str(tmp_path))
        # Should not raise
        result = pipeline.run({"targets": ["test"]})
        assert result.is_success()

    def test_retry_on_failure(self, tmp_path):
        """Stage should retry up to retry_on_failure times."""
        call_count = {"n": 0}

        class FlakyAgent(MockAgent):
            def run(self, message: CollaborationMessage) -> SubAgentResult:
                call_count["n"] += 1
                if call_count["n"] < 3:
                    return SubAgentResult(
                        agent_role=self.role,
                        status=SubAgentStatus.FAILED,
                        error=f"Transient error #{call_count['n']}",
                    )
                return super().run(message)

        stages = [
            PipelineStep(
                PipelineStage.CODE_GEN,
                FlakyAgent(AgentRole.CODE_GEN, state_dir=str(tmp_path)),
                retry_on_failure=2,
            ),
        ]
        pipeline = Pipeline(stages=stages, state_dir=str(tmp_path))
        result = pipeline.run({"targets": ["test"]})
        # Should succeed on 3rd attempt (1 + 2 retries)
        assert result.is_success()
        assert call_count["n"] == 3

    def test_retry_exhausted(self, tmp_path):
        """If all retries fail, pipeline should return FAILED."""
        class AlwaysFailAgent(MockAgent):
            def run(self, message: CollaborationMessage) -> SubAgentResult:
                return SubAgentResult(
                    agent_role=self.role,
                    status=SubAgentStatus.FAILED,
                    error="always fails",
                )

        stages = [
            PipelineStep(
                PipelineStage.CODE_GEN,
                AlwaysFailAgent(AgentRole.CODE_GEN, state_dir=str(tmp_path)),
                retry_on_failure=1,
            ),
        ]
        pipeline = Pipeline(stages=stages, state_dir=str(tmp_path))
        result = pipeline.run({"targets": ["test"]})
        assert result.is_failed()

    def test_persistence(self, tmp_path):
        """Each stage should be logged to pipeline_log.jsonl."""
        stages = [
            PipelineStep(PipelineStage.PLAN, MockAgent(AgentRole.PLANNER, state_dir=str(tmp_path))),
            PipelineStep(PipelineStage.CODE_GEN, MockAgent(AgentRole.CODE_GEN, state_dir=str(tmp_path))),
        ]
        pipeline = Pipeline(stages=stages, state_dir=str(tmp_path))
        pipeline.run({"targets": ["test"]})

        log_path = os.path.join(str(tmp_path), "pipeline_log.jsonl")
        assert os.path.exists(log_path)
        with open(log_path) as f:
            lines = f.readlines()
        # At least: pipeline_start + 2 stage logs + pipeline_complete
        assert len(lines) >= 3

    def test_build_default(self, tmp_path):
        """build_default should create a 4-stage pipeline."""
        pipeline = Pipeline.build_default(
            planner=MockAgent(AgentRole.PLANNER, state_dir=str(tmp_path)),
            code_gen=MockAgent(AgentRole.CODE_GEN, state_dir=str(tmp_path)),
            metric_analysis=MockAgent(AgentRole.METRIC_ANALYSIS, state_dir=str(tmp_path)),
            verification=MockAgent(AgentRole.VERIFICATION, state_dir=str(tmp_path)),
            state_dir=str(tmp_path),
        )
        assert len(pipeline._stages) == 4
        assert pipeline._stages[0].stage == PipelineStage.PLAN
        assert pipeline._stages[3].stage == PipelineStage.VERIFICATION

    def test_empty_target_spec(self, tmp_path):
        """Empty targets should cause planner to fail."""
        stages = [
            PipelineStep(PipelineStage.PLAN, MockAgent(
                AgentRole.PLANNER,
                result=SubAgentResult(
                    agent_role=AgentRole.PLANNER,
                    status=SubAgentStatus.FAILED,
                    error="No targets specified",
                ),
                state_dir=str(tmp_path),
            )),
        ]
        pipeline = Pipeline(stages=stages, state_dir=str(tmp_path))
        result = pipeline.run({"targets": []})
        assert result.is_failed()

    def test_sender_uses_prev_result_role(self, tmp_path):
        """BUG-NEW-1: sender should be prev_result.agent_role, not always PLANNER."""
        senders_seen = []

        class SenderCapturingAgent(MockAgent):
            def run(self, message: CollaborationMessage) -> SubAgentResult:
                senders_seen.append(message.sender)
                return super().run(message)

        stages = [
            PipelineStep(PipelineStage.PLAN, SenderCapturingAgent(
                AgentRole.PLANNER,
                result=SubAgentResult(AgentRole.PLANNER, SubAgentStatus.SUCCESS, data={"plan": "A"}),
                state_dir=str(tmp_path),
            )),
            PipelineStep(PipelineStage.CODE_GEN, SenderCapturingAgent(
                AgentRole.CODE_GEN,
                result=SubAgentResult(AgentRole.CODE_GEN, SubAgentStatus.SUCCESS, data={"code": "B"}),
                state_dir=str(tmp_path),
            )),
        ]
        pipeline = Pipeline(stages=stages, state_dir=str(tmp_path))
        pipeline.run({"targets": ["test"]})

        # First message sender should be PLANNER (no prev_result)
        assert senders_seen[0] == AgentRole.PLANNER
        # Second message sender should be PLANNER (from prev_result)
        assert senders_seen[1] == AgentRole.PLANNER

    def test_rejected_not_retried(self, tmp_path):
        """BUG-NEW-2: REJECTED status should NOT trigger retries."""
        call_count = {"n": 0}

        class RejectingAgent(BaseSubAgent):
            def run(self, message: CollaborationMessage) -> SubAgentResult:
                call_count["n"] += 1
                return SubAgentResult(
                    agent_role=self.role,
                    status=SubAgentStatus.REJECTED,
                    error="Methodology unsound",
                )

        agent = RejectingAgent(
            role=AgentRole.CODE_GEN,
            context_manager=ContextManager(max_tokens=4000),
            tool_registry=ToolRegistry(),
            state_dir=str(tmp_path),
            permission_mode=PermissionMode.DEFAULT,
        )
        # retry_on_failure=3, but REJECTED should not retry
        stages = [PipelineStep(PipelineStage.CODE_GEN, agent, retry_on_failure=3)]
        pipeline = Pipeline(stages=stages, state_dir=str(tmp_path))
        result = pipeline.run({"targets": ["test"]})

        assert result.is_failed() or result.status == SubAgentStatus.REJECTED
        # Should be called exactly once — no retries for REJECTED
        assert call_count["n"] == 1
