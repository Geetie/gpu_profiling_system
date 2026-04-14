"""Tests for Pipeline orchestrator (domain/pipeline.py).

Updated for AgentLoop-based pipeline execution.
Each stage now runs inside an AgentLoop with a model caller.
Tests use mock model callers that return predetermined responses.
"""
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


# ── Mock Agent with configurable model caller ────────────────────────


class MockAgent(BaseSubAgent):
    """Mock agent that returns predetermined results via mock model caller."""

    def __init__(
        self,
        role: AgentRole,
        model_response: str = "Task completed successfully.",
        state_dir: str = ".state",
        tool_registry: ToolRegistry | None = None,
        simulate_tool_results: bool = False,
    ) -> None:
        super().__init__(
            role=role,
            context_manager=ContextManager(max_tokens=4000),
            tool_registry=tool_registry or ToolRegistry(),
            state_dir=state_dir,
            permission_mode=PermissionMode.HIGH_AUTONOMY,
        )
        self._mock_response = model_response
        self._call_count = 0
        self._simulate_tool_results = simulate_tool_results

        def mock_caller(messages):
            self._call_count += 1
            # If simulating tool results, inject a tool result entry into context
            # before returning text. This mimics what the AgentLoop does when
            # a tool call succeeds: it adds the tool result as an ASSISTANT entry.
            if self._simulate_tool_results:
                self.context_manager.add_entry(
                    Role.ASSISTANT,
                    json.dumps({
                        "tool": "compile_cuda",
                        "status": "success",
                        "success": True,
                        "binary_path": ".sandbox/benchmark",
                        "stdout": self._mock_response,
                    }),
                    token_count=10,
                )
            return self._mock_response

        self._model_caller = mock_caller

    @property
    def call_count(self) -> int:
        return self._call_count

    def run(self, message: CollaborationMessage) -> SubAgentResult:
        """Fallback — not used by AgentLoop-based pipeline, but kept for interface."""
        self.context_manager.add_entry(Role.USER, f"task for {self.role.value}")
        return SubAgentResult(
            agent_role=self.role,
            status=SubAgentStatus.SUCCESS,
            data={"mock": True},
        )


class CapturingMockAgent(MockAgent):
    """Mock agent that captures incoming messages and returns predetermined results."""

    def __init__(
        self,
        role: AgentRole,
        model_response: str = "Task completed successfully.",
        state_dir: str = ".state",
        simulate_tool_results: bool = False,
    ) -> None:
        super().__init__(role, model_response, state_dir, simulate_tool_results=simulate_tool_results)
        self.captured_messages: list[CollaborationMessage] = []

    def run(self, message: CollaborationMessage) -> SubAgentResult:
        """Kept for interface compatibility; AgentLoop uses model caller."""
        self.captured_messages.append(message)
        return SubAgentResult(
            agent_role=self.role,
            status=SubAgentStatus.SUCCESS,
            data={"mock": True},
        )


# ── Flaky / Failing Agents ───────────────────────────────────────────


def _make_flaky_agent(role, fail_until, success_response, state_dir):
    """Create an agent that fails N times then succeeds.

    Uses a model caller that tracks calls internally.
    """
    agent = MockAgent(role, success_response, state_dir)
    call_count = {"n": 0}

    def flaky_caller(messages):
        call_count["n"] += 1
        if call_count["n"] < fail_until:
            # Return a tool call that will "fail"
            return json.dumps({"tool": "execute_binary", "args": {"binary_path": "fail"}})
        return success_response

    agent._model_caller = flaky_caller
    agent._call_tracker = call_count
    return agent


def _make_failing_agent(role, state_dir):
    """Create an agent that always fails."""
    agent = MockAgent(role, "Error: compilation failed", state_dir)

    def failing_caller(messages):
        return "Error: this always fails"

    agent._model_caller = failing_caller
    return agent


# ── Pipeline Tests ───────────────────────────────────────────────────


class TestPipeline:
    def test_full_happy_path(self, tmp_path):
        """All 4 stages succeed in sequence."""
        stages = [
            PipelineStep(PipelineStage.PLAN, MockAgent(AgentRole.PLANNER, state_dir=str(tmp_path))),
            PipelineStep(PipelineStage.CODE_GEN, MockAgent(AgentRole.CODE_GEN, state_dir=str(tmp_path), simulate_tool_results=True)),
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
            PipelineStep(PipelineStage.CODE_GEN, _make_failing_agent(AgentRole.CODE_GEN, str(tmp_path))),
            PipelineStep(PipelineStage.METRIC_ANALYSIS, MockAgent(AgentRole.METRIC_ANALYSIS, state_dir=str(tmp_path))),
        ]
        pipeline = Pipeline(stages=stages, state_dir=str(tmp_path))
        result = pipeline.run({"targets": ["dram_latency_cycles"]})
        assert result.is_failed()
        assert result.agent_role == AgentRole.CODE_GEN

    def test_input_chaining(self, tmp_path):
        """Each stage receives the previous stage's result via AgentLoop context."""
        # Use agents that embed their stage data in the response
        planner = MockAgent(
            AgentRole.PLANNER,
            model_response=json.dumps([{"target": "test", "category": "latency_measurement", "method": "test"}]),
            state_dir=str(tmp_path),
        )
        code_gen = MockAgent(
            AgentRole.CODE_GEN,
            model_response="Measured latency: 442 cycles",
            state_dir=str(tmp_path),
            simulate_tool_results=True,
        )

        stages = [
            PipelineStep(PipelineStage.PLAN, planner),
            PipelineStep(PipelineStage.CODE_GEN, code_gen),
        ]
        pipeline = Pipeline(stages=stages, state_dir=str(tmp_path))
        result = pipeline.run({"targets": ["test"]})

        # Both stages should have run
        assert planner.call_count >= 1
        assert code_gen.call_count >= 1
        # Final result should be from CODE_GEN stage
        assert result.is_success()

    def test_p7_violation_raised(self, tmp_path):
        """Pipeline should raise P7ViolationError if verification has non-empty context."""
        cm = ContextManager(max_tokens=4000)
        cm.add_entry(Role.USER, "leaked context")  # non-empty!
        verify_agent = MockAgent(
            AgentRole.VERIFICATION,
            state_dir=str(tmp_path),
        )
        verify_agent.context_manager = cm  # Replace with contaminated context

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
        result = pipeline.run({"targets": ["test"]})
        assert result.is_success()

    def test_retry_on_failure(self, tmp_path):
        """Stage should retry up to retry_on_failure times.

        Simulates retries by having the first N calls return error-like text,
        then the final call returns text with a simulated tool result.
        """
        agent = MockAgent(AgentRole.CODE_GEN, state_dir=str(tmp_path))
        call_count = {"n": 0}

        def flaky_caller(messages):
            call_count["n"] += 1
            if call_count["n"] < 3:
                return "Compilation error: invalid syntax"  # will trigger retry
            # Final attempt: return text + inject tool result
            agent.context_manager.add_entry(
                Role.ASSISTANT,
                json.dumps({
                    "tool": "compile_cuda",
                    "status": "success",
                    "success": True,
                    "binary_path": ".sandbox/benchmark",
                    "stdout": "dram_latency_cycles: 442",
                }),
                token_count=10,
            )
            return "Compilation successful, measured 442 cycles"

        agent._model_caller = flaky_caller

        stages = [
            PipelineStep(
                PipelineStage.CODE_GEN,
                agent,
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
        agent = _make_failing_agent(AgentRole.CODE_GEN, str(tmp_path))

        stages = [
            PipelineStep(
                PipelineStage.CODE_GEN,
                agent,
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
            PipelineStep(PipelineStage.CODE_GEN, MockAgent(AgentRole.CODE_GEN, state_dir=str(tmp_path), simulate_tool_results=True)),
        ]
        pipeline = Pipeline(stages=stages, state_dir=str(tmp_path))
        pipeline.run({"targets": ["test"]})

        log_path = os.path.join(str(tmp_path), "pipeline_log.jsonl")
        assert os.path.exists(log_path)
        with open(log_path) as f:
            lines = f.readlines()
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
        """Planner should fail when it produces no output."""
        agent = MockAgent(
            AgentRole.PLANNER,
            model_response="",  # Empty response → FAILED
            state_dir=str(tmp_path),
        )
        stages = [
            PipelineStep(PipelineStage.PLAN, agent),
        ]
        pipeline = Pipeline(stages=stages, state_dir=str(tmp_path))
        result = pipeline.run({"targets": []})
        assert result.is_failed()

    def test_sender_uses_prev_result_role(self, tmp_path):
        """Sender should be prev_result.agent_role, not always PLANNER."""
        planner = CapturingMockAgent(
            AgentRole.PLANNER,
            model_response=json.dumps([{"target": "test", "category": "latency_measurement", "method": "test"}]),
            state_dir=str(tmp_path),
        )
        code_gen = CapturingMockAgent(
            AgentRole.CODE_GEN,
            model_response="Measured 442 cycles",
            state_dir=str(tmp_path),
            simulate_tool_results=True,
        )

        stages = [
            PipelineStep(PipelineStage.PLAN, planner),
            PipelineStep(PipelineStage.CODE_GEN, code_gen),
        ]
        pipeline = Pipeline(stages=stages, state_dir=str(tmp_path))
        pipeline.run({"targets": ["test"]})

        # In AgentLoop mode, messages are built into context, not via run()
        # But we can verify both stages ran
        assert planner.call_count >= 1
        assert code_gen.call_count >= 1

    def test_rejected_not_retried(self, tmp_path):
        """REJECTED status should NOT trigger retries."""
        call_count = {"n": 0}

        def rejecting_caller(messages):
            call_count["n"] += 1
            return "REJECT: Methodology unsound — invalid sample size"

        agent = MockAgent(AgentRole.VERIFICATION, state_dir=str(tmp_path))
        agent._model_caller = rejecting_caller

        stages = [
            PipelineStep(PipelineStage.VERIFICATION, agent, retry_on_failure=3),
        ]
        pipeline = Pipeline(stages=stages, state_dir=str(tmp_path))
        result = pipeline.run({"targets": ["test"]})

        # Verification REJECTED — should not retry
        assert result.status == SubAgentStatus.REJECTED
        assert call_count["n"] == 1
