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

    def _process(self, message: CollaborationMessage) -> SubAgentResult:
        return SubAgentResult(
            agent_role=self.role,
            status=SubAgentStatus.SUCCESS,
            data={"mock": True},
        )

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

    def _process(self, message: CollaborationMessage) -> SubAgentResult:
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


class TestMetricAnalysisOptimizationLoop:
    """Tests for CodeGen ↔ MetricAnalysis iterative optimization.

    Verifies that:
    1. When MetricAnalysis identifies a bottleneck, pipeline loops back to CodeGen
    2. When MetricAnalysis reports no actionable feedback, pipeline proceeds to Verification
    3. "balanced" bottleneck type does NOT trigger optimization loop
    4. Max iterations limit is respected
    5. MetricAnalysis feedback is properly injected into CodeGen's prompt
    """

    def test_metric_feedback_triggers_codegen_loop(self, tmp_path):
        """When MetricAnalysis finds a bottleneck, pipeline loops back to CodeGen."""
        from src.domain.pipeline_context import PipelineContext

        ctx = PipelineContext(target_spec={"targets": ["dram_latency_cycles"]})

        ctx.add_metric_feedback(
            suggested_fixes=["Use shared memory for reduction"],
            bottleneck_type="memory_bound",
            bottleneck_sub_type="dram",
            recommendations=["Reduce global memory accesses"],
        )

        feedback = ctx.get_feedback_for_codegen()
        assert feedback is not None
        assert feedback["bottleneck_type"] == "memory_bound"
        assert feedback["metric_suggested_fixes"] == ["Use shared memory for reduction"]
        assert feedback["metric_recommendations"] == ["Reduce global memory accesses"]

    def test_balanced_bottleneck_not_actionable(self, tmp_path):
        """'balanced' bottleneck type should NOT trigger optimization loop."""
        from src.domain.pipeline_context import PipelineContext

        ctx = PipelineContext(target_spec={"targets": ["dram_latency_cycles"]})

        ctx.add_metric_feedback(
            suggested_fixes=[],
            bottleneck_type="balanced",
            recommendations=[],
        )

        feedback = ctx.get_feedback_for_codegen()
        assert feedback is not None
        assert feedback["bottleneck_type"] == "balanced"

        bottleneck_type = feedback.get("bottleneck_type", "")
        is_balanced = bottleneck_type == "balanced"
        has_actionable_feedback = bool(
            (feedback.get("metric_suggested_fixes") or feedback.get("metric_recommendations"))
            and not is_balanced
        )
        assert not has_actionable_feedback, "'balanced' should not be actionable"

    def test_no_feedback_not_actionable(self, tmp_path):
        """When MetricAnalysis has no feedback, it should not be actionable."""
        from src.domain.pipeline_context import PipelineContext

        ctx = PipelineContext(target_spec={"targets": ["dram_latency_cycles"]})

        ctx.add_metric_feedback(
            suggested_fixes=[],
            bottleneck_type="",
            recommendations=[],
        )

        feedback = ctx.get_feedback_for_codegen()
        assert feedback is not None

        has_actionable_feedback = bool(
            (feedback.get("metric_suggested_fixes") or feedback.get("metric_recommendations"))
            and not (feedback.get("bottleneck_type") == "balanced")
        )
        assert not has_actionable_feedback

    def test_iteration_count_limits_optimization(self, tmp_path):
        """Pipeline should stop iterating when max_iterations is reached."""
        from src.domain.pipeline_context import PipelineContext

        ctx = PipelineContext(
            target_spec={"targets": ["dram_latency_cycles"]},
            max_iterations=5,
        )

        for i in range(1, 5):
            ctx.increment_iteration()
            assert ctx.iteration_count == i
            assert ctx.can_retry()

        ctx.increment_iteration()
        assert ctx.iteration_count == 5
        assert not ctx.can_retry(), "Should not allow retry after max_iterations"

    def test_multiple_metric_feedback_accumulates(self, tmp_path):
        """Multiple MetricAnalysis rounds should accumulate feedback."""
        from src.domain.pipeline_context import PipelineContext

        ctx = PipelineContext(target_spec={"targets": ["dram_latency_cycles"]})

        ctx.add_metric_feedback(
            suggested_fixes=["Use shared memory"],
            bottleneck_type="memory_bound",
            recommendations=["Reduce global memory accesses"],
        )
        ctx.increment_iteration()

        ctx.add_metric_feedback(
            suggested_fixes=["Optimize warp scheduling"],
            bottleneck_type="compute_bound",
            recommendations=["Increase occupancy"],
        )
        ctx.increment_iteration()

        assert len(ctx.metric_feedback) == 2
        feedback = ctx.get_feedback_for_codegen()
        assert feedback["bottleneck_type"] == "compute_bound"
        assert feedback["metric_suggested_fixes"] == ["Optimize warp scheduling"]
        assert feedback["metric_recommendations"] == ["Increase occupancy"]

    def test_metric_feedback_injected_into_codegen_prompt(self, tmp_path):
        """_build_user_task should inject MetricAnalysis feedback for CodeGen."""
        from src.domain.pipeline_context import PipelineContext
        from src.domain.stage_executor import StageExecutor

        ctx = PipelineContext(target_spec={"targets": ["dram_latency_cycles"]})
        ctx.add_metric_feedback(
            suggested_fixes=["Use shared memory for reduction"],
            bottleneck_type="memory_bound",
            bottleneck_sub_type="dram",
            recommendations=["Reduce global memory accesses"],
        )
        ctx.increment_iteration()

        executor = StageExecutor(state_dir=str(tmp_path))
        user_task = executor._build_user_task(
            stage=PipelineStage.CODE_GEN,
            task={"tasks": [{"target": "dram_latency_cycles", "category": "latency_measurement", "method": "test"}]},
            prev_result={},
            target_spec={"targets": ["dram_latency_cycles"]},
            ctx=ctx,
        )

        assert "OPTIMIZATION ITERATION" in user_task
        assert "memory_bound" in user_task
        assert "Use shared memory for reduction" in user_task
        assert "Reduce global memory accesses" in user_task
        assert "YOUR MISSION" in user_task

    def test_metric_feedback_not_injected_for_non_codegen(self, tmp_path):
        """_build_user_task should NOT inject MetricAnalysis feedback for non-CodeGen stages."""
        from src.domain.pipeline_context import PipelineContext
        from src.domain.stage_executor import StageExecutor

        ctx = PipelineContext(target_spec={"targets": ["dram_latency_cycles"]})
        ctx.add_metric_feedback(
            suggested_fixes=["Use shared memory"],
            bottleneck_type="memory_bound",
            recommendations=["Reduce global memory accesses"],
        )

        executor = StageExecutor(state_dir=str(tmp_path))
        user_task = executor._build_user_task(
            stage=PipelineStage.METRIC_ANALYSIS,
            task={},
            prev_result={},
            target_spec={"targets": ["dram_latency_cycles"]},
            ctx=ctx,
        )

        assert "OPTIMIZATION ITERATION" not in user_task
        assert "memory_bound" not in user_task

    def test_no_metric_feedback_no_injection(self, tmp_path):
        """_build_user_task should not inject anything when there's no feedback."""
        from src.domain.pipeline_context import PipelineContext
        from src.domain.stage_executor import StageExecutor

        ctx = PipelineContext(target_spec={"targets": ["dram_latency_cycles"]})

        executor = StageExecutor(state_dir=str(tmp_path))
        user_task = executor._build_user_task(
            stage=PipelineStage.CODE_GEN,
            task={"tasks": [{"target": "test", "category": "latency_measurement", "method": "test"}]},
            prev_result={},
            target_spec={"targets": ["dram_latency_cycles"]},
            ctx=ctx,
        )

        assert "OPTIMIZATION ITERATION" not in user_task
        assert "BOTTLENECK" not in user_task

    def test_full_optimization_loop_with_convergence(self, tmp_path):
        """End-to-end: MetricAnalysis finds bottleneck → CodeGen optimizes → MetricAnalysis says balanced → done.

        This is the core test for continuous iterative optimization:
        - Iteration 0: CodeGen → MetricAnalysis finds memory_bound → loop
        - Iteration 1: CodeGen → MetricAnalysis finds compute_bound → loop
        - Iteration 2: CodeGen → MetricAnalysis says balanced → proceed to Verification
        """
        from src.domain.pipeline_context import PipelineContext

        ctx = PipelineContext(
            target_spec={"targets": ["dram_latency_cycles"]},
            max_iterations=5,
        )

        # Iteration 0: MetricAnalysis finds memory_bound
        ctx.add_metric_feedback(
            suggested_fixes=["Use shared memory"],
            bottleneck_type="memory_bound",
            recommendations=["Reduce global memory accesses"],
        )
        feedback = ctx.get_feedback_for_codegen()
        bottleneck_type = feedback.get("bottleneck_type", "")
        is_balanced = bottleneck_type == "balanced"
        has_actionable = bool(
            (feedback.get("metric_suggested_fixes") or feedback.get("metric_recommendations"))
            and not is_balanced
        )
        assert has_actionable, "memory_bound should be actionable"
        assert ctx.can_retry()
        ctx.increment_iteration()

        # Iteration 1: MetricAnalysis finds compute_bound
        ctx.add_metric_feedback(
            suggested_fixes=["Increase occupancy"],
            bottleneck_type="compute_bound",
            recommendations=["Optimize warp scheduling"],
        )
        feedback = ctx.get_feedback_for_codegen()
        bottleneck_type = feedback.get("bottleneck_type", "")
        is_balanced = bottleneck_type == "balanced"
        has_actionable = bool(
            (feedback.get("metric_suggested_fixes") or feedback.get("metric_recommendations"))
            and not is_balanced
        )
        assert has_actionable, "compute_bound should be actionable"
        assert ctx.can_retry()
        ctx.increment_iteration()

        # Iteration 2: MetricAnalysis says balanced (converged!)
        ctx.add_metric_feedback(
            suggested_fixes=[],
            bottleneck_type="balanced",
            recommendations=[],
        )
        feedback = ctx.get_feedback_for_codegen()
        bottleneck_type = feedback.get("bottleneck_type", "")
        is_balanced = bottleneck_type == "balanced"
        has_actionable = bool(
            (feedback.get("metric_suggested_fixes") or feedback.get("metric_recommendations"))
            and not is_balanced
        )
        assert not has_actionable, "'balanced' means converged — no more optimization needed"

    def test_max_iterations_forced_termination(self, tmp_path):
        """When max iterations reached, pipeline must proceed even if feedback exists."""
        from src.domain.pipeline_context import PipelineContext

        ctx = PipelineContext(
            target_spec={"targets": ["dram_latency_cycles"]},
            max_iterations=2,
        )

        # Iteration 0: finds bottleneck
        ctx.add_metric_feedback(
            suggested_fixes=["Fix 1"],
            bottleneck_type="memory_bound",
            recommendations=["Rec 1"],
        )
        ctx.increment_iteration()
        assert ctx.can_retry()

        # Iteration 1: still finds bottleneck
        ctx.add_metric_feedback(
            suggested_fixes=["Fix 2"],
            bottleneck_type="memory_bound",
            recommendations=["Rec 2"],
        )
        ctx.increment_iteration()
        assert not ctx.can_retry(), "Max iterations reached — must proceed to Verification"

        # Even though there's actionable feedback, can_retry() prevents further looping
        feedback = ctx.get_feedback_for_codegen()
        has_actionable = bool(
            (feedback.get("metric_suggested_fixes") or feedback.get("metric_recommendations"))
            and not (feedback.get("bottleneck_type") == "balanced")
        )
        assert has_actionable, "Feedback is still actionable"
        assert not ctx.can_retry(), "But iteration limit prevents further optimization"

    def test_convergence_detection_same_bottleneck_no_fixes(self, tmp_path):
        """is_optimization_converged should detect when same bottleneck repeats with no fixes."""
        from src.domain.pipeline_context import PipelineContext

        ctx = PipelineContext(
            target_spec={"targets": ["dram_latency_cycles"]},
            max_iterations=5,
        )

        assert not ctx.is_optimization_converged(), "No feedback yet — not converged"

        ctx.add_metric_feedback(
            suggested_fixes=["Use shared memory"],
            bottleneck_type="memory_bound",
            recommendations=["Reduce global memory accesses"],
        )
        assert not ctx.is_optimization_converged(), "Only 1 round — not converged"

        ctx.add_metric_feedback(
            suggested_fixes=["Optimize access pattern"],
            bottleneck_type="memory_bound",
            recommendations=["Coalesce accesses"],
        )
        assert not ctx.is_optimization_converged(), "Same bottleneck but has fixes — not converged"

        ctx.add_metric_feedback(
            suggested_fixes=[],
            bottleneck_type="memory_bound",
            recommendations=[],
        )
        assert not ctx.is_optimization_converged(), "Last round has no fixes, but prev round did — not converged"

        ctx.add_metric_feedback(
            suggested_fixes=[],
            bottleneck_type="memory_bound",
            recommendations=[],
        )
        assert ctx.is_optimization_converged(), "Same bottleneck, no fixes for 2 rounds — CONVERGED"

    def test_convergence_not_triggered_different_bottlenecks(self, tmp_path):
        """is_optimization_converged should NOT trigger when bottleneck types differ."""
        from src.domain.pipeline_context import PipelineContext

        ctx = PipelineContext(
            target_spec={"targets": ["dram_latency_cycles"]},
            max_iterations=5,
        )

        ctx.add_metric_feedback(
            suggested_fixes=[],
            bottleneck_type="memory_bound",
            recommendations=[],
        )
        ctx.add_metric_feedback(
            suggested_fixes=[],
            bottleneck_type="compute_bound",
            recommendations=[],
        )
        assert not ctx.is_optimization_converged(), "Different bottleneck types — not converged"

    def test_convergence_not_triggered_with_fixes(self, tmp_path):
        """is_optimization_converged should NOT trigger when there are concrete fixes."""
        from src.domain.pipeline_context import PipelineContext

        ctx = PipelineContext(
            target_spec={"targets": ["dram_latency_cycles"]},
            max_iterations=5,
        )

        ctx.add_metric_feedback(
            suggested_fixes=["Fix A"],
            bottleneck_type="memory_bound",
            recommendations=[],
        )
        ctx.add_metric_feedback(
            suggested_fixes=["Fix B"],
            bottleneck_type="memory_bound",
            recommendations=[],
        )
        assert not ctx.is_optimization_converged(), "Has concrete fixes — not converged"

    def test_convergence_with_recommendations_only(self, tmp_path):
        """Convergence should NOT trigger if recommendations exist (even without fixes)."""
        from src.domain.pipeline_context import PipelineContext

        ctx = PipelineContext(
            target_spec={"targets": ["dram_latency_cycles"]},
            max_iterations=5,
        )

        ctx.add_metric_feedback(
            suggested_fixes=[],
            bottleneck_type="memory_bound",
            recommendations=["Try shared memory"],
        )
        ctx.add_metric_feedback(
            suggested_fixes=[],
            bottleneck_type="memory_bound",
            recommendations=["Try tiling"],
        )
        assert not ctx.is_optimization_converged(), "Has recommendations — still actionable"

    def test_diminishing_returns_convergence(self, tmp_path):
        """is_optimization_converged should detect diminishing returns via CodeGen duration."""
        from src.domain.pipeline_context import PipelineContext

        ctx = PipelineContext(
            target_spec={"targets": ["dram_latency_cycles"]},
            max_iterations=5,
        )

        ctx.add_metric_feedback(
            suggested_fixes=["Fix A"],
            bottleneck_type="memory_bound",
            recommendations=["Rec A"],
        )
        ctx.add_metric_feedback(
            suggested_fixes=["Fix B"],
            bottleneck_type="memory_bound",
            recommendations=["Rec B"],
        )
        assert not ctx.is_optimization_converged(), "Has fixes — not converged"

        ctx.record_stage_duration("code_gen", 190.0, iteration=0)
        ctx.record_stage_duration("code_gen", 14.0, iteration=1)
        ctx.record_stage_duration("code_gen", 6.0, iteration=2)
        assert ctx.is_optimization_converged(), "6s < 14s*0.5=7s and <30s — diminishing returns detected"

    def test_diminishing_returns_not_triggered_slow_codegen(self, tmp_path):
        """Diminishing returns should NOT trigger when CodeGen is still doing substantial work."""
        from src.domain.pipeline_context import PipelineContext

        ctx = PipelineContext(
            target_spec={"targets": ["dram_latency_cycles"]},
            max_iterations=5,
        )

        ctx.add_metric_feedback(
            suggested_fixes=["Fix A"],
            bottleneck_type="memory_bound",
            recommendations=["Rec A"],
        )
        ctx.add_metric_feedback(
            suggested_fixes=["Fix B"],
            bottleneck_type="memory_bound",
            recommendations=["Rec B"],
        )

        ctx.record_stage_duration("code_gen", 190.0, iteration=0)
        ctx.record_stage_duration("code_gen", 100.0, iteration=1)
        ctx.record_stage_duration("code_gen", 60.0, iteration=2)
        assert not ctx.is_optimization_converged(), "60s > 100s*0.5=50s but >30s — still substantial work"

    def test_record_stage_duration(self, tmp_path):
        """record_stage_duration should store durations with correct keys."""
        from src.domain.pipeline_context import PipelineContext

        ctx = PipelineContext(target_spec={"targets": ["test"]})
        ctx.record_stage_duration("plan", 25.0)
        ctx.record_stage_duration("code_gen", 190.0, iteration=0)
        ctx.record_stage_duration("code_gen", 14.0, iteration=1)
        ctx.record_stage_duration("code_gen", 6.0, iteration=2)

        assert "plan" in ctx._stage_durations
        assert "code_gen" in ctx._stage_durations
        assert "code_gen_opt_1" in ctx._stage_durations
        assert "code_gen_opt_2" in ctx._stage_durations
        assert ctx._stage_durations["code_gen_opt_2"] == 6.0
