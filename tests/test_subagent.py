"""Tests for sub-agent domain model (domain/subagent.py)."""
import pytest
from src.domain.subagent import (
    AgentRole,
    BaseSubAgent,
    CollaborationMessage,
    P7ViolationError,
    PipelineStage,
    SubAgentResult,
    SubAgentStatus,
)
from src.application.context import ContextManager, Role
from src.domain.permission import PermissionMode
from src.domain.tool_contract import ToolRegistry


# ── Enum Tests ───────────────────────────────────────────────────────


class TestAgentRole:
    def test_all_roles_exist(self):
        assert AgentRole.PLANNER.value == "planner"
        assert AgentRole.CODE_GEN.value == "code_gen"
        assert AgentRole.METRIC_ANALYSIS.value == "metric_analysis"
        assert AgentRole.VERIFICATION.value == "verification"


class TestSubAgentStatus:
    def test_all_statuses_exist(self):
        assert SubAgentStatus.PENDING.value == "pending"
        assert SubAgentStatus.RUNNING.value == "running"
        assert SubAgentStatus.SUCCESS.value == "success"
        assert SubAgentStatus.FAILED.value == "failed"
        assert SubAgentStatus.REJECTED.value == "rejected"


class TestPipelineStage:
    def test_all_stages_exist(self):
        assert PipelineStage.PLAN.value == "plan"
        assert PipelineStage.CODE_GEN.value == "code_gen"
        assert PipelineStage.METRIC_ANALYSIS.value == "metric_analysis"
        assert PipelineStage.VERIFICATION.value == "verification"


# ── SubAgentResult Tests ─────────────────────────────────────────────


class TestSubAgentResult:
    def test_create_default(self):
        result = SubAgentResult(agent_role=AgentRole.PLANNER)
        assert result.status == SubAgentStatus.PENDING
        assert result.data == {}
        assert result.artifacts == []
        assert result.error is None

    def test_create_with_fields(self):
        result = SubAgentResult(
            agent_role=AgentRole.CODE_GEN,
            status=SubAgentStatus.SUCCESS,
            data={"key": "value"},
            artifacts=["benchmark.cu"],
            error=None,
        )
        assert result.is_success() is True
        assert result.is_failed() is False

    def test_failed_status(self):
        result = SubAgentResult(
            agent_role=AgentRole.VERIFICATION,
            status=SubAgentStatus.REJECTED,
            error="Methodology unsound",
        )
        assert result.is_failed() is True
        assert result.is_success() is False

    def test_to_dict_roundtrip(self):
        result = SubAgentResult(
            agent_role=AgentRole.METRIC_ANALYSIS,
            status=SubAgentStatus.SUCCESS,
            data={"bottleneck_type": "memory_bound"},
            artifacts=["trace.ncu"],
            context_fingerprint="abc123",
            metadata={"time": 42},
        )
        d = result.to_dict()
        restored = SubAgentResult.from_dict(d)
        assert restored.agent_role == result.agent_role
        assert restored.status == result.status
        assert restored.data == result.data
        assert restored.artifacts == result.artifacts
        assert restored.context_fingerprint == result.context_fingerprint
        assert restored.metadata == result.metadata

    def test_compute_fingerprint(self):
        cm = ContextManager(max_tokens=1000)
        cm.add_entry(Role.SYSTEM, "system prompt", token_count=5)
        cm.add_entry(Role.USER, "user query", token_count=5)
        result = SubAgentResult(agent_role=AgentRole.PLANNER)
        fp = result.compute_fingerprint(cm)
        assert isinstance(fp, str)
        assert len(fp) == 16  # truncated SHA-256

    def test_fingerprint_changes_with_content(self):
        cm1 = ContextManager(max_tokens=1000)
        cm1.add_entry(Role.USER, "content A")
        cm2 = ContextManager(max_tokens=1000)
        cm2.add_entry(Role.USER, "content B")
        r1 = SubAgentResult(agent_role=AgentRole.PLANNER)
        r2 = SubAgentResult(agent_role=AgentRole.PLANNER)
        assert r1.compute_fingerprint(cm1) != r2.compute_fingerprint(cm2)


# ── CollaborationMessage Tests ───────────────────────────────────────


class TestCollaborationMessage:
    def test_create_message(self):
        msg = CollaborationMessage(
            sender=AgentRole.PLANNER,
            receiver=AgentRole.CODE_GEN,
            message_type="task_dispatch",
            payload={"task": "generate kernel"},
        )
        assert msg.sender == AgentRole.PLANNER
        assert msg.receiver == AgentRole.CODE_GEN
        assert msg.message_type == "task_dispatch"
        assert msg.payload == {"task": "generate kernel"}

    def test_to_dict(self):
        msg = CollaborationMessage(
            sender=AgentRole.PLANNER,
            receiver=AgentRole.VERIFICATION,
            message_type="result",
            payload={},
        )
        d = msg.to_dict()
        assert d["sender"] == "planner"
        assert d["receiver"] == "verification"
        assert d["message_type"] == "result"
        assert "timestamp" in d

    def test_different_sender_receiver(self):
        """Message should allow same sender and receiver (for self-dispatch)."""
        msg = CollaborationMessage(
            sender=AgentRole.PLANNER,
            receiver=AgentRole.PLANNER,
            message_type="self_dispatch",
            payload={},
        )
        assert msg.sender == msg.receiver


# ── P7ViolationError Tests ───────────────────────────────────────────


class TestP7ViolationError:
    def test_exception_is_exception(self):
        with pytest.raises(P7ViolationError):
            raise P7ViolationError("test violation")

    def test_exception_message(self):
        try:
            raise P7ViolationError("context shared")
        except P7ViolationError as e:
            assert "context shared" in str(e)


# ── BaseSubAgent Tests ───────────────────────────────────────────────


class ConcreteTestAgent(BaseSubAgent):
    """Concrete implementation for testing BaseSubAgent."""

    def run(self, message: CollaborationMessage) -> SubAgentResult:
        self.context_manager.add_entry(
            Role.SYSTEM, self._build_system_prompt(), token_count=10
        )
        result = SubAgentResult(
            agent_role=self.role,
            status=SubAgentStatus.SUCCESS,
            data={"ran": True},
        )
        result.context_fingerprint = result.compute_fingerprint(self.context_manager)
        self._persist_result(result)
        return result


class TestBaseSubAgent:
    def test_context_isolation(self, tmp_path):
        """Each sub-agent gets its own ContextManager."""
        cm1 = ContextManager(max_tokens=1000)
        cm2 = ContextManager(max_tokens=1000)
        reg = ToolRegistry()
        a1 = ConcreteTestAgent(
            role=AgentRole.PLANNER,
            context_manager=cm1,
            tool_registry=reg,
            state_dir=str(tmp_path / "a1"),
        )
        a2 = ConcreteTestAgent(
            role=AgentRole.CODE_GEN,
            context_manager=cm2,
            tool_registry=reg,
            state_dir=str(tmp_path / "a2"),
        )
        # They are completely independent
        a1.context_manager.add_entry(Role.USER, "only in a1")
        assert len(a2.context_manager.get_entries()) == 0

    def test_role_specific_system_prompt(self, tmp_path):
        cm = ContextManager(max_tokens=1000)
        agent = ConcreteTestAgent(
            role=AgentRole.CODE_GEN,
            context_manager=cm,
            tool_registry=ToolRegistry(),
            state_dir=str(tmp_path),
        )
        prompt = agent._build_system_prompt()
        assert "Code Generation" in prompt

    def test_build_system_prompt_all_roles(self, tmp_path):
        for role in AgentRole:
            cm = ContextManager(max_tokens=1000)
            agent = ConcreteTestAgent(
                role=role,
                context_manager=cm,
                tool_registry=ToolRegistry(),
                state_dir=str(tmp_path / role.value),
            )
            prompt = agent._build_system_prompt()
            assert len(prompt) > 10

    def test_run_persists_result(self, tmp_path):
        reg = ToolRegistry()
        agent = ConcreteTestAgent(
            role=AgentRole.PLANNER,
            context_manager=ContextManager(max_tokens=1000),
            tool_registry=reg,
            state_dir=str(tmp_path),
        )
        msg = CollaborationMessage(
            sender=AgentRole.PLANNER,
            receiver=AgentRole.PLANNER,
            message_type="task_dispatch",
            payload={},
        )
        result = agent.run(msg)
        assert result.is_success()

    def test_set_model_caller(self, tmp_path):
        reg = ToolRegistry()
        agent = ConcreteTestAgent(
            role=AgentRole.PLANNER,
            context_manager=ContextManager(max_tokens=1000),
            tool_registry=reg,
            state_dir=str(tmp_path),
        )
        agent.set_model_caller(lambda msgs: "mock response")
        assert agent._model_caller is not None

    def test_persist_result_writes_file(self, tmp_path):
        reg = ToolRegistry()
        agent = ConcreteTestAgent(
            role=AgentRole.CODE_GEN,
            context_manager=ContextManager(max_tokens=1000),
            tool_registry=reg,
            state_dir=str(tmp_path),
        )
        msg = CollaborationMessage(
            sender=AgentRole.PLANNER,
            receiver=AgentRole.CODE_GEN,
            message_type="task_dispatch",
            payload={},
        )
        agent.run(msg)
        # Check that the agent-specific log file exists
        import os
        log_path = str(tmp_path / "agent_code_gen_log.jsonl")
        assert os.path.exists(log_path)
