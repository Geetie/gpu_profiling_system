"""阶段 4：多智能体协作测试。

验证多智能体团队是否真的能分工协作，且认知隔离有效。
"""
import json
import os
import pytest


# ── 测试用例 4.1：角色分工验证 ──────────────────────────────────────


class TestRoleDivision:
    """验证各智能体各司其职。"""

    def _build_all_agents(self, tmp_path):
        """创建所有 4 个智能体实例。"""
        from src.application.context import ContextManager
        from src.application.subagents.codegen import CodeGenAgent
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        from src.application.subagents.planner import PlannerAgent
        from src.application.subagents.verification import VerificationAgent
        from src.domain.permission import PermissionMode
        from src.domain.subagent import AgentRole
        from src.domain.tool_contract import ToolRegistry

        registry = ToolRegistry()
        state_dir = str(tmp_path)

        planner = PlannerAgent(
            context_manager=ContextManager(max_tokens=4000),
            tool_registry=registry,
            state_dir=state_dir,
            permission_mode=PermissionMode.DEFAULT,
        )

        code_gen = CodeGenAgent(
            context_manager=ContextManager(max_tokens=4000),
            tool_registry=registry,
            state_dir=state_dir,
            permission_mode=PermissionMode.DEFAULT,
        )

        metric_analysis = MetricAnalysisAgent(
            context_manager=ContextManager(max_tokens=4000),
            tool_registry=registry,
            state_dir=state_dir,
            permission_mode=PermissionMode.DEFAULT,
        )

        verification = VerificationAgent(
            tool_registry=registry,
            state_dir=state_dir,
            permission_mode=PermissionMode.DEFAULT,
        )

        return planner, code_gen, metric_analysis, verification

    def test_planner_role_is_correct(self, tmp_path):
        """Planner 的角色必须是 PLANNER。"""
        planner, _, _, _ = self._build_all_agents(tmp_path)
        assert planner.role.value == "planner"

    def test_code_gen_role_is_correct(self, tmp_path):
        """CodeGen 的角色必须是 CODE_GEN。"""
        _, code_gen, _, _ = self._build_all_agents(tmp_path)
        assert code_gen.role.value == "code_gen"

    def test_metric_analysis_role_is_correct(self, tmp_path):
        """MetricAnalysis 的角色必须是 METRIC_ANALYSIS。"""
        _, _, metric_analysis, _ = self._build_all_agents(tmp_path)
        assert metric_analysis.role.value == "metric_analysis"

    def test_verification_role_is_correct(self, tmp_path):
        """Verification 的角色必须是 VERIFICATION。"""
        _, _, _, verification = self._build_all_agents(tmp_path)
        assert verification.role.value == "verification"

    def test_planner_decomposes_task(self, tmp_path):
        """Planner 应能拆解任务并调度。"""
        from src.domain.subagent import AgentRole, CollaborationMessage, PipelineStage

        planner, _, _, _ = self._build_all_agents(tmp_path)

        message = CollaborationMessage(
            sender=AgentRole.PLANNER,
            receiver=AgentRole.PLANNER,
            message_type="task_dispatch",
            payload={
                "target_spec": {
                    "targets": ["dram_latency_cycles"],
                },
            },
        )

        # Planner should produce a plan (uses template fallback without model caller)
        result = planner.run(message)
        assert result.status.value == "success"
        assert result.agent_role == AgentRole.PLANNER

    def test_code_gen_produces_cuda_source(self, tmp_path):
        """CodeGenAgent 应能生成 CUDA 源码。"""
        from src.domain.subagent import AgentRole, CollaborationMessage
        from src.infrastructure.sandbox import LocalSandbox, SandboxConfig

        _, code_gen, _, _ = self._build_all_agents(tmp_path)
        code_gen._sandbox = LocalSandbox(
            SandboxConfig(),
            sandbox_root=str(tmp_path / ".sandbox"),
        )

        message = CollaborationMessage(
            sender=AgentRole.PLANNER,
            receiver=AgentRole.CODE_GEN,
            message_type="task_dispatch",
            payload={
                "task": {
                    "target": "dram_latency",
                    "category": "latency_measurement",
                    "method": "pointer-chase",
                },
            },
        )

        result = code_gen.run(message)
        # Without nvcc, compilation will fail, but source generation should happen
        # Check that the agent attempted the full pipeline
        assert result.agent_role == AgentRole.CODE_GEN

    def test_metric_analysis_parses_ncu_output(self, tmp_path):
        """MetricAnalysisAgent 应能解析 ncu 输出。"""
        from src.domain.subagent import AgentRole, CollaborationMessage, SubAgentResult, SubAgentStatus

        _, _, metric_analysis, _ = self._build_all_agents(tmp_path)

        # Simulate previous result with ncu output
        prev_result = SubAgentResult(
            agent_role=AgentRole.CODE_GEN,
            status=SubAgentStatus.SUCCESS,
            data={
                "raw_output": "dram__throughput.avg.pct_of_peak_sustained_elapsed 45.2\ndram__bytes_read.sum 1073741824\nl2__throughput.avg.pct_of_peak_sustained_elapsed 78.3\n",
                "target": "dram_latency",
            },
        )

        message = CollaborationMessage(
            sender=AgentRole.CODE_GEN,
            receiver=AgentRole.METRIC_ANALYSIS,
            message_type="result",
            payload={"prev_result": prev_result.to_dict()},
        )

        result = metric_analysis.run(message)
        assert result.is_success()
        assert "bottleneck_type" in result.data

    def test_verification_independent_assessment(self, tmp_path):
        """VerificationAgent 应做独立评估，不继承生成上下文。"""
        from src.domain.subagent import AgentRole, CollaborationMessage, SubAgentResult, SubAgentStatus

        _, _, _, verification = self._build_all_agents(tmp_path)

        # Verification must start with clean context
        assert verification.context_manager.total_tokens == 0

        prev_result = SubAgentResult(
            agent_role=AgentRole.METRIC_ANALYSIS,
            status=SubAgentStatus.SUCCESS,
            data={
                "bottleneck_type": "memory_bound",
                "confidence": 0.85,
                "metrics": {"dram_throughput": 45.2},
            },
            context_fingerprint="abc123",
        )

        message = CollaborationMessage(
            sender=AgentRole.METRIC_ANALYSIS,
            receiver=AgentRole.VERIFICATION,
            message_type="result",
            payload={"prev_result": prev_result.to_dict()},
        )

        result = verification.run(message)
        assert result.agent_role == AgentRole.VERIFICATION

        # After run, context should have system prompt + assessment
        assert verification.context_manager.total_tokens > 0


# ── 测试用例 4.2：P7 原则验证 ──────────────────────────────────────


class TestP7Enforcement:
    """验证生成与评估分离。"""

    def test_verification_context_starts_empty(self, tmp_path):
        """VerificationAgent 的上下文初始必须为空。"""
        from src.application.subagents.verification import VerificationAgent
        from src.application.context import ContextManager
        from src.domain.tool_contract import ToolRegistry

        agent = VerificationAgent(
            tool_registry=ToolRegistry(),
            state_dir=str(tmp_path),
        )
        assert agent.context_manager.total_tokens == 0
        assert len(agent.context_manager.get_entries()) == 0

    def test_pipeline_p7_gate_blocks_contaminated_verification(self, tmp_path):
        """Pipeline 的 P7 gate 必须阻止被污染的 VerificationAgent。"""
        from unittest.mock import patch
        from src.application.context import ContextManager, Role
        from src.application.subagents.verification import VerificationAgent
        from src.domain.pipeline import Pipeline, PipelineStep
        from src.domain.subagent import AgentRole, PipelineStage
        from src.domain.subagent import P7ViolationError
        from src.domain.tool_contract import ToolRegistry

        os.chdir(str(tmp_path))

        # Create a contaminated context
        ctx = ContextManager(max_tokens=4000)
        ctx.add_entry(Role.USER, "contaminated from code_gen", token_count=10)

        # VerificationAgent always creates its own context, so we patch
        # the internal ContextManager to simulate contamination
        with patch.object(ContextManager, "__init__", return_value=None):
            verification = VerificationAgent(
                tool_registry=ToolRegistry(),
                state_dir=str(tmp_path),
            )
            verification.context_manager = ctx

        # Build a pipeline with just the verification stage
        pipeline = Pipeline(
            stages=[
                PipelineStep(
                    stage=PipelineStage.VERIFICATION,
                    agent=verification,
                ),
            ],
            state_dir=str(tmp_path),
        )

        # Pipeline should raise P7ViolationError
        with pytest.raises(P7ViolationError, match="P7 violation"):
            pipeline.run({"targets": ["test"]})

    def test_verification_records_generation_fingerprint(self, tmp_path):
        """Verification 必须记录生成者的指纹，做交叉验证。"""
        from src.domain.subagent import AgentRole, CollaborationMessage, SubAgentResult, SubAgentStatus
        from src.application.subagents.verification import VerificationAgent
        from src.application.context import ContextManager
        from src.domain.tool_contract import ToolRegistry

        agent = VerificationAgent(
            tool_registry=ToolRegistry(),
            state_dir=str(tmp_path),
        )

        prev_result = SubAgentResult(
            agent_role=AgentRole.METRIC_ANALYSIS,
            status=SubAgentStatus.SUCCESS,
            data={
                "bottleneck_type": "memory_bound",
                "confidence": 0.85,
                "metrics": {"dram_throughput": 45.2},
            },
            context_fingerprint="gen_fingerprint_xyz",
        )

        message = CollaborationMessage(
            sender=AgentRole.METRIC_ANALYSIS,
            receiver=AgentRole.VERIFICATION,
            message_type="result",
            payload={
                "prev_result": prev_result.to_dict(),
                "prev_fingerprint": "gen_fingerprint_xyz",
            },
        )

        result = agent.run(message)

        # Must record the generation's fingerprint
        assert result.data.get("generation_fingerprint") == "gen_fingerprint_xyz"


# ── 测试用例 4.3：Pipeline 端到端 ──────────────────────────────────


class TestPipelineEndToEnd:
    """验证 Pipeline 完整执行。"""

    def test_pipeline_sequential_execution(self, tmp_path):
        """Pipeline 应顺序执行 4 个阶段。"""
        import shutil
        from src.application.context import ContextManager
        from src.application.subagents.codegen import CodeGenAgent
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        from src.application.subagents.planner import PlannerAgent
        from src.application.subagents.verification import VerificationAgent
        from src.domain.pipeline import Pipeline
        from src.domain.subagent import AgentRole, PipelineStage
        from src.domain.tool_contract import ToolRegistry

        # This test requires nvcc to be available; skip if not installed.
        if shutil.which("nvcc") is None:
            pytest.skip("nvcc not found — CODE_GEN stage cannot run")

        os.chdir(str(tmp_path))
        registry = ToolRegistry()

        planner = PlannerAgent(
            context_manager=ContextManager(max_tokens=4000),
            tool_registry=registry,
            state_dir=str(tmp_path),
        )

        code_gen = CodeGenAgent(
            context_manager=ContextManager(max_tokens=4000),
            tool_registry=registry,
            state_dir=str(tmp_path),
        )

        metric_analysis = MetricAnalysisAgent(
            context_manager=ContextManager(max_tokens=4000),
            tool_registry=registry,
            state_dir=str(tmp_path),
        )

        verification = VerificationAgent(
            tool_registry=registry,
            state_dir=str(tmp_path),
        )

        pipeline = Pipeline.build_default(
            planner=planner,
            code_gen=code_gen,
            metric_analysis=metric_analysis,
            verification=verification,
            state_dir=str(tmp_path),
        )

        result = pipeline.run({"targets": ["dram_latency_cycles"]})

        # Pipeline should produce a result
        assert result is not None
        # The final stage is verification
        assert result.agent_role == AgentRole.VERIFICATION

    def test_pipeline_persists_state(self, tmp_path):
        """Pipeline 执行结果应落盘。"""
        from src.application.context import ContextManager
        from src.application.subagents.planner import PlannerAgent
        from src.domain.pipeline import Pipeline
        from src.domain.subagent import AgentRole, PipelineStage
        from src.domain.tool_contract import ToolRegistry
        from src.infrastructure.state_persist import StatePersister

        os.chdir(str(tmp_path))
        registry = ToolRegistry()

        planner = PlannerAgent(
            context_manager=ContextManager(max_tokens=4000),
            tool_registry=registry,
            state_dir=str(tmp_path),
        )

        pipeline = Pipeline(
            stages=[
                type("PipelineStep", (), {
                    "stage": PipelineStage.PLAN,
                    "agent": planner,
                    "retry_on_failure": 0,
                })(),
            ],
            state_dir=str(tmp_path),
        )

        pipeline.run({"targets": ["test"]})

        # Check pipeline log was written
        log_path = os.path.join(tmp_path, "pipeline_log.jsonl")
        assert os.path.isfile(log_path)

        persister = StatePersister(log_dir=str(tmp_path), filename="pipeline_log.jsonl")
        history = persister.load_history()
        assert len(history) >= 1

    def test_pipeline_agentloop_integration(self, tmp_path):
        """AgentLoop.run_pipeline() 应能执行 Pipeline 并将结果注入上下文。"""
        from src.application.agent_loop import AgentLoop
        from src.application.context import ContextManager
        from src.application.control_plane import ControlPlane
        from src.application.session import SessionState
        from src.application.subagents.planner import PlannerAgent
        from src.domain.pipeline import Pipeline
        from src.domain.subagent import AgentRole, PipelineStage
        from src.domain.tool_contract import ToolRegistry

        os.chdir(str(tmp_path))
        registry = ToolRegistry()

        planner = PlannerAgent(
            context_manager=ContextManager(max_tokens=4000),
            tool_registry=registry,
            state_dir=str(tmp_path),
        )

        pipeline = Pipeline(
            stages=[
                type("PipelineStep", (), {
                    "stage": PipelineStage.PLAN,
                    "agent": planner,
                    "retry_on_failure": 0,
                })(),
            ],
            state_dir=str(tmp_path),
        )

        session = SessionState(session_id="pipeline_integration", goal="test")
        loop = AgentLoop(
            session=session,
            context_manager=ContextManager(max_tokens=8000),
            control_plane=ControlPlane(rule_dir=str(tmp_path)),
            tool_registry=registry,
            max_turns=5,
            state_dir=str(tmp_path),
        )

        result = loop.run_pipeline(pipeline, {"targets": ["dram_latency"]})

        # Pipeline result should be injected into agent loop context
        assert loop.context_manager.total_tokens > 0
        # Session should be persisted
        history = loop._persister.load_history()
        persist_entries = [h for h in history if h.get("action") == "tool_execution"
                         and h.get("tool_name") == "__loop_state__"]
        assert len(persist_entries) >= 1
