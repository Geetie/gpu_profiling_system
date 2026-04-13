"""Tests for application-level sub-agents."""
import json
import os
import pytest

from src.application.context import ContextManager
from src.domain.permission import PermissionMode
from src.domain.subagent import (
    AgentRole,
    CollaborationMessage,
    PipelineStage,
    SubAgentResult,
    SubAgentStatus,
)
from src.domain.tool_contract import ToolRegistry
from src.infrastructure.sandbox import LocalSandbox, SandboxConfig


# ── PlannerAgent Tests ───────────────────────────────────────────────


class TestPlannerAgent:
    def test_plan_with_targets(self, tmp_path):
        from src.application.subagents.planner import PlannerAgent
        agent = PlannerAgent(state_dir=str(tmp_path))
        msg = CollaborationMessage(
            sender=AgentRole.PLANNER,
            receiver=AgentRole.PLANNER,
            message_type="task_dispatch",
            payload={"target_spec": {"targets": ["dram_latency_cycles"]}},
        )
        result = agent.run(msg)
        assert result.is_success()
        assert result.data["targets"] == ["dram_latency_cycles"]
        assert len(result.data["tasks"]) == 1
        assert result.data["tasks"][0]["target"] == "dram_latency_cycles"
        assert result.data["tasks"][0]["category"] == "latency_measurement"

    def test_plan_empty_targets_fails(self, tmp_path):
        from src.application.subagents.planner import PlannerAgent
        agent = PlannerAgent(state_dir=str(tmp_path))
        msg = CollaborationMessage(
            sender=AgentRole.PLANNER,
            receiver=AgentRole.PLANNER,
            message_type="task_dispatch",
            payload={"target_spec": {"targets": []}},
        )
        result = agent.run(msg)
        assert result.is_failed()
        assert "No targets" in (result.error or "")

    def test_classify_all_categories(self, tmp_path):
        from src.application.subagents.planner import PlannerAgent
        agent = PlannerAgent(state_dir=str(tmp_path))
        targets = [
            ("dram_latency_cycles", "latency_measurement"),
            ("max_shmem_per_block_kb", "capacity_measurement"),
            ("actual_boost_clock_mhz", "clock_measurement"),
            ("dram_bandwidth_gbps", "bandwidth_measurement"),
            ("unknown_metric", "unknown"),
        ]
        for target, expected_category in targets:
            task = agent._classify_target(target)
            assert task["category"] == expected_category, f"Failed for {target}"

    def test_create_plan_routes_to_codegen(self, tmp_path):
        from src.application.subagents.planner import PlannerAgent
        agent = PlannerAgent(state_dir=str(tmp_path))
        tasks = [{"target": "dram_latency_cycles", "category": "latency_measurement", "method": "test"}]
        plan = agent.create_plan(tasks)
        assert len(plan) == 1
        assert plan[0].receiver == AgentRole.CODE_GEN

    def test_plan_persists_result(self, tmp_path):
        from src.application.subagents.planner import PlannerAgent
        agent = PlannerAgent(state_dir=str(tmp_path))
        msg = CollaborationMessage(
            sender=AgentRole.PLANNER,
            receiver=AgentRole.PLANNER,
            message_type="task_dispatch",
            payload={"target_spec": {"targets": ["l2_cache_size_kb"]}},
        )
        agent.run(msg)
        log_path = os.path.join(str(tmp_path), "agent_planner_log.jsonl")
        assert os.path.exists(log_path)


# ── CodeGenAgent Tests ──────────────────────────────────────────────


class TestCodeGenAgent:
    def test_generate_latency_kernel(self, tmp_path):
        from src.application.subagents.codegen import CodeGenAgent
        sandbox = LocalSandbox(sandbox_root=str(tmp_path))
        agent = CodeGenAgent(state_dir=str(tmp_path), sandbox=sandbox)
        msg = CollaborationMessage(
            sender=AgentRole.PLANNER,
            receiver=AgentRole.CODE_GEN,
            message_type="task_dispatch",
            payload={"task": {"target": "dram_latency_cycles", "category": "latency_measurement", "method": "pointer-chasing"}},
        )
        # Without nvcc, compilation will fail but code generation works
        result = agent.run(msg)
        # Should fail on compilation (no nvcc) but the kernel was generated
        assert result.is_failed()
        assert "Compilation failed" in (result.error or "")

    def test_template_all_kernel_types(self, tmp_path):
        from src.application.subagents.codegen import CodeGenAgent
        agent = CodeGenAgent(state_dir=str(tmp_path))
        for category in ["latency_measurement", "capacity_measurement", "clock_measurement", "bandwidth_measurement"]:
            source = agent._template_kernel("test_target", category, "test_method")
            assert "__global__" in source
            assert "test_target" in source

    def test_generic_kernel_fallback(self, tmp_path):
        from src.application.subagents.codegen import CodeGenAgent
        agent = CodeGenAgent(state_dir=str(tmp_path))
        source = agent._template_kernel("weird_metric", "unknown", "custom")
        assert "generic_kernel" in source

    def test_model_caller_used_when_available(self, tmp_path):
        from src.application.subagents.codegen import CodeGenAgent
        agent = CodeGenAgent(state_dir=str(tmp_path))
        agent.set_model_caller(lambda msgs: "// custom model-generated kernel")
        source = agent._generate_kernel("test", "latency_measurement", "test")
        assert "custom model-generated kernel" in source


# ── MetricAnalysisAgent Tests ────────────────────────────────────────


class TestMetricAnalysisAgent:
    def test_parse_kv_metrics(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        raw = "L2 Hit Rate: 85.5\nDRAM Latency: 320\n"
        metrics = agent._parse_output(raw)
        assert metrics["L2 Hit Rate"] == 85.5
        assert metrics["DRAM Latency"] == 320.0

    def test_parse_plain_numbers(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        raw = "100\n200\n350\n"
        metrics = agent._parse_output(raw)
        assert "result" in metrics
        assert metrics["result"] == [100.0, 200.0, 350.0]

    def test_parse_ignores_comments(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        raw = "# header\n--- separator ---\nvalue: 42\n"
        metrics = agent._parse_output(raw)
        assert metrics["value"] == 42.0
        assert len(metrics) == 1

    def test_parse_fallback_raw(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        raw = "no structured data here at all"
        metrics = agent._parse_output(raw)
        assert "raw" in metrics

    def test_identify_latency_bound(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        assert agent.identify_bottleneck({"dram_latency_cycles": 320}) == "latency_bound"
        assert agent.identify_bottleneck({"l2_cycles": 50}) == "latency_bound"

    def test_identify_memory_bound(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        assert agent.identify_bottleneck({"dram_bandwidth_gbps": 800}) == "memory_bound"
        assert agent.identify_bottleneck({"memory_throughput": 500}) == "memory_bound"

    def test_identify_compute_bound(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        assert agent.identify_bottleneck({"ipc": 2.5}) == "compute_bound"
        assert agent.identify_bottleneck({"total_flops": 1e9}) == "compute_bound"

    def test_identify_cache_capacity_cliff(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        # Large jump triggers cliff detection
        assert agent.identify_bottleneck({"a": 10, "b": 100, "c": 500}) == "cache_capacity"
        assert agent.identify_bottleneck({"l2_cache_miss": 500}) == "cache_capacity"

    def test_identify_unknown(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        assert agent.identify_bottleneck({"name": "test"}) == "unknown"

    def test_confidence_scaling(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        # 0 metrics → 0.0 confidence
        assert agent._assess_confidence({}) == 0.0
        # 5 numeric metrics → 1.0 (capped)
        assert agent._assess_confidence({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}) == 1.0
        # 2 numeric metrics → 0.4
        assert agent._assess_confidence({"a": 1, "b": 2, "name": "test"}) == 0.4

    def test_no_raw_output_fails(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        msg = CollaborationMessage(
            sender=AgentRole.CODE_GEN,
            receiver=AgentRole.METRIC_ANALYSIS,
            message_type="task_dispatch",
            payload={"prev_result": {"data": {"raw_output": ""}}},
        )
        result = agent.run(msg)
        assert result.is_failed()
        assert "No raw output" in (result.error or "")

    def test_full_analysis_flow(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        msg = CollaborationMessage(
            sender=AgentRole.CODE_GEN,
            receiver=AgentRole.METRIC_ANALYSIS,
            message_type="task_dispatch",
            payload={"prev_result": {"data": {"raw_output": "DRAM Latency: 320\nL2 Hit Rate: 85.5\n"}}},
        )
        result = agent.run(msg)
        assert result.is_success()
        assert result.data["bottleneck_type"] == "latency_bound"


# ── VerificationAgent Tests ─────────────────────────────────────────


class TestVerificationAgent:
    def test_accept_valid_data(self, tmp_path):
        from src.application.subagents.verification import VerificationAgent
        agent = VerificationAgent(state_dir=str(tmp_path))
        msg = CollaborationMessage(
            sender=AgentRole.METRIC_ANALYSIS,
            receiver=AgentRole.VERIFICATION,
            message_type="task_dispatch",
            payload={"prev_result": {"data": {"bottleneck_type": "memory_bound", "confidence": 0.8}, "artifacts": ["trace.ncu"], "status": "success", "agent_role": "metric_analysis"}},
        )
        result = agent.run(msg)
        assert result.is_success()
        assert result.data["accepted"] is True

    def test_reject_empty_data(self, tmp_path):
        from src.application.subagents.verification import VerificationAgent
        agent = VerificationAgent(state_dir=str(tmp_path))
        msg = CollaborationMessage(
            sender=AgentRole.CODE_GEN,
            receiver=AgentRole.VERIFICATION,
            message_type="task_dispatch",
            payload={"prev_result": {"data": {}, "artifacts": [], "status": "success", "agent_role": "code_gen"}},
        )
        result = agent.run(msg)
        assert result.status == SubAgentStatus.REJECTED
        assert result.data["accepted"] is False

    def test_reject_negative_values(self, tmp_path):
        from src.application.subagents.verification import VerificationAgent
        agent = VerificationAgent(state_dir=str(tmp_path))
        msg = CollaborationMessage(
            sender=AgentRole.METRIC_ANALYSIS,
            receiver=AgentRole.VERIFICATION,
            message_type="task_dispatch",
            payload={"prev_result": {"data": {"latency": -50, "bottleneck_type": "memory_bound"}, "artifacts": ["trace.ncu"], "status": "success", "agent_role": "metric_analysis"}},
        )
        result = agent.run(msg)
        assert result.status == SubAgentStatus.REJECTED

    def test_reject_suspiciously_large_values(self, tmp_path):
        from src.application.subagents.verification import VerificationAgent
        agent = VerificationAgent(state_dir=str(tmp_path))
        msg = CollaborationMessage(
            sender=AgentRole.METRIC_ANALYSIS,
            receiver=AgentRole.VERIFICATION,
            message_type="task_dispatch",
            payload={"prev_result": {"data": {"cycles": 9e12, "bottleneck_type": "compute_bound"}, "artifacts": ["trace.ncu"], "status": "success", "agent_role": "metric_analysis"}},
        )
        result = agent.run(msg)
        assert result.status == SubAgentStatus.REJECTED

    def test_reject_unknown_bottleneck_type(self, tmp_path):
        from src.application.subagents.verification import VerificationAgent
        agent = VerificationAgent(state_dir=str(tmp_path))
        msg = CollaborationMessage(
            sender=AgentRole.METRIC_ANALYSIS,
            receiver=AgentRole.VERIFICATION,
            message_type="task_dispatch",
            payload={"prev_result": {"data": {"bottleneck_type": "magic_wand"}, "artifacts": [], "status": "success", "agent_role": "metric_analysis"}},
        )
        result = agent.run(msg)
        assert result.status == SubAgentStatus.REJECTED
        assert "Unknown bottleneck type" in str(result.data.get("concerns", ""))

    def test_p7_guard_context_must_be_empty(self, tmp_path):
        from src.application.subagents.verification import VerificationAgent
        from src.domain.subagent import P7ViolationError
        cm = ContextManager(max_tokens=4000)
        cm.add_entry("user", "leaked context", token_count=10)
        agent = VerificationAgent(state_dir=str(tmp_path))
        agent.context_manager = cm
        msg = CollaborationMessage(
            sender=AgentRole.METRIC_ANALYSIS,
            receiver=AgentRole.VERIFICATION,
            message_type="task_dispatch",
            payload={"prev_result": {}},
        )
        with pytest.raises(P7ViolationError, match="context must be empty"):
            agent.run(msg)

    def test_records_generation_fingerprint(self, tmp_path):
        from src.application.subagents.verification import VerificationAgent
        agent = VerificationAgent(state_dir=str(tmp_path))
        msg = CollaborationMessage(
            sender=AgentRole.METRIC_ANALYSIS,
            receiver=AgentRole.VERIFICATION,
            message_type="task_dispatch",
            payload={"prev_result": {"data": {"bottleneck_type": "compute_bound"}, "artifacts": ["out"], "status": "success", "agent_role": "metric_analysis"}, "prev_fingerprint": "abc123"},
        )
        result = agent.run(msg)
        assert result.data["generation_fingerprint"] == "abc123"

    def test_persists_result(self, tmp_path):
        from src.application.subagents.verification import VerificationAgent
        agent = VerificationAgent(state_dir=str(tmp_path))
        msg = CollaborationMessage(
            sender=AgentRole.METRIC_ANALYSIS,
            receiver=AgentRole.VERIFICATION,
            message_type="task_dispatch",
            payload={"prev_result": {"data": {"bottleneck_type": "memory_bound"}, "artifacts": ["trace"], "status": "success", "agent_role": "metric_analysis"}},
        )
        agent.run(msg)
        log_path = os.path.join(str(tmp_path), "agent_verification_log.jsonl")
        assert os.path.exists(log_path)
