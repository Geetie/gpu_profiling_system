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


class TestCodeGenAgent:
    def test_generate_latency_kernel(self, tmp_path):
        from src.application.subagents.codegen import CodeGenAgent
        sandbox = LocalSandbox(sandbox_root=str(tmp_path))
        agent = CodeGenAgent(state_dir=str(tmp_path), sandbox=sandbox)
        agent.set_model_caller(lambda msgs: "__global__ void latency_kernel() { clock(); }")
        msg = CollaborationMessage(
            sender=AgentRole.PLANNER,
            receiver=AgentRole.CODE_GEN,
            message_type="task_dispatch",
            payload={"task": {"target": "dram_latency_cycles", "category": "latency_measurement", "method": "pointer-chasing"}},
        )
        result = agent.run(msg)
        assert result.is_failed()

    def test_no_model_caller_raises_error(self, tmp_path):
        from src.application.subagents.codegen import CodeGenAgent
        agent = CodeGenAgent(state_dir=str(tmp_path))
        import pytest
        with pytest.raises(RuntimeError, match="No LLM configured"):
            source = agent._generate_kernel("dram_latency_cycles", "latency_measurement", "pointer-chasing")

    def test_llm_call_failure_raises_error(self, tmp_path):
        from src.application.subagents.codegen import CodeGenAgent
        agent = CodeGenAgent(state_dir=str(tmp_path))
        def failing_caller(msgs):
            raise RuntimeError("API error: 500")
        agent.set_model_caller(failing_caller)
        import pytest
        with pytest.raises(RuntimeError, match="LLM code generation failed"):
            agent._generate_kernel("dram_latency_cycles", "latency_measurement", "pointer-chasing")

    def test_model_caller_used_when_available(self, tmp_path):
        from src.application.subagents.codegen import CodeGenAgent
        agent = CodeGenAgent(state_dir=str(tmp_path))
        agent.set_model_caller(lambda msgs: "// custom model-generated kernel")
        source = agent._generate_kernel("test", "latency_measurement", "test")
        assert "custom model-generated kernel" in source


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
        assert agent.identify_bottleneck({"a": 10, "b": 100, "c": 500}) == "cache_capacity"
        assert agent.identify_bottleneck({"l2_cache_miss": 500}) == "cache_capacity"

    def test_identify_unknown(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        assert agent.identify_bottleneck({"name": "test"}) == "unknown"

    def test_confidence_scaling(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        assert agent._assess_confidence({}) == 0.0
        assert agent._assess_confidence({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}) == 1.0
        assert agent._assess_confidence({"a": 1, "b": 2, "name": "test"}) == 0.4

    def test_no_raw_output_no_binary_fails(self, tmp_path):
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
        assert "No binary paths or raw output" in (result.error or "")

    def test_full_analysis_flow(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        msg = CollaborationMessage(
            sender=AgentRole.CODE_GEN,
            receiver=AgentRole.METRIC_ANALYSIS,
            message_type="task_dispatch",
            payload={
                "prev_result": {"data": {"raw_output": "DRAM Latency: 320\nL2 Hit Rate: 85.5\n"}},
                "target_spec": {"target": "dram_latency_cycles"},
            },
        )
        result = agent.run(msg)
        assert result.is_success()
        assert result.data["bottleneck_type"] == "latency_bound"
        assert result.data["bottleneck_sub_type"] == "dram"

    def test_roofline_analysis_compute_bound(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        metrics = {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 85.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 30.0,
            "sm__pipe_tensor_op_hmma_cycle_active.avg.pct_of_peak_sustained_active": 90.0,
        }
        result = agent.analyze_roofline(metrics, "dram_bandwidth_gbps")
        assert result["bottleneck_type"] == "compute_bound"
        assert result["bottleneck_sub_type"] == "tensor_core"
        assert "evidence" in result
        assert "recommendations" in result

    def test_roofline_analysis_memory_bound(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        metrics = {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 30.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 85.0,
            "dram__throughput.avg.pct_of_peak_sustained_elapsed": 90.0,
            "l2__throughput.avg.pct_of_peak_sustained_elapsed": 40.0,
        }
        result = agent.analyze_roofline(metrics, "dram_bandwidth_gbps")
        assert result["bottleneck_type"] == "memory_bound"
        assert result["bottleneck_sub_type"] == "dram"
        assert len(result["recommendations"]) > 0

    def test_roofline_analysis_memory_bound_l2(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        metrics = {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 30.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 85.0,
            "dram__throughput.avg.pct_of_peak_sustained_elapsed": 40.0,
            "l2__throughput.avg.pct_of_peak_sustained_elapsed": 85.0,
        }
        result = agent.analyze_roofline(metrics, "l2_latency_cycles")
        assert result["bottleneck_type"] == "memory_bound"
        assert result["bottleneck_sub_type"] == "l2"

    def test_roofline_analysis_balanced(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        metrics = {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 60.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 65.0,
        }
        result = agent.analyze_roofline(metrics, "dram_bandwidth_gbps")
        assert result["bottleneck_type"] == "balanced"

    def test_roofline_target_based_classification(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        result = agent.analyze_roofline({}, "dram_latency_cycles")
        assert result["bottleneck_type"] == "latency_bound"
        assert result["bottleneck_sub_type"] == "dram"

        result = agent.analyze_roofline({}, "l2_cache_size_mb")
        assert result["bottleneck_type"] == "cache_capacity"
        assert result["bottleneck_sub_type"] == "l2"

        result = agent.analyze_roofline({}, "bank_conflict_penalty_ratio")
        assert result["bottleneck_type"] == "latency_bound"
        assert result["bottleneck_sub_type"] == "bank_conflict"

    def test_select_metrics_for_target(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        latency_metrics = agent._select_metrics_for_target("dram_latency_cycles")
        assert "dram__throughput.avg.pct_of_peak_sustained_elapsed" in latency_metrics
        bandwidth_metrics = agent._select_metrics_for_target("dram_bandwidth_gbps")
        assert "dram__bytes.sum" in bandwidth_metrics
        default_metrics = agent._select_metrics_for_target("unknown_target")
        assert "sm__throughput.avg.pct_of_peak_sustained_elapsed" in default_metrics

    def test_generate_recommendations_memory_bound(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        recs = agent._generate_recommendations("memory_bound", "dram")
        assert len(recs) > 0
        assert any("shared memory" in r.lower() or "tiling" in r.lower() for r in recs)

        recs_l2 = agent._generate_recommendations("memory_bound", "l2")
        assert len(recs_l2) > 0
        assert any("L2" in r or "spatial locality" in r for r in recs_l2)

        recs_l1 = agent._generate_recommendations("memory_bound", "l1")
        assert len(recs_l1) > 0
        assert any("shared memory" in r.lower() or "registers" in r.lower() for r in recs_l1)

    def test_generate_recommendations_compute_bound(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        recs_tc = agent._generate_recommendations("compute_bound", "tensor_core")
        assert len(recs_tc) > 0
        assert any("Tensor Core" in r or "WMMA" in r or "CUTLASS" in r for r in recs_tc)

        recs_fp32 = agent._generate_recommendations("compute_bound", "fp32")
        assert len(recs_fp32) > 0
        assert any("FP16" in r or "mixed precision" in r or "register" in r.lower() for r in recs_fp32)

        recs_fp64 = agent._generate_recommendations("compute_bound", "fp64")
        assert len(recs_fp64) > 0
        assert any("FP64" in r or "FP32" in r for r in recs_fp64)

    def test_generate_recommendations_latency_bound(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        recs = agent._generate_recommendations("latency_bound", None)
        assert len(recs) > 0
        assert any("latency" in r.lower() or "occupancy" in r.lower() for r in recs)

    def test_generate_recommendations_cache_capacity(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        recs = agent._generate_recommendations("cache_capacity", None)
        assert len(recs) > 0
        assert any("cache" in r.lower() or "tiling" in r.lower() or "blocking" in r.lower() for r in recs)

    def test_output_format_has_all_fields(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        msg = CollaborationMessage(
            sender=AgentRole.CODE_GEN,
            receiver=AgentRole.METRIC_ANALYSIS,
            message_type="task_dispatch",
            payload={
                "prev_result": {"data": {"raw_output": "DRAM Latency: 320\n"}},
                "target_spec": {"target": "dram_latency_cycles"},
            },
        )
        result = agent.run(msg)
        assert result.is_success()
        data = result.data
        assert "bottleneck_type" in data
        assert "bottleneck_sub_type" in data
        assert "parsed_metrics" in data
        assert "evidence" in data
        assert "recommendations" in data
        assert "suggested_fixes" in data
        assert "confidence" in data
        assert "confidence_reason" in data
        assert "analysis_method" in data

    def test_confidence_detailed_with_ncu(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        metrics = {"sm__throughput": 85.0, "dram__throughput": 30.0}
        conf, reason = agent._assess_confidence_detailed(metrics, ncu_available=True)
        assert conf >= 0.7
        assert "ncu" in reason.lower()

    def test_confidence_detailed_without_ncu(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        metrics = {"latency": 320}
        conf, reason = agent._assess_confidence_detailed(metrics, ncu_available=False)
        assert conf <= 0.5
        assert "printf" in reason.lower() or "ncu" in reason.lower()

    def test_confidence_detailed_with_cross_validation(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        metrics = {"sm__throughput": 85.0, "dram__throughput": 30.0}
        cv_agree = {"agreement": True, "agreements_count": 3, "discrepancies_count": 0}
        conf, reason = agent._assess_confidence_detailed(metrics, ncu_available=True, cross_validation=cv_agree)
        assert conf >= 0.8
        assert "cross-validation" in reason.lower() or "consistent" in reason.lower()

        cv_disagree = {"agreement": False, "agreements_count": 1, "discrepancies_count": 2}
        conf2, reason2 = agent._assess_confidence_detailed(metrics, ncu_available=True, cross_validation=cv_disagree)
        assert conf2 < conf
        assert "discrepan" in reason2.lower()

    def test_cross_validate_agreement(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        codegen = {"dram_latency_cycles": 320}
        ncu = {"dram__throughput.avg.pct_of_peak_sustained_elapsed": 85.0}
        result = agent.cross_validate(codegen, ncu, "dram_latency_cycles")
        assert result is not None

    def test_cross_validate_empty(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        assert agent.cross_validate({}, {"a": 1}, "test") is None
        assert agent.cross_validate({"a": 1}, {}, "test") is None

    def test_extract_binary_paths_from_tool_results(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        prev_data = {"binary_path": "/tmp/bench1"}
        tool_results = [
            {"tool": "compile_cuda", "success": True, "binary_path": "/tmp/bench2"},
            {"tool": "execute_binary", "executable": "/tmp/bench3"},
        ]
        paths = agent._extract_binary_paths(prev_data, tool_results)
        assert "/tmp/bench1" in paths
        assert "/tmp/bench2" in paths
        assert "/tmp/bench3" in paths
        assert len(paths) == 3

    def test_extract_binary_paths_dedup(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        prev_data = {"binary_path": "/tmp/bench1"}
        tool_results = [
            {"tool": "compile_cuda", "success": True, "binary_path": "/tmp/bench1"},
        ]
        paths = agent._extract_binary_paths(prev_data, tool_results)
        assert len(paths) == 1

    def test_extract_metric_value(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        metrics = {"sm__throughput.avg.pct_of_peak_sustained_elapsed": 85.0}
        val = agent._extract_metric_value(metrics, ["sm__throughput.avg.pct_of_peak_sustained_elapsed"])
        assert val == 85.0

        val2 = agent._extract_metric_value(metrics, ["nonexistent_key"])
        assert val2 == 0.0

        val3 = agent._extract_metric_value(
            {"SM Throughput": "85.5 %"},
            ["SM Throughput"]
        )
        assert val3 == 85.5

    def test_identify_compute_unit(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        assert agent._identify_compute_unit(90.0, {}) == "tensor_core"
        assert agent._identify_compute_unit(50.0, {
            "sm__pipe_fma_cycle_active.avg.pct_of_peak_sustained_active": 70.0,
        }) == "fp64"
        assert agent._identify_compute_unit(50.0, {}) == "fp32"

    def test_identify_memory_level(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        assert agent._identify_memory_level(90.0, 40.0, {}) == "dram"
        assert agent._identify_memory_level(40.0, 90.0, {}) == "l2"
        assert agent._identify_memory_level(40.0, 50.0, {}) == "l2"

    def test_l1_hit_rate_computation(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        metrics = {"l1tex__t_sectors.sum": 1000, "l1tex__t_sectors_hit.sum": 400}
        hit_rate = agent._compute_l1_hit_rate(metrics)
        assert hit_rate == 40.0

        assert agent._compute_l1_hit_rate({"other": 1}) is None

    def test_collect_evidence(self, tmp_path):
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        agent = MetricAnalysisAgent(state_dir=str(tmp_path))
        metrics = {
            "dram__throughput.avg.pct_of_peak_sustained_elapsed": 85.0,
            "l2__throughput.avg.pct_of_peak_sustained_elapsed": 40.0,
        }
        util_data = {"compute_utilization": 30.0, "memory_utilization": 85.0}
        evidence = agent._collect_evidence(metrics, "memory_bound", "dram", util_data)
        assert evidence["bottleneck_type"] == "memory_bound"
        assert evidence["bottleneck_sub_type"] == "dram"
        assert any("dram__throughput" in k for k in evidence["key_metrics"])
        assert "analysis" in evidence


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
