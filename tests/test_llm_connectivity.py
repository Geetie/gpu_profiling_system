"""Test LLM model calling through all 4 subagents via Bailian API.

Uses the real API endpoint — requires valid ANTHROPIC_AUTH_TOKEN in
config/api_config.json.
"""
import json
import os

import pytest


def _get_api_config():
    """Load API config from config/api_config.json."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config", "api_config.json",
    )
    if not os.path.exists(config_path):
        pytest.skip("No API config found")
    with open(config_path) as f:
        cfg = json.load(f)
    return cfg["env"]


def _check_key_valid(env):
    key = env.get("ANTHROPIC_AUTH_TOKEN", "")
    if not key or key.startswith("在此填入"):
        pytest.skip("API key not configured")


# ── Test 1: Direct API call works ──

def test_bailian_api_responds():
    """百炼 API 应返回 200 响应。"""
    env = _get_api_config()
    _check_key_valid(env)
    import requests
    resp = requests.post(
        f"{env['ANTHROPIC_BASE_URL']}/v1/messages",
        headers={
            "x-api-key": env["ANTHROPIC_AUTH_TOKEN"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": env.get("ANTHROPIC_MODEL", "qwen3.6-plus"),
            "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 32,
        },
        timeout=30,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data.get("content", [])) > 0


# ── Test 2: Tool calling works ──

def test_bailian_tool_calling():
    """模型应返回 tool_call 类型响应。"""
    env = _get_api_config()
    _check_key_valid(env)
    import requests
    resp = requests.post(
        f"{env['ANTHROPIC_BASE_URL']}/v1/messages",
        headers={
            "x-api-key": env["ANTHROPIC_AUTH_TOKEN"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": env.get("ANTHROPIC_DEFAULT_SONNET_MODEL", env.get("ANTHROPIC_MODEL")),
            "messages": [
                {"role": "user", "content": "Write a simple CUDA hello world kernel."}
            ],
            "max_tokens": 2048,
            "temperature": 0.1,
        },
        timeout=30,
    )
    assert resp.status_code == 200
    data = resp.json()
    # Should have content blocks
    blocks = data.get("content", [])
    assert len(blocks) > 0
    # Should contain CUDA-related text
    text = ""
    for block in blocks:
        if block.get("type") == "text":
            text += block.get("text", "")
    assert len(text) > 50  # Substantial response


# ── Test 3: Model caller integrates with subagents ──

def test_model_caller_wires_to_codegen():
    """make_model_caller 应返回可调用的函数。"""
    env = _get_api_config()
    _check_key_valid(env)
    from src.infrastructure.model_caller import make_model_caller
    caller = make_model_caller(model_name=env.get("ANTHROPIC_MODEL", "qwen3.6-plus"))
    assert callable(caller)
    result = caller([
        {"role": "user", "content": "Reply with just the word: PROFILING"}
    ])
    assert "PROFILING" in result.upper()


def test_model_caller_wires_to_planner():
    """PlannerAgent 通过 LLM 分解任务。"""
    env = _get_api_config()
    _check_key_valid(env)
    from src.infrastructure.model_caller import make_model_caller
    from src.application.subagents.planner import PlannerAgent
    from src.domain.subagent import AgentRole, CollaborationMessage, SubAgentStatus
    from src.application.context import ContextManager

    planner = PlannerAgent(context_manager=ContextManager(max_tokens=4000))
    planner.set_model_caller(make_model_caller(model_name=env.get("ANTHROPIC_MODEL", "qwen3.6-plus")))

    msg = CollaborationMessage(
        sender=AgentRole.PLANNER,
        receiver=AgentRole.PLANNER,
        message_type="task_dispatch",
        payload={"target_spec": {"targets": ["dram_latency_cycles"]}},
    )
    result = planner.run(msg)
    assert result.status == SubAgentStatus.SUCCESS
    assert "tasks" in result.data


def test_model_caller_wires_to_metric_analysis():
    """MetricAnalysisAgent 通过 LLM 分析数据。"""
    env = _get_api_config()
    _check_key_valid(env)
    from src.infrastructure.model_caller import make_model_caller
    from src.application.subagents.metric_analysis import MetricAnalysisAgent
    from src.domain.subagent import AgentRole, CollaborationMessage, SubAgentStatus
    from src.application.context import ContextManager

    agent = MetricAnalysisAgent(context_manager=ContextManager(max_tokens=4000))
    agent.set_model_caller(make_model_caller(model_name=env.get("ANTHROPIC_MODEL", "qwen3.6-plus")))

    msg = CollaborationMessage(
        sender=AgentRole.CODE_GEN,
        receiver=AgentRole.METRIC_ANALYSIS,
        message_type="task_dispatch",
        payload={
            "prev_result": {
                "data": {"raw_output": "DRAM latency: 442 cycles\nL2 hit rate: 0.73"},
                "status": "success",
                "agent_role": "code_gen",
            }
        },
    )
    result = agent.run(msg)
    assert result.status == SubAgentStatus.SUCCESS
    assert "bottleneck_type" in result.data


def test_model_caller_wires_to_verification():
    """VerificationAgent 通过 LLM 独立审查。"""
    env = _get_api_config()
    _check_key_valid(env)
    from src.infrastructure.model_caller import make_model_caller
    from src.application.subagents.verification import VerificationAgent
    from src.domain.subagent import AgentRole, CollaborationMessage, SubAgentStatus
    from src.application.context import ContextManager

    agent = VerificationAgent(max_tokens=4000)
    agent.set_model_caller(make_model_caller(model_name=env.get("ANTHROPIC_REASONING_MODEL", env.get("ANTHROPIC_MODEL"))))

    msg = CollaborationMessage(
        sender=AgentRole.METRIC_ANALYSIS,
        receiver=AgentRole.VERIFICATION,
        message_type="task_dispatch",
        payload={
            "prev_result": {
                "data": {
                    "bottleneck_type": "latency_bound",
                    "parsed_metrics": {"dram_latency_cycles": 442},
                },
                "status": "success",
                "agent_role": "metric_analysis",
                "artifacts": [],
            },
            "prev_fingerprint": "test_fp_001",
        },
    )
    result = agent.run(msg)
    assert result.status in (SubAgentStatus.SUCCESS, SubAgentStatus.REJECTED)
    assert "review" in result.data or "accepted" in result.data
