"""Model caller — calls LLM via Anthropic-compatible API.

Reads config from config/api_config.json.
Supports both AgentLoop (single-turn) and SubAgent (multi-turn) modes.
"""
from __future__ import annotations

import json
import os
from typing import Any


def load_config() -> dict[str, Any]:
    """Load API configuration from config/api_config.json."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "config", "api_config.json",
    )
    with open(config_path, "r") as f:
        return json.load(f)


def check_key(api_key: str) -> None:
    if not api_key or api_key.startswith("在此填入"):
        raise ValueError(
            "API Key 未配置。\n"
            "请编辑 config/api_config.json，将 ANTHROPIC_AUTH_TOKEN 设置为你的 API Key。\n"
            "格式示例: \"sk-xxxxxxxxxxxxxxxxxxxxxxxx\""
        )


def make_model_caller(model_name: str | None = None, max_tokens: int = 4096):
    """Create a model caller function compatible with AgentLoop.set_model_caller().

    Returns a callable: (messages: list[dict]) -> str

    Usage:
        caller = make_model_caller()
        agent_loop.set_model_caller(caller)
    """
    config = load_config()
    env = config["env"]

    api_key = env.get("ANTHROPIC_AUTH_TOKEN", "")
    check_key(api_key)

    base_url = env.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    model = model_name or env.get("ANTHROPIC_MODEL", "qwen3.6-plus")

    def _caller(messages: list[dict[str, Any]]) -> str:
        """Call the LLM API and return the assistant's text response."""
        # Build Anthropic-compatible request
        system_msg = ""
        user_messages = []
        for m in messages:
            if m.get("role") == "system":
                system_msg = m.get("content", "")
            else:
                user_messages.append(m)

        # Use requests to call the API (no external dependency)
        import requests

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        body = {
            "model": model,
            "messages": user_messages,
            "max_tokens": max_tokens,
            "temperature": 0.1,  # Low temperature for consistent tool calls
        }
        if system_msg:
            body["system"] = system_msg

        response = requests.post(
            f"{base_url}/v1/messages",
            headers=headers,
            json=body,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        # Extract text content
        for block in data.get("content", []):
            if block.get("type") == "text":
                return block["text"]

        return ""

    return _caller


def make_subagent_model_caller(model_name: str | None = None, max_tokens: int = 4096):
    """Create a model caller for subagents (same API, same config)."""
    return make_model_caller(model_name=model_name, max_tokens=max_tokens)
