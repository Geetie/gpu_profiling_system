"""Model caller — calls LLM via multi-provider API system.

Supports multiple LLM providers:
- Aliyun Bailian (DASHSCOPE)
- Anthropic Claude
- LongCat API

Uses provider_manager for unified configuration management,
with fallback to direct api_config.json reading for compatibility.
"""
from __future__ import annotations

import json
import os
import time
import requests
from typing import Any, Dict, List, Optional


def _retry_request(url, headers, json_payload, max_retries=2, timeout=60):
    """Retry POST request with exponential backoff for transient errors."""
    import urllib3
    last_exc = None
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url, headers=headers, json=json_payload, timeout=timeout
            )
            if response.status_code in (429, 500, 502, 503, 504):
                wait = min(2 ** attempt * 3, 30)  # 减少等待时间
                print(f"[model_caller] HTTP {response.status_code}, retrying in {wait}s ({attempt+1}/{max_retries})")
                time.sleep(wait)
                last_exc = RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
                continue
            return response
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            wait = min(2 ** attempt * 3, 30)  # 减少等待时间
            print(f"[model_caller] Connection error: {type(e).__name__}, retrying in {wait}s ({attempt+1}/{max_retries})")
            time.sleep(wait)
            last_exc = e
    raise last_exc if last_exc else RuntimeError("All retries exhausted")


def load_config() -> dict[str, Any]:
    """Load API configuration from config/api_config.json.

    Fallback for environments without provider_manager (e.g. Kaggle).
    """
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "config", "api_config.json",
    )
    with open(config_path, "r") as f:
        return json.load(f)


def check_key(api_key: str, provider_name: str = "API") -> None:
    """Check API key validity."""
    if not api_key:
        raise ValueError(
            f"{provider_name} API Key not configured.\n"
            f"Please set the corresponding environment variable or check provider config."
        )

    # Check common placeholder patterns
    # Note: "ak_" removed — LongCat/Aliyun real API keys start with "ak_"
    placeholders = ["在此填入", "YOUR_API_KEY", "sk-<", "在此处填入", "mock"]
    if any(api_key.startswith(ph) for ph in placeholders):
        raise ValueError(
            f"{provider_name} API Key is a placeholder.\n"
            f"Please configure a valid API key."
        )


def make_model_caller(model_name: str | None = None, max_tokens: int = 4096):
    """Create a model caller function compatible with AgentLoop.set_model_caller().

    Returns a callable: (messages: list[dict], tools: list[dict] | None = None) -> str

    Strategy:
    1. Try provider_manager (multi-provider support)
    2. Fall back to direct api_config.json reading (Kaggle compatibility)
    """
    # Try provider_manager first
    provider_manager = None
    try:
        from src.infrastructure.provider_manager import get_provider_manager
        provider_manager = get_provider_manager()
        if provider_manager.get_provider() is None:
            provider_manager = None
    except Exception:
        provider_manager = None

    # Fall back to api_config.json
    use_fallback = provider_manager is None

    def _caller(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """Actual model call function."""
        if use_fallback:
            return _caller_from_config(messages, model_name, max_tokens, tools)
        else:
            return _caller_from_provider(messages, model_name, max_tokens, provider_manager, tools)

    return _caller


def _caller_from_config(
    messages: List[Dict[str, Any]],
    model_name: str | None,
    max_tokens: int,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Fallback: read config directly from api_config.json."""
    config = load_config()
    env = config["env"]

    api_key = env.get("ANTHROPIC_AUTH_TOKEN", "")
    check_key(api_key, "API")

    base_url = env.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    model = model_name or env.get("ANTHROPIC_MODEL", "qwen3.6-plus")

    return _call_api(base_url, api_key, model, messages, max_tokens, tools)


def _caller_from_provider(
    messages: List[Dict[str, Any]],
    model_name: str | None,
    max_tokens: int,
    provider_manager,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Use provider_manager for multi-provider support with failover."""
    # 尝试所有可用的提供商
    provider_priority = ["longcat", "aliyun_bailian", "anthropic"]
    
    for provider_name in provider_priority:
        try:
            # 手动设置提供商
            if provider_manager.set_provider(provider_name):
                provider = provider_manager.get_provider()
                if not provider:
                    continue

                unified_config = provider_manager.create_unified_config()
                if not unified_config:
                    continue

                env = unified_config["env"]
                headers = unified_config["headers"]

                api_key = env.get("ANTHROPIC_AUTH_TOKEN", "")
                try:
                    check_key(api_key, unified_config["provider_name"])
                except ValueError:
                    continue

                base_url = env.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
                model = model_name or env.get("ANTHROPIC_MODEL", provider.get_model("default"))

                print(f"[model_caller] 使用提供商: {provider_name}")
                
                if provider.provider == "anthropic":
                    result = _call_anthropic(base_url, headers, model, messages, max_tokens, tools)
                elif provider.provider in ("aliyun_bailian", "longcat"):
                    result = _call_openai_compatible(base_url, headers, model, messages, max_tokens, tools)
                else:
                    result = _call_api(base_url, api_key, model, messages, max_tokens, tools)
                
                # 检查结果是否有效
                if result.strip():
                    return result
                else:
                    print(f"[model_caller] 提供商 {provider_name} 返回空内容，尝试下一个...")
                    
        except Exception as e:
            print(f"[model_caller] 提供商 {provider_name} 失败: {str(e)}")
            continue
    
    # 所有提供商都失败
    raise ValueError("All LLM providers failed")


def _call_api(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Call API — detect endpoint type from base_url."""
    # Anthropic native API
    if "anthropic.com" in base_url:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        return _call_anthropic(base_url, headers, model, messages, max_tokens, tools)
    else:
        # OpenAI-compatible API
        headers = {
            "Authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }
        return _call_openai_compatible(base_url, headers, model, messages, max_tokens, tools)


def _call_anthropic(
    base_url: str,
    headers: Dict[str, str],
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Call Anthropic-compatible API endpoint."""
    # Convert messages: extract system message
    anthropic_messages = []
    system_message = None

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if content is None:
            content = ""
        if role == "system":
            system_message = content
        elif role == "user":
            anthropic_messages.append({"role": "user", "content": str(content)})
        elif role == "assistant":
            anthropic_messages.append({"role": "assistant", "content": str(content)})

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": anthropic_messages,
    }
    if system_message:
        payload["system"] = system_message
    if tools:
        # Convert {"type": "function", "function": {"name": ...}} to Anthropic format
        payload["tools"] = [
            {"name": t["function"]["name"]}
            for t in tools if t.get("type") == "function" and "function" in t
        ]

    response = _retry_request(base_url, headers=headers, json_payload=payload, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(f"API call failed: {response.status_code} - {response.text[:500]}")

    result = response.json()
    # Check for tool_calls first (Anthropic function calling)
    for block in result.get("content", []):
        if block.get("type") == "tool_use":
            return json.dumps({
                "tool": block["name"],
                "args": block.get("input", {}),
            })
    # Fallback to text
    for block in result.get("content", []):
        if block.get("type") == "text":
            return block["text"]
    return ""


def _call_openai_compatible(
    base_url: str,
    headers: Dict[str, str],
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Call OpenAI-compatible API endpoint."""
    # LongCat/DashScope apply_chat_template crashes on None content.
    # Sanitize all messages defensively before sending.
    cleaned_messages = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role and content is not None:
            cleaned_messages.append({"role": role, "content": str(content)})
        elif role:
            cleaned_messages.append({"role": role, "content": ""})
        # Skip messages with no role — malformed, safer to omit

    payload = {
        "model": model,
        "messages": cleaned_messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    response = _retry_request(base_url, headers=headers, json_payload=payload, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(f"API call failed: {response.status_code} - {response.text[:500]}")

    result = response.json()
    choice = result["choices"][0]

    # Check for tool_calls (OpenAI function calling)
    tool_calls = choice.get("message", {}).get("tool_calls")
    if tool_calls:
        # Return the first tool call in AgentLoop's expected format
        tc = tool_calls[0]
        if tc.get("type") == "function":
            func = tc["function"]
            try:
                args = json.loads(func["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = {}
            return json.dumps({"tool": func["name"], "args": args})

    # Fallback to text content
    return choice.get("message", {}).get("content", "")


def make_subagent_model_caller(model_name: str | None = None, max_tokens: int = 4096):
    """Create a model caller for subagents (same API, same config)."""
    return make_model_caller(model_name=model_name, max_tokens=max_tokens)
