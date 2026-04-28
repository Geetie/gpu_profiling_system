"""Model caller — calls LLM via OpenAI-compatible API.

Design based on mlsys-project-main reference implementation:
- Priority 1: Environment variables (API_KEY + BASE_URL + BASE_MODEL)
- Priority 2: api_config.json (development only)
- Priority 3: provider_manager (legacy, for backward compatibility)

The evaluation environment provides: API_KEY, BASE_URL, BASE_MODEL
"""
from __future__ import annotations

import json
import os
import time
import requests
from typing import Any, Dict, List, Optional


def _retry_request(url, headers, json_payload, max_retries=2, timeout=90):
    """Retry POST request with exponential backoff for transient errors."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url, headers=headers, json=json_payload, timeout=timeout
            )
            if response.status_code in (429, 500, 502, 503, 504):
                wait = 3 * (attempt + 1)
                print(f"[model_caller] HTTP {response.status_code}, retrying in {wait}s ({attempt+1}/{max_retries})")
                time.sleep(wait)
                last_exc = RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
                continue
            return response
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            wait = 3 * (attempt + 1)
            print(f"[model_caller] Connection error: {type(e).__name__}, retrying in {wait}s ({attempt+1}/{max_retries})")
            time.sleep(wait)
            last_exc = e
    print(f"[model_caller] All retries exhausted")
    raise last_exc if last_exc else RuntimeError("All retries exhausted")


def _get_api_config() -> tuple[str, str, str]:
    """Get API configuration from environment variables or config file.

    Returns:
        (api_key, base_url, model_name)

    Priority:
    1. Environment variables: API_KEY, BASE_URL, BASE_MODEL
    2. api_config.json file
    3. provider_manager (legacy)
    """
    env_api_key = os.getenv("API_KEY", "").strip()
    env_base_url = os.getenv("BASE_URL", "").strip()
    env_base_model = os.getenv("BASE_MODEL", "").strip()

    if env_api_key and env_base_model:
        effective_url = env_base_url or "https://api.openai.com/v1"
        print(f"[model_caller] Using env vars: model={env_base_model}, url={effective_url}")
        return env_api_key, effective_url, env_base_model

    try:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config", "api_config.json",
        )
        with open(config_path, "r") as f:
            config = json.load(f)
        env = config.get("env", {})
        api_key = env.get("ANTHROPIC_AUTH_TOKEN", "")
        base_url = env.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        model = env.get("ANTHROPIC_MODEL", "")
        if api_key and model:
            print(f"[model_caller] Using api_config.json: model={model}, url={base_url}")
            return api_key, base_url, model
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"[model_caller] api_config.json not available: {e}")

    try:
        from src.infrastructure.provider_manager import get_provider_manager
        pm = get_provider_manager()
        if pm.providers:
            for pn in ["longcat", "aliyun_bailian", "anthropic"]:
                provider = pm.providers.get(pn)
                if provider:
                    api_key = provider.get_api_key()
                    if api_key:
                        base_url = provider.endpoints.get("chat", "")
                        model = provider.get_model("default")
                        pm.current_provider = provider
                        print(f"[model_caller] Using provider: {pn}, model={model}")
                        return api_key, base_url, model
    except Exception as e:
        print(f"[model_caller] provider_manager not available: {e}")

    raise ValueError(
        "No API credentials found!\n"
        "Set environment variables: API_KEY + BASE_MODEL (+ optional BASE_URL)\n"
        "Or create config/api_config.json"
    )


def make_model_caller(model_name: str | None = None, max_tokens: int = 4096):
    """Create a model caller function compatible with AgentLoop.set_model_caller().

    Returns a callable: (messages: list[dict], tools: list[dict] | None = None) -> str
    """
    try:
        api_key, base_url, resolved_model = _get_api_config()
    except ValueError as e:
        print(f"[model_caller] FATAL: {e}")
        def _error_caller(messages, tools=None):
            raise RuntimeError(f"No API credentials: {e}")
        return _error_caller

    effective_model = model_name or resolved_model

    if "anthropic.com" in base_url:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        print(f"[model_caller] Configured: Anthropic API, model={effective_model}")
    else:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        print(f"[model_caller] Configured: OpenAI-compatible API, model={effective_model}")

    def _caller(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> str:
        if "anthropic.com" in base_url:
            return _call_anthropic(base_url, headers, effective_model, messages, max_tokens, tools)
        else:
            return _call_openai_compatible(base_url, headers, effective_model, messages, max_tokens, tools)

    return _caller


def _is_reasoning_model(model: str) -> bool:
    """Check if the model is a reasoning model that doesn't support temperature.

    GPT-5 family and o-series models reject explicit temperature values
    with HTTP 400 errors. The temperature parameter must be omitted entirely.
    See: https://github.com/dotCMS/core/issues/34814
    """
    model_lower = model.lower()
    reasoning_prefixes = ("gpt-5", "o1", "o3", "o4")
    return any(model_lower.startswith(p) for p in reasoning_prefixes)


def _call_openai_compatible(
    base_url: str,
    headers: Dict[str, str],
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Call OpenAI-compatible API endpoint."""
    cleaned_messages = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")
        if content is None:
            content = ""
        else:
            content = str(content)
        if not role:
            continue
        cleaned_messages.append({"role": role, "content": content})

    payload = {
        "model": model,
        "messages": cleaned_messages,
        "max_completion_tokens": max_tokens,
    }

    if not _is_reasoning_model(model):
        payload["temperature"] = 0.1
        payload["stream"] = False

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    chat_url = base_url if base_url.endswith("/chat/completions") else f"{base_url.rstrip('/')}/chat/completions"
    print(f"[model_caller] POST {chat_url} model={model} msgs={len(cleaned_messages)} tools={len(tools) if tools else 0}")

    response = _retry_request(chat_url, headers=headers, json_payload=payload, timeout=120)

    if response.status_code != 200:
        print(f"[model_caller] API error: {response.status_code} - {response.text[:300]}")
        raise RuntimeError(f"API call failed: {response.status_code} - {response.text[:500]}")

    result = response.json()
    choice = result["choices"][0]

    tool_calls = choice.get("message", {}).get("tool_calls")
    if tool_calls:
        tc = tool_calls[0]
        if tc.get("type") == "function":
            func = tc["function"]
            try:
                args = json.loads(func["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = {}
            print(f"[model_caller] Got tool_call: {func['name']}")
            return json.dumps({"tool": func["name"], "args": args})

    content = choice.get("message", {}).get("content", "")
    print(f"[model_caller] Got text response: {len(content)} chars")
    return content


def _call_anthropic(
    base_url: str,
    headers: Dict[str, str],
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Call Anthropic-compatible API endpoint."""
    anthropic_messages = []
    system_message = None

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if content is None:
            content = ""
        if role == "system":
            system_message = str(content)
        elif role == "user":
            anthropic_messages.append({"role": "user", "content": str(content)})
        elif role == "assistant":
            anthropic_messages.append({"role": "assistant", "content": str(content)})

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "messages": anthropic_messages,
    }
    if system_message:
        payload["system"] = system_message
    if tools:
        payload["tools"] = [
            {"name": t["function"]["name"]}
            for t in tools if t.get("type") == "function" and "function" in t
        ]

    response = _retry_request(base_url, headers=headers, json_payload=payload, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(f"API call failed: {response.status_code} - {response.text[:500]}")

    result = response.json()
    for block in result.get("content", []):
        if block.get("type") == "tool_use":
            return json.dumps({
                "tool": block["name"],
                "args": block.get("input", {}),
            })
    for block in result.get("content", []):
        if block.get("type") == "text":
            return block["text"]
    return ""


def make_subagent_model_caller(model_name: str | None = None, max_tokens: int = 4096):
    """Create a model caller for subagents (same API, same config)."""
    return make_model_caller(model_name=model_name, max_tokens=max_tokens)
