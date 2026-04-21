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
    print(f"[model_caller] Sending request to: {url}")
    for attempt in range(max_retries):
        try:
            print(f"[model_caller] Attempt {attempt+1}/{max_retries}")
            response = requests.post(
                url, headers=headers, json=json_payload, timeout=timeout
            )
            print(f"[model_caller] Response status: {response.status_code}")
            if response.status_code in (429, 500, 502, 503, 504):
                wait = 5 + attempt  # 5s, 6s
                print(f"[model_caller] HTTP {response.status_code}, retrying in {wait}s ({attempt+1}/{max_retries})")
                time.sleep(wait)
                last_exc = RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
                continue
            return response
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            wait = 5 + attempt  # 5s, 6s
            print(f"[model_caller] Connection error: {type(e).__name__}, retrying in {wait}s ({attempt+1}/{max_retries})")
            time.sleep(wait)
            last_exc = e
    print(f"[model_caller] All retries exhausted")
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

    IMPORTANT: Never call provider_manager.get_provider() here because it
    triggers detect_provider() which does a health-check HTTP request that
    can block for minutes.  Instead, check current_provider directly.
    """
    provider_manager = None
    try:
        from src.infrastructure.provider_manager import get_provider_manager
        provider_manager = get_provider_manager()
        # 直接检查 current_provider，不触发 detect_provider() / 健康检查
        if provider_manager.current_provider is None and not provider_manager.providers:
            provider_manager = None
    except Exception:
        provider_manager = None

    use_fallback = provider_manager is None

    if provider_manager and provider_manager.current_provider:
        print(f"[model_caller] Using provider: {provider_manager.current_provider.provider}")
    elif provider_manager and provider_manager.providers:
        print(f"[model_caller] Provider manager loaded but no current_provider set yet")
    else:
        print(f"[model_caller] No provider_manager, using api_config.json fallback")

    def _caller(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> str:
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
    """Fallback: read config directly from api_config.json with OpenAI env var support.

    COMPLIANCE (Requirement #2 & #3):
    - Priority 1: Read from OpenAI-compatible environment variables (for GPT-5.4 evaluation)
    - Priority 2: Fallback to api_config.json (for development/testing only)
    """
    import os

    # 🆕 PRIORITY 1: OpenAI-compatible environment variables (GPT-5.4 evaluation requirement)
    env_api_key = os.getenv("API_KEY", "").strip()
    env_base_url = os.getenv("BASE_URL", "").strip()
    env_base_model = os.getenv("BASE_MODEL", "").strip()

    if env_api_key and env_base_model:
        print(f"[model_caller] ✅ Using OpenAI-compatible environment variables")
        print(f"[model_caller]   Model: {env_base_model}")
        print(f"[model_caller]   Base URL: {env_base_url or 'https://api.openai.com/v1'}")

        effective_url = env_base_url or "https://api.openai.com/v1"
        headers = {
            "Authorization": f"Bearer {env_api_key}",
            "Content-Type": "application/json",
        }

        return _call_openai_compatible(
            base_url=effective_url,
            headers=headers,
            model=env_base_model,
            messages=messages,
            max_tokens=max_tokens,
            tools=tools
        )

    # PRIORITY 2: Fallback to api_config.json (development environment only)
    try:
        config = load_config()
        env = config["env"]

        api_key = env.get("ANTHROPIC_AUTH_TOKEN", "")
        check_key(api_key, "API")

        base_url = env.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        model = model_name or env.get("ANTHROPIC_MODEL", "qwen3.6-plus")

        return _call_api(base_url, api_key, model, messages, max_tokens, tools)

    except (FileNotFoundError, ValueError) as e:
        print(f"[model_caller] ⚠️ Config file not found or invalid: {e}")
        if not env_api_key:
            raise ValueError(
                "No API credentials found!\n\n"
                "Please set one of the following:\n"
                "1. Environment variables: API_KEY + BASE_MODEL (recommended for submission)\n"
                "2. Config file: config/api_config.json (for development only)\n\n"
                "For evaluation submission, use:\n"
                "  export API_KEY=<your-key>\n"
                "  export BASE_MODEL=gpt-5.4\n"
                "  export BASE_URL=https://api.openai.com/v1"
            )
        raise


def _caller_from_provider(
    messages: List[Dict[str, Any]],
    model_name: str | None,
    max_tokens: int,
    provider_manager,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Use provider_manager for multi-provider support with failover.

    IMPORTANT: Never call provider_manager.get_provider() or
    create_unified_config() because they trigger detect_provider()
    which does a health-check HTTP request that can block for minutes.
    Instead, access provider config directly.
    """
    provider_priority = ["longcat"]

    tried_providers = set()

    # 首先尝试 current_provider（如果已设置）
    if provider_manager.current_provider:
        provider = provider_manager.current_provider
        tried_providers.add(provider.provider)
        print(f"[model_caller] Trying current provider: {provider.provider}")
        try:
            api_key = provider.get_api_key()
            if api_key:
                base_url = provider.endpoints["chat"]
                model = model_name or provider.get_model("default")
                headers = provider.get_headers(api_key)

                print(f"[model_caller] Calling {provider.provider}: model={model}, url={base_url}")
                print(f"[model_caller] Messages: {len(messages)}, Tools: {len(tools) if tools else 0}")
                # 打印每个 message 的 role 和 content 长度
                for i, msg in enumerate(messages):
                    content = msg.get('content')
                    content_len = len(content) if content else 0
                    content_type = type(content).__name__
                    print(f"[model_caller]   Msg {i}: role={msg['role']}, content_type={content_type}, len={content_len}, is_none={content is None}")
                # 保存实际消息到文件以便调试
                import json
                debug_file = f"debug_messages_{provider.provider}_{len(messages)}msg_{len(tools) if tools else 0}tool.json"
                with open(debug_file, "w", encoding="utf-8") as f:
                    json.dump({"messages": messages, "tools": tools}, f, ensure_ascii=False, indent=2)
                print(f"[model_caller] Debug: Saved messages to {debug_file}")

                if provider.provider == "anthropic":
                    result = _call_anthropic(base_url, headers, model, messages, max_tokens, tools)
                elif provider.provider in ("aliyun_bailian", "longcat"):
                    result = _call_openai_compatible(base_url, headers, model, messages, max_tokens, tools)
                else:
                    result = _call_api(base_url, api_key, model, messages, max_tokens, tools)

                if result and result.strip():
                    return result
                else:
                    print(f"[model_caller] Provider {provider.provider} returned empty content")
            else:
                print(f"[model_caller] Provider {provider.provider} has no API key")
        except Exception as e:
            print(f"[model_caller] Current provider {provider.provider} failed: {str(e)[:200]}")

    # 遍历优先级列表尝试其他提供商
    for provider_name in provider_priority:
        if provider_name in tried_providers:
            continue

        try:
            provider = provider_manager.providers.get(provider_name)
            if not provider:
                continue

            api_key = provider.get_api_key()
            if not api_key:
                print(f"[model_caller] Provider {provider_name} has no API key, skipping")
                continue

            print(f"[model_caller] Trying provider: {provider_name}")
            provider_manager.current_provider = provider

            base_url = provider.endpoints["chat"]
            model = model_name or provider.get_model("default")
            headers = provider.get_headers(api_key)

            print(f"[model_caller] Calling {provider_name}: model={model}, url={base_url}")

            if provider.provider == "anthropic":
                result = _call_anthropic(base_url, headers, model, messages, max_tokens, tools)
            elif provider.provider in ("aliyun_bailian", "longcat"):
                result = _call_openai_compatible(base_url, headers, model, messages, max_tokens, tools)
            else:
                result = _call_api(base_url, api_key, model, messages, max_tokens, tools)

            if result and result.strip():
                return result
            else:
                print(f"[model_caller] Provider {provider_name} returned empty content, trying next...")
        except Exception as e:
            print(f"[model_caller] Provider {provider_name} failed: {str(e)[:200]}")
            continue

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
    # 严格清理消息：确保所有 content 都是字符串
    cleaned_messages = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")
        
        # 处理 content 为 None 的情况
        if content is None:
            content = ""
        else:
            content = str(content)
        
        # 跳过没有 role 的消息
        if not role:
            continue
            
        cleaned_messages.append({"role": role, "content": content})

    payload = {
        "model": model,
        "messages": cleaned_messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    # Ensure URL has the correct endpoint path for OpenAI-compatible APIs
    chat_url = base_url if base_url.endswith("/chat/completions") else f"{base_url.rstrip('/')}/chat/completions"
    print(f"[model_caller] POST {chat_url} model={model} msgs={len(cleaned_messages)} tools={len(tools) if tools else 0}")

    response = _retry_request(chat_url, headers=headers, json_payload=payload, timeout=120)

    if response.status_code != 200:
        print(f"[model_caller] API error: {response.status_code} - {response.text[:300]}")
        raise RuntimeError(f"API call failed: {response.status_code} - {response.text[:500]}")

    result = response.json()
    choice = result["choices"][0]

    # Check for tool_calls (OpenAI function calling)
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

    # Fallback to text content
    content = choice.get("message", {}).get("content", "")
    print(f"[model_caller] Got text response: {len(content)} chars")
    return content


def make_subagent_model_caller(model_name: str | None = None, max_tokens: int = 4096):
    """Create a model caller for subagents (same API, same config)."""
    return make_model_caller(model_name=model_name, max_tokens=max_tokens)
