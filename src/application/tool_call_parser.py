"""Tool call parsing strategies for AgentLoop.

Extracted from agent_loop.py using Strategy pattern.
Each parser handles a different format of tool call representation
in LLM output, and they are tried in priority order.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any

from src.domain.tool_contract import ToolRegistry


class ToolCall:
    """Represents a parsed tool call from model output."""

    __slots__ = ("name", "arguments")

    def __init__(self, name: str, arguments: dict[str, Any]) -> None:
        self.name = name
        self.arguments = arguments

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        return cls(
            name=data["name"],
            arguments=data.get("arguments", {}),
        )


class ToolCallParser(ABC):
    """Strategy interface for parsing tool calls from model output."""

    @abstractmethod
    def parse(self, text: str, tool_registry: ToolRegistry) -> ToolCall | None:
        ...


class JsonToolCallParser(ToolCallParser):
    """Parse pure JSON tool calls: {"tool": "name", "args": {...}}"""

    def parse(self, text: str, tool_registry: ToolRegistry) -> ToolCall | None:
        if not text:
            return None
        parsed = _try_parse_json(text)
        if parsed:
            result = _extract_tool_call(parsed)
            if result:
                return result

        json_blocks = re.findall(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
        for block in json_blocks:
            parsed = _try_parse_json(block)
            if parsed:
                result = _extract_tool_call(parsed)
                if result:
                    return result
        return None


class BraceScanToolCallParser(ToolCallParser):
    """Scan for JSON objects embedded in text by tracking brace depth."""

    def parse(self, text: str, tool_registry: ToolRegistry) -> ToolCall | None:
        if not text:
            return None
        depth = 0
        start = None
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = text[start:i + 1]
                    parsed = _try_parse_json(candidate)
                    if parsed:
                        result = _extract_tool_call(parsed)
                        if result:
                            return result
                    start = None
        return None


class FuzzyToolCallParser(ToolCallParser):
    """Fuzzy parser for tool calls in natural language output.

    Handles patterns like:
    - "I'll call compile_cuda with source=..."
    - "compile_cuda(source='...', flags=[...])"
    - "Let me use the run_ncu tool on ..."
    """

    def parse(self, text: str, tool_registry: ToolRegistry) -> ToolCall | None:
        if not text:
            return None
        known_tools = set(tool_registry.list_tools()) if tool_registry else {
            "run_ncu", "compile_cuda", "execute_binary",
            "write_file", "read_file", "generate_microbenchmark",
        }

        for tool_name in known_tools:
            patterns = [
                rf'\b{re.escape(tool_name)}\s*\(([^)]*)\)',
                rf'\b{re.escape(tool_name)}\s*\{{([^}}]*)\}}',
                rf'(?:call|use|invoke|run)\s+(?:the\s+)?{re.escape(tool_name)}\b',
                rf'{re.escape(tool_name)}\s+with\s',
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    args: dict[str, Any] = {}
                    if match.lastindex and match.group(1):
                        arg_text = match.group(1).strip()
                        args = _parse_fuzzy_args(arg_text, tool_name, tool_registry)
                    return ToolCall(name=tool_name, arguments=args)
        return None


class CompositeToolCallParser(ToolCallParser):
    """Chain of parsers tried in priority order.

    Order: JSON → Brace scan → Fuzzy
    """

    def __init__(self, parsers: list[ToolCallParser] | None = None) -> None:
        self._parsers = parsers or [
            JsonToolCallParser(),
            BraceScanToolCallParser(),
            FuzzyToolCallParser(),
        ]

    def parse(self, text: str, tool_registry: ToolRegistry) -> ToolCall | None:
        for parser in self._parsers:
            result = parser.parse(text, tool_registry)
            if result is not None:
                return result
        return None


def _try_parse_json(text: str) -> dict | None:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _extract_tool_call(data: dict) -> ToolCall | None:
    tool_name = (
        data.get("tool")
        or data.get("tool_name")
        or data.get("name")
        or data.get("action")
        or data.get("command")
    )
    if not tool_name:
        return None
    if isinstance(tool_name, str):
        tool_name = tool_name.strip()

    arguments = (
        data.get("args")
        or data.get("arguments")
        or data.get("params")
        or data.get("parameters")
        or data.get("input")
        or {}
    )
    if not isinstance(arguments, dict):
        arguments = {}

    return ToolCall(name=tool_name, arguments=arguments)


def _parse_fuzzy_args(arg_text: str, tool_name: str, tool_registry: ToolRegistry) -> dict[str, Any]:
    args: dict[str, Any] = {}
    pairs = re.findall(
        r'(\w+)\s*[=:]\s*(?:["\']([^"\']*)["\']|(\[[^\]]*\])|(\{[^}]*\})|(\S+))',
        arg_text,
    )
    for pair in pairs:
        key = pair[0]
        value = pair[1] or pair[2] or pair[3] or pair[4]
        if value is None:
            continue
        if value.startswith('['):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = [v.strip().strip("'\"") for v in value[1:-1].split(',')]
        elif value.startswith('{'):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        args[key] = value

    if not args and arg_text.strip():
        contract = None
        try:
            contract = tool_registry.get(tool_name)
        except KeyError:
            pass
        if contract and "source" in contract.input_schema:
            args["source"] = arg_text
        elif contract and "executable" in contract.input_schema:
            args["executable"] = arg_text.strip().strip("'\"")
        elif contract and "file_path" in contract.input_schema:
            args["file_path"] = arg_text.strip().strip("'\"")

    return args
