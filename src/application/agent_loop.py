"""Dual-Layer Agent Loop — application layer.

Outer loop (session management): lifecycle, sub-agent coordination.
Inner loop (single-turn executor): context assembly → model inference
→ tool parsing → execution → repeat.

The inner loop is designed as an async-style generator (synchronous
for now, with an async-ready structure).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from src.application.control_plane import ControlPlane, InjectedContext
from src.application.context import ContextManager, Role
from src.application.session import SessionState, SessionManager
from src.application.tool_runner import ApprovalRequiredError
from src.domain.tool_contract import ToolContract, ToolRegistry
from src.domain.permission import PermissionChecker, PermissionMode, InvariantTracker
from src.infrastructure.state_persist import StatePersister


# ── Events ───────────────────────────────────────────────────────────


class EventKind(Enum):
    START = "start"
    TURN = "turn"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    COMPRESS = "compress"
    STOP = "stop"
    ERROR = "error"
    PERSIST = "persist"
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"


@dataclass
class LoopEvent:
    kind: EventKind
    payload: dict[str, Any] = field(default_factory=dict)


# ── Tool Call ────────────────────────────────────────────────────────


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        return cls(
            name=data["name"],
            arguments=data.get("arguments", {}),
        )


# ── Loop State ───────────────────────────────────────────────────────


@dataclass
class LoopState:
    session_id: str
    is_running: bool = False
    turn_count: int = 0
    pending_tool_call: ToolCall | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        pt = None
        if self.pending_tool_call:
            pt = {
                "name": self.pending_tool_call.name,
                "arguments": self.pending_tool_call.arguments,
            }
        return {
            "session_id": self.session_id,
            "is_running": self.is_running,
            "turn_count": self.turn_count,
            "pending_tool_call": pt,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoopState:
        tc = None
        pt = data.get("pending_tool_call")
        if pt:
            if isinstance(pt, dict):
                tc = ToolCall(name=pt["name"], arguments=pt.get("arguments", {}))
            else:
                # Legacy format: just a string name
                tc = ToolCall(name=pt, arguments={})
        return cls(
            session_id=data["session_id"],
            is_running=data.get("is_running", False),
            turn_count=data.get("turn_count", 0),
            pending_tool_call=tc,
            last_error=data.get("last_error"),
        )


# ── Tool Execution Callback ──────────────────────────────────────────

# Signature: (tool_name, arguments) -> result_dict
ToolExecutor = Callable[[str, dict[str, Any]], dict[str, Any]]

# Signature: (user_input) -> str
ModelCaller = Callable[[list[dict[str, Any]]], str]


# ── Agent Loop ───────────────────────────────────────────────────────


class AgentLoop:
    """Dual-layer agent loop.

    Outer loop: manages session lifecycle and coordinates sub-agents.
    Inner loop: executes a single turn (context → model → tool → repeat).
    """

    def __init__(
        self,
        session: SessionState,
        context_manager: ContextManager,
        control_plane: ControlPlane,
        tool_registry: ToolRegistry,
        max_turns: int = 20,
        state_dir: str = ".state",
        permission_mode: PermissionMode = PermissionMode.DEFAULT,
    ) -> None:
        self.session = session
        self.context_manager = context_manager
        self.control_plane = control_plane
        self.tool_registry = tool_registry
        self.max_turns = max_turns
        self._permission_checker = PermissionChecker(mode=permission_mode)

        self.loop_state = LoopState(session_id=session.session_id)
        self._event_handlers: list[Callable[[LoopEvent], None]] = []
        self._state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)

        self._persister = StatePersister(log_dir=state_dir)
        self._session_mgr = SessionManager(state_dir=state_dir)
        self._failure_tracker = InvariantTracker()
        self._failure_pattern: str | None = None

        # Overridable hooks for model/tool execution
        self._model_output: str = ""
        self._model_tool_call: ToolCall | None = None
        self._model_caller: ModelCaller | None = None
        self._tool_executor: ToolExecutor | None = None
        self._approval_callback: Callable[[Any], bool] | None = None
        self._available_tools: list[dict[str, Any]] | None = None

    # ── Event System ─────────────────────────────────────────────────

    def on_event(self, handler: Callable[[LoopEvent], None]) -> None:
        self._event_handlers.append(handler)

    def _emit(self, kind: EventKind, payload: dict[str, Any] | None = None) -> None:
        ev = LoopEvent(kind=kind, payload=payload or {})
        for h in self._event_handlers:
            h(ev)

    # ── Lifecycle (Outer Loop) ───────────────────────────────────────

    def start(self) -> None:
        """Begin the outer loop — run until completion or max turns."""
        self.loop_state.is_running = True
        self._emit(EventKind.START, {"session_id": self.session.session_id})

        while self.loop_state.is_running:
            self._inner_loop_step()

    def stop(self) -> None:
        self.loop_state.is_running = False
        self._emit(EventKind.STOP, {
            "session_id": self.session.session_id,
            "turn_count": self.loop_state.turn_count,
        })

    # ── Inner Loop Step ──────────────────────────────────────────────

    def _inner_loop_step(self) -> None:
        """Execute a single turn of the inner loop.

        Resilience improvements:
        - Non-tool output does NOT immediately stop the loop.
          The model may be explaining before calling a tool.
        - M4 anti-loop protects against infinite repetition.
        - All model caller exceptions are caught and recovered.
        - Completion is detected via explicit patterns, not just
          the absence of a tool call.
        """
        if not self.loop_state.is_running:
            return

        if self.loop_state.turn_count >= self.max_turns:
            self._emit(EventKind.STOP, {"reason": "max_turns_reached"})
            self.stop()
            return

        self.loop_state.turn_count += 1
        self.session.increment_step()

        if self._failure_pattern and self._failure_tracker.should_terminate(self._failure_pattern):
            self._emit(EventKind.STOP, {
                "reason": "M4_anti_loop",
                "pattern": self._failure_pattern,
            })
            self.stop()
            return

        injected = self.control_plane.inject()
        self.context_manager.update_system_entry(
            injected.render(), token_count=50
        )

        if self.context_manager.is_over_budget():
            removed = self.context_manager.compress()
            self._emit(EventKind.COMPRESS, {"entries_removed": removed})

        try:
            if self._model_caller:
                messages = self.context_manager.to_messages()
                try:
                    self._model_output = self._model_caller(messages, self._available_tools)
                except TypeError:
                    self._model_output = self._model_caller(messages)
            # _model_output is pre-set for testing
        except Exception as e:
            self.loop_state.last_error = str(e)
            self._failure_pattern = f"api_error:{type(e).__name__}"
            self._failure_tracker.record_failure(self._failure_pattern)
            self._emit(EventKind.ERROR, {"error": str(e), "type": type(e).__name__})
            self._persister.log_error(
                error_type=type(e).__name__,
                context="model_caller",
                message=str(e),
            )
            self.context_manager.add_entry(
                Role.ASSISTANT,
                f"[API Error - will retry] {type(e).__name__}: {str(e)[:200]}",
                token_count=20,
            )
            self._persist_state()
            return

        if not self._model_output or not self._model_output.strip():
            self._failure_pattern = "empty_output"
            self._failure_tracker.record_failure(self._failure_pattern)
            self._emit(EventKind.ERROR, {"error": "Model returned empty output"})
            self.context_manager.add_entry(
                Role.ASSISTANT,
                "[Empty model output - will retry next turn]",
                token_count=10,
            )
            self._persist_state()
            return

        tool_call = self._parse_tool_call()
        self._model_tool_call = tool_call

        if tool_call is not None:
            self._failure_pattern = None
            self._emit(EventKind.TOOL_CALL, {
                "tool": tool_call.name,
                "args": tool_call.arguments,
            })
            try:
                result = self._execute_tool_call(tool_call)
                self._emit(EventKind.TOOL_RESULT, {
                    "tool": tool_call.name,
                    "status": "success",
                })
                self.context_manager.add_entry(
                    Role.ASSISTANT,
                    json.dumps(result, ensure_ascii=False),
                    token_count=20,
                )
            except Exception as e:
                self.loop_state.last_error = str(e)
                self._failure_pattern = f"tool_error:{tool_call.name}"
                self._failure_tracker.record_failure(self._failure_pattern)
                self._emit(EventKind.ERROR, {"error": str(e)})
                if self.loop_state.last_error:
                    self._persister.log_error(
                        error_type=type(e).__name__,
                        context=f"tool:{tool_call.name}",
                        message=str(e),
                    )
                self.context_manager.add_entry(
                    Role.ASSISTANT,
                    f"Tool execution failed: {e}",
                    token_count=20,
                )
        else:
            self.context_manager.add_entry(
                Role.ASSISTANT,
                self._model_output,
                token_count=20,
            )
            self._emit(EventKind.TURN, {
                "turn": self.loop_state.turn_count,
                "output": self._model_output[:200],
            })
            if self._is_completion_signal(self._model_output):
                self._emit(EventKind.STOP, {"reason": "completion_signal"})
                self.stop()
                return
            no_tool_pattern = "no_tool_call"
            self._failure_tracker.record_failure(no_tool_pattern)
            if self._failure_tracker.should_terminate(no_tool_pattern):
                self._emit(EventKind.STOP, {
                    "reason": "M4_no_tool_repeat",
                    "pattern": no_tool_pattern,
                })
                self.stop()
                return

        self._persist_state()

    def _is_completion_signal(self, text: str) -> bool:
        """Detect if the model's text output signals task completion.

        Instead of stopping on ANY non-tool output, we only stop when
        the model explicitly signals it's done. This prevents premature
        termination when the model explains before calling tools.
        """
        lower = text.lower().strip()
        if len(lower) < 10:
            return False
        completion_patterns = [
            "verdict: accept",
            "verdict: reject",
            "all targets measured",
            "all targets have been measured",
            "measurement complete",
            "profiling complete",
            "task complete",
            "final answer:",
            "final results:",
            "summary of findings",
            "verification report",
        ]
        for pattern in completion_patterns:
            if pattern in lower:
                return True
        has_key_value_pairs = False
        has_done_statement = False
        import re
        kv_matches = re.findall(r'^[a-z_]+:\s*\d+\.?\d*', lower, re.MULTILINE)
        if len(kv_matches) >= 3:
            has_key_value_pairs = True
        done_phrases = [
            "i have completed", "i am done", "i'm done",
            "here are the final", "here is the final",
            "these are the measured", "the measured values",
        ]
        for phrase in done_phrases:
            if phrase in lower:
                has_done_statement = True
                break
        return has_key_value_pairs and has_done_statement

    # ── Tool Parsing & Execution ─────────────────────────────────────

    def _parse_tool_call(self) -> ToolCall | None:
        """Parse a tool call from the model's output.

        Accepts (in priority order):
        1. Pure JSON: {"tool": "name", "args": {...}}
        2. JSON with alternative keys: {"tool_name": "...", "arguments": {...}}
        3. Markdown-wrapped JSON: ```json\n{...}\n``` or ```\n{...}\n```
        4. Multiple JSON blocks — returns the first one with a tool name
        5. Fuzzy: looks for tool-like patterns in text (e.g., "compile_cuda(...)")
        """
        if not self._model_output:
            return None

        parsed = self._try_parse_json(self._model_output)
        if parsed:
            result = self._extract_tool_call(parsed)
            if result:
                return result

        import re
        text = self._model_output
        json_blocks = re.findall(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
        for block in json_blocks:
            parsed = self._try_parse_json(block)
            if parsed:
                result = self._extract_tool_call(parsed)
                if result:
                    return result

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
                    candidate = text[start:i+1]
                    parsed = self._try_parse_json(candidate)
                    if parsed:
                        result = self._extract_tool_call(parsed)
                        if result:
                            return result
                    start = None

        return self._fuzzy_parse_tool_call(text)

    def _try_parse_json(self, text: str) -> dict | None:
        """Try to parse text as JSON, return dict or None."""
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    def _fuzzy_parse_tool_call(self, text: str) -> ToolCall | None:
        """Fuzzy parser for tool calls in natural language output.

        Handles patterns like:
        - "I'll call compile_cuda with source=..."
        - "compile_cuda(source='...', flags=[...])"
        - "Let me use the run_ncu tool on ..."
        """
        import re
        known_tools = set(self.tool_registry.list_tools()) if self.tool_registry else {
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
                    args = {}
                    if match.lastindex and match.group(1):
                        arg_text = match.group(1).strip()
                        args = self._parse_fuzzy_args(arg_text, tool_name)
                    return ToolCall(name=tool_name, arguments=args)

        return None

    def _parse_fuzzy_args(self, arg_text: str, tool_name: str) -> dict[str, Any]:
        """Parse fuzzy argument text into a dict.

        Handles: key=value, key='value', key="value", key: value
        """
        import re
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
                contract = self.tool_registry.get(tool_name)
            except KeyError:
                pass
            if contract and "source" in contract.input_schema:
                args["source"] = arg_text
            elif contract and "executable" in contract.input_schema:
                args["executable"] = arg_text.strip().strip("'\"")
            elif contract and "file_path" in contract.input_schema:
                args["file_path"] = arg_text.strip().strip("'\"")

        return args

    def _extract_tool_call(self, data: dict) -> ToolCall | None:
        """Extract tool name and arguments from parsed JSON dict.

        Supports multiple key naming conventions:
        - "tool" or "tool_name" or "name" or "action" or "command" for the tool name
        - "args" or "arguments" or "params" or "parameters" or "input" for the arguments
        """
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

    def _execute_tool_call(self, tool_call: ToolCall) -> dict[str, Any]:
        """Execute a tool call through the registry and executor hook.

        INT-2 fix: Only checks is_allowed() (P2 fail-closed).
        Approval checking is delegated to ToolRunner which has the
        full ApprovalQueue integration.
        """
        # Validate tool exists (P2: fail-closed)
        contract = self.tool_registry.get(tool_call.name)

        # INT-2 fix: Only check permission is_allowed, not requires_approval.
        # The approval flow is handled by ToolRunner + ApprovalQueue.
        for perm in contract.permissions:
            if not self._permission_checker.is_allowed(perm):
                self._persister.log_permission_decision(
                    permission=perm,
                    mode=self._permission_checker.mode.value,
                    decision="denied",
                    reason=f"Tool '{tool_call.name}' requires '{perm}' "
                           f"(mode={self._permission_checker.mode.value})",
                )
                raise PermissionError(
                    f"P2 fail-closed: permission '{perm}' denied for tool '{tool_call.name}' "
                    f"under mode '{self._permission_checker.mode.value}'"
                )

        # Use hook if provided
        if self._tool_executor is not None:
            return self._execute_with_approval(tool_call)

        # No executor hook installed — log a warning instead of silent stub
        self._persister.log_error(
            error_type="NoToolExecutor",
            context=f"tool:{tool_call.name}",
            message="No tool executor hook installed; returning contract info only",
        )
        return {
            "tool": tool_call.name,
            "status": "no_executor_installed",
            "contract": contract.to_dict(),
        }

    def _execute_with_approval(self, tool_call: ToolCall) -> dict[str, Any]:
        """Execute a tool call with approval flow support.

        INT-8 fix: After user approves/rejects via callback, respond to
        the ApprovalQueue so its state and persistence (P6) are updated.
        """
        if self._tool_executor is None:
            # Fallback handled by caller
            return {}

        try:
            return self._tool_executor(tool_call.name, tool_call.arguments)
        except ApprovalRequiredError as e:
            self._emit(EventKind.APPROVAL_REQUEST, {
                "tool": tool_call.name,
                "request_id": e.request.id,
            })

            # Use approval callback if registered
            if self._approval_callback is not None:
                approved = self._approval_callback(e.request)
            else:
                # No callback — auto-reject (fail-closed)
                approved = False

            # INT-8 fix: respond to ApprovalQueue to update state and persist
            self._respond_to_approval_queue(e.request, approved)

            if approved:
                self._emit(EventKind.APPROVAL_GRANTED, {
                    "tool": tool_call.name,
                    "request_id": e.request.id,
                })
                # Re-execute after approval
                return self._tool_executor(tool_call.name, tool_call.arguments)
            else:
                self._emit(EventKind.APPROVAL_DENIED, {
                    "tool": tool_call.name,
                    "request_id": e.request.id,
                })
                raise PermissionError(
                    f"Tool '{tool_call.name}' approval denied"
                ) from e

    def _respond_to_approval_queue(self, request, approved: bool) -> None:
        """Respond to the ApprovalQueue (separate method for testability)."""
        from src.application.approval_queue import ApprovalQueue
        # The approval_queue is typically accessible through the tool_runner
        # which is set as _tool_executor. We need to find it there.
        if hasattr(self._tool_executor, "_approval_queue"):
            self._tool_executor._approval_queue.respond(request.id, approved)

    # ── Persistence (P6) ─────────────────────────────────────────────

    def _persist_state(self) -> None:
        """Persist session and loop state to disk."""
        self._session_mgr.save_session(self.session)
        self._persister.log_tool_execution(
            tool_name="__loop_state__",
            inputs=self.loop_state.to_dict(),
            status="persisted",
        )
        self._emit(EventKind.PERSIST, {
            "turn": self.loop_state.turn_count,
        })

    # ── Resume ───────────────────────────────────────────────────────

    @classmethod
    def from_resume(
        cls,
        session_id: str,
        control_plane: ControlPlane,
        context_manager: ContextManager,
        tool_registry: ToolRegistry,
        state_dir: str = ".state",
        new_goal: str | None = None,
        max_turns: int = 20,
        permission_mode: PermissionMode = PermissionMode.DEFAULT,
    ) -> AgentLoop:
        """Resume a session from a persisted checkpoint.

        This is the entry point for `--resume` functionality.
        """
        mgr = SessionManager(state_dir=state_dir)
        session = mgr.resume(session_id, new_goal=new_goal)
        return cls(
            session=session,
            context_manager=context_manager,
            control_plane=control_plane,
            tool_registry=tool_registry,
            max_turns=max_turns,
            state_dir=state_dir,
            permission_mode=permission_mode,
        )

    # ── Hook Registration ────────────────────────────────────────────

    def set_model_caller(self, caller: ModelCaller) -> None:
        """Set the function that calls the LLM."""
        self._model_caller = caller

    def set_available_tools(self, tools: list[dict[str, Any]]) -> None:
        """Set tool definitions for OpenAI function calling.

        Format: [{"type": "function", "function": {"name": "tool_name"}}]
        """
        self._available_tools = tools

    def set_tool_executor(self, executor: ToolExecutor) -> None:
        """Set the function that executes tool calls."""
        self._tool_executor = executor

    def set_permission_mode(self, mode: PermissionMode) -> None:
        """Change the permission mode at runtime."""
        self._permission_checker.set_mode(mode)

    def set_approval_callback(self, callback: Callable[[Any], bool]) -> None:
        """Set the function that handles approval requests.

        The callback receives an ApprovalRequest and returns True to approve,
        False to reject. It is responsible for responding to the ApprovalQueue.
        """
        self._approval_callback = callback

    # ── Pipeline Integration (INT-3) ─────────────────────────────────

    def run_pipeline(
        self,
        pipeline: "Pipeline",
        target_spec: dict[str, Any],
    ) -> Any:
        """Run a multi-agent Pipeline within this session.

        This connects the Pipeline orchestrator to AgentLoop,
        allowing the system to switch between single-agent loop
        and multi-agent pipeline modes.

        Args:
            pipeline: A configured Pipeline instance.
            target_spec: Target specification dict (from target_spec.json).

        Returns:
            SubAgentResult from the final verification stage.
        """
        from src.domain.subagent import SubAgentStatus

        self._emit(EventKind.START, {
            "session_id": self.session.session_id,
            "mode": "pipeline",
            "targets": target_spec.get("targets", []),
        })

        result = pipeline.run(target_spec)

        # Persist pipeline result into session context
        if result.is_success():
            self.context_manager.add_entry(
                Role.ASSISTANT,
                f"Pipeline completed: {result.agent_role.value} -> "
                f"status={result.status.value}, "
                f"fingerprint={result.context_fingerprint}",
                token_count=30,
            )
        else:
            self.session.mark_error(result.error or "Pipeline failed")
            self.context_manager.add_entry(
                Role.ASSISTANT,
                f"Pipeline FAILED: {result.error}",
                token_count=20,
            )

        self._persist_state()
        self._emit(EventKind.STOP, {
            "session_id": self.session.session_id,
            "mode": "pipeline",
            "status": result.status.value,
            "error": result.error,
        })

        return result
