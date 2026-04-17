"""Dual-Layer Agent Loop — application layer.

Outer loop (session management): lifecycle, sub-agent coordination.
Inner loop (single-turn executor): context assembly → model inference
→ tool parsing → execution → repeat.

Refactored with Strategy pattern:
- Tool call parsing delegated to CompositeToolCallParser
- Completion detection delegated to CompletionDetector
- Event dispatch delegated to EventBus
- AgentLoop focuses on loop control and orchestration
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from src.application.completion_detector import CompletionDetector
from src.application.control_plane import ControlPlane, InjectedContext
from src.application.context import ContextManager, Role
from src.application.event_bus import EventBus, EventKind, LoopEvent
from src.application.session import SessionState, SessionManager
from src.application.tool_call_parser import ToolCall, CompositeToolCallParser
from src.application.tool_runner import ApprovalRequiredError
from src.domain.tool_contract import ToolContract, ToolRegistry
from src.domain.permission import PermissionChecker, PermissionMode, InvariantTracker
from src.infrastructure.state_persist import StatePersister


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
                tc = ToolCall(name=pt, arguments={})
        return cls(
            session_id=data["session_id"],
            is_running=data.get("is_running", False),
            turn_count=data.get("turn_count", 0),
            pending_tool_call=tc,
            last_error=data.get("last_error"),
        )


ToolExecutor = Callable[[str, dict[str, Any]], dict[str, Any]]
ModelCaller = Callable[[list[dict[str, Any]]], str]


class AgentLoop:
    """Dual-layer agent loop.

    Outer loop: manages session lifecycle and coordinates sub-agents.
    Inner loop: executes a single turn (context → model → tool → repeat).

    Parsing is delegated to CompositeToolCallParser (Strategy pattern).
    Completion detection is delegated to CompletionDetector.
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
        self._event_bus = EventBus()
        self._state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)

        self._persister = StatePersister(log_dir=state_dir)
        self._session_mgr = SessionManager(state_dir=state_dir)
        self._failure_tracker = InvariantTracker()
        self._failure_pattern: str | None = None

        self._model_output: str = ""
        self._model_tool_call: ToolCall | None = None
        self._model_caller: ModelCaller | None = None
        self._tool_executor: ToolExecutor | None = None
        self._approval_callback: Callable[[Any], bool] | None = None
        self._available_tools: list[dict[str, Any]] | None = None

        self._tool_call_parser = CompositeToolCallParser()
        self._completion_detector = CompletionDetector()

    def on_event(
        self,
        handler: Callable[[LoopEvent], None],
        kinds: set[EventKind] | None = None,
        priority: int = 0,
    ) -> None:
        """Subscribe to loop events.

        Args:
            handler: Callback receiving LoopEvent.
            kinds: If None, receives all events. Otherwise only listed kinds.
            priority: Higher priority handlers execute first.
        """
        self._event_bus.subscribe(handler, kinds=kinds, priority=priority)

    @property
    def event_bus(self) -> EventBus:
        """Access the underlying EventBus for advanced subscription."""
        return self._event_bus

    def _emit(self, kind: EventKind, payload: dict[str, Any] | None = None) -> None:
        self._event_bus.emit(LoopEvent(kind=kind, payload=payload or {}))

    def start(self) -> None:
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

    def _inner_loop_step(self) -> None:
        if not self.loop_state.is_running:
            return

        if self.loop_state.turn_count >= self.max_turns:
            self._emit(EventKind.STOP, {"reason": "max_turns_reached"})
            self.stop()
            return

        self.loop_state.turn_count += 1
        self.session.increment_step()
        print(f"[AgentLoop] Turn {self.loop_state.turn_count}/{self.max_turns} (session: {self.session.session_id})")

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
                print(f"[AgentLoop] Calling model with {len(messages)} messages")
                try:
                    self._model_output = self._model_caller(messages, self._available_tools)
                except TypeError:
                    self._model_output = self._model_caller(messages)
                print(f"[AgentLoop] Model response: {len(self._model_output)} chars")
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
            # 不要将 API 错误保存到 ASSISTANT 消息中，避免污染上下文
            # 使用 SYSTEM 角色记录错误信息
            self.context_manager.add_entry(
                Role.SYSTEM,
                f"[Internal Error Log: {type(e).__name__}] {str(e)[:200]}",
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

        tool_call = self._tool_call_parser.parse(self._model_output, self.tool_registry)
        self._model_tool_call = tool_call

        if tool_call is not None:
            self._failure_pattern = None
            print(f"[AgentLoop] Tool call: {tool_call.name}({list(tool_call.arguments.keys())})")
            self._emit(EventKind.TOOL_CALL, {
                "tool": tool_call.name,
                "args": tool_call.arguments,
            })
            try:
                result = self._execute_tool_call(tool_call)
                print(f"[AgentLoop] Tool result: {tool_call.name} -> {str(result)[:200]}")
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
                error_result = {
                    "tool": tool_call.name,
                    "status": "error",
                    "error": str(e)[:500],
                    "error_type": type(e).__name__,
                }
                self.context_manager.add_entry(
                    Role.ASSISTANT,
                    json.dumps(error_result, ensure_ascii=False),
                    token_count=50,
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
            if self._completion_detector.is_completion(self._model_output):
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

    def _execute_tool_call(self, tool_call: ToolCall) -> dict[str, Any]:
        contract = self.tool_registry.get(tool_call.name)

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

        if self._tool_executor is not None:
            return self._execute_with_approval(tool_call)

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
        if self._tool_executor is None:
            return {}

        try:
            return self._tool_executor(tool_call.name, tool_call.arguments)
        except ApprovalRequiredError as e:
            self._emit(EventKind.APPROVAL_REQUEST, {
                "tool": tool_call.name,
                "request_id": e.request.id,
            })

            if self._approval_callback is not None:
                approved = self._approval_callback(e.request)
            else:
                approved = False

            self._respond_to_approval_queue(e.request, approved)

            if approved:
                self._emit(EventKind.APPROVAL_GRANTED, {
                    "tool": tool_call.name,
                    "request_id": e.request.id,
                })
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
        if hasattr(self._tool_executor, "_approval_queue"):
            self._tool_executor._approval_queue.respond(request.id, approved)

    def _persist_state(self) -> None:
        self._session_mgr.save_session(self.session)
        self._persister.log_tool_execution(
            tool_name="__loop_state__",
            inputs=self.loop_state.to_dict(),
            status="persisted",
        )
        self._emit(EventKind.PERSIST, {
            "turn": self.loop_state.turn_count,
        })

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

    def set_model_caller(self, caller: ModelCaller) -> None:
        self._model_caller = caller

    def set_available_tools(self, tools: list[dict[str, Any]]) -> None:
        self._available_tools = tools

    def set_tool_executor(self, executor: ToolExecutor) -> None:
        self._tool_executor = executor

    def set_permission_mode(self, mode: PermissionMode) -> None:
        self._permission_checker.set_mode(mode)

    def set_approval_callback(self, callback: Callable[[Any], bool]) -> None:
        self._approval_callback = callback

    def set_tool_call_parser(self, parser: Any) -> None:
        """Override the default tool call parser (Strategy pattern)."""
        self._tool_call_parser = parser

    def set_completion_detector(self, detector: CompletionDetector) -> None:
        """Override the default completion detector."""
        self._completion_detector = detector

    def run_pipeline(
        self,
        pipeline: "Pipeline",
        target_spec: dict[str, Any],
    ) -> Any:
        from src.domain.subagent import SubAgentStatus

        self._emit(EventKind.START, {
            "session_id": self.session.session_id,
            "mode": "pipeline",
            "targets": target_spec.get("targets", []),
        })

        result = pipeline.run(target_spec)

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
