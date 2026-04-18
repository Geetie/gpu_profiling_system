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
            # P2 Harness: auto-inject binary_path when LLM omits it
            # But only if this binary hasn't already been executed (prevent re-exec loop)
            if tool_call.name == "execute_binary" and not tool_call.arguments.get("binary_path"):
                last_bp = self._find_last_binary_path()
                if last_bp and not self._already_executed_binary(last_bp):
                    tool_call.arguments["binary_path"] = last_bp
                    print(f"[AgentLoop] P2 auto-inject: binary_path={last_bp}")
                elif last_bp:
                    print(f"[AgentLoop] P2 auto-inject SKIPPED: {last_bp} already executed, LLM should compile new code")
                    self.context_manager.add_entry(
                        Role.SYSTEM,
                        f"⚠️ You already executed {last_bp}. Do NOT execute the same binary again.\n"
                        f"If you need to measure a DIFFERENT target, write NEW CUDA code and call compile_cuda first.\n"
                        f"If you have measured all targets, output your final results as: target_name: value",
                        token_count=50,
                    )
            print(f"[AgentLoop] Tool call: {tool_call.name}({list(tool_call.arguments.keys())})")
            self._emit(EventKind.TOOL_CALL, {
                "tool": tool_call.name,
                "args": tool_call.arguments,
            })
            try:
                result = self._execute_tool_call(tool_call)
                print(f"[AgentLoop] Tool result: {tool_call.name} -> {str(result)[:200]}")
                tool_status = result.get("status", "success") if isinstance(result, dict) else "success"
                self._emit(EventKind.TOOL_RESULT, {
                    "tool": tool_call.name,
                    "status": tool_status,
                })
                # Estimate token count from content length
                result_str = json.dumps(result, ensure_ascii=False)
                # Truncate large tool outputs to prevent context bloat
                MAX_TOOL_RESULT_CHARS = 3000
                if len(result_str) > MAX_TOOL_RESULT_CHARS:
                    truncated_result = dict(result) if isinstance(result, dict) else result
                    if isinstance(truncated_result, dict):
                        for key in ("raw_output", "stdout", "stderr", "output"):
                            val = truncated_result.get(key, "")
                            if isinstance(val, str) and len(val) > 1500:
                                truncated_result[key] = val[:1500] + f"...[truncated, {len(val)} chars total]"
                    result_str = json.dumps(truncated_result, ensure_ascii=False)
                # Estimate token count: code/JSON is denser (~2 chars/token), text is ~4 chars/token
                code_ratio = 2.5 if any(c in result_str for c in "{}[];=<>") else 4.0
                estimated_tokens = max(20, int(len(result_str) / code_ratio))
                self.context_manager.add_entry(
                    Role.ASSISTANT,
                    result_str,
                    token_count=estimated_tokens,
                )
                # When tool returns error status, add system guidance to help LLM learn
                if isinstance(result, dict) and result.get("status") == "error":
                    error_msg = result.get("errors", result.get("error", ""))
                    hint = result.get("hint", "")
                    guidance = f"⚠️ Tool '{tool_call.name}' returned an error: {error_msg[:300]}"
                    if hint:
                        guidance += f"\n💡 HINT: {hint}"
                    if tool_call.name == "run_ncu":
                        guidance += (
                            "\n💡 Common fixes: Use real ncu metric names like 'sm__cycles', "
                            "'dram__throughput', 'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum'. "
                            "Do NOT use '.' or empty strings as metric names."
                        )
                        # Track specific invalid metric patterns for stronger anti-loop
                        metrics_arg = tool_call.arguments.get("metrics", [])
                        if isinstance(metrics_arg, list):
                            invalid_metrics = [m for m in metrics_arg if not isinstance(m, str) or m.strip() in ("", ".")]
                            if invalid_metrics:
                                specific_pattern = f"tool_error:run_ncu:invalid_metric"
                                self._failure_tracker.record_failure(specific_pattern)
                                if self._failure_tracker.should_terminate(specific_pattern):
                                    self._emit(EventKind.STOP, {
                                        "reason": "M4_repeated_invalid_metric",
                                        "pattern": specific_pattern,
                                    })
                                    self.stop()
                                    return
                    elif tool_call.name == "compile_cuda":
                        guidance += (
                            "\n💡 Common fixes: Check source code for syntax errors. "
                            "Ensure flags include the correct -arch=sm_XX for this GPU. "
                            "Do NOT use '-arch=0' or '-arch=sm_0'."
                        )
                    self.context_manager.add_entry(
                        Role.SYSTEM,
                        guidance,
                        token_count=60,
                    )
                    # Track failure for anti-loop
                    self._failure_pattern = f"tool_error:{tool_call.name}"
                    self._failure_tracker.record_failure(self._failure_pattern)
                elif isinstance(result, dict) and result.get("success") is True:
                    self._failure_pattern = None
                    # Auto-inject binary_path hint after compile_cuda success
                    if tool_call.name == "compile_cuda" and result.get("binary_path"):
                        bp = result["binary_path"]
                        auto_hint = (
                            f"✅ Compilation succeeded! Binary saved to: {bp}\n"
                            f"👉 To run it, call: "
                            f'{{"tool": "execute_binary", "args": {{"binary_path": "{bp}"}}}}'
                        )
                        self.context_manager.add_entry(
                            Role.SYSTEM,
                            auto_hint,
                            token_count=40,
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
            
            # Dynamic system feedback based on context - guide LLM to the right next tool
            # Check what tools have been called so far
            context_entries = self.context_manager.get_entries()
            has_compiled = False
            has_executed = False
            binary_path = ""
            for entry in context_entries:
                if entry.role.value == "assistant":
                    try:
                        d = json.loads(entry.content)
                        if isinstance(d, dict):
                            if d.get("tool") == "compile_cuda" and d.get("success"):
                                has_compiled = True
                                binary_path = d.get("binary_path", "")
                            if d.get("tool") == "execute_binary":
                                has_executed = True
                    except (json.JSONDecodeError, TypeError):
                        pass
            
            if has_compiled and not has_executed and binary_path:
                guidance = (
                    f"⚠️ ERROR: You compiled the code but did NOT execute the binary!\n"
                    f"You MUST now call execute_binary with the compiled binary:\n"
                    f'{{"tool": "execute_binary", "args": {{"binary_path": "{binary_path}"}}}}\n'
                    f"Do NOT output natural language — CALL the tool now."
                )
            elif has_compiled and has_executed:
                guidance = (
                    f"⚠️ ERROR: You did not call any tool in this turn.\n"
                    f"If you have measurements, output them as: target_name: value\n"
                    f"If not, call the appropriate tool to get more data."
                )
            else:
                guidance = (
                    f"⚠️ ERROR: You did not call any tool in this turn. "
                    f"You MUST output a JSON tool call like: "
                    f'{{\"tool\": \"compile_cuda\", \"args\": {{\"source\": \"...\", \"flags\": [\"-O3\"]}}}}\n'
                    f"Do NOT output natural language — ACTUALLY CALL the tools."
                )
            
            self.context_manager.add_entry(
                Role.SYSTEM,
                guidance,
                token_count=60,
            )
            
            if self._failure_tracker.should_terminate(no_tool_pattern):
                self._emit(EventKind.STOP, {
                    "reason": "M4_no_tool_repeat",
                    "pattern": no_tool_pattern,
                })
                self.stop()
                return

        self._persist_state()

    def _find_last_binary_path(self) -> str:
        """P2 Harness: find last binary_path from compile_cuda results in context."""
        entries = self.context_manager.get_entries()
        for entry in reversed(entries):
            if entry.role.value == "assistant":
                try:
                    data = json.loads(entry.content)
                    if isinstance(data, dict) and data.get("binary_path"):
                        bp = data["binary_path"]
                        if isinstance(bp, str) and bp.strip():
                            return bp.strip()
                except (json.JSONDecodeError, TypeError):
                    pass
        return ""

    def _already_executed_binary(self, binary_path: str) -> bool:
        """Check if a binary has already been executed in this session."""
        entries = self.context_manager.get_entries()
        exec_count = 0
        for entry in entries:
            if entry.role.value == "assistant":
                try:
                    data = json.loads(entry.content)
                    if isinstance(data, dict) and data.get("tool") == "execute_binary":
                        bp = data.get("binary_path", data.get("args", {}).get("binary_path", ""))
                        if bp == binary_path:
                            exec_count += 1
                except (json.JSONDecodeError, TypeError):
                    pass
        return exec_count >= 1

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
