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
import re
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

            if tool_call.name == "compile_cuda":
                source = tool_call.arguments.get("source", "")
                validation_error = tool_call.arguments.get("_validation_error", "")
                if not source or (isinstance(source, str) and not source.strip()):
                    error_msg = validation_error or (
                        "compile_cuda REQUIRES a non-empty 'source' parameter with FULL CUDA code. "
                        "You provided an empty source."
                    )
                    self.context_manager.add_entry(
                        Role.ASSISTANT,
                        json.dumps({
                            "tool": "compile_cuda",
                            "status": "error",
                            "errors": error_msg,
                            "success": False,
                        }),
                        token_count=30,
                    )
                    self.context_manager.add_entry(
                        Role.SYSTEM,
                        "⛔ compile_cuda was called with EMPTY source code. This is a CRITICAL error.\n"
                        "You MUST provide the COMPLETE CUDA source code as a string in the 'source' parameter.\n\n"
                        "CORRECT FORMAT:\n"
                        '{"tool": "compile_cuda", "args": {"source": "#include <cuda_runtime.h>\\n'
                        '#include <cstdio>\\n#include <cstdint>\\n...your full kernel code...", '
                        '"flags": ["-O3"]}}\n\n'
                        "❌ WRONG: compile_cuda with source=[] (empty array)\n"
                        "❌ WRONG: compile_cuda with source=\"\" (empty string)\n"
                        "❌ WRONG: compile_cuda without source parameter\n\n"
                        "Do NOT call compile_cuda again until you have written the FULL CUDA source code.",
                        token_count=80,
                    )
                    empty_compile_pattern = "compile_cuda_empty_source"
                    self._failure_tracker.record_failure(empty_compile_pattern)
                    if self._failure_tracker.should_terminate(empty_compile_pattern):
                        self._emit(EventKind.STOP, {
                            "reason": "M4_repeated_empty_compile",
                            "pattern": empty_compile_pattern,
                        })
                        self.stop()
                        return
                    self._persist_state()
                    return

            if tool_call.name == "execute_binary":
                bp_arg = tool_call.arguments.get("binary_path", "")
                
                DEFAULT_BINARY_PATHS = {
                    "bin/benchmark", "./benchmark", "benchmark",
                    "/bin/benchmark", "./bin/benchmark", "output/benchmark"
                }
                
                should_inject = False
                inject_reason = ""
                
                if not bp_arg or (isinstance(bp_arg, str) and not bp_arg.strip()):
                    should_inject = True
                    inject_reason = "empty path"
                elif isinstance(bp_arg, str) and bp_arg.strip() in DEFAULT_BINARY_PATHS:
                    should_inject = True
                    inject_reason = f"default value '{bp_arg}' detected (will be replaced)"
                
                if should_inject:
                    last_bp = self._find_last_binary_path()
                    last_tool_was_compile = self._last_tool_was_compile()
                    already_ran = self._already_executed_binary(last_bp) if last_bp else False

                    if last_bp and (not already_ran or last_tool_was_compile):
                        tool_call.arguments["binary_path"] = last_bp
                        reason = ("latest compile not yet executed" if last_tool_was_compile else f"replaced {inject_reason}")
                        print(f"[AgentLoop] P2 auto-inject: binary_path={last_bp} (reason: {reason})")
                    elif last_bp and already_ran and not last_tool_was_compile:
                        print(f"[AgentLoop] P2 auto-inject SKIPPED: {last_bp} already executed after latest compile")
                        unmeasured = self._find_unmeasured_targets()
                        if unmeasured:
                            from src.domain.design_principles import get_design_principle
                            next_target = unmeasured[0]
                            next_principle = get_design_principle(next_target)
                            next_brief = next_principle[:300] if len(next_principle) > 300 else next_principle
                            guidance = (
                                f"⚠️ You already executed {last_bp} after its latest compilation.\n"
                                f"But you have NOT measured all targets! Remaining: {unmeasured}\n\n"
                                f"👉 NEXT TARGET: '{next_target}'\n"
                                f"You MUST write NEW CUDA code for '{next_target}' and call compile_cuda.\n\n"
                                f"Design principle for '{next_target}':\n{next_brief}\n\n"
                                f"WORKFLOW:\n"
                                f'  1. compile_cuda(source="...full .cu source for {next_target}...", flags=["-O3"])\n'
                                f"  2. execute_binary(binary_path='<from compile>')\n\n"
                                f"Do NOT call execute_binary again with the old binary."
                            )
                        else:
                            guidance = (
                                f"⚠️ You already executed {last_bp} after its latest compilation.\n"
                                f"All targets have been measured. Output your final results as: target_name: value"
                            )
                        self.context_manager.add_entry(
                            Role.SYSTEM,
                            guidance,
                            token_count=60,
                        )
                        already_ran_pattern = "execute_binary_already_ran"
                        self._failure_tracker.record_failure(already_ran_pattern)
                        if self._failure_tracker.should_terminate(already_ran_pattern):
                            self._emit(EventKind.STOP, {
                                "reason": "M4_repeated_already_ran",
                                "pattern": already_ran_pattern,
                            })
                            self.stop()
                            return
                        self._persist_state()
                        return
                    else:
                        self.context_manager.add_entry(
                            Role.SYSTEM,
                            "⚠️ execute_binary requires a 'binary_path' parameter, but no compiled binary exists.\n"
                            "You MUST call compile_cuda FIRST to compile your CUDA code, then execute_binary.",
                            token_count=40,
                        )
                        empty_exec_pattern = "execute_binary_no_path"
                        self._failure_tracker.record_failure(empty_exec_pattern)
                        if self._failure_tracker.should_terminate(empty_exec_pattern):
                            self._emit(EventKind.STOP, {
                                "reason": "M4_repeated_empty_exec",
                                "pattern": empty_exec_pattern,
                            })
                            self.stop()
                        return
                else:
                    print(f"[AgentLoop] Using LLM-provided binary_path: {bp_arg}")
                        self._persist_state()
                        return
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
                # Track which tool was called (for _already_executed_binary detection)
                if isinstance(result, dict) and "tool" not in result:
                    result["tool"] = tool_call.name
                if tool_call.name == "execute_binary" and isinstance(result, dict):
                    bp = tool_call.arguments.get("binary_path", "")
                    if bp:
                        result["binary_path"] = bp
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
                # Update Control Plane progress after each tool call
                if isinstance(result, dict):
                    self._update_control_plane_progress(result)
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
                        stderr = result.get("stderr", "")
                        if "ERR_NVGPUCTRPERM" in stderr or "permission" in stderr.lower():
                            guidance = (
                                "⛔ NCU PERMISSION DENIED (ERR_NVGPUCTRPERM).\n"
                                "You CANNOT use run_ncu in this environment.\n"
                                "Instead, analyze CodeGen's measurements directly from your task description.\n"
                                "Provide bottleneck classification and confidence based on available data.\n"
                                "Do NOT call run_ncu again."
                            )
                            ncu_perm_pattern = "tool_error:run_ncu:permission"
                            self._failure_tracker.record_failure(ncu_perm_pattern)
                            if self._failure_tracker.should_terminate(ncu_perm_pattern):
                                self._emit(EventKind.STOP, {
                                    "reason": "M4_ncu_permission_denied",
                                    "pattern": ncu_perm_pattern,
                                })
                                self.stop()
                                return
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
                        errors = result.get("errors", result.get("error", ""))
                        if "No source code provided" in str(errors) or "source" not in tool_call.arguments:
                            guidance = (
                                "⚠️ compile_cuda REQUIRES a 'source' parameter with FULL CUDA code.\n"
                                "You MUST provide the complete .cu source code as a string.\n\n"
                                "CORRECT FORMAT:\n"
                                '{"tool": "compile_cuda", "args": {"source": "#include <cuda_runtime.h>\\n#include <cstdio>\\n#include <cstdint>\\n...your full kernel code...", "flags": ["-O3"]}}\n\n'
                                "❌ WRONG: compile_cuda with empty source or no source parameter\n"
                                "❌ WRONG: compile_cuda with just a file path\n"
                                "The 'source' parameter must contain the ENTIRE CUDA source code as a string."
                            )
                        else:
                            guidance += self._build_compile_error_guidance(str(errors))
                    self.context_manager.add_entry(
                        Role.SYSTEM,
                        guidance,
                        token_count=60,
                    )
                    self._failure_pattern = f"tool_error:{tool_call.name}"
                    self._failure_tracker.record_failure(self._failure_pattern)
                elif isinstance(result, dict) and result.get("success") is True:
                    self._failure_pattern = None
                    # Auto-inject binary_path hint after compile_cuda success
                    if tool_call.name == "compile_cuda" and result.get("binary_path"):
                        bp = result["binary_path"]
                        compile_count = sum(1 for e in self.context_manager.get_entries()
                                            if e.role.value == "assistant"
                                            and _safe_get_tool(e) == "compile_cuda"
                                            and _safe_get_success(e))
                        auto_hint = (
                            f"✅ Compilation #{compile_count} succeeded! Binary saved to: {bp}\n"
                            f"👉 IMMEDIATELY call execute_binary to run this binary:\n"
                            f'{{"tool": "execute_binary", "args": {{"binary_path": "{bp}"}}}}\n\n'
                            f"⚠️ Do NOT output text. Do NOT write more code. CALL execute_binary NOW."
                        )
                        self.context_manager.add_entry(
                            Role.SYSTEM,
                            auto_hint,
                            token_count=50,
                        )
                    # Detect implausible measurements (all zeros) after execute_binary
                    if tool_call.name == "execute_binary":
                        stdout = result.get("stdout", "")
                        if stdout and ": 0" in stdout:
                            zero_lines = [l for l in stdout.splitlines()
                                          if l.strip() and ": 0" in l and not l.strip().startswith("//")]
                            if len(zero_lines) >= 1:
                                self.context_manager.add_entry(
                                    Role.SYSTEM,
                                    "⚠️ MEASUREMENT WARNING: Output contains zero value(s). "
                                    "This usually means the compiler optimized away the measurement loop.\n"
                                    "FIX: Add 'volatile' qualifiers and asm volatile barriers:\n"
                                    "  volatile uint64_t sink64 = (uint64_t)idx;\n"
                                    "  asm volatile(\"\" : \"+l\"(sink64) : : \"memory\");\n"
                                    "Also add #pragma unroll 1 before loops to prevent unrolling.\n"
                                    "Pass arrays as 'volatile type*' to prevent register caching.\n"
                                    "If this measurement is already zero, you MUST rewrite the kernel "
                                    "with these anti-optimization techniques before compiling again.",
                                    token_count=80,
                                )

                        # After successful execution, prompt for next unmeasured target
                        unmeasured = self._find_unmeasured_targets()
                        if unmeasured and result.get("return_code", -1) == 0:
                            next_target = unmeasured[0]
                            from src.domain.design_principles import get_design_principle
                            next_principle = get_design_principle(next_target)
                            next_brief = next_principle[:400] if len(next_principle) > 400 else next_principle
                            self.context_manager.add_entry(
                                Role.SYSTEM,
                                f"✅ Measurement successful! Now measure the NEXT target.\n\n"
                                f"Remaining unmeasured targets: {unmeasured}\n\n"
                                f"👉 NEXT: Write CUDA code for '{next_target}' and call compile_cuda.\n"
                                f"Design principle for '{next_target}':\n{next_brief}\n\n"
                                f"Do NOT output text — CALL compile_cuda NOW with new source code for '{next_target}'.",
                                token_count=80,
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
            has_tools = len(self.tool_registry.list_tools()) > 0
            if self._completion_detector.is_completion(self._model_output):
                if has_tools:
                    unmeasured = self._find_unmeasured_targets()
                    if unmeasured:
                        from src.domain.design_principles import get_design_principle
                        next_target = unmeasured[0]
                        next_principle = get_design_principle(next_target)
                        next_brief = next_principle[:400] if len(next_principle) > 400 else next_principle
                        self.context_manager.add_entry(
                            Role.SYSTEM,
                            f"⚠️ STOP — You said you're done, but NOT all targets are measured!\n"
                            f"Missing targets: {unmeasured}\n\n"
                            f"👉 You MUST measure '{next_target}' next.\n"
                            f"Design principle for '{next_target}':\n{next_brief}\n\n"
                            f"Call compile_cuda with NEW source code for '{next_target}'.\n"
                            f"Do NOT output text — CALL the tool NOW.",
                            token_count=80,
                        )
                        no_tool_pattern = "no_tool_call"
                        self._failure_tracker.record_failure(no_tool_pattern)
                        if self._failure_tracker.should_terminate(no_tool_pattern):
                            self._emit(EventKind.STOP, {
                                "reason": "M4_no_tool_repeat",
                                "pattern": no_tool_pattern,
                            })
                            self.stop()
                            return
                    else:
                        self._emit(EventKind.STOP, {"reason": "completion_signal"})
                        self.stop()
                        return
                else:
                    self._emit(EventKind.STOP, {"reason": "completion_signal"})
                    self.stop()
                    return
            if has_tools:
                no_tool_pattern = "no_tool_call"
                self._failure_tracker.record_failure(no_tool_pattern)
                
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
                    unmeasured = self._find_unmeasured_targets()
                    if unmeasured:
                        next_target = unmeasured[0]
                        from src.domain.design_principles import get_design_principle
                        principle_hint = get_design_principle(next_target)
                        principle_brief = principle_hint[:500] if len(principle_hint) > 500 else principle_hint
                        guidance = (
                            f"⚠️ CRITICAL: You have NOT measured all targets yet!\n"
                            f"✅ Measured targets are done. Remaining: {unmeasured}\n\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"👉 NEXT TARGET: '{next_target}'\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                            f"You MUST write NEW CUDA code for '{next_target}' and compile it.\n"
                            f"Do NOT output text — CALL compile_cuda NOW.\n\n"
                            f"Design principle for '{next_target}':\n{principle_brief}\n\n"
                            f"MANDATORY WORKFLOW:\n"
                            f'  1. compile_cuda(source="...full .cu source for {next_target}...", flags=["-O3"])\n'
                            f"  2. execute_binary(binary_path='<from compile>')\n"
                            f"  3. Record the measured value\n\n"
                            f"⚠️ Do NOT skip '{next_target}'. The pipeline will FAIL if any target is missing."
                        )
                    else:
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
        """P2 Harness: find last binary_path from successful compile_cuda results.

        Only searches compile_cuda tool results (not execute_binary or others)
        to ensure we always get the path from the most recent compilation.
        This prevents edge cases where execute_binary results might
        contain a stale binary_path from an earlier compilation.

        Enhanced to handle multiple result formats for robustness:
        - Standard format: {"tool": "compile_cuda", "binary_path": "...", ...}
        - Alternative format: {"name": "compile_cuda", "output": {"binary_path": "..."}}
        - Artifacts format: {"artifacts": {"binary": "..."}}
        """
        entries = self.context_manager.get_entries()
        for entry in reversed(entries):
            if entry.role.value != "assistant":
                continue

            try:
                data = json.loads(entry.content)
                if not isinstance(data, dict):
                    continue

                bp = ""

                if data.get("tool") == "compile_cuda" or data.get("name") == "compile_cuda":
                    bp = data.get("binary_path", "")
                    if not bp and isinstance(data.get("output"), dict):
                        bp = data["output"].get("binary_path", "")

                if not bp and isinstance(data.get("artifacts"), dict):
                    bp = data["artifacts"].get("binary", "")
                    if not bp:
                        artifacts_list = data["artifacts"]
                        if isinstance(artifacts_list, list) and len(artifacts_list) > 0:
                            for artifact in artifacts_list:
                                if isinstance(artifact, str) and ("benchmark" in artifact or "bin/" in artifact):
                                    bp = artifact
                                    break

                if not bp:
                    continue

                success_val = data.get("success")
                is_success = (
                    success_val is True or
                    (isinstance(success_val, str) and success_val.lower() == "true") or
                    (data.get("status", "") in ["success", "success_with_warning"])
                )

                if isinstance(bp, str) and bp.strip() and (is_success or success_val is None):
                    return bp.strip()

            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

        return ""

    def _already_executed_binary(self, binary_path: str) -> bool:
        """Check if a binary has already been executed AFTER its most recent compilation.

        Uses POSITIONAL CHECKING instead of global counting:
        1. Find the index of the last successful compile_cuda in entries
        2. Check if any execute_binary with the same binary_path appears AFTER that index
        3. If yes → the latest compilation has been executed (return True)
        4. If no → the latest compilation has NOT been executed (return False)

        This is more precise than global counting because it correctly handles
        cases where LLM skips an execution and later re-executes an old binary.
        """
        entries = self.context_manager.get_entries()
        last_compile_idx = -1

        for idx, entry in enumerate(entries):
            if entry.role.value == "assistant":
                try:
                    data = json.loads(entry.content)
                    if isinstance(data, dict):
                        tool = data.get("tool", "")
                        success_val = data.get("success")
                        is_success = success_val is True or (isinstance(success_val, str) and success_val.lower() == "true")
                        if tool == "compile_cuda" and is_success:
                            last_compile_idx = idx
                except (json.JSONDecodeError, TypeError):
                    pass

        if last_compile_idx < 0:
            print(f"[AgentLoop] _already_executed_binary: path={binary_path}, "
                  f"no successful compile found, result=False")
            return False

        for idx in range(last_compile_idx + 1, len(entries)):
            entry = entries[idx]
            if entry.role.value == "assistant":
                try:
                    data = json.loads(entry.content)
                    if isinstance(data, dict):
                        tool = data.get("tool", "")
                        bp = data.get("binary_path", "")
                        if tool == "execute_binary" and bp == binary_path:
                            print(f"[AgentLoop] _already_executed_binary: path={binary_path}, "
                                  f"last_compile_idx={last_compile_idx}, exec_at={idx}, result=True")
                            return True
                except (json.JSONDecodeError, TypeError):
                    pass

        print(f"[AgentLoop] _already_executed_binary: path={binary_path}, "
              f"last_compile_idx={last_compile_idx}, no exec after, result=False")
        return False

    def _last_tool_was_compile(self) -> bool:
        """Check if the most recent successful tool call was compile_cuda.

        This is a safety net: even if _already_executed_binary returns True
        due to edge cases, if the last tool was compile_cuda, we should
        still auto-inject the binary_path for execution.
        """
        entries = self.context_manager.get_entries()
        for entry in reversed(entries):
            if entry.role.value == "assistant":
                try:
                    data = json.loads(entry.content)
                    if isinstance(data, dict) and "tool" in data:
                        tool = data["tool"]
                        success_val = data.get("success")
                        is_success = success_val is True or (isinstance(success_val, str) and success_val.lower() == "true")
                        if tool == "compile_cuda" and is_success:
                            return True
                        if tool == "execute_binary":
                            return False
                except (json.JSONDecodeError, TypeError):
                    pass
        return False

    def _find_unmeasured_targets(self) -> list[str]:
        """Find targets that have not yet been measured in this session."""
        all_targets = []
        measured = set()
        entries = self.context_manager.get_entries()
        for entry in entries:
            content = entry.content
            if entry.role.value == "system" and "targets" in content:
                m = re.search(r'"targets"\s*:\s*\[([^\]]+)\]', content)
                if m:
                    all_targets = re.findall(r'"([^"]+)"', m.group(1))
            if entry.role.value == "assistant":
                for line in content.splitlines():
                    m2 = re.match(r'\s*([\w_]+)\s*[:=]\s*[\d.]+', line)
                    if m2:
                        measured.add(m2.group(1))
                    try:
                        data = json.loads(content)
                        if isinstance(data, dict) and "stdout" in data:
                            for stdout_line in data["stdout"].splitlines():
                                m3 = re.match(r'\s*([\w_]+)\s*[:=]\s*[\d.]+', stdout_line)
                                if m3:
                                    measured.add(m3.group(1))
                    except (json.JSONDecodeError, TypeError):
                        pass
        return [t for t in all_targets if t not in measured]

    def _update_control_plane_progress(self, result: dict) -> None:
        """Update Control Plane with current CodeGen progress after each tool call."""
        try:
            entries = self.context_manager.get_entries()
            compile_count = sum(1 for e in entries
                                if e.role.value == "assistant"
                                and _safe_get_tool(e) == "compile_cuda"
                                and _safe_get_success(e))
            exec_count = sum(1 for e in entries
                             if e.role.value == "assistant"
                             and _safe_get_tool(e) == "execute_binary")
            measured = []
            remaining = []
            all_targets = []
            for entry in entries:
                if entry.role.value == "system" and "targets" in entry.content:
                    m = re.search(r'"targets"\s*:\s*\[([^\]]+)\]', entry.content)
                    if m:
                        all_targets = re.findall(r'"([^"]+)"', m.group(1))
            measured_set = set()
            for entry in entries:
                if entry.role.value == "assistant":
                    try:
                        data = json.loads(entry.content)
                        if isinstance(data, dict) and "stdout" in data:
                            for line in data["stdout"].splitlines():
                                m2 = re.match(r'\s*([\w_]+)\s*[:=]\s*[\d.]+', line)
                                if m2:
                                    measured_set.add(m2.group(1))
                    except (json.JSONDecodeError, TypeError):
                        pass
            measured = [t for t in all_targets if t in measured_set]
            remaining = [t for t in all_targets if t not in measured_set]
            self.control_plane.update_progress(measured, remaining, compile_count, exec_count)
        except Exception:
            pass

    @staticmethod
    def _build_compile_error_guidance(errors: str) -> str:
        """Build specific guidance for common CUDA compilation errors.

        Parses the error output and provides targeted fix suggestions
        to help the LLM fix the code in a single turn.
        """
        guidance_parts = []
        errors_lower = errors.lower()

        if "std" in errors and "sort" in errors_lower:
            guidance_parts.append(
                "\n💡 FIX: Add #include <algorithm> at the top of your .cu file. "
                "std::sort requires this header."
            )
        if "asm operand type size" in errors_lower and "constraint" in errors_lower:
            guidance_parts.append(
                "\n💡 FIX: The +l asm constraint requires a 64-bit (8-byte) operand. "
                "If your variable is uint32_t or int (4 bytes), cast it to uint64_t first:\n"
                "  volatile uint64_t sink64 = (uint64_t)idx;\n"
                "  asm volatile(\"\" : \"+l\"(sink64) : : \"memory\");\n"
                "Or use +r constraint for 32-bit variables:\n"
                "  asm volatile(\"\" : \"+r\"(idx) : : \"memory\");"
            )
        if "undefined reference to clock(" in errors_lower:
            guidance_parts.append(
                "\n💡 FIX: Use clock64() instead of clock(). "
                "clock() returns 0 on Pascal+ GPUs. clock64() returns the actual GPU cycle counter."
            )
        if "cannot bind" in errors_lower and "volatile" in errors_lower:
            guidance_parts.append(
                "\n💡 FIX: Remove 'volatile' from the loop variable itself. "
                "Only use 'volatile' for the sink variable and timing variables (start/end). "
                "The loop variable (idx) should be plain uint32_t, not volatile uint32_t."
            )
        if "was set but never used" in errors_lower or "set but not used" in errors_lower:
            guidance_parts.append(
                "\n💡 FIX: The compiler sees a variable that is assigned but never read. "
                "Add a sink variable to consume the result:\n"
                "  volatile uint64_t sink64 = (uint64_t)result_variable;\n"
                "  asm volatile(\"\" : \"+l\"(sink64) : : \"memory\");"
            )
        if "-arch" in errors_lower or "sm_" in errors_lower:
            guidance_parts.append(
                "\n💡 FIX: Remove or fix the -arch flag. Do NOT use -arch=0 or -arch=sm_0. "
                "Use flags=[\"-O3\"] only, the system will auto-detect the correct architecture."
            )
        if "expected a" in errors_lower and (";" in errors or "{" in errors):
            guidance_parts.append(
                "\n💡 FIX: Syntax error in your CUDA code. Check for missing semicolons, "
                "mismatched braces, or incorrect type declarations."
            )

        if not guidance_parts:
            guidance_parts.append(
                "\n💡 Common fixes:\n"
                "1. Check for missing #include headers (add <algorithm>, <cstring>)\n"
                "2. Check asm volatile constraint types (+l needs 64-bit, +r for 32-bit)\n"
                "3. Use clock64() not clock()\n"
                "4. Ensure all braces and semicolons are matched\n"
                "5. Do NOT use -arch=0 or -arch=sm_0 in flags"
            )

        guidance_parts.append(
            "\n\n⚠️ IMPORTANT: Fix the error and call compile_cuda AGAIN with the corrected source. "
            "Do NOT give up — compilation errors are normal and fixable!"
        )

        return "".join(guidance_parts)

    def _execute_tool_call(self, tool_call: ToolCall) -> dict[str, Any]:
        contract = self.tool_registry.get(tool_call.name)

        if contract is None:
            return {
                "tool": tool_call.name,
                "status": "error",
                "error": f"Tool '{tool_call.name}' is not available for your role. "
                         f"Use only the tools listed in your system prompt.",
            }

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

        max_approval_retries = 3
        retry_count = 0

        while retry_count < max_approval_retries:
            try:
                result = self._tool_executor(tool_call.name, tool_call.arguments)
                return result

            except ApprovalRequiredError as e:
                retry_count += 1
                self._emit(EventKind.APPROVAL_REQUEST, {
                    "tool": tool_call.name,
                    "request_id": e.request.id,
                    "retry_count": retry_count,
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
                        "retry_count": retry_count,
                    })
                    continue
                else:
                    self._emit(EventKind.APPROVAL_DENIED, {
                        "tool": tool_call.name,
                        "request_id": e.request.id,
                    })
                    raise PermissionError(
                        f"Tool '{tool_call.name}' approval denied"
                    ) from e

        raise RuntimeError(
            f"Approval loop detected for '{tool_call.name}' after {max_approval_retries} retries. "
            f"This indicates a systemic issue with the approval system."
        )

    def _respond_to_approval_queue(self, request, approved: bool) -> None:
        if hasattr(self._tool_executor, "_approval_queue"):
            self._tool_executor._approval_queue.respond(request.id, approved)

    def _test_approval_flow(self) -> dict:
        """Test the approval flow end-to-end.

        Used by StageExecutor to verify approval callback connectivity
        before starting pipeline execution.
        """
        try:
            if self._approval_callback is None:
                return {"success": False, "error": "No approval callback set"}

            from src.application.approval_queue import ApprovalRequest
            test_request = ApprovalRequest(
                id="test_001",
                tool_name="test_tool",
                arguments={},
                permissions=["test"],
            )

            approved = self._approval_callback(test_request)
            if not approved:
                return {"success": False, "error": "Callback returned False"}

            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

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


def _safe_get_tool(entry) -> str:
    try:
        data = json.loads(entry.content)
        if isinstance(data, dict):
            return data.get("tool", "")
    except (json.JSONDecodeError, TypeError):
        pass
    return ""


def _safe_get_success(entry) -> bool:
    """Check if a tool execution was successful.

    Handles both Python bool (True) and string ("true") success values
    for consistency with _already_executed_binary and _find_last_binary_path.
    """
    try:
        data = json.loads(entry.content)
        if isinstance(data, dict):
            success_val = data.get("success")
            return success_val is True or (isinstance(success_val, str) and success_val.lower() == "true")
    except (json.JSONDecodeError, TypeError):
        pass
    return False

