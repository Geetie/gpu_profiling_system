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

    # Target state machine fields (for CodeGen stage)
    current_target: str | None = None
    completed_targets: list[str] = field(default_factory=list)
    target_retry_count: dict[str, int] = field(default_factory=dict)

    # C-01 FIX: LLM stall detection fields
    consecutive_no_tool_calls: int = 0
    last_tool_call_turn: int = 0
    stall_recovery_triggered: bool = False

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
            # Include target state in serialization
            "current_target": self.current_target,
            "completed_targets": self.completed_targets,
            "target_retry_count": self.target_retry_count,
            # C-01 FIX: Include stall detection state
            "consecutive_no_tool_calls": self.consecutive_no_tool_calls,
            "last_tool_call_turn": self.last_tool_call_turn,
            "stall_recovery_triggered": self.stall_recovery_triggered,
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
            # Restore target state from dict
            current_target=data.get("current_target"),
            completed_targets=data.get("completed_targets", []),
            target_retry_count=data.get("target_retry_count", {}),
            # C-01 FIX: Restore stall detection state
            consecutive_no_tool_calls=data.get("consecutive_no_tool_calls", 0),
            last_tool_call_turn=data.get("last_tool_call_turn", 0),
            stall_recovery_triggered=data.get("stall_recovery_triggered", False),
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

    def _init_target_state(self, target_spec: dict[str, Any] | None = None) -> None:
        """Initialize the target state machine for CodeGen stage.

        Args:
            target_spec: Dictionary containing 'targets' list from target_spec.json
        """
        if not target_spec:
            return

        targets = target_spec.get("targets", [])
        if targets:
            self.loop_state.current_target = targets[0]
            self.loop_state.completed_targets = []
            self.loop_state.target_retry_count = {t: 0 for t in targets}
            print(f"[AgentLoop] Target state initialized: current={targets[0]}, total={len(targets)}")

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

        # C-02 FIX: Independent state machine synchronization
        # BUG#3 FIX: Reduced from every 3 turns to every 2 turns for faster detection
        if self.loop_state.turn_count % 2 == 0:
            print(f"[AgentLoop] Triggering C-02 state sync (Turn {self.loop_state.turn_count})")
            self._sync_target_state_machine()

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
            # C-01 FIX: Reset stall counter on successful tool call detection
            if self.loop_state.consecutive_no_tool_calls > 0:
                print(f"[AgentLoop] ✅ Tool call detected: {tool_call.name} - "
                      f"resetting stall counter (was: {self.loop_state.consecutive_no_tool_calls})")
                self.loop_state.consecutive_no_tool_calls = 0
                self.loop_state.last_tool_call_turn = self.loop_state.turn_count
                self.loop_state.stall_recovery_triggered = False

            if tool_call.name == "compile_cuda":
                # Step 4: Check per-target retry limit to prevent infinite recompilation loops
                MAX_RETRIES_PER_TARGET = 2  # Reduced from 3 to handle LLM syntax errors faster
                current_target = self.loop_state.current_target

                if current_target:
                    retry_count = self.loop_state.target_retry_count.get(current_target, 0)

                    # Check if we've exceeded max retries
                    should_force_switch = False
                    force_reason = ""

                    if retry_count >= MAX_RETRIES_PER_TARGET:
                        should_force_switch = True
                        force_reason = f"max retries ({MAX_RETRIES_PER_TARGET})"

                    # Find next unmeasured target
                    remaining = self._find_unmeasured_targets()

                    if should_force_switch and remaining:
                        next_target = remaining[0]

                        # Build context-aware guidance based on failure reason
                        force_guidance = (
                            f"🚨 FORCE SWITCH TRIGGERED for '{current_target}' ({force_reason})\n\n"
                            f"⚠️ You have attempted '{current_target}' {retry_count} times.\n"
                            f"⚠️ The pipeline is FORCING you to move to the next target.\n\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"👉 IMMEDIATE ACTION REQUIRED:\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                            f"You MUST now measure: **{next_target}**\n\n"
                            f"STEP 1: Call compile_cuda with COMPLETELY NEW CUDA code for '{next_target}'\n"
                            f"  → Do NOT reuse any code from '{current_target}'\n"
                            f"  → Each target needs a UNIQUE kernel design\n\n"
                            f"STEP 2: After successful compilation, call execute_binary IMMEDIATELY\n\n"
                            f"⚠️ CRITICAL WARNINGS:\n"
                            f"  • Do NOT attempt to fix '{current_target}' again - it's marked as FAILED\n"
                            f"  • Do NOT output text explanations - CALL compile_cuda NOW\n"
                            f"  • The pipeline will NOT wait for perfect code - move on to next target\n"
                        )
                        self.context_manager.add_entry(Role.SYSTEM, force_guidance, token_count=80)

                        # Update state machine
                        self.loop_state.current_target = next_target
                        self.loop_state.target_retry_count[next_target] = 0
                        print(f"[AgentLoop] FORCE SWITCH: {current_target} -> {next_target} "
                              f"(reason: {force_reason}, attempts: {retry_count})")

                        # Block this compile_cuda call by returning early
                        self._persist_state()
                        return

                # Increment retry count for current target
                if current_target:
                    self.loop_state.target_retry_count[current_target] = \
                        self.loop_state.target_retry_count.get(current_target, 0) + 1
# P0 FIX: Inject current_target into compile_cuda arguments for target-specific binary names
                if tool_call.name == "compile_cuda" and current_target:
                    tool_call.arguments["target"] = current_target

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
                
                last_bp = self._find_last_binary_path()
                last_tool_was_compile = self._last_tool_was_compile()
                already_ran = self._already_executed_binary(last_bp) if last_bp else False
                
                if not last_bp:
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
                
                if already_ran and not last_tool_was_compile:
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
                
                tool_call.arguments["binary_path"] = last_bp
                reason = "latest compile not yet executed" if last_tool_was_compile else "auto-replaced with actual compiled binary_path"
                print(f"[AgentLoop] P2 auto-inject: binary_path={last_bp} (reason: {reason})")
            print(f"[AgentLoop] Tool call: {tool_call.name}({list(tool_call.arguments.keys())})")
            self._emit(EventKind.TOOL_CALL, {
                "tool": tool_call.name,
                "args": tool_call.arguments,
            })
            try:
                result = self._execute_tool_call(tool_call)
                print(f"[AgentLoop] Tool result: {tool_call.name} -> {str(result)[:200]}")

                # CRITICAL FIX: After successful compile_cuda, FORCE execute_binary before next compile
                if (tool_call.name == "compile_cuda" and isinstance(result, dict)
                    and result.get("success") == True):

                    # Check if we already executed binary after last compilation
                    entries = self.context_manager.get_entries()
                    has_recent_execute = False
                    for entry in reversed(entries[-5:]):  # Check last 5 entries
                        if (entry.role.value == "assistant"
                            and '"tool": "execute_binary"' in entry.content):
                            has_recent_execute = True
                            break

                    if not has_recent_execute:
                        # Find the compiled binary path from result
                        binary_path = None
                        output = result.get("output", "")
                        if output and "binary_path" in output:
                            import re as re_module
                            bp_match = re_module.search(r'binary_path["\']?\s*[:=]\s*["\']?([^\s"\',]+)', output)
                            if bp_match:
                                binary_path = bp_match.group(1)

                        if not binary_path:
                            # Fallback to target-specific path
                            current_target = self.loop_state.current_target or "unknown"
                            safe_target = current_target.replace(" ", "_").replace("-", "_").lower()
                            binary_path = f".kaggle_sandbox/bin/benchmark_{safe_target}"

                        force_exec_guidance = (
                            f"🎯 MANDATORY: EXECUTE COMPILED BINARY\n\n"
                            f"✅ Compilation SUCCESSFUL for '{self.loop_state.current_target}'!\n"
                            f"⚠️ You MUST now execute the compiled binary BEFORE compiling again.\n\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"👉 IMMEDIATE ACTION: Call execute_binary NOW\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                            f'Step 1: Call execute_binary with:\n'
                            f'  {{"tool": "execute_binary", "args": {{"binary_path": "{binary_path}"}}}}\n\n'
                            f"⚠️ FORBIDDEN:\n"
                            f"  • Do NOT call compile_cuda again without executing first\n"
                            f"  • Do NOT modify your CUDA code - it already compiles successfully\n"
                            f"  • Do NOT output text explanations - CALL execute_binary NOW\n\n"
                            f"After execution, you will receive measurement results.\n"
                            f"Then you can proceed to the next target."
                        )
                        self.context_manager.add_entry(
                            Role.SYSTEM,
                            force_exec_guidance,
                            token_count=90,
                        )
                        print(f"[AgentLoop] Forced execute_binary guidance after successful compilation")
                tool_status = result.get("status", "success") if isinstance(result, dict) else "success"
                self._emit(EventKind.TOOL_RESULT, {
                    "tool": tool_call.name,
                    "status": tool_status,
                })
                # Estimate token count from content length
                # Track which tool was called (for _already_executed_binary detection)
                if isinstance(result, dict) and "tool" not in result:
                    result["tool"] = tool_call.name
                # P1 FIX: Add target metadata to tool results for better traceability
                if (isinstance(result, dict) and self.loop_state.current_target 
                    and "target" not in result):
                    result["target"] = self.loop_state.current_target
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

                # Phase C: Smart compilation error detection and guidance for compile_cuda
                if (tool_call.name == "compile_cuda" and isinstance(result, dict)
                    and result.get("success") == False):

                    error_text = result.get("errors", "")
                    smart_guidance = self._detect_cuda_syntax_error(error_text)

                    if smart_guidance:
                        self.context_manager.add_entry(
                            Role.SYSTEM,
                            smart_guidance,
                            token_count=70,
                        )
                        print(f"[AgentLoop] Smart error guidance injected for compile_cuda failure")
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
                        current_target = self.loop_state.current_target
                        compile_count = sum(1 for e in self.context_manager.get_entries()
                                            if e.role.value == "assistant"
                                            and _safe_get_tool(e) == "compile_cuda"
                                            and _safe_get_success(e))

                        # BUG#2 FIX: Detect repeated compilation of already-measured target
                        if current_target and current_target in self.loop_state.completed_targets:
                            print(f"[AgentLoop] ⚠️ BUG#2 DETECTED: Re-compiling already-measured target '{current_target}' "
                                  f"(compile #{compile_count})")

                            unmeasured = self._find_unmeasured_targets()
                            if unmeasured:
                                next_target = unmeasured[0]
                                from src.domain.design_principles import get_design_principle
                                next_principle = get_design_principle(next_target)
                                next_brief = next_principle[:400] if len(next_principle) > 400 else next_principle

                                force_switch_msg = (
                                    f"🚨🚨🚨 FORCED TARGET SWITCH 🚨🚨🚨\n\n"
                                    f"⛔ STOP! You are RE-COMPILING an already-measured target!\n"
                                    f"✅ Target '{current_target}' was ALREADY measured successfully.\n\n"
                                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                    f"👉 FORCED: Switch to NEXT UNMEASURED TARGET\n"
                                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                                    f"You MUST now measure: **{next_target}**\n\n"
                                    f"Remaining unmeasured targets: {unmeasured}\n\n"
                                    f"STEP 1 (MANDATORY): Call compile_cuda with NEW code:\n"
                                    f'  {{"tool": "compile_cuda", "args": {{"source": "...new kernel for {next_target}...", "flags": ["-O3"]}}}}\n\n'
                                    f"⚠️ ABSOLUTELY FORBIDDEN:\n"
                                    f"  • Do NOT compile '{current_target}' again - it's DONE\n"
                                    f"  • Do NOT output text explanations\n"
                                    f"  • Do NOT reuse old CUDA code\n\n"
                                    f"Design principle for '{next_target}':\n{next_brief}"
                                )
                                self.context_manager.add_entry(
                                    Role.SYSTEM,
                                    force_switch_msg,
                                    token_count=120,
                                )
                                # Force switch the state machine
                                self.loop_state.current_target = next_target
                                self.loop_state.target_retry_count[next_target] = 0
                                print(f"[AgentLoop] Force-switched from '{current_target}' to '{next_target}' "
                                      f"(reason: repeated compilation of measured target)")

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

                        # Step 7: Auto-parse execute_binary output and record measurements
                        if stdout:
                            measurements = {}
                            for line in stdout.splitlines():
                                # Skip comment lines and empty lines
                                if line.strip().startswith("//") or line.strip().startswith("#"):
                                    continue
                                # Match patterns like "dram_latency_cycles: 450.5" or "sm_count = 56"
                                m = re.match(r'\s*([\w_]+)\s*[:=]\s*([\d.]+[eE]?[\d]*)', line)
                                if m:
                                    key, val_str = m.group(1), m.group(2)
                                    try:
                                        val = float(val_str)
                                        measurements[key] = val
                                    except ValueError:
                                        pass

                            print(f"[AgentLoop] Parsed {len(measurements)} measurements from stdout: {list(measurements.keys())}")

                            if measurements:
                                newly_measured = []
                                for key, val in measurements.items():
                                    if key not in self.loop_state.completed_targets:
                                        self.loop_state.completed_targets.append(key)
                                        newly_measured.append(key)

                                # Generate structured measurement summary for visibility
                                summary_lines = ["✅ MEASUREMENTS RECORDED:"]
                                for key, val in measurements.items():
                                    marker = " [NEW]" if key in newly_measured else ""
                                    summary_lines.append(f"  • {key}: {val}{marker}")

                                # Add progress info
                                total_targets = len(self.loop_state.completed_targets) + len(self._find_unmeasured_targets())
                                completed_count = len(self.loop_state.completed_targets)
                                summary_lines.append(f"\nProgress: {completed_count}/{total_targets} targets measured")

                                summary = "\n".join(summary_lines)

                                self.context_manager.add_entry(
                                    Role.SYSTEM,
                                    summary,
                                    token_count=30,
                                )
                                print(f"[AgentLoop] Recorded {len(newly_measured)} new measurements: {newly_measured}")

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

                            # Update target state machine: mark current as completed, switch to next
                            prev_target = self.loop_state.current_target
                            if prev_target and prev_target not in self.loop_state.completed_targets:
                                self.loop_state.completed_targets.append(prev_target)
                                print(f"[AgentLoop] Target '{prev_target}' marked as COMPLETED")

                            self.loop_state.current_target = next_target
                            print(f"[AgentLoop] Switching to target: {next_target}")

                            # Build MANDATORY-level guidance message with high token weight
                            guidance = (
                                f"🛑 MANDATORY TARGET SWITCH 🛑\n\n"
                                f"✅ Target '{prev_target}' is now COMPLETED.\n"
                                f"❌ You still have NOT measured these targets: {unmeasured}\n\n"
                                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                f"👉 IMMEDIATE NEXT ACTION REQUIRED:\n"
                                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                                f"You MUST now measure: **{next_target}**\n\n"
                                f"STEP 1: Call compile_cuda with NEW CUDA code for '{next_target}'\n"
                                f"  → Do NOT reuse the previous kernel — each target needs unique code\n"
                                f"  → Use the design principle below for guidance\n\n"
                                f"STEP 2: Call execute_binary with the new binary_path from Step 1\n\n"
                                f"⚠️ FORBIDDEN:\n"
                                f"  • Do NOT continue optimizing or recompile '{prev_target}'\n"
                                f"  • Do NOT output text explanations — CALL compile_cuda NOW\n"
                                f"  • Do NOT skip this target — pipeline WILL FAIL if missing\n\n"
                                f"Design principle for '{next_target}':\n{next_brief}"
                            )
                            self.context_manager.add_entry(
                                Role.SYSTEM,
                                guidance,
                                token_count=100,  # Higher weight to ensure visibility
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
            # C-01 FIX: Enhanced LLM stall detection and forced recovery
            self.loop_state.consecutive_no_tool_calls += 1
            print(f"[AgentLoop] ⚠️ NO TOOL CALL in Turn {self.loop_state.turn_count} "
                  f"(consecutive: {self.loop_state.consecutive_no_tool_calls})")

            self.context_manager.add_entry(
                Role.ASSISTANT,
                self._model_output,
                token_count=20,
            )
            self._emit(EventKind.TURN, {
                "turn": self.loop_state.turn_count,
                "output": self._model_output[:200],
                "stall_detected": True,
                "consecutive_no_tool": self.loop_state.consecutive_no_tool_calls,
            })
            has_tools = len(self.tool_registry.list_tools()) > 0

            # C-01: Aggressive stall detection - force recovery after 2 consecutive no-tool calls
            MAX_CONSECUTIVE_NO_TOOL = 2  # Reduced from default failure_tracker threshold
            if self.loop_state.consecutive_no_tool_calls >= MAX_CONSECUTIVE_NO_TOOL:
                print(f"[AgentLoop] 🚨 STALL DETECTED: {self.loop_state.consecutive_no_tool_calls} "
                      f"consecutive turns without tool calls - triggering FORCED RECOVERY")

                unmeasured = self._find_unmeasured_targets()
                if unmeasured:
                    next_target = unmeasured[0]
                    from src.domain.design_principles import get_design_principle
                    next_principle = get_design_principle(next_target)
                    next_brief = next_principle[:400] if len(next_principle) > 400 else next_principle

                    # C-02 FIX: Independent state machine update - don't wait for tool call
                    prev_target = self.loop_state.current_target
                    if prev_target and prev_target not in self.loop_state.completed_targets:
                        # Mark current target as failed/attempted and move on
                        self.loop_state.completed_targets.append(prev_target)
                        print(f"[AgentLoop] STALL RECOVERY: Force-marking '{prev_target}' as completed "
                              f"(stalled after {self.loop_state.consecutive_no_tool_calls} turns)")

                    self.loop_state.current_target = next_target
                    self.loop_state.target_retry_count[next_target] = 0
                    self.loop_state.consecutive_no_tool_calls = 0  # Reset stall counter
                    self.loop_state.stall_recovery_triggered = True

                    # Build CRITICAL-level forced recovery guidance
                    forced_recovery_guidance = (
                        f"🚨🚨🚨 EMERGENCY STALL RECOVERY ACTIVATED 🚨🚨🚨\n\n"
                        f"⚠️ SYSTEM ALERT: You have stopped calling tools for "
                        f"{MAX_CONSECUTIVE_NO_TOOL} consecutive turns!\n"
                        f"⚠️ This is causing the pipeline to STALL and will eventually CRASH.\n\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"🔥 IMMEDIATE ACTION REQUIRED - NON-NEGOTIABLE 🔥\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                        f"The system is FORCE-SWITCHING you to the next target.\n"
                        f"Previous target '{prev_target}' has been marked as ATTEMPTED.\n\n"
                        f"👉 YOU MUST NOW MEASURE: **{next_target}**\n\n"
                        f"STEP 1 (IMMEDIATE): Call compile_cuda with NEW CUDA code:\n"
                        f'  {{"tool": "compile_cuda", "args": {{"source": "...full .cu source for {next_target}...", "flags": ["-O3"]}}}}\n\n'
                        f"STEP 2 (AFTER COMPILE): Call execute_binary immediately:\n"
                        f'  {{"tool": "execute_binary", "args": {{"binary_path": "<from compile>"}}}}\n\n'
                        f"⛔ ABSOLUTELY FORBIDDEN:\n"
                        f"  • Do NOT output text explanations under any circumstances\n"
                        f"  • Do NOT call any tool other than compile_cuda or execute_binary\n"
                        f"  • Do NOT attempt to fix or optimize the previous target\n"
                        f"  • Do NOT say 'I understand' or 'I will do X' - JUST CALL THE TOOL\n\n"
                        f"Design principle for '{next_target}':\n{next_brief}\n\n"
                        f"💥 FAILURE TO COMPLY WILL RESULT IN PIPELINE TERMINATION 💥"
                    )
                    self.context_manager.add_entry(
                        Role.SYSTEM,
                        forced_recovery_guidance,
                        token_count=150,  # Maximum visibility
                    )
                    print(f"[AgentLoop] Forced recovery activated: switching to '{next_target}'")
                    self._persist_state()
                    return  # Exit this turn to allow recovery on next turn

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
        """Find targets that have not yet been measured in this session.

        Enhanced with robust JSON parsing for both target extraction and measurement detection.
        Supports multiple output formats:
        - JSON stdout from execute_binary results
        - Direct text output with 'key: value' format
        - Various numeric formats (int, float, scientific notation)

        BUG#1 FIX: Now extracts targets from MULTIPLE sources to ensure completeness:
        1. LoopState initialization data (most reliable)
        2. User messages (target_spec from Planner)
        3. System messages (progress reports)
        """
        all_targets = []
        measured = set()
        entries = self.context_manager.get_entries()

        # P0 FIX: Extract from LoopState first (if initialized via _init_target_state)
        if self.loop_state.completed_targets or self.loop_state.current_target:
            # Try to reconstruct full target list from completed + current + unmeasured
            known_targets = set(self.loop_state.completed_targets)
            if self.loop_state.current_target:
                known_targets.add(self.loop_state.current_target)
            if len(known_targets) >= 2:  # At least 2 targets suggests we have the list
                all_targets = list(known_targets)

        # Extract all requested targets from user messages (contains target_spec from Planner)
        if not all_targets:
            for entry in entries:
                if entry.role.value == "user":
                    content = entry.content
                    # Pattern 1: Target specification dict
                    m = re.search(r"'targets'\s*:\s*\[([^\]]+)\]", content)
                    if m:
                        found_targets = re.findall(r"'([^']+)'", m.group(1))
                        if found_targets and len(found_targets) >= 2:  # Valid multi-target spec
                            all_targets = found_targets
                            print(f"[AgentLoop] _find_unmeasured: Found {len(all_targets)} targets from user message")
                            break

                    # Pattern 2: JSON format target specification
                    try:
                        data = json.loads(content)
                        if isinstance(data, dict):
                            targets = data.get("targets", [])
                            if isinstance(targets, list) and len(targets) >= 2:
                                all_targets = [t for t in targets if isinstance(t, str)]
                                print(f"[AgentLoop] _find_unmeasured: Found {len(all_targets)} targets from user JSON")
                                break

                        # Also check nested "Target specification" field
                        target_spec = data.get("target_spec", {})
                        if isinstance(target_spec, dict):
                            targets = target_spec.get("targets", [])
                            if isinstance(targets, list) and len(targets) >= 2:
                                all_targets = [t for t in targets if isinstance(t, str)]
                                print(f"[AgentLoop] _find_unmeasured: Found {len(all_targets)} targets from target_spec")
                                break
                    except (json.JSONDecodeError, TypeError):
                        pass

        # Fallback: Extract from system messages (original logic)
        if not all_targets:
            for entry in entries:
                if entry.role.value == "system" and "targets" in entry.content:
                    try:
                        data = json.loads(entry.content)
                        if isinstance(data, dict):
                            targets = data.get("targets", [])
                            if isinstance(targets, list) and targets:
                                all_targets = [t for t in targets if isinstance(t, str)]
                                print(f"[AgentLoop] _find_unmeasured: Found {len(all_targets)} targets from system message (fallback)")
                                break
                    except (json.JSONDecodeError, TypeError):
                        m = re.search(r'"targets"\s*:\s*\[([^\]]+)\]', entry.content)
                        if m:
                            all_targets = re.findall(r'"([^"]+)"', m.group(1))

        # CRITICAL: If still no targets found but we have measurements, use measurements as reference
        if not all_targets and self.loop_state.completed_targets:
            print(f"[AgentLoop] _find_unmeasured WARNING: No target list found, "
                  f"using {len(self.loop_state.completed_targets)} completed targets as baseline")
            all_targets = list(self.loop_state.completed_targets)

        # Enhanced measurement detection with multiple format support
        for entry in entries:
            if entry.role.value != "assistant":
                continue

            content = entry.content

            # Format 1: JSON structured result (from tool calls)
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    stdout = data.get("stdout", "")
                    if isinstance(stdout, str) and stdout:
                        measurements = self._parse_measurements_from_text(stdout)
                        measured.update(measurements)

                    output = data.get("output", "")
                    if isinstance(output, str) and output:
                        measurements = self._parse_measurements_from_text(output)
                        measured.update(measurements)

                    measurements_field = data.get("measurements", {})
                    if isinstance(measurements_dict := measurements_field, dict):
                        for key, val in measurements_dict.items():
                            if isinstance(key, str) and isinstance(val, (int, float)):
                                measured.add(key)
            except (json.JSONDecodeError, TypeError):
                pass

            # Format 2: Plain text with key-value pairs
            measurements = self._parse_measurements_from_text(content)
            measured.update(measurements)

        unmeasured = [t for t in all_targets if t not in measured]

        # DEBUG: Log the results for troubleshooting
        if all_targets:
            print(f"[AgentLoop] _find_unmeasured: total={len(all_targets)}, "
                  f"measured={len(measured & set(all_targets))}, unmeasured={len(unmeasured)}, "
                  f"all_targets={all_targets}")

        return unmeasured

    def _sync_target_state_machine(self) -> None:
        """C-02 FIX: Independent state machine synchronization.

        Periodically checks and fixes inconsistencies between:
        1. Actual measurements recorded in context
        2. State machine's completed_targets list
        3. Current target pointer

        This prevents the state machine from becoming stale when LLM
        stops calling tools or when tool calls fail silently.
        """
        try:
            # Get actual measured targets from context
            entries = self.context_manager.get_entries()
            measured_from_context = set()

            for entry in entries:
                if entry.role.value != "assistant":
                    continue

                content = entry.content
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        # Check stdout field
                        stdout = data.get("stdout", "")
                        if isinstance(stdout, str) and stdout:
                            measurements = self._parse_measurements_from_text(stdout)
                            measured_from_context.update(measurements)

                        # Check output field
                        output = data.get("output", "")
                        if isinstance(output, str) and output:
                            measurements = self._parse_measurements_from_text(output)
                            measured_from_context.update(measurements)
                except (json.JSONDecodeError, TypeError):
                    pass

                # Also check plain text format
                measurements = self._parse_measurements_from_text(content)
                measured_from_context.update(measurements)

            # Get all expected targets (use same enhanced logic as _find_unmeasured_targets)
            all_targets = []
            # BUG#3 FIX: Use multiple sources like _find_unmeasured_targets does
            if self.loop_state.completed_targets or self.loop_state.current_target:
                known_targets = set(self.loop_state.completed_targets)
                if self.loop_state.current_target:
                    known_targets.add(self.loop_state.current_target)
                if len(known_targets) >= 2:
                    all_targets = list(known_targets)

            if not all_targets:
                for entry in entries:
                    if entry.role.value == "user":
                        content = entry.content
                        m = re.search(r"'targets'\s*:\s*\[([^\]]+)\]", content)
                        if m:
                            found = re.findall(r"'([^']+)'", m.group(1))
                            if found and len(found) >= 2:
                                all_targets = found
                                break
                        try:
                            data = json.loads(content)
                            if isinstance(data, dict):
                                targets = data.get("targets", [])
                                if isinstance(targets, list) and len(targets) >= 2:
                                    all_targets = [t for t in targets if isinstance(t, str)]
                                    break
                        except (json.JSONDecodeError, TypeError):
                            pass

            if not all_targets:
                for entry in entries:
                    if entry.role.value == "system" and "targets" in entry.content:
                        try:
                            data = json.loads(entry.content)
                            if isinstance(data, dict):
                                targets = data.get("targets", [])
                                if isinstance(targets, list) and targets:
                                    all_targets = [t for t in targets if isinstance(t, str)]
                                    break
                        except (json.JSONDecodeError, TypeError):
                            m = re.search(r'"targets"\s*:\s*\[([^\]]+)\]', entry.content)
                            if m:
                                all_targets = re.findall(r'"([^"]+)"', m.group(1))

            # BUG#3 FIX: Don't return early if no targets found - try to use what we have
            if not all_targets:
                if self.loop_state.completed_targets:
                    print(f"[AgentLoop] C-02 SYNC WARNING: No target list found in context, "
                          f"using {len(self.loop_state.completed_targets)} completed targets")
                    all_targets = list(self.loop_state.completed_targets)
                else:
                    print(f"[AgentLoop] C-02 SYNC: No targets found anywhere, skipping sync")
                    return  # Only skip if absolutely no target information available

            # Compare state machine with reality
            state_completed = set(self.loop_state.completed_targets)
            newly_measured = measured_from_context - state_completed
            incorrectly_marked_completed = state_completed - measured_from_context

            if newly_measured:
                print(f"[AgentLoop] C-02 SYNC: Found {len(newly_measured)} newly measured targets "
                      f"not in state machine: {newly_measured}")
                for target in newly_measured:
                    if target not in self.loop_state.completed_targets:
                        self.loop_state.completed_targets.append(target)

            if incorrectly_marked_completed:
                print(f"[AgentLoop] C-02 SYNC: Found {len(incorrectly_marked_completed)} targets "
                      f"incorrectly marked as completed: {incorrectly_marked_completed}")
                # Remove incorrectly marked targets
                self.loop_state.completed_targets = [
                    t for t in self.loop_state.completed_targets
                    if t in measured_from_context or t in all_targets  # Keep if measured or is a valid target
                ]

            # Check if current_target needs update
            unmeasured = [t for t in all_targets if t not in measured_from_context]
            current = self.loop_state.current_target

            if current and current in measured_from_context and unmeasured:
                # Current target is already measured but we haven't moved to next
                next_target = unmeasured[0]
                print(f"[AgentLoop] C-02 SYNC: Current target '{current}' is already measured, "
                      f"force-switching to '{next_target}'")
                self.loop_state.current_target = next_target
                self.loop_state.target_retry_count[next_target] = 0

            elif not current and unmeasured:
                # No current target but there are unmeasured ones
                next_target = unmeasured[0]
                print(f"[AgentLoop] C-02 SYNC: No current target set, initializing to '{next_target}'")
                self.loop_state.current_target = next_target
                self.loop_state.target_retry_count[next_target] = 0

            elif current and current not in all_targets:
                # Current target is invalid (not in target list)
                if unmeasured:
                    next_target = unmeasured[0]
                    print(f"[AgentLoop] C-02 SYNC: Current target '{current}' is invalid, "
                          f"switching to '{next_target}'")
                    self.loop_state.current_target = next_target
                    self.loop_state.target_retry_count[next_target] = 0

            # Log sync status for debugging
            total = len(all_targets)
            completed = len(measured_from_context & set(all_targets))
            remaining = len(unmeasured)
            print(f"[AgentLoop] C-02 SYNC: Status {completed}/{total} measured, "
                  f"{remaining} remaining, current='{self.loop_state.current_target}'")

        except Exception as e:
            print(f"[AgentLoop] C-02 SYNC ERROR: {e}")
            # Don't let sync errors crash the loop

    def _parse_measurements_from_text(self, text: str) -> set[str]:
        """Parse measurement values from text content.

        Args:
            text: Text content to parse (stdout, output, or plain text)

        Returns:
            set[str]: Set of measurement keys found
        """
        measurements = set()
        if not text or not isinstance(text, str):
            return measurements

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Skip comment lines
            if line.startswith("//") or line.startswith("#"):
                continue
            # Match patterns like "dram_latency_cycles: 450.5" or "sm_count = 56"
            m = re.match(r'\s*([\w_]+)\s*[:=]\s*[\d.]+[eE]?[\d]*', line)
            if m:
                measurements.add(m.group(1))

        return measurements

    def _detect_cuda_syntax_error(self, error_text: str) -> str | None:
        """Detect common CUDA compilation errors and provide targeted fix guidance.

        Args:
            error_text: The compilation error message from nvcc

        Returns:
            str | None: Targeted fix guidance, or None if no pattern matched
        """
        if not error_text or not isinstance(error_text, str):
            return None

        error_lower = error_text.lower()

        # Pattern 1: asm volatile syntax error (missing colon)
        if 'asm volatile' in error_lower and ('expected a ";"' in error_lower or 'expected ";"' in error_lower):
            return (
                "🔧 DETECTED: asm volatile() SYNTAX ERROR\n\n"
                "❌ Your code has this WRONG pattern:\n"
                '   asm volatile("") : "+l"(var) : : "memory");  // Extra quote before first colon\n\n'
                "✅ FIX - Change to:\n"
                '   asm volatile("" : "+l"(var) : : "memory"); // No extra quote!\n\n'
                "💡 The FIRST colon must come IMMEDIATELY after the closing quote.\n"
                "   Do NOT put any characters (including quotes) between ) and :\n\n"
                "⚠️ This is the #1 cause of compilation failures. Fix it and retry."
            )

        # Pattern 2: Missing #include
        if ('printf was not declared' in error_lower or "'printf'" in error_lower
            or 'sort was not declared' in error_lower or "'std::sort'" in error_lower):

            missing_includes = []
            if 'printf' in error_lower:
                missing_includes.append('#include <cstdio>')
            if 'sort' in error_lower or 'std::sort' in error_lower:
                missing_includes.append('#include <algorithm>')

            includes_str = '\n'.join(f"   {inc}" for inc in missing_includes)

            return (
                f"🔧 DETECTED: MISSING #include STATEMENT(S)\n\n"
                f"You are using functions without including their headers.\n\n"
                f"✅ FIX - Add these lines at the TOP of your file (before any other code):\n\n"
                f"{includes_str}\n\n"
                f"⚠️ Make sure these are the VERY FIRST lines after your existing #includes."
            )

        # Pattern 3: Format specifier warning/error
        if ("expects argument of type" in error_lower and ('%llu' in error_text or '%lu' in error_text)):
            return (
                "🔧 DETECTED: FORMAT SPECIFIER MISMATCH\n\n"
                "❌ Using wrong format specifier for uint64_t.\n\n"
                "✅ FIX - Use one of these instead:\n"
                '   printf("%lu", (unsigned long)value);  // Cast to unsigned long\n'
                '   // OR include <cinttypes> and use:\n'
                '   printf("%" PRIu64 "\\n", value);  // Portable but requires macro\n\n'
                "⚠️ On Tesla P100 (sm_60), use %lu with (unsigned long) cast."
            )

        # Pattern 4: General syntax error with line number
        if 'error:' in error_lower and 'source.cu(' in error_text:
            import re as re_module
            line_match = re_module.search(r'source\.cu\((\d+)\)', error_text)
            if line_match:
                line_num = line_match.group(1)
                return (
                    f"🔧 COMPILATION ERROR at line {line_num}:\n\n"
                    f"{error_text[:400]}\n\n"
                    f"✅ ACTION REQUIRED:\n"
                    f"1. Look at line {line_num} in your CUDA source\n"
                    f"2. Fix the specific syntax error shown above\n"
                    f"3. Common fixes: check semicolons, brackets, quotes, colons\n"
                    f"4. Do NOT rewrite the entire kernel - just fix this line\n\n"
                    f"⚠️ If you cannot fix it in 1 attempt, the system will force target switch."
                )

        # No pattern matched - return generic guidance
        return None

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

        instance = cls(
            session=session,
            context_manager=context_manager,
            control_plane=control_plane,
            tool_registry=tool_registry,
            max_turns=max_turns,
            state_dir=state_dir,
            permission_mode=permission_mode,
        )

        # CRITICAL FIX: Restore loop_state from persisted data
        # This ensures target_retry_count, current_target, completed_targets
        # are preserved across turns (otherwise FORCE SWITCH never triggers!)
        try:
            persister = StatePersister(log_dir=state_dir)
            last_state_log = persister.get_last_tool_execution("__loop_state__")
            if last_state_log and isinstance(last_state_log.get("inputs"), dict):
                restored_state = LoopState.from_dict(last_state_log["inputs"])
                instance.loop_state.current_target = restored_state.current_target
                instance.loop_state.completed_targets = restored_state.completed_targets
                instance.loop_state.target_retry_count = restored_state.target_retry_count
                print(f"[AgentLoop] Restored target state from persistence: "
                      f"current={restored_state.current_target}, "
                      f"completed={restored_state.completed_targets}, "
                      f"retries={restored_state.target_retry_count}")
        except Exception as e:
            print(f"[AgentLoop] WARNING: Could not restore loop_state from persistence: {e}")

        return instance

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

