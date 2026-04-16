"""Stage executor — runs a single pipeline stage inside an AgentLoop.

Encapsulates the retry loop, AgentLoop wiring, and result extraction
that were previously embedded in the Pipeline God Object.

Design pattern: Template Method — execute() defines the skeleton;
subclasses could override individual steps if needed.
"""
from __future__ import annotations

import json as _json
import re
import uuid
from typing import Any

from src.application.agent_loop import AgentLoop
from src.application.context import Role
from src.application.control_plane import ControlPlane
from src.application.session import SessionState
from src.domain.permission import PermissionMode
from src.domain.pipeline_context import PipelineContext
from src.domain.prompt_builder import StagePromptBuilder
from src.domain.subagent import (
    AgentRole,
    BaseSubAgent,
    CollaborationMessage,
    PipelineStage,
    SubAgentResult,
    SubAgentStatus,
)


class StageExecutor:
    """Executes a single pipeline stage with retry support.

    Responsibilities:
    - Build collaboration message from context
    - Create and wire AgentLoop for LLM-driven iteration
    - Handle retry logic for failed stages
    - Extract structured SubAgentResult from AgentLoop output
    """

    def __init__(
        self,
        state_dir: str,
        sandbox: Any | None = None,
        tool_handlers: dict | None = None,
        max_turns_per_stage: int = 15,
        persister: Any | None = None,
        prompt_builder: StagePromptBuilder | None = None,
    ) -> None:
        self._state_dir = state_dir
        self._sandbox = sandbox
        self._tool_handlers = tool_handlers or {}
        self._max_turns = max_turns_per_stage
        self._persister = persister
        self._prompt_builder = prompt_builder or StagePromptBuilder()

    def execute(
        self,
        step: Any,
        ctx: PipelineContext,
    ) -> SubAgentResult:
        """Execute a pipeline stage with retries.

        Args:
            step: PipelineStep with stage, agent, and retry_on_failure.
            ctx: Current pipeline context for data flow.

        Returns:
            SubAgentResult from the last attempt.
        """
        self._log("pipeline_stage_start", {"stage": step.stage.value, "retry_limit": step.retry_on_failure})
        print(f"[StageExecutor] Starting stage {step.stage.value} with {step.retry_on_failure} retries")

        last_result: SubAgentResult | None = None

        for attempt in range(1 + step.retry_on_failure):
            if attempt > 0:
                self._log("pipeline_retry", {"stage": step.stage.value, "attempt": attempt})
                print(f"[StageExecutor] Retry {attempt} for stage {step.stage.value}")

            message = self._build_collaboration_message(step, ctx)
            last_result = self._run_with_agent_loop(step, message, ctx)

            print(f"[StageExecutor] Attempt status: {last_result.status.value}")
            if last_result.is_success():
                break
            if last_result.status == SubAgentStatus.REJECTED:
                break

            self._log("pipeline_attempt_failed", {
                "stage": step.stage.value,
                "attempt": attempt + 1,
                "error": last_result.error,
            })

        if last_result is None:
            last_result = SubAgentResult(
                agent_role=step.agent.role,
                status=SubAgentStatus.FAILED,
                error="Stage produced no result after all retries",
            )

        print(f"[StageExecutor] Stage {step.stage.value} finished: {last_result.status.value}")
        return last_result

    def _build_collaboration_message(
        self, step: Any, ctx: PipelineContext,
    ) -> CollaborationMessage:
        """Build the collaboration message from pipeline context."""
        payload: dict[str, Any] = {}
        if ctx.target_spec:
            payload["target_spec"] = ctx.target_spec
        if ctx.prev_result is not None:
            payload["prev_result"] = ctx.prev_result.to_dict()
            payload["prev_fingerprint"] = ctx.prev_result.context_fingerprint

        return CollaborationMessage(
            sender=ctx.prev_result.agent_role if ctx.prev_result else AgentRole.PLANNER,
            receiver=step.agent.role,
            message_type="task_dispatch",
            payload=payload,
        )

    def _run_with_agent_loop(
        self, step: Any, message: CollaborationMessage, ctx: PipelineContext,
    ) -> SubAgentResult:
        """Run a pipeline stage inside an AgentLoop for iteration."""
        agent = step.agent
        stage_name = step.stage.value

        payload = message.payload
        task = payload.get("task", {})
        prev_result = payload.get("prev_result", {})
        target_spec = payload.get("target_spec", {})

        if not task and stage_name == PipelineStage.CODE_GEN.value:
            tasks_list = prev_result.get("data", {}).get("tasks", [])
            if tasks_list:
                task = {"tasks": tasks_list}

        system_prompt = self._build_system_prompt(agent, step.stage)
        user_task = self._build_user_task(step.stage, task, prev_result, target_spec)

        session_id = f"pipeline_{stage_name}_{uuid.uuid4().hex[:6]}"
        session = SessionState(session_id=session_id, goal=f"Pipeline stage: {stage_name}")
        control_plane = ControlPlane(rule_dir=None)

        loop = AgentLoop(
            session=session,
            context_manager=agent.context_manager,
            control_plane=control_plane,
            tool_registry=agent.tool_registry,
            max_turns=self._max_turns,
            state_dir=self._state_dir,
            permission_mode=agent.permission_mode,
        )

        if agent._model_caller is not None:
            loop.set_model_caller(agent._model_caller)
        else:
            print(f"[StageExecutor] WARNING: No model caller for {stage_name}!")

        if self._sandbox and self._tool_handlers:
            handlers = dict(self._tool_handlers)
            loop.set_tool_executor(lambda tool_name, args: handlers[tool_name](args))
            loop.set_available_tools(self._build_tool_schemas(handlers, agent.tool_registry))

        if agent.permission_mode == PermissionMode.HIGH_AUTONOMY:
            loop.set_approval_callback(lambda request: True)

        agent.context_manager.add_entry(Role.SYSTEM, system_prompt, token_count=50)
        agent.context_manager.add_entry(Role.USER, user_task, token_count=30)

        try:
            print(f"[StageExecutor] Starting AgentLoop for {stage_name}")
            loop.start()
            print(f"[StageExecutor] AgentLoop finished for {stage_name} (turns={loop.loop_state.turn_count})")
        except Exception as e:
            print(f"[StageExecutor] AgentLoop CRASHED in {stage_name}: {e}")
            return SubAgentResult(
                agent_role=agent.role,
                status=SubAgentStatus.FAILED,
                error=f"AgentLoop failed in {stage_name}: {e}",
            )

        return self._extract_result(agent, step.stage, loop)

    def _build_system_prompt(self, agent: BaseSubAgent, stage: PipelineStage) -> str:
        """Build system prompt for a pipeline stage's AgentLoop."""
        available_tools = agent.tool_registry.list_tools()
        tool_list = _json.dumps(available_tools, indent=2) if available_tools else "(no tools registered)"

        tool_guidance = self._get_tool_guidance(stage)

        return (
            f"You are the {stage.value} stage in a GPU hardware profiling pipeline.\n"
            f"{agent._build_system_prompt()}\n\n"
            f"Available tools: {tool_list}\n\n"
            f"Tool call format: {{\"tool\": \"tool_name\", \"args\": {{\"key\": \"value\"}}}}\n"
            f"After each tool call result, you may call more tools or give your final answer.\n"
            f"When done, give your final answer as plain text (not JSON)."
            f"{tool_guidance}"
        )

    def _build_user_task(
        self,
        stage: PipelineStage,
        task: dict,
        prev_result: dict,
        target_spec: dict,
    ) -> str:
        """Build the user task prompt for a stage."""
        if stage == PipelineStage.PLAN:
            return (
                f"You are the PLANNER stage. Analyze these GPU profiling targets "
                f"and decompose them into actionable tasks.\n\n"
                f"Targets: {_json.dumps(target_spec, indent=2)}\n\n"
                f"Return a JSON array of task objects with: "
                f'"target", "category" (latency_measurement, capacity_measurement, '
                f'clock_measurement, bandwidth_measurement, or unknown), '
                f'"method" (detailed description of the measurement approach).'
            )

        if task:
            return self._prompt_builder.build_task_prompt(stage, target_spec, prev_result)
        if prev_result:
            return self._prompt_builder.build_task_prompt(stage, target_spec, prev_result)

        return f"You are the {stage.value} stage. Execute your task."

    @staticmethod
    def _get_tool_guidance(stage: PipelineStage) -> str:
        """Return stage-specific tool usage instructions."""
        if stage == PipelineStage.CODE_GEN:
            return (
                "\n\n🛠️ YOUR TOOLS: compile_cuda, execute_binary, write_file, read_file\n"
                "🎯 YOUR JOB: Write CUDA code → compile → execute → report values\n"
                "❌ DO NOT: run ncu (that's MetricAnalysis's job)\n"
                "❌ DO NOT: verify results (that's Verification's job)\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "⚠️  CRITICAL: FILE PATH FORMAT (READ THIS FIRST)\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "You MUST use the .sandbox directory for ALL file operations.\n\n"
                "✅ CORRECT: .sandbox/benchmark.cu\n"
                "❌ WRONG: /kaggle/working/gpu_profiling_system/benchmark.cu (path escape)\n"
                "❌ WRONG: benchmark.cu (missing .sandbox prefix)\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "📋 MANDATORY WORKFLOW — Process EACH Target Separately\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "For each target:\n"
                '  1. compile_cuda(source="...full .cu source...", flags=["-O3"])\n'
                '  2. execute_binary(binary_path="<path from compile_cuda>")\n'
                "  3. Record the measured value from stdout\n\n"
                "⚠️  CRITICAL: compile_cuda OVERWRITES the previous binary each time.\n"
                "So you MUST execute_binary IMMEDIATELY after each compile_cuda.\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "🔧 TOOL CALL FORMAT\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                'You MUST call tools as JSON objects, one at a time:\n\n'
                '{{"tool": "write_file", "args": {{"file_path": ".sandbox/benchmark.cu", "content": "...code..."}}}}\n'
                '{{"tool": "compile_cuda", "args": {{"source": "...full .cu source code...", "flags": ["-O3"]}}}}\n'
                '{{"tool": "execute_binary", "args": {{"binary_path": "<path from compile_cuda>"}}}}\n\n'
                "After each tool call, you will see the result. Then call the next tool.\n"
                "❌ DO NOT just describe what you would do — ACTUALLY CALL the tools.\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "🔄 ERROR RECOVERY\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "- Compilation error → fix source code → retry compile_cuda\n"
                "- Execution error → fix binary path or code → recompile → retry\n"
                "- Implausible output (0, negative, NaN) → fix measurement logic → retry\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "✅ MANDATORY REQUIREMENT\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "You MUST call compile_cuda and execute_binary for EACH target.\n"
                "The pipeline will FAIL if you do not call compile_cuda.\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "🛡️  ANTI-CHEAT AWARENESS\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "- ❌ Do NOT rely solely on cudaGetDeviceProperties — may return virtualized data\n"
                "- ✅ Use clock64() + cudaEventElapsedTime to measure actual hardware behavior\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "📊 OUTPUT FORMAT\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "After all targets, list results as:\n"
                "target_name: numeric_value\n"
                "for each target measured.\n"
            )
        if stage == PipelineStage.METRIC_ANALYSIS:
            return (
                "\n\nYOUR TOOLS: run_ncu, read_file\n"
                "YOUR JOB: Profile CodeGen's binaries with ncu → analyze bottlenecks → extract metrics\n"
                "DO NOT: write/compile CUDA code (that's CodeGen's job)\n"
                "DO NOT: verify results (that's Verification's job)\n\n"
                "WORKFLOW:\n"
                "1. Review CodeGen's stdout output provided in the task description\n"
                "2. If binary paths are available AND ncu is installed, use run_ncu on each binary\n"
                "3. If ncu is NOT available OR binaries are not found:\n"
                "   - Analyze the raw printf output from CodeGen directly\n"
                "   - Extract numeric values and validate against expected ranges\n"
                "   - Report confidence as 'low' with note 'ncu not available'\n"
                "4. Classify bottleneck: compute_bound, memory_bound, latency_bound, cache_capacity\n"
                "5. Report metrics with confidence levels\n\n"
                "OUTPUT FORMAT: For each target:\n"
                "  target_name: measured_value (confidence: high/medium/low) [bottleneck_type]"
            )
        if stage == PipelineStage.VERIFICATION:
            return (
                "\n\nYOUR TOOL: read_file ONLY\n"
                "YOUR JOB: Independently review all previous stage results\n"
                "You CANNOT: compile, execute, profile, write files, or generate measurements\n\n"
                "IMPORTANT: All previous stage data is provided in the task description below.\n"
                "You do NOT need to use read_file to find data — it is already given to you.\n"
                "Only use read_file if you need to check a specific evidence file.\n\n"
                "VERIFICATION CHECKS (perform in order):\n"
                "1. Data completeness — are ALL targets measured?\n"
                "2. Numeric sanity — are values in plausible GPU hardware ranges?\n"
                "3. Latency hierarchy — L1 < L2 < DRAM (when both measured)\n"
                "4. Cross-validation — do CodeGen and MetricAnalysis agree?\n"
                "5. Methodology soundness — were correct techniques used?\n\n"
                "State your verdict as: Verdict: ACCEPT or Verdict: REJECT"
            )
        return ""

    @staticmethod
    def _build_tool_schemas(handlers: dict, tool_registry: Any) -> list[dict]:
        """Build OpenAI-format tool schemas from handler dict + registry contracts.
        
        Only includes tools that are both in handlers AND registered in tool_registry.
        This ensures agents only see tools they are allowed to use per their role.
        """
        tools: list[dict] = []
        for name in handlers:
            # Skip tools not in registry (agent doesn't have permission for this tool)
            if not tool_registry.has_tool(name):
                print(f"[StageExecutor] Tool '{name}' not in registry for this agent role, skipping")
                continue
            try:
                contract = tool_registry.get(name)
                properties = {}
                for key, type_str in contract.input_schema.items():
                    properties[key] = {"type": type_str}
                tools.append({
                    "type": "function",
                    "function": {
                        "name": contract.name,
                        "description": contract.description,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": list(contract.input_schema.keys()),
                        },
                    },
                })
            except KeyError:
                # This should not happen since we checked has_tool() above, but handle gracefully
                print(f"[StageExecutor] WARNING: Tool contract for '{name}' not found in registry")
                tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": f"Tool: {name}",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    },
                })
        return tools

    def _extract_result(
        self,
        agent: BaseSubAgent,
        stage: PipelineStage,
        loop: AgentLoop,
    ) -> SubAgentResult:
        """Extract a SubAgentResult from the AgentLoop's final context."""
        entries = agent.context_manager.get_entries()

        tool_results: list[dict] = []
        assistant_outputs: list[str] = []
        model_prompt_prefix = ("You are the", "Available tools", "Instructions:")

        for entry in entries:
            if entry.role.value != "assistant":
                continue
            try:
                data = _json.loads(entry.content)
                if isinstance(data, dict) and ("status" in data or "tool" in data):
                    tool_results.append(data)
                elif isinstance(data, (dict, list)):
                    assistant_outputs.append(entry.content)
            except (_json.JSONDecodeError, TypeError):
                content = entry.content.strip()
                if content and not content.startswith(model_prompt_prefix):
                    assistant_outputs.append(content)

        final_text = assistant_outputs[-1] if assistant_outputs else ""
        data: dict[str, Any] = {
            "tool_results": tool_results,
            "final_output": final_text[:3000],
            "num_tool_calls": len(tool_results),
        }

        _empty_placeholders = {
            "[Empty model output - will retry next turn]",
            "[Empty model output]",
        }
        effective_text = final_text if final_text not in _empty_placeholders else ""

        status = self._determine_status(stage, effective_text, tool_results, data)

        error_msg = self._build_error_message(stage, status, data, effective_text, tool_results)

        result = SubAgentResult(
            agent_role=agent.role,
            status=status,
            data=data,
            artifacts=[],
            error=error_msg or None,
        )

        result.context_fingerprint = result.compute_fingerprint(agent.context_manager)
        self._log("stage_result", {
            "stage": stage.value,
            "status": status.value,
            "tool_calls": len(tool_results),
            "output_length": len(final_text),
        })

        return result

    @staticmethod
    def _determine_status(
        stage: PipelineStage,
        final_text: str,
        tool_results: list[dict],
        data: dict[str, Any],
    ) -> SubAgentStatus:
        """Determine the SubAgentStatus for a stage result."""
        if stage == PipelineStage.PLAN:
            data["plan_text"] = final_text[:2000]
            if not final_text and not tool_results:
                data["error_detail"] = "Planner produced no output"
            return SubAgentStatus.SUCCESS if final_text else SubAgentStatus.FAILED

        if stage == PipelineStage.CODE_GEN:
            return StageExecutor._codegen_status(final_text, tool_results, data)

        if stage == PipelineStage.METRIC_ANALYSIS:
            data["analysis_output"] = final_text[:2000]
            return SubAgentStatus.SUCCESS if final_text else SubAgentStatus.FAILED

        if stage == PipelineStage.VERIFICATION:
            return StageExecutor._verification_status(final_text, data)

        return SubAgentStatus.SUCCESS

    @staticmethod
    def _codegen_status(
        final_text: str, tool_results: list[dict], data: dict[str, Any],
    ) -> SubAgentStatus:
        """Determine CodeGen-specific status and extract measurements."""
        data["code_gen_output"] = final_text[:2000]

        has_compile = any(
            r.get("tool") == "compile_cuda" or r.get("binary_path")
            for r in tool_results
        )
        tool_succeeded = any(
            r.get("status") in ("success", True) or r.get("success") is True
            for r in tool_results
        )
        has_binary = any(r.get("binary_path") for r in tool_results)
        has_output = any(r.get("stdout") for r in tool_results)
        exec_succeeded = any(
            r.get("return_code", -1) == 0 for r in tool_results
            if "return_code" in r or "stdout" in r
        )

        if tool_results and (tool_succeeded or has_binary or has_output or exec_succeeded):
            status = SubAgentStatus.SUCCESS
        else:
            status = SubAgentStatus.FAILED
            if not tool_results and not final_text:
                data["error_detail"] = "CodeGen produced no output and made no tool calls"
            elif not tool_results:
                data["error_detail"] = (
                    f"CodeGen output was text-only (no tool calls). "
                    f"Model must call compile_cuda to generate benchmarks. "
                    f"Output preview: {final_text[:200]}"
                )
            elif not tool_succeeded and not has_binary:
                data["error_detail"] = "CodeGen compilation failed — check tool call results for errors"

        measurements: dict[str, float] = {}
        methodology_parts: list[str] = []
        for r in tool_results:
            if "stdout" in r:
                stdout = r.get("stdout", "")
                for line in stdout.splitlines():
                    line = line.strip()
                    if ":" in line and not line.startswith("//"):
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            val_str = parts[1].strip()
                            try:
                                measurements[key] = float(val_str)
                                continue
                            except ValueError:
                                pass
            if "output" in r and r.get("output"):
                out = r["output"]
                if len(out) < 500:
                    methodology_parts.append(out[:200])

        if measurements:
            data["measurements"] = measurements
        if final_text:
            methodology_parts.append(final_text[:1000])
        if methodology_parts:
            data["analysis_method"] = "\n---\n".join(methodology_parts[:5])

        return status

    @staticmethod
    def _verification_status(final_text: str, data: dict[str, Any]) -> SubAgentStatus:
        """Determine Verification-specific status from verdict text."""
        data["review_text"] = final_text[:2000]
        lower = final_text.lower()

        has_accept_word = bool(re.search(r'\baccept(?:ed|ing)?\b', lower))
        has_reject_word = bool(re.search(r'\breject(?:ed|ing|ion)?\b', lower))
        has_not_valid = "not valid" in lower or "is not valid" in lower
        has_cannot_accept = "cannot accept" in lower or "do not accept" in lower or "don't accept" in lower
        verdict_accept = bool(re.search(r'\bverdict\s*:\s*accept\b', lower))
        verdict_reject = bool(re.search(r'\bverdict\s*:\s*reject\b', lower))

        if verdict_reject or has_reject_word or has_not_valid or has_cannot_accept:
            return SubAgentStatus.REJECTED
        if verdict_accept or has_accept_word:
            return SubAgentStatus.SUCCESS
        return SubAgentStatus.SUCCESS

    @staticmethod
    def _build_error_message(
        stage: PipelineStage,
        status: SubAgentStatus,
        data: dict[str, Any],
        final_text: str,
        tool_results: list[dict],
    ) -> str:
        """Build error message for failed/rejected results."""
        if status == SubAgentStatus.FAILED:
            error_msg = data.get("error_detail", "")
            if not error_msg and not final_text and not tool_results:
                error_msg = f"Stage {stage.value} produced no output"
            elif not error_msg and final_text:
                error_msg = f"Stage {stage.value} failed. Output: {final_text[:500]}"
            else:
                error_msg = error_msg or f"Stage {stage.value} failed"
            return error_msg

        if status == SubAgentStatus.REJECTED:
            concerns = data.get("concerns", [])
            review = data.get("review", [])
            review_text = data.get("review_text", "")[:300]
            if concerns:
                return f"Verification rejected: {'; '.join(concerns)}"
            if review:
                return f"Verification rejected: {'; '.join(str(r) for r in review[:5])}"
            if review_text:
                return f"Verification rejected: {review_text}"
            return "Verification rejected (no reason provided)"

        return ""

    def _log(self, action: str, details: dict | None = None) -> None:
        """Persist a log entry if persister is available."""
        if self._persister:
            self._persister.log_entry(action, details=details)
