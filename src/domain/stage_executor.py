"""Stage executor — runs a single pipeline stage inside an AgentLoop.

Encapsulates the retry loop, AgentLoop wiring, and result extraction
that were previously embedded in the Pipeline God Object.

Design pattern: Template Method — execute() defines the skeleton;
subclasses could override individual steps if needed.
"""
from __future__ import annotations

import json as _json
import logging
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

logger = logging.getLogger(__name__)


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
        logger.info("[StageExecutor] Starting stage %s with %d retries", step.stage.value, step.retry_on_failure)

        last_result: SubAgentResult | None = None

        for attempt in range(1 + step.retry_on_failure):
            if attempt > 0:
                self._log("pipeline_retry", {"stage": step.stage.value, "attempt": attempt})
                logger.info("[StageExecutor] Retry %d for stage %s", attempt, step.stage.value)

            feedback = ctx.get_feedback_for_codegen()
            if feedback and step.stage == PipelineStage.CODE_GEN:
                message = self._build_retry_message(step, ctx, feedback)
            else:
                message = self._build_collaboration_message(step, ctx)
            last_result = self._run_with_agent_loop(step, message, ctx)

            logger.info("[StageExecutor] Attempt status: %s", last_result.status.value)
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

        logger.info("[StageExecutor] Stage %s finished: %s", step.stage.value, last_result.status.value)
        return last_result

    def _build_collaboration_message(
        self, step: Any, ctx: PipelineContext,
    ) -> CollaborationMessage:
        """Build the collaboration message from pipeline context.

        Includes MetricAnalysis feedback when available to enable
        the MetricAnalysis → CodeGen optimization loop.
        Includes CodeGen data for downstream stages (MetricAnalysis, Verification).
        """
        payload: dict[str, Any] = {}
        if ctx.target_spec:
            payload["target_spec"] = ctx.target_spec
        if ctx.prev_result is not None:
            payload["prev_result"] = ctx.prev_result.to_dict()
            payload["prev_fingerprint"] = ctx.prev_result.context_fingerprint

        # Bug fix: Include CodeGen data for downstream stages
        # MetricAnalysis needs CodeGen's tool_results (execute_binary stdout)
        # Verification needs CodeGen's measurements for review
        if ctx.code_gen_data and step.stage in (
            PipelineStage.METRIC_ANALYSIS, PipelineStage.VERIFICATION
        ):
            payload["codegen_data"] = ctx.code_gen_data

        metric_feedback = self._extract_metric_feedback(ctx)
        if metric_feedback and step.stage == PipelineStage.CODE_GEN:
            payload["metric_feedback"] = metric_feedback

        return CollaborationMessage(
            sender=ctx.prev_result.agent_role if ctx.prev_result else AgentRole.PLANNER,
            receiver=step.agent.role,
            message_type="task_dispatch",
            payload=payload,
        )

    @staticmethod
    def _extract_metric_feedback(ctx: PipelineContext) -> str:
        """Extract and format MetricAnalysis feedback for CodeGen injection."""
        if not ctx.metric_feedback:
            return ""

        last = ctx.metric_feedback[-1]
        bottleneck_type = last.get("bottleneck_type", "")
        bottleneck_sub_type = last.get("bottleneck_sub_type", "")
        recommendations = last.get("recommendations", [])
        suggested_fixes = last.get("suggested_fixes", [])

        if not bottleneck_type and not recommendations and not suggested_fixes:
            return ""

        parts = [
            "📊 MetricAnalysis identified bottleneck:",
            f"   Type: {bottleneck_type}",
        ]
        if bottleneck_sub_type:
            parts.append(f"   Sub-type: {bottleneck_sub_type}")

        if recommendations:
            parts.append("")
            parts.append("💡 Optimization suggestions from MetricAnalysis:")
            for rec in recommendations:
                parts.append(f"  - {rec}")

        if suggested_fixes:
            parts.append("")
            parts.append("🔧 Suggested fixes:")
            for fix in suggested_fixes:
                parts.append(f"  → {fix}")

        return "\n".join(parts)

    def _build_retry_message(
        self, step: Any, ctx: PipelineContext, feedback: dict[str, Any],
    ) -> CollaborationMessage:
        """Build a retry collaboration message with rejection and MetricAnalysis feedback."""
        concerns = feedback.get("concerns", [])
        suggested_fixes = feedback.get("suggested_fixes", [])
        iteration = feedback.get("iteration", 0)

        feedback_parts = [
            "⚠️  VERIFICATION REJECTED YOUR PREVIOUS OUTPUT",
            f"Iteration: {iteration}/{ctx.max_iterations}",
            "Please fix the following concerns and regenerate:",
            "",
            *[f"- {concern}" for concern in concerns],
        ]
        if suggested_fixes:
            feedback_parts.append("")
            feedback_parts.append("Suggested fixes:")
            for fix in suggested_fixes:
                feedback_parts.append(f"  → {fix}")

        metric_recommendations = feedback.get("metric_recommendations", [])
        metric_suggested_fixes = feedback.get("metric_suggested_fixes", [])
        bottleneck_type = feedback.get("bottleneck_type", "")
        bottleneck_sub_type = feedback.get("bottleneck_sub_type", "")

        if bottleneck_type or metric_recommendations or metric_suggested_fixes:
            feedback_parts.append("")
            feedback_parts.append("📊 MetricAnalysis feedback:")
            if bottleneck_type:
                sub_info = f"/{bottleneck_sub_type}" if bottleneck_sub_type else ""
                feedback_parts.append(f"   Bottleneck identified: {bottleneck_type}{sub_info}")
            if metric_recommendations:
                feedback_parts.append("   Optimization recommendations:")
                for rec in metric_recommendations:
                    feedback_parts.append(f"  - {rec}")
            if metric_suggested_fixes:
                feedback_parts.append("   Suggested fixes:")
                for fix in metric_suggested_fixes:
                    feedback_parts.append(f"  → {fix}")

        payload: dict[str, Any] = {}
        if ctx.target_spec:
            payload["target_spec"] = ctx.target_spec
        if ctx.code_gen_data:
            payload["prev_result"] = {
                "agent_role": AgentRole.CODE_GEN.value,
                "status": "rejected_retry",
                "data": ctx.code_gen_data,
            }
        payload["rejection_feedback"] = "\n".join(feedback_parts)
        payload["metric_recommendations"] = metric_recommendations
        payload["metric_suggested_fixes"] = metric_suggested_fixes
        payload["bottleneck_type"] = bottleneck_type
        payload["bottleneck_sub_type"] = bottleneck_sub_type

        return CollaborationMessage(
            sender=AgentRole.VERIFICATION,
            receiver=step.agent.role,
            message_type="feedback",
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
        rejection_feedback = payload.get("rejection_feedback", "")

        if not task and stage_name == PipelineStage.CODE_GEN.value:
            tasks_list = prev_result.get("data", {}).get("tasks", [])
            if tasks_list:
                task = {"tasks": tasks_list}

        system_prompt = self._build_system_prompt(agent, step.stage)
        user_task = self._build_user_task(step.stage, task, prev_result, target_spec)

        if rejection_feedback:
            user_task = f"{rejection_feedback}\n\n---\n\n{user_task}"

        if ctx.conversation_history:
            history_summary = self._format_conversation_history(ctx)
            if history_summary:
                user_task = f"{user_task}\n\n--- Conversation History from Previous Iteration ---\n{history_summary}"

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
            logger.warning("[StageExecutor] No model caller for %s!", stage_name)

        if self._sandbox and self._tool_handlers:
            handlers = dict(self._tool_handlers)
            loop.set_tool_executor(lambda tool_name, args: handlers[tool_name](args))
            loop.set_available_tools(self._build_tool_schemas(handlers, agent.tool_registry))

        if agent.permission_mode == PermissionMode.HIGH_AUTONOMY:
            loop.set_approval_callback(lambda request: True)

        agent.context_manager.add_entry(Role.SYSTEM, system_prompt, token_count=50)
        agent.context_manager.add_entry(Role.USER, user_task, token_count=30)

        try:
            logger.info("[StageExecutor] Starting AgentLoop for %s", stage_name)
            loop.start()
            logger.info("[StageExecutor] AgentLoop finished for %s (turns=%d)", stage_name, loop.loop_state.turn_count)
        except Exception as e:
            logger.error("[StageExecutor] AgentLoop CRASHED in %s: %s", stage_name, e)
            return SubAgentResult(
                agent_role=agent.role,
                status=SubAgentStatus.FAILED,
                error=f"AgentLoop failed in {stage_name}: {e}",
            )

        self._save_conversation_history(agent, ctx)

        return self._extract_result(agent, step.stage, loop)

    @staticmethod
    def _save_conversation_history(agent: BaseSubAgent, ctx: PipelineContext) -> None:
        """Save agent's conversation entries to PipelineContext for cross-Stage inheritance."""
        entries = agent.context_manager.get_entries()
        for entry in entries:
            ctx.append_history(entry.role.value, entry.content[:500])

    @staticmethod
    def _format_conversation_history(ctx: PipelineContext) -> str:
        """Format conversation history for injection into a retry prompt."""
        history = ctx.get_history(limit=10)
        if not history:
            return ""
        parts = []
        for entry in history:
            role = entry.get("role", "unknown")
            content = entry.get("content", "")
            if content and len(content) > 20:
                parts.append(f"[{role}]: {content[:300]}")
        return "\n".join(parts)

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
                "⚠️  CRITICAL: 'success_with_warning' means compilation SUCCEEDED.\n"
                "If compile_cuda returns status='success_with_warning', the binary IS valid.\n"
                "You MUST still call execute_binary — warnings do NOT prevent execution.\n"
                "Only status='error' means compilation failed and you need to fix the code.\n\n"
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
                "⚠️ CRITICAL: You MUST call tools as JSON objects. DO NOT just describe analysis.\n"
                "ACTUALLY CALL the tools to get data.\n\n"
                "WORKFLOW:\n"
                "1. Review CodeGen's stdout output provided in the task description\n"
                "2. If binary paths are available AND ncu is installed, call run_ncu:\n"
                '   {"tool": "run_ncu", "args": {"executable": "<binary_path>", "metrics": ["dram__throughput", "l2__throughput"]}}\n'
                "3. If ncu is NOT available OR binaries are not found:\n"
                "   - Analyze the raw printf output from CodeGen directly\n"
                "   - Extract numeric values and validate against expected ranges\n"
                "   - Report confidence as 'low' with note 'ncu not available'\n"
                "4. Classify bottleneck: compute_bound, memory_bound, latency_bound, cache_capacity\n"
                "5. Report metrics with confidence levels\n\n"
                "OUTPUT FORMAT: For each target:\n"
                "  target_name: measured_value (confidence: high/medium/low) [bottleneck_type]\n\n"
                "MANDATORY: Your output MUST include:\n"
                "- bottleneck_type: one of compute_bound, memory_bound, latency_bound, cache_capacity\n"
                "- confidence: high/medium/low\n"
                "- At least one measured value per target\n"
            )
        if stage == PipelineStage.VERIFICATION:
            return (
                "\n\nYOUR TOOL: read_file ONLY\n"
                "YOUR JOB: Independently review all previous stage results\n"
                "You CANNOT: compile, execute, profile, write files, or generate measurements\n\n"
                "⚠️ CRITICAL: DO NOT call read_file! All data is already in the task description above.\n"
                "The task description contains CodeGen measurements and MetricAnalysis results.\n"
                "You only need to ANALYZE the data provided — do NOT try to read files.\n\n"
                "VERIFICATION CHECKS (perform in order):\n"
                "1. Data completeness — are ALL targets measured?\n"
                "2. Numeric sanity — are values in plausible GPU hardware ranges?\n"
                "3. Latency hierarchy — L1 < L2 < DRAM (when both measured)\n"
                "4. Cross-validation — do CodeGen and MetricAnalysis agree?\n"
                "5. Methodology soundness — were correct techniques used?\n\n"
                "⚠️ CRITICAL: You MUST output a clear verdict.\n"
                "State your verdict as: Verdict: ACCEPT or Verdict: REJECT\n\n"
                "MANDATORY OUTPUT FORMAT:\n"
                "Verdict: ACCEPT or REJECT\n"
                "Findings: [list of issues found]\n"
                "Concerns: [list of concerns]\n"
                "If REJECT, provide suggested fixes.\n"
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
                logger.debug("[StageExecutor] Tool '%s' not in registry for this agent role, skipping", name)
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
                logger.warning("[StageExecutor] Tool contract for '%s' not found in registry", name)
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
            "final_output": final_text,
            "num_tool_calls": len(tool_results),
        }

        _empty_placeholders = {
            "[Empty model output - will retry next turn]",
            "[Empty model output]",
        }
        effective_text = final_text if final_text not in _empty_placeholders else ""

        # P0 FIX: For PLAN stage, parse LLM output into structured tasks
        # This bridges AgentLoop's text output with HandoffValidator's contract
        if stage == PipelineStage.PLAN:
            # Pass ALL assistant outputs so _parse_planner_tasks can find JSON in any turn
            tasks = self._parse_planner_tasks(effective_text, agent, assistant_outputs)
            if tasks:
                data["tasks"] = tasks
                logger.info("[StageExecutor] Extracted %d tasks from Planner LLM output", len(tasks))
            else:
                logger.warning("[StageExecutor] Failed to extract tasks from Planner output, "
                               "HandoffValidator will likely reject this")

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
    def _parse_planner_tasks(
        final_text: str, agent: BaseSubAgent, all_assistant_outputs: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Parse Planner LLM output into structured task list.

        Multiple fallback strategies to ensure tasks are always extracted:
        1. Scan ALL assistant outputs (not just final) for valid JSON arrays
        2. Parse final output as direct JSON
        3. Extract JSON array from mixed content
        4. Try code block JSON
        5. Rule-based classification as ultimate fallback
        """
        # Strategy 1: Scan ALL assistant outputs for the FIRST valid JSON task list
        # This handles cases where early turns produced valid JSON but later turns did not
        all_outputs = all_assistant_outputs or []
        if final_text and final_text not in all_outputs:
            all_outputs = list(all_outputs) + [final_text]

        for i, output in enumerate(all_outputs):
            if not output or not output.strip():
                continue
            tasks = StageExecutor._try_extract_tasks(output)
            if tasks:
                logger.info("[StageExecutor] Found valid tasks in assistant output #%d (%d chars)", i+1, len(output))
                return tasks

        logger.info("[StageExecutor] LLM output parsing failed, using rule-based task classification")
        try:
            if hasattr(agent, 'context_manager'):
                cm = agent.context_manager
                entries = cm.get_entries()
                for entry in entries:
                    if entry.role.value == "user":
                        content = entry.content
                        if "target_spec" in content[:500] or "PLANNER" in content[:100]:
                            spec_match = re.search(r'\{[^{}]*(?:"targets"\s*:\s*\[[^\]]*\]|"tasks"\s*:\s*\[[^\]]*\])[^{}]*\}', content, re.DOTALL)
                            if spec_match:
                                spec = _json.loads(spec_match.group())
                                targets = spec.get("targets", [])
                                if targets and hasattr(agent, "_classify_target"):
                                    tasks = [agent._classify_target(t) for t in targets]
                                    logger.info("[StageExecutor] Rule-based fallback produced %d tasks", len(tasks))
                                    return tasks
        except Exception as e:
            logger.warning("[StageExecutor] Rule-based fallback also failed: %s", e)

        return []

    @staticmethod
    def _try_extract_tasks(text: str) -> list[dict[str, Any]]:
        """Try to extract task list from a single text output.

        Returns normalized task list if successful, empty list otherwise.
        Uses bracket-matching algorithm for nested JSON structures.
        """
        if not text or not text.strip():
            return []

        # Strategy A: Try to parse entire output as JSON
        try:
            parsed = _json.loads(text)
            result = StageExecutor._try_parse_json_value(parsed)
            if result:
                return result
        except (_json.JSONDecodeError, TypeError):
            pass

        # Strategy B: Extract JSON from code blocks first (highest signal-to-noise)
        try:
            code_block_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?\s*```', text)
            if code_block_match:
                json_content = code_block_match.group(1).strip()
                parsed = _json.loads(json_content)
                result = StageExecutor._try_parse_json_value(parsed)
                if result:
                    return result
        except (_json.JSONDecodeError, TypeError, ValueError):
            pass

        # Strategy C: Scan for JSON objects/arrays using bracket matching
        # Find all potential JSON starts (either { or [)
        candidates = []
        for i, ch in enumerate(text):
            if ch in ('{', '['):
                candidates.append(i)

        # Try each candidate position, using bracket matching to find the end
        for start_pos in candidates:
            for end_pos in StageExecutor._find_json_end(text, start_pos):
                json_str = text[start_pos:end_pos]
                try:
                    parsed = _json.loads(json_str)
                    result = StageExecutor._try_parse_json_value(parsed)
                    if result:
                        return result
                except (_json.JSONDecodeError, TypeError, ValueError):
                    continue

        # Strategy D: Simple first-[ to last-] extraction (deprecated, kept as fallback)
        try:
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start + 2:
                json_str = text[start:end]
                parsed = _json.loads(json_str)
                if isinstance(parsed, list):
                    tasks = StageExecutor._normalize_tasks(parsed)
                    if tasks:
                        return tasks
        except (_json.JSONDecodeError, TypeError, ValueError):
            pass

        return []

    @staticmethod
    def _find_json_end(text: str, start: int) -> list[int]:
        """Find potential JSON end positions using bracket matching.

        Returns a list of end positions (exclusive) that form valid bracket matching.
        Only returns the first valid match to keep it efficient.
        """
        if start >= len(text):
            return []

        open_char = text[start]
        if open_char == '{':
            close_char = '}'
        elif open_char == '[':
            close_char = ']'
        else:
            return []

        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            ch = text[i]

            if escape_next:
                escape_next = False
                continue

            if ch == '\\' and in_string:
                escape_next = True
                continue

            if ch == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    # Found a balanced bracket sequence
                    # Try a few variations: this exact end, and also with trailing whitespace
                    yield i + 1
                    return

    @staticmethod
    def _try_parse_json_value(parsed: Any) -> list[dict[str, Any]]:
        """Try to extract tasks from a parsed JSON value.

        Handles: list of tasks, dict with 'tasks' key.
        Does NOT treat arbitrary dicts with 'target' key as tasks
        to avoid false positives on non-task JSON objects.
        """
        if isinstance(parsed, list):
            tasks = StageExecutor._normalize_tasks(parsed)
            if tasks:
                return tasks
        if isinstance(parsed, dict):
            # Check for 'tasks' key
            if "tasks" in parsed:
                tasks = StageExecutor._normalize_tasks(parsed["tasks"])
                if tasks:
                    return tasks
            # Only treat as single task if it looks like a complete task object
            # (has target AND category, not just a stray target field)
            if "target" in parsed and "category" in parsed and "method" in parsed:
                normalized = StageExecutor._normalize_tasks([parsed])
                if normalized:
                    return normalized
        return []

    @staticmethod
    def _normalize_tasks(raw_tasks: list) -> list[dict[str, Any]]:
        """Normalize raw task dicts to ensure required fields.

        Validates and normalizes:
        - target: must be non-empty string
        - category: must be one of the valid categories, defaults to "unknown"
        - method: defaults to "custom micro-benchmark" if missing
        """
        valid_categories = {
            "latency_measurement", "capacity_measurement",
            "clock_measurement", "bandwidth_measurement", "unknown",
        }
        tasks = []
        for t in raw_tasks:
            if not isinstance(t, dict):
                continue
            target = t.get("target", "")
            if not isinstance(target, str) or not target.strip():
                continue
            category = t.get("category", "unknown")
            if category not in valid_categories:
                category = "unknown"
            tasks.append({
                "target": target.strip(),
                "category": category,
                "method": t.get("method", "custom micro-benchmark"),
            })
        return tasks

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
            has_tasks = "tasks" in data and isinstance(data.get("tasks"), list) and len(data.get("tasks", [])) > 0
            if not final_text and not tool_results and not has_tasks:
                data["error_detail"] = "Planner produced no output"
                return SubAgentStatus.FAILED
            if has_tasks:
                return SubAgentStatus.SUCCESS
            if final_text:
                return SubAgentStatus.SUCCESS
            return SubAgentStatus.FAILED

        if stage == PipelineStage.CODE_GEN:
            return StageExecutor._codegen_status(final_text, tool_results, data)

        if stage == PipelineStage.METRIC_ANALYSIS:
            data["analysis_output"] = final_text[:2000]
            # Check for meaningful analysis output
            has_tool_result = any(
                r.get("status") in ("success", "success_with_warning", True) or r.get("success") is True
                for r in tool_results
            )
            has_bottleneck = "bottleneck" in final_text.lower()
            has_metrics = any(
                kw in final_text.lower() 
                for kw in ["dram_latency", "dram_bandwidth", "l2_cache", "compute_bound", 
                           "memory_bound", "latency_bound", "cache_capacity", "confidence"]
            )
            if has_tool_result or (final_text and (has_bottleneck or has_metrics)):
                return SubAgentStatus.SUCCESS
            if final_text:
                return SubAgentStatus.SUCCESS
            return SubAgentStatus.FAILED

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
            r.get("status") in ("success", "success_with_warning", True) or r.get("success") is True
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
