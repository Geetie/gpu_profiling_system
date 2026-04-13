"""Pipeline orchestrator — domain layer.

Coordinates the sequential execution of sub-agents with P7 enforcement,
retry support, and persistence hooks.

Each stage runs inside an AgentLoop, enabling LLM-driven iteration:
model calls tool → sees result → retries/refines → calls next tool → ...
until the model signals completion or max turns reached.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.domain.subagent import (
    AgentRole,
    BaseSubAgent,
    CollaborationMessage,
    PipelineStage,
    P7ViolationError,
    SubAgentResult,
    SubAgentStatus,
)
from src.infrastructure.state_persist import StatePersister


@dataclass
class PipelineStep:
    """A single stage in the pipeline."""
    stage: PipelineStage
    agent: BaseSubAgent
    retry_on_failure: int = 0


class Pipeline:
    """Orchestrates the multi-agent collaboration flow.

    Flow: target_spec → PLAN → CODE_GEN → METRIC_ANALYSIS → VERIFICATION

    Each stage runs inside an AgentLoop, allowing the LLM to iterate:
    - Call tools (compile_cuda, execute_binary, run_ncu, etc.)
    - See results and adapt
    - Retry/refine until task is complete or max turns reached

    P7 enforcement:
    - VerificationAgent must have an empty ContextManager before execution.
    - The pipeline checks this explicitly and raises P7ViolationError.
    """

    def __init__(
        self,
        stages: list[PipelineStep],
        state_dir: str,
        sandbox=None,
        tool_handlers: dict | None = None,
        max_turns_per_stage: int = 15,
    ) -> None:
        self._stages = stages
        self._state_dir = state_dir
        self._persister = StatePersister(log_dir=state_dir, filename="pipeline_log.jsonl")
        self._sandbox = sandbox
        self._tool_handlers = tool_handlers or {}
        self._max_turns_per_stage = max_turns_per_stage

    def run(self, target_spec: dict[str, Any]) -> SubAgentResult:
        """Execute the full pipeline.

        Args:
            target_spec: The target specification (from target_spec.json).

        Returns:
            The final SubAgentResult from the VERIFICATION stage,
            or a FAILED result if any stage fails permanently.
        """
        self._persister.log_entry("pipeline_start", details={"target_spec": target_spec})

        prev_result: SubAgentResult | None = None

        for step in self._stages:
            # P7 gate
            self._check_p7(step.stage, prev_result)

            # Execute with retries
            result = self._execute_stage(step, prev_result, target_spec)

            if result.is_failed():
                self._persister.log_entry(
                    "pipeline_stage_failed",
                    details={
                        "stage": step.stage.value,
                        "error": result.error,
                    },
                )
                return result

            prev_result = result

        # Persist final result
        if prev_result:
            self._persister.log_entry("pipeline_complete", details=prev_result.to_dict())

        return prev_result or SubAgentResult(
            agent_role=AgentRole.PLANNER,
            status=SubAgentStatus.FAILED,
            error="Pipeline produced no result",
        )

    def _execute_stage(
        self, step: PipelineStep, prev_result: SubAgentResult | None,
        target_spec: dict[str, Any] | None = None,
    ) -> SubAgentResult:
        """Execute a single pipeline stage with AgentLoop iteration.

        Instead of calling step.agent.run() once, this creates an AgentLoop
        that allows the LLM to iterate: call tools, see results, adapt.
        """
        self._persister.log_entry(
            "pipeline_stage_start",
            details={"stage": step.stage.value, "retry_limit": step.retry_on_failure},
        )
        last_result: SubAgentResult | None = None

        for attempt in range(1 + step.retry_on_failure):
            if attempt > 0:
                self._persister.log_entry(
                    "pipeline_retry",
                    details={"stage": step.stage.value, "attempt": attempt},
                )

            # Build the collaboration message
            payload: dict[str, Any] = {}
            if prev_result is not None:
                payload["prev_result"] = prev_result.to_dict()
                payload["prev_fingerprint"] = prev_result.context_fingerprint
            else:
                payload["target_spec"] = target_spec

            message = CollaborationMessage(
                sender=prev_result.agent_role if prev_result else AgentRole.PLANNER,
                receiver=step.agent.role,
                message_type="task_dispatch",
                payload=payload,
            )

            # Use AgentLoop for iteration within this stage
            last_result = self._run_with_agent_loop(step, message)

            if last_result.is_success():
                break

            if last_result.status == SubAgentStatus.REJECTED:
                break

            self._persister.log_entry(
                "pipeline_attempt_failed",
                details={
                    "stage": step.stage.value,
                    "attempt": attempt + 1,
                    "error": last_result.error,
                },
            )

        return last_result or SubAgentResult(
            agent_role=step.agent.role,
            status=SubAgentStatus.FAILED,
            error="Stage produced no result after all retries",
        )

    def _check_p7(
        self, current_stage: PipelineStage, prev_result: SubAgentResult | None
    ) -> None:
        """P7 gate: verify that VerificationAgent has clean context."""
        if current_stage != PipelineStage.VERIFICATION:
            return

        verify_agent = self._get_stage_agent(PipelineStage.VERIFICATION)
        if verify_agent is None:
            return

        tokens = verify_agent.context_manager.total_tokens
        if tokens > 0:
            raise P7ViolationError(
                f"P7 violation: VerificationAgent has non-empty context "
                f"({tokens} tokens). Verification must not inherit generation context."
            )

        # Log the audit trail
        self._persister.log_entry(
            "p7_audit",
            details={
                "generation_fingerprint": prev_result.context_fingerprint if prev_result else None,
                "verification_context_tokens": 0,
                "status": "clean",
            },
        )

    def _get_stage_agent(self, stage: PipelineStage) -> BaseSubAgent | None:
        """Find the agent for a given stage."""
        for step in self._stages:
            if step.stage == stage:
                return step.agent
        return None

    def _run_with_agent_loop(
        self, step: PipelineStep, message: CollaborationMessage
    ) -> SubAgentResult:
        """Run a pipeline stage inside an AgentLoop for iteration.

        Creates an AgentLoop with the subagent's context and tools,
        lets the LLM iterate through tool calls, then extracts result.
        """
        import json as _json
        import uuid
        from src.application.agent_loop import AgentLoop
        from src.application.context import Role
        from src.application.control_plane import ControlPlane
        from src.application.session import SessionState
        from src.domain.permission import PermissionMode

        agent = step.agent
        stage_name = step.stage.value

        # Build task description for the LLM
        if message.message_type == "task_dispatch":
            task = message.payload.get("task", {})
            prev_result = message.payload.get("prev_result", {})
            target_spec = message.payload.get("target_spec", {})
        else:
            task = message.payload
            prev_result = None
            target_spec = None

        # System prompt for the AgentLoop stage
        system_prompt = f"""You are the {stage_name} stage in a GPU profiling pipeline.

Your role: {agent._build_system_prompt()}

Available tools: {agent.tool_registry.list_tools()}

Instructions:
1. Use the available tools to complete your task
2. For each tool call, respond with ONLY valid JSON:
   {{"tool": "tool_name", "args": {{"arg1": "value1"}}}}
3. After using tools, analyze results and produce a final answer
4. When done, output your final answer as plain text (no JSON)

"""

        # Build user task description
        if target_spec:
            user_task = f"Process these profiling targets:\n{_json.dumps(target_spec, indent=2)}"
        elif prev_result:
            user_task = f"Process results from {prev_result.get('agent_role', 'previous')} stage:\n{_json.dumps(prev_result.get('data', {}), indent=2)[:2000]}"
        elif task:
            user_task = f"Task: {_json.dumps(task, indent=2)}"
        else:
            user_task = "Execute your stage task."

        # Create session and control plane
        session_id = f"pipeline_{stage_name}_{uuid.uuid4().hex[:6]}"
        session = SessionState(session_id=session_id, goal=f"Pipeline stage: {stage_name}")
        control_plane = ControlPlane(rule_dir=None)

        # Create AgentLoop for this stage
        loop = AgentLoop(
            session=session,
            context_manager=agent.context_manager,
            control_plane=control_plane,
            tool_registry=agent.tool_registry,
            max_turns=self._max_turns_per_stage,
            state_dir=self._state_dir,
            permission_mode=agent.permission_mode,
        )

        # Wire model caller
        if agent._model_caller is not None:
            loop.set_model_caller(agent._model_caller)

        # Wire tool executor if sandbox available
        if self._sandbox and self._tool_handlers:
            loop.set_tool_executor(
                lambda tool_name, args: self._tool_handlers[tool_name](args)
            )

        # Set up context and run
        agent.context_manager.add_entry(
            Role.SYSTEM, system_prompt, token_count=50
        )
        agent.context_manager.add_entry(
            Role.USER, user_task, token_count=30
        )

        # Run the loop — iterates until model stops calling tools or max turns
        try:
            loop.start()
        except Exception as e:
            return SubAgentResult(
                agent_role=agent.role,
                status=SubAgentStatus.FAILED,
                error=f"AgentLoop failed in {stage_name}: {e}",
            )

        # Extract result from context
        return self._extract_stage_result(agent, step.stage, loop)

    def _extract_stage_result(
        self,
        agent: BaseSubAgent,
        stage: PipelineStage,
        loop: AgentLoop,
    ) -> SubAgentResult:
        """Extract a SubAgentResult from the AgentLoop's final context."""
        import json as _json

        entries = agent.context_manager.get_entries()

        # Collect all tool call results
        tool_results = []
        final_text = ""

        for entry in entries:
            try:
                data = _json.loads(entry.content)
                if isinstance(data, dict):
                    if "tool" in data or "status" in data:
                        tool_results.append(data)
            except (_json.JSONDecodeError, AttributeError):
                # Non-JSON content = model's text output
                if entry.content and not entry.content.startswith("You are") and not entry.content.startswith("Process"):
                    final_text = entry.content

        # Build structured result based on stage
        data = {"tool_results": tool_results, "final_output": final_text[:3000]}

        # Stage-specific extraction
        if stage == PipelineStage.PLAN:
            # Try to extract plan from final output
            data["plan_text"] = final_text[:2000]
            status = SubAgentStatus.SUCCESS
        elif stage == PipelineStage.CODE_GEN:
            # Extract compilation/execution results
            data["code_gen_output"] = final_text[:2000]
            # Check if any tool succeeded
            succeeded = any(r.get("status") == "success" for r in tool_results)
            status = SubAgentStatus.SUCCESS if succeeded else SubAgentStatus.FAILED
            if not succeeded and tool_results:
                errors = [r.get("error", "") for r in tool_results if r.get("status") != "success"]
                data["errors"] = errors
        elif stage == PipelineStage.METRIC_ANALYSIS:
            data["analysis_output"] = final_text[:2000]
            status = SubAgentStatus.SUCCESS if final_text else SubAgentStatus.FAILED
        elif stage == PipelineStage.VERIFICATION:
            # Check if verification accepted or rejected
            accepted = "accept" in final_text.lower() or "valid" in final_text.lower()
            rejected = "reject" in final_text.lower() or "invalid" in final_text.lower()
            if rejected and not accepted:
                status = SubAgentStatus.REJECTED
            else:
                status = SubAgentStatus.SUCCESS
            data["review_text"] = final_text[:2000]
        else:
            status = SubAgentStatus.SUCCESS

        result = SubAgentResult(
            agent_role=agent.role,
            status=status,
            data=data,
            artifacts=[],
        )

        result.context_fingerprint = result.compute_fingerprint(agent.context_manager)
        self._persister.log_entry(
            "stage_result",
            details={"stage": stage.value, "status": status.value, "tool_calls": len(tool_results)},
        )

        return result

    @classmethod
    def build_default(
        cls,
        planner: BaseSubAgent,
        code_gen: BaseSubAgent,
        metric_analysis: BaseSubAgent,
        verification: BaseSubAgent,
        state_dir: str,
        sandbox=None,
        tool_handlers: dict | None = None,
        max_turns_per_stage: int = 15,
    ) -> Pipeline:
        """Build a standard pipeline with all 4 agents.

        Each stage runs inside an AgentLoop for LLM-driven iteration.
        """
        return cls(
            stages=[
                PipelineStep(stage=PipelineStage.PLAN, agent=planner),
                PipelineStep(stage=PipelineStage.CODE_GEN, agent=code_gen),
                PipelineStep(stage=PipelineStage.METRIC_ANALYSIS, agent=metric_analysis),
                PipelineStep(
                    stage=PipelineStage.VERIFICATION,
                    agent=verification,
                    retry_on_failure=0,  # Verification failures are final
                ),
            ],
            state_dir=state_dir,
            sandbox=sandbox,
            tool_handlers=tool_handlers,
            max_turns_per_stage=max_turns_per_stage,
        )
