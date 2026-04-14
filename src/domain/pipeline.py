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
        code_gen_data: dict | None = None  # Preserve CodeGen measurements for final output

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

            # Preserve CodeGen measurements for final result assembly
            if step.stage == PipelineStage.CODE_GEN and result.is_success():
                code_gen_data = dict(result.data)

            prev_result = result

        # Bubble up CodeGen measurements to final result
        if code_gen_data and prev_result and prev_result.is_success():
            if "measurements" in code_gen_data:
                prev_result.data["measurements"] = code_gen_data["measurements"]
            if "analysis_method" in code_gen_data:
                prev_result.data["analysis_method"] = code_gen_data["analysis_method"]
            if "code_gen_output" in code_gen_data:
                prev_result.data["code_gen_output"] = code_gen_data["code_gen_output"]

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
            # CRITICAL: target_spec must be available to ALL stages, not just Planner
            payload: dict[str, Any] = {}
            if target_spec:
                payload["target_spec"] = target_spec
            if prev_result is not None:
                payload["prev_result"] = prev_result.to_dict()
                payload["prev_fingerprint"] = prev_result.context_fingerprint

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

    def _build_stage_system_prompt(self, agent: BaseSubAgent, stage_name: str) -> str:
        """Build system prompt for a pipeline stage's AgentLoop."""
        import json as _json
        available_tools = agent.tool_registry.list_tools()
        tool_list = _json.dumps(available_tools, indent=2) if available_tools else "(no tools registered)"

        # Stage-specific tool usage instructions
        tool_guidance = ""
        if stage_name == PipelineStage.CODE_GEN.value:
            tool_guidance = (
                f"\n\nWORKFLOW: For each target in the task list:\n"
                f"1. compile_cuda(source=\"...full .cu code...\", flags=[\"-O3\", \"-arch=sm_XX\"])\n"
                f"2. execute_binary(binary_path=\"./benchmark\")\n"
                f"3. Parse the output for 'target_name: numeric_value'\n"
                f"4. If compilation fails or output is invalid, fix the code and retry\n"
                f"5. After all targets are done, report all measured values"
            )
        elif stage_name == PipelineStage.METRIC_ANALYSIS.value:
            tool_guidance = (
                f"\n\nWORKFLOW: Parse CodeGen output, extract numeric metrics, "
                f"assess confidence. Use read_file if evidence files exist."
            )
        elif stage_name == PipelineStage.VERIFICATION.value:
            tool_guidance = (
                f"\n\nWORKFLOW: Review all previous stage data independently. "
                f"Use read_file to check evidence files. State ACCEPT or REJECT."
            )

        return (
            f"You are the {stage_name} stage in a GPU hardware profiling pipeline.\n"
            f"{agent._build_system_prompt()}\n\n"
            f"Available tools: {tool_list}\n\n"
            f"Tool call format: {{\"tool\": \"tool_name\", \"args\": {{\"key\": \"value\"}}}}\n"
            f"After each tool call result, you may call more tools or give your final answer.\n"
            f"When done, give your final answer as plain text (not JSON)."
            f"{tool_guidance}"
        )

    def _build_task_prompt(
        self, task: dict, prev_result: dict, stage_name: str
    ) -> str:
        """Build task-specific prompt for CODE_GEN, METRIC_ANALYSIS, VERIFICATION."""
        import json as _json

        if stage_name == PipelineStage.CODE_GEN.value:
            # CodeGen receives ALL tasks from Planner as a list
            tasks_list = task.get("tasks", [])

            if tasks_list:
                # Build design guide for ALL targets
                design_guides = []
                for t in tasks_list:
                    target = t.get("target", "unknown")
                    guide = _get_design_principle(target)
                    design_guides.append(f"--- Task: {target} ---\n{guide}")
                all_design = "\n\n".join(design_guides)

                task_summary = "\n".join(
                    f"- {t.get('target', 'unknown')}: {t.get('category', 'unknown')} "
                    f"(method: {t.get('method', 'custom')})"
                    for t in tasks_list
                )

                return (
                    f"You must generate and execute CUDA micro-benchmarks for "
                    f"{len(tasks_list)} targets:\n\n{task_summary}\n\n"
                    f"PROCESS EACH TARGET ONE BY ONE:\n"
                    f"1. Pick the first target from the list above\n"
                    f"2. Read its design guide below\n"
                    f"3. Write complete CUDA C++ source code\n"
                    f"4. Use compile_cuda to compile\n"
                    f"5. Use execute_binary to run\n"
                    f"6. Parse the numeric output\n"
                    f"7. Move to the next target\n\n"
                    f"CODING RULES:\n"
                    f"- Use clock64() for cycle timing (NOT clock())\n"
                    f"- Use cudaEventElapsedTime for wall-clock timing\n"
                    f"- Each binary must output: target_name: numeric_value\n"
                    f"- Include: cuda_runtime.h, main(), cudaMalloc, cudaMemcpy, kernel launch\n"
                    f"- Detect GPU arch: compile with -arch=sm_XX (from nvidia-smi, e.g. sm_60)\n\n"
                    f"DESIGN GUIDES (one per target):\n\n{all_design}"
                )
            else:
                # Fallback: single task format
                target = task.get("target", "unknown")
                category = task.get("category", "unknown")
                method = task.get("method", "custom micro-benchmark")
                design_guide = _get_design_principle(target)

                return (
                    f"Generate and execute a CUDA micro-benchmark to measure '{target}'.\n"
                    f"Category: {category}\n"
                    f"Method: {method}\n\n"
                    f"{design_guide}\n\n"
                    f"Steps:\n"
                    f"1. Write complete CUDA C++ source code implementing the above design\n"
                    f"   - Use clock64() for cycle timing (NOT clock())\n"
                    f"   - Use cudaEventElapsedTime for wall-clock timing when needed\n"
                    f"   - Output parseable key: value pairs via printf\n"
                    f"   - Include: cuda_runtime.h, main(), cudaMalloc, cudaMemcpy, kernel launch\n"
                    f"2. Use compile_cuda tool to compile the CUDA source\n"
                    f"3. Use execute_binary tool to run the compiled benchmark\n"
                    f"4. Parse the numeric output from printf results\n"
                    f"5. If results are invalid (0, negative, implausible), revise the code and retry\n\n"
                    f"Output the measured value with: {target}: <numeric_value>\n"
                    f"Also report: confidence (0.0-1.0), method_used, num_trials."
                )

        elif stage_name == PipelineStage.METRIC_ANALYSIS.value:
            # MetricAnalysis receives CodeGen's raw output + parsed data
            code_gen_data = prev_result.get("data", {})
            raw_outputs = {}
            for key, val in code_gen_data.items():
                if isinstance(val, str):
                    raw_outputs[key] = val[:1000]

            return (
                f"You are analyzing GPU micro-benchmark results.\n\n"
                f"Benchmark output from CodeGen stage:\n"
                f"{_json.dumps(raw_outputs, indent=2, ensure_ascii=False)[:2000]}\n\n"
                f"YOUR TASK:\n"
                f"1. Parse the raw output — extract all numeric metrics (latency cycles, "
                f"bandwidth GB/s, cache size MB, clock MHz, SM count, etc.)\n"
                f"2. For each metric: identify the target name and measured value\n"
                f"3. Assess confidence: high (>3 trials, consistent), medium (1-2 trials), "
                f"low (single run, high variance)\n"
                f"4. Check for anomalies: negative values, zero cycles, implausible ranges\n\n"
                f"EXPECTED RANGES (for reference):\n"
                f"- L1 latency: 50-300 cycles, L2: 100-500, DRAM: 300-1000\n"
                f"- L2 cache: 2-100 MB (power of 2 typically)\n"
                f"- DRAM bandwidth: 100-900 GB/s\n"
                f"- SM clock: 1000-2500 MHz\n"
                f"- SM count: 8-256\n"
                f"- Shared memory per block: 48-164 KB\n\n"
                f"Return: for each metric, the target name, measured value, confidence, "
                f"and whether it passes sanity check."
            )

        elif stage_name == PipelineStage.VERIFICATION.value:
            # Verification receives the FULL chain: Planner targets + CodeGen output + MetricAnalysis
            return (
                f"You are the independent Verification Agent for GPU hardware profiling.\n\n"
                f"You do NOT trust any previous stage's conclusions.\n"
                f"Review from first principles using GPU hardware knowledge.\n\n"
                f"Previous stage data:\n"
                f"{_json.dumps(prev_result.get('data', {}), indent=2, ensure_ascii=False)[:2000]}\n\n"
                f"VERIFICATION CHECKS (perform each):\n"
                f"1. DATA COMPLETENESS: Are all requested targets measured? Any missing?\n"
                f"2. NUMERIC SANITY: Are values within plausible GPU hardware ranges?\n"
                f"3. LATENCY HIERARCHY: DRAM latency > L2 latency > L1 latency?\n"
                f"4. L2 CAPACITY: Is it a power of 2 (2, 4, 8, 40, 50, 60, 72 MB)?\n"
                f"5. BANDWIDTH: DRAM BW matches known GPU specs (±30%)?\n"
                f"6. CLOCK FREQ: 1000-2500 MHz range?\n"
                f"7. SM COUNT: Known configuration (56=P100, 108=A100, 256=H100, etc.)?\n"
                f"8. SHARED MEMORY: 48 KB or 64 KB or 164 KB per block?\n"
                f"9. BANK CONFLICTS: ratio >= 1.0 (strided >= sequential)?\n\n"
                f"State clearly: ACCEPT (all checks pass) or REJECT (specify which failed).\n"
                f"If REJECT: explain what needs to be re-measured."
            )

        else:
            return f"Execute your {stage_name} stage task."

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

        # Extract payload
        payload = message.payload
        task = payload.get("task", {})
        prev_result = payload.get("prev_result", {})
        target_spec = payload.get("target_spec", {})

        # Fix: Planner outputs "tasks" (list) in data, not "task" (singular)
        # Extract tasks list if this is CodeGen stage
        if not task and stage_name == PipelineStage.CODE_GEN.value:
            tasks_list = prev_result.get("data", {}).get("tasks", [])
            if tasks_list:
                task = {"tasks": tasks_list}  # Pass ALL tasks to CodeGen

        # System prompt for the AgentLoop stage
        system_prompt = self._build_stage_system_prompt(agent, stage_name)

        # Build user task description — CRITICAL: route by stage, not by payload
        if stage_name == PipelineStage.PLAN.value:
            # First stage: Planner receives targets
            user_task = (
                f"You are the PLANNER stage. Analyze these GPU profiling targets "
                f"and decompose them into actionable tasks.\n\n"
                f"Targets: {_json.dumps(target_spec, indent=2)}\n\n"
                f"Return a JSON array of task objects with: "
                f'"target", "category" (latency_measurement, capacity_measurement, '
                f'clock_measurement, bandwidth_measurement, or unknown), '
                f'"method" (detailed description of the measurement approach).'
            )
        elif task:
            # Subsequent stages: receive a specific task
            user_task = self._build_task_prompt(task, prev_result, stage_name)
        elif prev_result:
            # No task but has prev_result (MetricAnalysis, Verification)
            user_task = self._build_task_prompt(task, prev_result, stage_name)
        else:
            user_task = f"You are the {stage_name} stage. Execute your task."

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
            handlers = dict(self._tool_handlers)
            loop.set_tool_executor(
                lambda tool_name, args: handlers[tool_name](args)
            )

        # Auto-approve tool calls in HIGH_AUTONOMY mode (no human in loop)
        if agent.permission_mode == PermissionMode.HIGH_AUTONOMY:
            loop.set_approval_callback(lambda request: True)

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
        loop: "AgentLoop",
    ) -> SubAgentResult:
        """Extract a SubAgentResult from the AgentLoop's final context."""
        import json as _json

        entries = agent.context_manager.get_entries()

        # Collect tool call results and model's text outputs
        tool_results = []
        assistant_outputs = []
        model_prompt_prefix = ("You are the", "Available tools", "Instructions:")

        for entry in entries:
            if entry.role.value != "assistant":
                continue

            # Try to parse as JSON
            try:
                data = _json.loads(entry.content)
                if isinstance(data, dict) and ("status" in data or "tool" in data):
                    tool_results.append(data)
                elif isinstance(data, (dict, list)):
                    # JSON output from model (not a tool result)
                    assistant_outputs.append(entry.content)
            except (_json.JSONDecodeError, TypeError):
                # Plain text from model
                content = entry.content.strip()
                if content and not content.startswith(model_prompt_prefix):
                    assistant_outputs.append(content)

        # Final output = last non-empty assistant text
        final_text = assistant_outputs[-1] if assistant_outputs else ""

        # Build structured result
        data = {
            "tool_results": tool_results,
            "final_output": final_text[:3000],
            "num_tool_calls": len(tool_results),
        }

        # Stage-specific extraction and status determination
        if stage == PipelineStage.PLAN:
            data["plan_text"] = final_text[:2000]
            status = SubAgentStatus.SUCCESS if final_text else SubAgentStatus.FAILED
            if not final_text and not tool_results:
                data["error_detail"] = "Planner produced no output"

        elif stage == PipelineStage.CODE_GEN:
            data["code_gen_output"] = final_text[:2000]
            # Fix: tool handlers return "success" (boolean), not "status" (string)
            # compile_cuda → {"success": True, "output": "...", "binary_path": "..."}
            # execute_binary → {"stdout": "...", "stderr": "...", "return_code": 0}
            tool_succeeded = any(
                r.get("status") in ("success", True) or
                r.get("success") is True
                for r in tool_results
            )
            has_binary = any(r.get("binary_path") for r in tool_results)
            has_output = any(r.get("stdout") for r in tool_results)
            # Also check execute_binary success: return_code == 0
            exec_succeeded = any(
                r.get("return_code", -1) == 0 for r in tool_results
                if "return_code" in r or "stdout" in r
            )
            output_succeeded = bool(final_text) and not any(
                kw in final_text.lower() for kw in ["failed", "error:", "could not", "cannot"]
            )
            if tool_succeeded or has_binary or has_output or exec_succeeded or output_succeeded:
                status = SubAgentStatus.SUCCESS
            else:
                status = SubAgentStatus.FAILED
                if not final_text and not tool_results:
                    data["error_detail"] = "CodeGen produced no output"
                elif not tool_succeeded and not has_binary:
                    data["error_detail"] = "CodeGen compilation failed"

            # Extract structured measurements from CodeGen tool results
            # This feeds into _assemble_final_results in main.py
            measurements = {}
            methodology_parts = []
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

        elif stage == PipelineStage.METRIC_ANALYSIS:
            data["analysis_output"] = final_text[:2000]
            status = SubAgentStatus.SUCCESS if final_text else SubAgentStatus.FAILED

        elif stage == PipelineStage.VERIFICATION:
            data["review_text"] = final_text[:2000]
            lower = final_text.lower()
            # Use explicit phrases to avoid substring collisions (e.g., "invalid" contains "valid")
            rejected = "reject" in lower or "not valid" in lower or "not accept" in lower
            accepted = "accept" in lower and "reject" not in lower and "not valid" not in lower and "not accept" not in lower
            if rejected and not accepted:
                status = SubAgentStatus.REJECTED
            else:
                status = SubAgentStatus.SUCCESS

        else:
            status = SubAgentStatus.SUCCESS

        # Propagate error_detail to result.error for UI visibility
        if status == SubAgentStatus.FAILED:
            error_msg = data.get("error_detail", "")
            if not error_msg and not final_text and not tool_results:
                error_msg = f"Stage {stage.value} produced no output"
            elif not error_msg and final_text:
                # Show a snippet of what the model actually output
                error_msg = f"Stage {stage.value} failed. Output: {final_text[:500]}"

        result = SubAgentResult(
            agent_role=agent.role,
            status=status,
            data=data,
            artifacts=[],
            error=error_msg if status == SubAgentStatus.FAILED else None,
        )

        result.context_fingerprint = result.compute_fingerprint(agent.context_manager)
        self._persister.log_entry(
            "stage_result",
            details={
                "stage": stage.value,
                "status": status.value,
                "tool_calls": len(tool_results),
                "output_length": len(final_text),
            },
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


def _get_design_principle(target: str) -> str:
    """Return design principle for a specific profiling target.

    This is the design thinking injected into the CodeGen agent's prompt.
    The agent generates CUDA code based on these principles — no hardcoded source.
    """
    principles: dict[str, str] = {
        "dram_latency_cycles": (
            "Design: Pointer-chasing with random permutation chains (LCG-seeded Knuth shuffle).\n"
            "- Allocate 32M uint32_t indices (=128 MB, >> any L2), fill with random permutation\n"
            "- Single thread follows chain: idx = next[idx] for 10M iterations\n"
            "- Use clock64() to measure total cycles; latency = cycles / iterations\n"
            "- Why random: hardware prefetchers detect strided patterns; random permutation defeats all prefetching\n"
            "- Run 3 trials, report median cycles/access"
        ),
        "l2_latency_cycles": (
            "Design: Pointer-chasing with 2 MB working set (512K uint32_t).\n"
            "- Working set fits in L2 cache but exceeds L1 on all GPUs\n"
            "- Same random permutation chain approach as DRAM latency\n"
            "- Use clock64(); latency = total_cycles / iterations\n"
            "- Expected: 100-500 cycles"
        ),
        "l1_latency_cycles": (
            "Design: Pointer-chasing with 8 KB working set (2K uint32_t).\n"
            "- Working set fits in all GPU L1 data caches\n"
            "- Use clock64(); latency = total_cycles / iterations\n"
            "- Expected: 50-300 cycles"
        ),
        "l2_cache_size_mb": (
            "Design: Working-set sweep with pointer-chasing at 14 sizes.\n"
            "- Sizes: 1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128 MB\n"
            "- At each size: run pointer-chasing, measure cycles/access with clock64()\n"
            "- Detect 'cliff': the size where latency jumps >3x (L2 miss → DRAM)\n"
            "- L2 size = last size before cliff (typically power of 2: 4, 8, 40, 50, 60, 72 MB)"
        ),
        "actual_boost_clock_mhz": (
            "Design: SM clock cycles / wall-clock time.\n"
            "- Kernel: 10M iterations of random permutation, measure total clock64() cycles\n"
            "- Host timing: cudaEventElapsedTime (GPU-side wall-clock, NOT host clock)\n"
            "- Record events before/after kernel, elapsed_us = cudaEventElapsedTime(stop, start)\n"
            "- freq_MHz = total_cycles / elapsed_us\n"
            "- Run 3 trials, report median frequency\n"
            "- Expected: 1000-2500 MHz"
        ),
        "dram_bandwidth_gbps": (
            "Design: STREAM copy — sequential memory write saturation.\n"
            "- Kernel: dst[i] = src[i] for 32M floats (128 MB)\n"
            "- Launch 65535 blocks to fully saturate memory bandwidth\n"
            "- Use cudaEventElapsedTime for timing (GPU-side, more precise than host)\n"
            "- BW = bytes_copied / elapsed_ns * 1e9 / 1e9 = bytes / ns = GB/s"
        ),
        "max_shmem_per_block_kb": (
            "Design: CUDA occupancy API sweep — no timing needed.\n"
            "- For each shmem size (1K, 2K, 4K, 8K, 16K, 32K, 48K, 64K, 96K, 100K):\n"
            "  Call cudaOccupancyMaxActiveBlocksPerMultiprocessor with dummy kernel\n"
            "- Max shmem where blocks_per_sm > 0 = per-block capacity\n"
            "- This is a direct CUDA API query — most reliable probe in the suite"
        ),
        "bank_conflict_penalty_ratio": (
            "Design: Shared memory bank conflict via strided vs sequential access.\n"
            "- Use cudaEventElapsedTime (NOT clock64()) — bank conflicts are too fast for clock64()\n"
            "- Run TWO separate kernel calls in same program:\n"
            "  (a) Strided: thread t accesses shared_mem[t * 32] — all threads hit bank 0\n"
            "  (b) Sequential: thread t accesses shared_mem[(t + offset) % 256] — one thread per bank\n"
            "- 1 block, 256 threads, __shared__ uint32_t[256]\n"
            "- ratio = strided_ms / sequential_ms (>1.0 means bank conflicts)\n"
            "- Run 3 trials, report minimum ratio (eliminates noise)"
        ),
        "shmem_bandwidth_gbps": (
            "Design: Cooperative shared memory read/write within single block.\n"
            "- 1 block, 256 threads, __shared__ uint32_t[256]\n"
            "- Each iteration: all threads cooperatively read + write shared memory\n"
            "- Use cudaEventElapsedTime for timing\n"
            "- BW = (iterations * 256 threads * 2 ops * 4 bytes) / elapsed_ns GB/s\n"
            "- Note: this measures per-SM bandwidth (shared memory is per-SM resource)"
        ),
        "sm_count": (
            "Design: cudaDeviceGetAttribute for MultiProcessorCount.\n"
            "- Use cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device)\n"
            "- This directly queries hardware — NOT cudaGetDeviceProperties (can be virtualized)\n"
            "- Also query: maxThreadsPerBlock, warpSize, maxThreadsPerMultiProcessor\n"
            "- Print all values for cross-validation"
        ),
    }
    return principles.get(target, (
        f"Design: Custom micro-benchmark for '{target}'.\n"
        "- Write a complete CUDA .cu file with proper includes and main()\n"
        "- Use clock64() for cycle timing, cudaEventElapsedTime for wall-clock timing\n"
        "- Output must be parseable: printf(\"key: value\\n\")\n"
        "- Run multiple trials, report median"
    ))
