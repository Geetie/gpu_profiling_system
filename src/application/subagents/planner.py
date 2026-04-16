"""Planner Agent — global coordinator.

Receives target_spec.json, decomposes targets into sub-tasks, dispatches
to specialist agents, and integrates final results.
"""
from __future__ import annotations

from typing import Any

from src.application.context import ContextManager, Role
from src.domain.permission import PermissionMode
from src.domain.subagent import (
    AgentRole,
    BaseSubAgent,
    CollaborationMessage,
    SubAgentResult,
    SubAgentStatus,
)
from src.domain.tool_contract import ToolRegistry


class PlannerAgent(BaseSubAgent):
    """Main planning agent with global view.

    Decomposes targets and orchestrates the collaboration flow.
    """

    def __init__(
        self,
        context_manager: ContextManager | None = None,
        tool_registry: ToolRegistry | None = None,
        state_dir: str = ".state",
        permission_mode: PermissionMode = PermissionMode.DEFAULT,
        max_tokens: int = 16000,
    ) -> None:
        super().__init__(
            role=AgentRole.PLANNER,
            context_manager=context_manager or ContextManager(max_tokens=max_tokens),
            tool_registry=tool_registry or ToolRegistry(),
            state_dir=state_dir,
            permission_mode=permission_mode,
            max_tokens=max_tokens,
        )

    def _process(self, message: CollaborationMessage) -> SubAgentResult:
        """Parse target spec and create a task plan."""
        target_spec = message.payload.get("target_spec", {})
        targets = target_spec.get("targets", [])

        if not targets:
            return SubAgentResult(
                agent_role=self.role,
                status=SubAgentStatus.FAILED,
                error="No targets specified in target_spec",
            )

        # Try LLM-based planning first, fall back to rule-based
        if self._model_caller is not None:
            tasks, plan = self._llm_plan(target_spec, targets)
        else:
            # Decompose targets into sub-tasks
            tasks = self.parse_targets(target_spec)
            plan = self.create_plan(tasks)

        result = SubAgentResult(
            agent_role=self.role,
            status=SubAgentStatus.SUCCESS,
            data={
                "targets": targets,
                "tasks": tasks,
                "plan": [p.to_dict() for p in plan],
            },
            metadata={"num_targets": len(targets), "num_tasks": len(tasks)},
        )

        return result

    def _llm_plan(
        self, target_spec: dict[str, Any], targets: list[str]
    ) -> tuple[list[dict[str, Any]], list[CollaborationMessage]]:
        """Use LLM to decompose targets into a plan."""
        import json as _json

        user_msg = (
            f"Analyze these GPU profiling targets and decompose them into "
            f"actionable tasks:\n\n{_json.dumps(target_spec, indent=2)}\n\n"
            f"Return a JSON array of task objects, each with: "
            f'"target", "category" (one of: latency_measurement, '
            f'capacity_measurement, clock_measurement, bandwidth_measurement, unknown), '
            f'"method" (detailed description of the measurement approach — '
            f"include key techniques: pointer-chasing with random permutation, "
            f"clock64() timing, cudaEventElapsedTime for wall-clock, working-set sweep, "
            f"STREAM copy, occupancy API, etc.)."
        )
        messages = self.context_manager.to_messages()
        messages.append({"role": "user", "content": user_msg})

        try:
            response = self._model_caller(messages)
            # Try to extract JSON from response
            tasks = []
            # Find JSON array in response
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                task_list = _json.loads(response[start:end])
                for t in task_list:
                    tasks.append({
                        "target": t.get("target", "unknown"),
                        "category": t.get("category", "unknown"),
                        "method": t.get("method", "custom micro-benchmark"),
                    })
        except Exception:
            # Fallback to rule-based
            tasks = self.parse_targets(target_spec)

        # If LLM returned no tasks, fallback
        if not tasks:
            tasks = self.parse_targets(target_spec)

        plan = self.create_plan(tasks)
        return tasks, plan

    def parse_targets(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        """Decompose target_spec into individual task descriptions."""
        targets = spec.get("targets", [])
        tasks: list[dict[str, Any]] = []

        for target in targets:
            task = self._classify_target(target)
            tasks.append(task)

        return tasks

    def create_plan(self, tasks: list[dict[str, Any]]) -> list[CollaborationMessage]:
        """Create dispatch messages for each task."""
        messages: list[CollaborationMessage] = []

        for task in tasks:
            receiver = self._route_task(task)
            msg = CollaborationMessage(
                sender=AgentRole.PLANNER,
                receiver=receiver,
                message_type="task_dispatch",
                payload={"task": task},
            )
            messages.append(msg)

        return messages

    def _classify_target(self, target: str) -> dict[str, Any]:
        """Classify a target metric into a task description."""
        # Known target categories
        latency_targets = {"dram_latency_cycles", "l2_latency_cycles", "shmem_latency_cycles"}
        capacity_targets = {"max_shmem_per_block_kb", "l2_cache_size_kb", "l1_cache_size_kb"}
        clock_targets = {"actual_boost_clock_mhz", "base_clock_mhz", "sm_clock_mhz"}
        bandwidth_targets = {"dram_bandwidth_gbps", "l2_bandwidth_gbps"}

        category = "unknown"
        if target in latency_targets:
            category = "latency_measurement"
        elif target in capacity_targets:
            category = "capacity_measurement"
        elif target in clock_targets:
            category = "clock_measurement"
        elif target in bandwidth_targets:
            category = "bandwidth_measurement"

        return {
            "target": target,
            "category": category,
            "method": self._suggest_method(target, category),
        }

    def _suggest_method(self, target: str, category: str) -> str:
        """Suggest a measurement method based on target category.

        Returns a detailed design methodology description that flows through
        to the CodeGen agent's prompt (see pipeline.py _build_task_prompt).
        The CodeGen agent uses this to write CUDA code from design principles.
        """
        methods = {
            "latency_measurement": (
                "pointer-chasing with random permutation chains (LCG-seeded Knuth shuffle), "
                "clock64() for cycle timing, latency = total_cycles / iterations, "
                "working set sized for target memory level"
            ),
            "capacity_measurement": (
                "working-set sweep with pointer-chasing at multiple sizes "
                "(1, 2, 4, 8, 16, 32, 64, 128 MB), detect latency cliff "
                "where cycles/access jumps >3x (L2 miss → DRAM)"
            ),
            "clock_measurement": (
                "SM clock cycles divided by wall-clock time: "
                "kernel measures 10M iterations of random permutation with clock64(), "
                "host measures elapsed microseconds with cudaEventElapsedTime, "
                "freq_MHz = total_cycles / elapsed_us"
            ),
            "bandwidth_measurement": (
                "STREAM copy (dst[i] = src[i]) with large arrays (32M floats = 128 MB), "
                "cudaEventElapsedTime for timing, BW = bytes / elapsed_ns GB/s"
            ),
        }
        return methods.get(category, "custom micro-benchmark with clock64() timing and parseable printf output")

    def _route_task(self, task: dict[str, Any]) -> AgentRole:
        """Route a task to the appropriate specialist agent."""
        # All measurement tasks go through CodeGen first, then MetricAnalysis
        return AgentRole.CODE_GEN
