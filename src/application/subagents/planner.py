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
                data={"targets": [], "tasks": [], "plan": []},
            )

        # Try LLM-based planning first, fall back to rule-based
        if self._model_caller is not None:
            tasks, plan = self._llm_plan(target_spec, targets)
        else:
            # Decompose targets into sub-tasks
            tasks = self.parse_targets(target_spec)
            plan = self.create_plan(tasks)

        # CRITICAL SAFETY NET: Ensure tasks is never empty (per spec.md P6)
        # This prevents Handoff Validation failure which blocks the entire pipeline
        if not tasks:
            print(f"[Planner] WARNING: _llm_plan returned {len(tasks)} tasks, forcing rule-based fallback")
            tasks = self.parse_targets(target_spec)
        if not tasks:
            print("[Planner] CRITICAL: parse_targets also returned empty! Force-creating minimal tasks")
            tasks = [
                {"target": t, "category": "unknown", "method": "custom micro-benchmark"}
                for t in targets
            ]
        if not plan:
            plan = self.create_plan(tasks)

        # FINAL CONTRACT VALIDATION: Ensure required keys exist before returning
        # Per spec.md §5.1: Planner must produce structured task list for CodeGen dispatch
        data = {
            "targets": targets,
            "tasks": tasks,
            "plan": [p.to_dict() for p in plan],
        }

        print(f"[Planner] Output contract validated: "
              f"{len(targets)} targets → {len(tasks)} tasks → {len(plan)} plan items")

        return SubAgentResult(
            agent_role=self.role,
            status=SubAgentStatus.SUCCESS,
            data=data,
            metadata={"num_targets": len(targets), "num_tasks": len(tasks)},
        )

    def _llm_plan(
        self, target_spec: dict[str, Any], targets: list[str]
    ) -> tuple[list[dict[str, Any]], list[CollaborationMessage]]:
        """Use LLM to decompose targets into a plan."""
        import json as _json

        user_msg = (
            "You are a PLANNING AGENT. Your ONLY job is to decompose targets into tasks.\n"
            "DO NOT write code files. DO NOT use write_file tool.\n"
            "DO NOT compile or execute anything. DO NOT generate measurements.\n\n"
            f"Analyze these GPU profiling targets and decompose them into "
            f"actionable tasks:\n\n{_json.dumps(target_spec, indent=2)}\n\n"
            f"Return ONLY a JSON array of task objects (nothing else), each with:\n"
            f'  - "target": string (the metric name)\n'
            f'  - "category": one of: latency_measurement, capacity_measurement, '
            f'clock_measurement, bandwidth_measurement, unknown\n'
            f'  - "method": detailed description of measurement approach '
            f"(pointer-chasing, clock64(), STREAM copy, working-set sweep, etc.)\n\n"
            f"Example output format:\n"
            f'[{{"target": "dram_latency_cycles", "category": "latency_measurement", '
            f'"method": "pointer-chasing with random permutation"}}]\n'
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
                    target_name = t.get("target", "unknown")
                    if not isinstance(target_name, str) or len(target_name.strip()) == 0:
                        continue
                    tasks.append({
                        "target": target_name,
                        "category": t.get("category", "unknown"),
                        "method": t.get("method", "custom micro-benchmark"),
                    })
                print(f"[Planner] LLM returned {len(tasks)} valid tasks")
            else:
                print(f"[Planner] LLM response did not contain JSON array, falling back to rule-based")
        except _json.JSONDecodeError as e:
            print(f"[Planner] LLM output JSON parse error: {e}, falling back to rule-based")
        except Exception as e:
            print(f"[Planner] LLM planning failed: {e}, using rule-based fallback")

        # Triple-safety fallback: ensure we always have at least one task per target
        if not tasks:
            print("[Planner] LLM produced no valid tasks, using rule-based classification")
            tasks = self.parse_targets(target_spec)

        if not tasks or len(tasks) < len(targets):
            missing_count = max(0, len(targets) - len(tasks))
            print(f"[Planner] WARNING: Only {len(tasks)}/{len(targets)} tasks, "
                  f"force-creating {missing_count} fallback task(s)")
            existing_targets = {t["target"] for t in tasks}
            for t in targets:
                if t not in existing_targets:
                    tasks.append({
                        "target": t,
                        "category": "unknown",
                        "method": "custom micro-benchmark for " + t,
                    })

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
