"""Planner Agent — global coordinator.

Receives target_spec.json, decomposes targets into sub-tasks, dispatches
to specialist agents, and integrates final results.
"""
from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


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
            tool_registry=ToolRegistry(),
            state_dir=state_dir,
            permission_mode=permission_mode,
            max_tokens=max_tokens,
        )
        if self.tool_registry.list_tools():
            logger.warning("[Planner] WARNING: Planner received non-empty tool_registry, "
                           "but Planner MUST have zero tools (P1/P2 enforcement). "
                           "Forcing empty registry.")
        logger.info("[Planner] Tool isolation active: 0 tools registered. "
                     "Planner outputs pure JSON text only, no tool calls.")

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

        valid_categories = {
            "latency_measurement", "capacity_measurement",
            "clock_measurement", "bandwidth_measurement",
            "ncu_throughput_measurement", "unknown",
        }

        # Try LLM-based planning first, fall back to rule-based
        if self._model_caller is not None:
            tasks, plan = self._llm_plan(target_spec, targets)
        else:
            # Decompose targets into sub-tasks
            tasks = self.parse_targets(target_spec)
            plan = self.create_plan(tasks)

        # SAFETY NET 1: Ensure tasks is never empty
        if not tasks:
            tasks = self.parse_targets(target_spec)
        if not tasks:
            tasks = [
                {"target": t, "category": "unknown", "method": "custom micro-benchmark"}
                for t in targets
            ]
        if not plan:
            plan = self.create_plan(tasks)

        # SAFETY NET 2: Validate and normalize every task
        validated_tasks = []
        for t in tasks:
            if not isinstance(t, dict):
                continue
            target = t.get("target", "")
            if not isinstance(target, str) or not target.strip():
                continue
            category = t.get("category", "unknown")
            if category not in valid_categories:
                category = "unknown"
            validated_tasks.append({
                "target": target.strip(),
                "category": category,
                "method": t.get("method", "custom micro-benchmark"),
            })
        tasks = validated_tasks if validated_tasks else [
            {"target": t, "category": "unknown", "method": "custom micro-benchmark"}
            for t in targets
        ]

        data = {
            "targets": targets,
            "tasks": tasks,
            "plan": [p.to_dict() for p in plan],
        }

        # ASSERT: Verify output contract before returning
        assert "tasks" in data, "[P0 BUG] data missing 'tasks' key"
        assert isinstance(data["tasks"], list), "[P0 BUG] data['tasks'] is not a list"
        assert len(data["tasks"]) > 0, "[P0 BUG] data['tasks'] is empty"
        for t in data["tasks"]:
            assert "target" in t, f"[P0 BUG] task missing 'target': {t}"
            assert "category" in t, f"[P0 BUG] task missing 'category': {t}"

        logger.info(f"[Planner] Output contract validated: "
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
        """Use LLM to decompose targets into a plan with retry logic."""
        import json as _json
        import time

        max_retries = 3
        base_delay = 1.0

        # ⚡ FIX-C: Plan阶段Context优化 - 清理历史消息以减少API处理时间
        # Plan阶段只需要简洁的system prompt + target_spec，不需要累积的历史对话
        try:
            # 记录优化前的context size (用于监控)
            original_messages = self.context_manager.to_messages()
            original_total_chars = sum(len(m.get("content", "")) for m in original_messages)
            logger.info(f"[Planner] 🔍 Context optimization check: "
                        f"{len(original_messages)} messages, {original_total_chars} chars total")

            if len(original_messages) > 2 or original_total_chars > 2000:
                # Context过大，执行清理
                logger.warning(f"[Planner] ⚡ Optimizing context for faster API response "
                              f"({original_total_chars} chars → target <500 chars)")

                # 保留system prompt（第一条），清除所有历史
                if original_messages and original_messages[0].get("role") == "system":
                    system_msg = original_messages[0]
                    # 使用精简版system prompt（如果原版太长）
                    if len(system_msg.get("content", "")) > 500:
                        minimal_system = (
                            "You are a GPU Profiling Planner Agent.\n\n"
                            "CONSTRAINTS:\n"
                            "- Output ONLY JSON text (no tool calls)\n"
                            "- Decompose targets into tasks with: target, category, method\n"
                            "- Categories: latency_measurement, capacity_measurement, "
                            "clock_measurement, bandwidth_measurement, "
                            "ncu_throughput_measurement, unknown\n\n"
                            "Your role is ANALYSIS only — not code generation or execution."
                        )
                        system_msg["content"] = minimal_system
                        logger.info("[Planner] Replaced verbose system prompt with minimal version")

                    # 重置context为仅包含精简的system prompt
                    self.context_manager.clear_history()
                    self.context_manager.add_entry(
                        Role.SYSTEM,
                        system_msg.get("content", ""),
                        token_count=100
                    )

                    optimized_messages = self.context_manager.to_messages()
                    optimized_chars = sum(len(m.get("content", "")) for m in optimized_messages)
                    logger.info(f"[Planner] ✅ Context optimized: {original_total_chars} → "
                                f"{optimized_chars} chars ({(1-optimized_chars/original_total_chars)*100:.0f}% reduction)")
            else:
                logger.info(f"[Planner] ✅ Context already optimal ({original_total_chars} chars)")
        except Exception as e:
            logger.warning(f"[Planner] ⚠️ Context optimization failed (non-fatal): {e}")
            # 继续使用原始context，不会影响功能

        user_msg = (
            "You are a PLANNING AGENT. Your ONLY job is to decompose targets into tasks.\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "CRITICAL CONSTRAINTS (VIOLATION WILL CAUSE PIPELINE FAILURE)\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "- 🚫 DO NOT write any files — NEVER call write_file or any file operation tool\n"
            "- 🚫 DO NOT compile or execute anything — NEVER call compile_cuda or execute_binary\n"
            "- 🚫 DO NOT run profiling tools — NEVER call run_ncu or ncu\n"
            "- 🚫 DO NOT call ANY tools at all — your output is PURE JSON TEXT ONLY\n"
            "- 🚫 DO NOT generate CUDA code or measurements\n"
            "- ✅ Your SOLE output: a JSON array of task objects as plain text\n\n"
            "If you attempt to call any tool, the pipeline will FAIL.\n"
            "Your role is ANALYSIS and CLASSIFICATION only — not code generation.\n\n"
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
            f'"method": "pointer-chasing with random permutation"}}]\n\n'
            f"IMPORTANT: Output ONLY the JSON array. No explanations, no tool calls, "
            f"no file writes. Just the JSON."
        )
        messages = self.context_manager.to_messages()
        messages.append({"role": "user", "content": user_msg})

        # 添加最终发送的message size日志
        final_total_chars = sum(len(m.get("content", "")) for m in messages)
        logger.info(f"[Planner] 📤 Sending {len(messages)} messages ({final_total_chars} chars) to LLM")

        tasks: list[dict[str, Any]] = []
        last_error = ""

        for attempt in range(max_retries):
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(f"[Planner] LLM planning retry {attempt}/{max_retries} "
                               f"(waiting {delay}s, last error: {last_error})")
                time.sleep(delay)

                retry_msg = {
                    "role": "user",
                    "content": (
                        f"Your previous response did not contain valid task JSON. "
                        f"Please return ONLY a JSON array of task objects with "
                        f'"target", "category", and "method" fields. '
                        f"NO explanations, NO tool calls, just the JSON array."
                    ),
                }
                messages.append(retry_msg)

            try:
                response = self._model_caller(messages)

                if response is None or not isinstance(response, str):
                    last_error = f"Invalid response type: {type(response).__name__}"
                    raise ValueError(last_error)

                response = response.strip()
                if len(response) == 0:
                    last_error = "Empty response"
                    raise ValueError(last_error)

                if len(response) < 20:
                    last_error = f"Response too short ({len(response)} chars): {response[:100]}"
                    raise ValueError(last_error)

                tasks = self._extract_tasks_from_response(response)
                if tasks:
                    logger.info(f"[Planner] LLM returned {len(tasks)} valid tasks from JSON extraction")
                    return tasks, self.create_plan(tasks)
                else:
                    last_error = f"No valid tasks found in response (first 200 chars: {response[:200]})"
                    raise ValueError(last_error)

            except (_json.JSONDecodeError, ValueError) as e:
                if attempt == max_retries - 1:
                    logger.error(f"[Planner] LLM output parsing failed after {max_retries} attempts: {e}")
                else:
                    logger.warning(f"[Planner] LLM output parsing failed on attempt {attempt + 1}: {e}")
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"[Planner] LLM planning exception after {max_retries} attempts: "
                                 f"{type(e).__name__}: {e}")
                else:
                    logger.warning(f"[Planner] LLM planning exception on attempt {attempt + 1}: "
                                   f"{type(e).__name__}: {e}")

        logger.warning("[Planner] All LLM retries exhausted, using rule-based classification")
        tasks = self.parse_targets(target_spec)

        if not tasks or len(tasks) < len(targets):
            missing_count = max(0, len(targets) - len(tasks))
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

    @staticmethod
    def _extract_tasks_from_response(response: str) -> list[dict[str, Any]]:
        """Extract task list from LLM response using robust parsing.

        Strategies (in order):
        1. Parse entire response as JSON
        2. Code block extraction
        3. Bracket-matching for JSON arrays
        4. First-[ to last-] fallback
        """
        import json as _json
        import re

        # Strategy 1: Parse entire response as JSON
        try:
            parsed = _json.loads(response)
            tasks = PlannerAgent._normalize_response_tasks(parsed)
            if tasks:
                return tasks
        except (_json.JSONDecodeError, TypeError):
            pass

        # Strategy 2: Extract from code blocks
        code_block_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?\s*```', response)
        if code_block_match:
            json_content = code_block_match.group(1).strip()
            try:
                parsed = _json.loads(json_content)
                tasks = PlannerAgent._normalize_response_tasks(parsed)
                if tasks:
                    return tasks
            except (_json.JSONDecodeError, TypeError):
                pass

        # Strategy 3: Bracket-matching for JSON arrays
        for i, ch in enumerate(response):
            if ch == '[':
                depth = 0
                in_string = False
                escape_next = False
                for j in range(i, len(response)):
                    cj = response[j]
                    if escape_next:
                        escape_next = False
                        continue
                    if cj == '\\' and in_string:
                        escape_next = True
                        continue
                    if cj == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    if in_string:
                        continue
                    if cj == '[':
                        depth += 1
                    elif cj == ']':
                        depth -= 1
                        if depth == 0:
                            json_str = response[i:j+1]
                            try:
                                parsed = _json.loads(json_str)
                                if isinstance(parsed, list):
                                    tasks = PlannerAgent._normalize_response_tasks(parsed)
                                    if tasks:
                                        return tasks
                            except (_json.JSONDecodeError, TypeError):
                                pass
                            break

        # Strategy 4: Simple first-[ to last-] extraction
        start = response.find("[")
        end = response.rfind("]") + 1
        if start >= 0 and end > start + 2:
            try:
                parsed = _json.loads(response[start:end])
                if isinstance(parsed, list):
                    tasks = PlannerAgent._normalize_response_tasks(parsed)
                    if tasks:
                        return tasks
            except (_json.JSONDecodeError, TypeError):
                pass

        return []

    @staticmethod
    def _normalize_response_tasks(parsed: Any) -> list[dict[str, Any]]:
        """Normalize parsed JSON into task list."""
        valid_categories = {
            "latency_measurement", "capacity_measurement",
            "clock_measurement", "bandwidth_measurement",
            "ncu_throughput_measurement", "unknown",
        }
        if isinstance(parsed, list):
            tasks = []
            for t in parsed:
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
        if isinstance(parsed, dict) and "tasks" in parsed:
            return PlannerAgent._normalize_response_tasks(parsed["tasks"])
        return []

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
        """Classify a target metric into a task description.

        Classification rules MUST stay in sync with _PLANNER prompt TASK CLASSIFICATION RULES
        in agent_prompts.py. When updating either location, update both.
        """
        latency_targets = {"dram_latency_cycles", "l2_latency_cycles", "l1_latency_cycles",
                           "shmem_latency_targets"}
        capacity_targets = {"max_shmem_per_block_kb", "l2_cache_size_kb", "l1_cache_size_kb",
                            "l2_cache_size_mb"}
        clock_targets = {"actual_boost_clock_mhz", "base_clock_mhz", "sm_clock_mhz"}
        bandwidth_targets = {"dram_bandwidth_gbps", "l2_bandwidth_gbps", "shmem_bandwidth_gbps"}
        ncu_throughput_targets = {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        }

        category = "unknown"
        if target in latency_targets:
            category = "latency_measurement"
        elif target in capacity_targets:
            category = "capacity_measurement"
        elif target in clock_targets:
            category = "clock_measurement"
        elif target in bandwidth_targets:
            category = "bandwidth_measurement"
        elif target in ncu_throughput_targets:
            category = "ncu_throughput_measurement"

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

        Special-case targets (classified as 'unknown' but with known measurement
        approaches) are handled first, before the generic category-based lookup.
        """
        special_methods = {
            "sm_count": (
                "cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount) API call "
                "to query SM count directly, or occupancy-based estimation via "
                "cudaOccupancyMaxActiveBlocksPerMultiprocessor with a test kernel"
            ),
            "bank_conflict_penalty_ratio": (
                "strided access vs sequential access timing comparison in shared memory: "
                "measure elapsed cycles for 32-thread warp reading from shmem with "
                "stride=1 (no conflicts) vs stride=32 (maximum conflicts), "
                "penalty_ratio = conflicted_cycles / unconflicted_cycles"
            ),
            "shmem_latency_cycles": (
                "shared memory latency via pointer-chasing in shared memory space: "
                "allocate uint32_t array in __shared__ memory, build random permutation "
                "chain, single thread follows chain with clock64() timing, "
                "latency = total_cycles / iterations"
            ),
            "l1_latency_cycles": (
                "L1 cache latency via pointer-chasing with working set sized to "
                "L1 cache (typically 16-48 KB per SM), use clock64() for cycle timing, "
                "ensure working set fits in L1 but exceeds register file capacity"
            ),
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": (
                "Compute-bound metric: write a PURELY compute-bound kernel with double-precision "
                "FMA chain (result += a * b + c) where a,b,c are register doubles initialized "
                "BEFORE the timed loop. Use volatile double* sink to prevent dead-code elimination. "
                "Add asm volatile('' : '+d'(sink) : : 'memory') after the FMA loop. "
                "Use #pragma unroll 1 before the loop. Launch sm_count*4 blocks x 256 threads. "
                "Run a WARMUP kernel before timing. Inside kernel: record clock64() before/after FMA loop, "
                "output cycle count. Compute actual_freq_mhz = cycle_count / (elapsed_ms * 1000.0). "
                "COMPUTE the actual percentage: "
                "achieved_flops / (sm_count * fp64_per_sm * actual_freq_mhz * 1e6 * 2) * 100. "
                "Determine fp64_per_sm from compute capability: SM70=32, SM80=32, SM90=64, SM75=2, SM86+=2. "
                "Do NOT use cudaDevAttrClockRate for peak — it may report base clock, not boost!"
            ),
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": (
                "Compute-memory metric: write a FUSED read-compute-write kernel that stresses both "
                "memory and compute. Read input[i] via __restrict__ pointer, compute 8 FMA per "
                "element USING THE READ VALUE (not register-only!), write to volatile output[i]. "
                "WRONG: val = val * 1.0001f + 0.001f where val is register-only → 0.09%! "
                "RIGHT: val = input[i]; then val = val * 1.0001f + 0.001f; then output[i] = val; "
                "Use 64MB+ buffer to ensure DRAM access. "
                "Launch sm_count*4 blocks x 256 threads. Run a WARMUP kernel before timing. "
                "COMPUTE the actual percentage: achieved_bw / peak_bw * 100. "
                "peak_bw = (mem_clock_khz/1000.0) * 1e6 * (bus_width_bits/8) * 2 / 1e9. "
                "Query mem_clock_khz and bus_width_bits via cudaDeviceGetAttribute. "
                "If pct < 5%, the kernel is FUNDAMENTALLY WRONG — not stressing memory at all!"
            ),
        }
        if target in special_methods:
            return special_methods[target]

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
            "ncu_throughput_measurement": (
                "Throughput metric: write a stress kernel that exercises the "
                "target hardware unit, then COMPUTE the actual percentage: "
                "achieved_throughput / peak_throughput * 100. "
                "For sm__throughput: use clock64() inside kernel to measure actual running frequency, "
                "actual_freq_mhz = cycle_count / (elapsed_ms * 1000.0), "
                "achieved_flops / (sm_count * fp64_per_sm * actual_freq_mhz * 1e6 * 2) * 100. "
                "Determine fp64_per_sm from compute capability: SM70=32, SM80=32, SM90=64, SM75=2, SM86+=2. "
                "Do NOT use cudaDevAttrClockRate for peak — it may report base clock, not boost! "
                "For gpu__compute_memory_throughput: achieved_bw / peak_bw * 100. "
                "FMA chain MUST use value read from memory, NOT register-only. "
                "Query hardware params via cudaDeviceGetAttribute at runtime."
            ),
        }
        return methods.get(category, "custom micro-benchmark with clock64() timing and parseable printf output")

    def _route_task(self, task: dict[str, Any]) -> AgentRole:
        """Route a task to the appropriate specialist agent."""
        # All measurement tasks go through CodeGen first, then MetricAnalysis
        return AgentRole.CODE_GEN
