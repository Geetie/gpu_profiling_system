"""Sub-agent domain model — abstract base, roles, results, and messages.

P7 (生成与评估分离): This module defines the structural guarantees that
a VerificationAgent can never inherit a generator's context.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.application.context import ContextManager
from src.domain.permission import PermissionMode
from src.domain.tool_contract import ToolRegistry
from src.infrastructure.state_persist import StatePersister


# ── Enums ────────────────────────────────────────────────────────────


class AgentRole(Enum):
    """Roles in the multi-agent team."""
    PLANNER = "planner"
    CODE_GEN = "code_gen"
    METRIC_ANALYSIS = "metric_analysis"
    VERIFICATION = "verification"


class SubAgentStatus(Enum):
    """Lifecycle status of a sub-agent result."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    REJECTED = "rejected"   # Verification rejected the result


class PipelineStage(Enum):
    """Sequential stages in the collaboration pipeline."""
    PLAN = "plan"
    CODE_GEN = "code_gen"
    METRIC_ANALYSIS = "metric_analysis"
    VERIFICATION = "verification"


# ── Result & Message ─────────────────────────────────────────────────


@dataclass
class SubAgentResult:
    """Structured output from a sub-agent execution.

    Agents communicate ONLY through these objects — no shared mutable state.
    """
    agent_role: AgentRole
    status: SubAgentStatus = SubAgentStatus.PENDING
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None
    context_fingerprint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.context_fingerprint is None:
            self.context_fingerprint = "none"

    # ── Serialization ──────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_role": self.agent_role.value,
            "status": self.status.value,
            "data": self.data,
            "artifacts": self.artifacts,
            "error": self.error,
            "context_fingerprint": self.context_fingerprint,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubAgentResult:
        return cls(
            agent_role=AgentRole(data["agent_role"]),
            status=SubAgentStatus(data["status"]),
            data=data.get("data", {}),
            artifacts=data.get("artifacts", []),
            error=data.get("error"),
            context_fingerprint=data.get("context_fingerprint", "none"),
            metadata=data.get("metadata", {}),
        )

    def compute_fingerprint(self, context_manager: ContextManager) -> str:
        """SHA-256 hash of the agent's context for P7 audit trail."""
        entries = context_manager.get_entries()
        content = "|".join(f"{e.role.value}:{e.content}" for e in entries)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def is_success(self) -> bool:
        return self.status == SubAgentStatus.SUCCESS

    def is_failed(self) -> bool:
        return self.status in (SubAgentStatus.FAILED, SubAgentStatus.REJECTED)


@dataclass
class CollaborationMessage:
    """Message passed between agents during collaboration."""
    sender: AgentRole
    receiver: AgentRole
    message_type: str       # "task_dispatch", "result", "error", "retry"
    payload: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender": self.sender.value,
            "receiver": self.receiver.value,
            "message_type": self.message_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }


# ── P7 Violation ─────────────────────────────────────────────────────


class P7ViolationError(Exception):
    """Raised when generation and verification contexts are improperly shared."""
    pass


# ── Base Sub-Agent ───────────────────────────────────────────────────

# Signature: (messages) -> str
ModelCaller = Any  # Callable[[list[dict[str, Any]]], str]


class BaseSubAgent(ABC):
    """Abstract base for all sub-agents.

    Each sub-agent owns its own ContextManager (context isolation).
    Sub-agents communicate through SubAgentResult objects only.
    """

    def __init__(
        self,
        role: AgentRole,
        context_manager: ContextManager,
        tool_registry: ToolRegistry,
        state_dir: str,
        permission_mode: PermissionMode = PermissionMode.DEFAULT,
        max_tokens: int = 8000,
    ) -> None:
        self.role = role
        self.context_manager = context_manager
        self.tool_registry = tool_registry
        self.state_dir = state_dir
        self.permission_mode = permission_mode
        self._persister = StatePersister(log_dir=state_dir, filename=f"agent_{role.value}_log.jsonl")
        self._model_caller: ModelCaller | None = None

    def set_model_caller(self, caller: ModelCaller) -> None:
        self._model_caller = caller

    @abstractmethod
    def run(self, message: CollaborationMessage) -> SubAgentResult:
        """Execute the sub-agent's task and return a structured result."""
        ...

    def _build_system_prompt(self) -> str:
        """Build a role-specific system prompt prefix."""
        role_prompts: dict[AgentRole, str] = {
            AgentRole.PLANNER: (
                "You are the Planner Agent. You receive GPU profiling targets, "
                "decompose them into sub-tasks with measurement methodology descriptions, "
                "dispatch to specialists, and integrate final results."
            ),
            AgentRole.CODE_GEN: (
                "You are the Code Generation Agent in a GPU hardware profiling pipeline. "
                "You write CUDA C++ micro-benchmark kernels from scratch, compile them with "
                "compile_cuda, execute them with execute_binary, and parse the numeric output.\n\n"
                "CUDA DESIGN PRINCIPLES (apply these — do NOT copy-paste):\n\n"
                "1. TIMING — always use clock64() (NOT clock()):\n"
                "   - clock() returns 0 under PTX JIT on Pascal+ GPUs, clock64() is reliable\n"
                "   - Measure cycles: t0=clock64(); ...kernel work...; t1=clock64(); cycles=t1-t0\n"
                "   - Frequency-independent: cycle counts are the same regardless of GPU clock speed\n\n"
                "2. LATENCY MEASUREMENT (DRAM, L2, L1) — pointer chasing with random permutation:\n"
                "   - Allocate array of uint32_t indices, fill with random permutation (LCG seeded)\n"
                "   - Single thread follows chain: idx=next[idx] for N iterations\n"
                "   - DRAM: working set 128 MB (32M ints, >> any L2 cache)\n"
                "   - L2: working set 2 MB (512K ints, fits L2, exceeds L1)\n"
                "   - L1: working set 8 KB (2K ints, fits in all GPU L1 caches)\n"
                "   - Latency = total_cycles / iterations (cycles per access)\n\n"
                "3. CACHE CAPACITY (L2 size) — working-set sweep with cliff detection:\n"
                "   - Run pointer-chasing kernel at multiple sizes: 1, 2, 4, 8, 16, 32, 64, 128 MB\n"
                "   - Measure cycles/access at each size\n"
                "   - When size exceeds L2 capacity, latency jumps 3-10x (DRAM access)\n"
                "   - Last size before jump = L2 cache size (typically power of 2)\n\n"
                "4. CLOCK FREQUENCY — cycle count divided by wall-clock time:\n"
                "   - Kernel: 10M iterations of random permutation chain, measure total clock64() cycles\n"
                "   - Host: measure elapsed microseconds with cudaEventElapsedTime or host timing\n"
                "   - freq_MHz = total_cycles / elapsed_us\n\n"
                "5. DRAM BANDWIDTH — STREAM copy:\n"
                "   - Simple kernel: dst[i] = src[i] for large arrays (32M floats = 128 MB)\n"
                "   - Many blocks (65535) to saturate memory bandwidth\n"
                "   - Use cudaEventElapsedTime or ncu timing for elapsed time\n"
                "   - BW_GB/s = bytes_copied / time_us / 1000\n\n"
                "6. SHARED MEMORY CAPACITY — occupancy API sweep:\n"
                "   - Use cudaOccupancyMaxActiveBlocksPerMultiprocessor with increasing shmem sizes\n"
                "   - Max shmem where blocks_per_sm > 0 = per-block capacity\n"
                "   - No timing needed — direct CUDA API query\n\n"
                "7. BANK CONFLICTS — strided vs sequential access comparison:\n"
                "   - Use cudaEventElapsedTime (NOT clock64()) — bank conflicts are fast, events more precise\n"
                "   - Run TWO separate kernel calls in same execution:\n"
                "     (a) Strided: thread t accesses shared[t * 32] — all threads hit same bank\n"
                "     (b) Sequential: thread t accesses shared[t + offset] — each thread hits different bank\n"
                "   - ratio = strided_ms / sequential_ms (>1.0 means bank conflicts exist)\n"
                "   - Use __shared__ uint32_t array, 1 block, 256 threads\n\n"
                "8. SHARED MEMORY BANDWIDTH — cooperative read/write:\n"
                "   - Single block, all threads cooperatively read/write __shared__ array\n"
                "   - Each iteration: every thread reads + writes one element\n"
                "   - BW = (iterations * threads * 2 * sizeof(uint32_t)) / elapsed_us / 1000 GB/s\n"
                "   - Note: per-SM measurement, does NOT scale across SMs\n\n"
                "9. SM COUNT — cudaDeviceGetAttribute + microbenchmark backup:\n"
                "   - Primary: use CUDA runtime API cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount)\n"
                "   - Backup: launch progressive block count, detect queuing time jump with cudaEventElapsedTime\n\n"
                "CRITICAL RULES:\n"
                "- Kernel must be a complete .cu file with #include <cuda_runtime.h>, main(), cudaMalloc, cudaMemcpy\n"
                "- All kernels must use __global__ qualifier\n"
                "- Output must be parseable: use printf(\"key: value\\n\") format\n"
                "- Use cudaEventCreate, cudaEventRecord, cudaEventSynchronize, cudaEventElapsedTime for wall-clock timing\n"
                "- Compile flag: -arch=sm_XX (detect GPU arch from nvidia-smi, e.g. sm_60 for P100)\n"
                "- Each binary must run standalone: allocate GPU memory, launch kernel, copy result back, print\n"
            ),
            AgentRole.METRIC_ANALYSIS: (
                "You are the Metric Analysis Agent. You parse benchmark output to identify "
                "performance bottlenecks: compute-bound, memory-bound, latency-bound, or cache-capacity cliffs. "
                "Parse printf output from CUDA kernels, extract numeric metrics, and assess confidence."
            ),
            AgentRole.VERIFICATION: (
                "You are the Verification Agent. You independently review "
                "experimental results and methodology. You do NOT trust "
                "the generator's reasoning — you verify from first principles.\n\n"
                "GPU REFERENCE VALUES (for sanity checking):\n"
                "- L1 latency: 50-300 cycles, L2: 100-500 cycles, DRAM: 300-1000 cycles\n"
                "- L2 cache: typically power-of-2 MB (2, 4, 8, 40, 50, 60, 72 MB)\n"
                "- DRAM bandwidth: 100-900 GB/s for discrete GPUs\n"
                "- SM count: 256 (H100), 108 (A100), 56 (P100), 32 (V100), 2048 (T4)\n"
                "- Shared memory per block: 48 KB or 64 KB or 164 KB (varies by arch)\n"
                "- Clock frequency: 1000-2500 MHz\n"
                "- Bank conflict ratio: >1.0 for conflicting access, 1.0 for sequential\n"
            ),
        }
        return role_prompts.get(self.role, f"You are the {self.role.value} agent.")

    def _persist_result(self, result: SubAgentResult) -> None:
        """P6: persist result to agent-specific log."""
        self._persister.log_entry(
            action="subagent_result",
            details={
                "role": result.agent_role.value,
                "status": result.status.value,
                "fingerprint": result.context_fingerprint,
            },
            result_data=result.to_dict(),
        )
