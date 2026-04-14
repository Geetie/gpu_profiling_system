"""Tool Contract data structures — domain layer.

Implements P1 (工具定义能力边界): every operation must be declared through
a registered ToolContract. The model cannot invent ad-hoc behaviours.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolContract:
    """Structured definition of a single tool's interface and boundaries.

    This is the bridge between probabilistic cognition (the model) and
    deterministic execution (the runtime).
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    permissions: list[str]
    requires_approval: bool = False
    is_blocking: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Tool name must not be empty")
        if not self.permissions:
            raise ValueError(
                f"Tool '{self.name}' must declare at least one permission (P2: fail-closed)"
            )

    # ── Serialization ──────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "permissions": self.permissions,
            "requires_approval": self.requires_approval,
            "is_blocking": self.is_blocking,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolContract:
        return cls(
            name=data["name"],
            description=data["description"],
            input_schema=data["input_schema"],
            output_schema=data["output_schema"],
            permissions=data["permissions"],
            requires_approval=data.get("requires_approval", False),
            is_blocking=data.get("is_blocking", True),
        )


class ToolRegistry:
    """Central registry of all available tool contracts.

    Enforces P2: any unregistered tool lookup raises KeyError (fail-closed).
    Enforces P5: disabled tools are removed at build time, not hidden behind
    runtime gates.
    """

    def __init__(self, disabled_tools: set[str] | None = None) -> None:
        self._tools: dict[str, ToolContract] = {}
        self._disabled: frozenset[str] = frozenset(disabled_tools or set())

    @property
    def disabled_tools(self) -> frozenset[str]:
        return self._disabled

    def register(self, contract: ToolContract) -> None:
        if contract.name in self._tools:
            raise ValueError(f"Tool '{contract.name}' is already registered")
        # P5: disabled tools are excluded at build time
        if contract.name in self._disabled:
            return  # silently skip — tool is compile-time eliminated
        self._tools[contract.name] = contract

    def register_bulk(self, contracts: list[ToolContract]) -> None:
        for c in contracts:
            self.register(c)

    def get(self, name: str) -> ToolContract:
        """P2: KeyError if the tool is not registered."""
        if name not in self._tools:
            raise KeyError(
                f"Tool '{name}' is not registered. "
                "Unregistered tools are rejected (P2: fail-closed)."
            )
        return self._tools[name]

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())


# ── Standard Tool Set ──────────────────────────────────────────────


def build_standard_registry(
    disabled_tools: set[str] | None = None,
) -> ToolRegistry:
    """Build and return a registry preloaded with the standard tool set
    defined in spec.md §4.2.

    P5: disabled tools are eliminated at build time — they are never
    registered and therefore cannot be looked up at runtime.
    """
    return build_agent_registry(
        {"run_ncu", "compile_cuda", "execute_binary", "read_file", "write_file",
         "generate_microbenchmark", "kaggle_push"},
        disabled_tools=disabled_tools,
    )


def build_agent_registry(
    allowed_tools: set[str],
    disabled_tools: set[str] | None = None,
) -> ToolRegistry:
    """Build a tool registry for a specific agent role.

    P2 (fail-closed): agents can ONLY use tools explicitly allowed for their role.
    P1 (tools define boundaries): an agent cannot invent behaviors outside its
    registered tool contracts.

    Agent tool assignments per spec.md §5.1:
    - Planner: read_file, write_file (coordination only)
    - CodeGen: compile_cuda, execute_binary, write_file, read_file
    - MetricAnalysis: run_ncu, read_file
    - Verification: read_file (evidence review only)
    """
    disabled = disabled_tools or set()
    registry = ToolRegistry(disabled_tools=disabled)

    all_contracts = [
        ToolContract(
            name="run_ncu",
            description="Execute NVIDIA Nsight Compute analysis on a target binary",
            input_schema={"executable": "string", "metrics": ["string"]},
            output_schema={"raw_output": "string", "parsed_metrics": "object"},
            permissions=["file:read", "process:exec"],
            requires_approval=False,
            is_blocking=True,
        ),
        ToolContract(
            name="compile_cuda",
            description="Compile CUDA source code via nvcc",
            input_schema={"source": "string", "flags": ["string"]},
            output_schema={"success": "boolean", "output": "string", "errors": "string"},
            permissions=["file:read", "file:write", "process:exec"],
            requires_approval=True,
            is_blocking=True,
        ),
        ToolContract(
            name="execute_binary",
            description="Run a compiled binary and capture output",
            input_schema={"binary_path": "string", "args": ["string"]},
            output_schema={"stdout": "string", "stderr": "string", "return_code": "integer"},
            permissions=["process:exec", "file:read"],
            requires_approval=True,
            is_blocking=True,
        ),
        ToolContract(
            name="write_file",
            description="Write content to a file (restricted paths only)",
            input_schema={"file_path": "string", "content": "string"},
            output_schema={"bytes_written": "integer"},
            permissions=["file:write"],
            requires_approval=True,
            is_blocking=True,
        ),
        ToolContract(
            name="read_file",
            description="Read a file from disk",
            input_schema={"file_path": "string"},
            output_schema={"content": "string", "lines": "integer"},
            permissions=["file:read"],
            requires_approval=False,
            is_blocking=True,
        ),
        ToolContract(
            name="generate_microbenchmark",
            description="Auto-generate pointer-chasing or probe CUDA kernels",
            input_schema={"benchmark_type": "string", "parameters": "object"},
            output_schema={"source_code": "string", "file_path": "string"},
            permissions=["file:write"],
            requires_approval=False,
            is_blocking=True,
        ),
        ToolContract(
            name="kaggle_push",
            description="Push kernel source to Kaggle and optionally monitor execution",
            input_schema={
                "kernel_text": "string",
                "kernel_id": "integer (optional, for updates)",
                "kernel_title": "string",
                "kernel_slug": "string",
                "language": "string (default: python3)",
                "kernel_type": "string (default: notebook)",
                "enable_gpu": "boolean (default: true)",
                "enable_internet": "boolean (default: true)",
                "is_private": "boolean (default: true)",
                "monitor": "boolean (default: false) — wait for completion",
                "timeout_min": "integer (default: 90) — monitor timeout",
            },
            output_schema={
                "success": "boolean",
                "kernel_id": "integer",
                "kernel_url": "string",
                "session_id": "integer (if monitor=true)",
                "session_status": "string (if monitor=true)",
                "output_files": "array of strings (if monitor=true and complete)",
                "error": "string (if failure)",
            },
            permissions=["process:exec", "network:outbound"],
            requires_approval=True,
            is_blocking=True,
        ),
    ]

    # P2: only register tools allowed for this agent role
    for contract in all_contracts:
        if contract.name not in allowed_tools:
            continue  # P2: tool is compile-time eliminated for this agent
        if contract.name in disabled:
            continue  # P5: disabled tools eliminated at build time
        registry.register(contract)

    return registry
