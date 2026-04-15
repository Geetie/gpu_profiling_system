"""Domain-layer Protocol definitions — dependency inversion boundary.

These Protocols define the contracts that infrastructure and application
layers must satisfy.  High-level domain logic depends ONLY on these
abstractions, never on concrete implementations.

Design rationale:
  - typing.Protocol enables structural subtyping (duck typing with
    static checking).  Concrete classes don't need to inherit from
    these Protocols — they just need to implement the same methods.
  - This follows the Dependency Inversion Principle (DIP): domain
    layer defines what it needs; infrastructure layer provides it.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


# ── Persistence ──────────────────────────────────────────────────────


@runtime_checkable
class CanPersist(Protocol):
    """Append-only structured log — P6 (状态必须落盘)."""

    def log_entry(
        self, action: str, details: dict[str, Any] | None = None, **extra: Any
    ) -> None: ...

    def log_tool_execution(
        self,
        tool_name: str,
        inputs: dict[str, Any] | None = None,
        status: str = "success",
        output: Any = None,
    ) -> None: ...

    def log_error(
        self, error_type: str, context: str, message: str
    ) -> None: ...

    def log_invariant_violation(self, invariant: str, detail: str) -> None: ...

    def load_history(self) -> list[dict[str, Any]]: ...


# ── Model Calling ────────────────────────────────────────────────────


@runtime_checkable
class CanCallModel(Protocol):
    """LLM inference callable — (messages, tools?) -> response text."""

    def __call__(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> str: ...


# ── Tool Execution ───────────────────────────────────────────────────


@runtime_checkable
class CanExecuteTool(Protocol):
    """Execute a registered tool by name with arguments."""

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


# ── Sandbox ──────────────────────────────────────────────────────────


@runtime_checkable
class CanRunSandbox(Protocol):
    """Isolated execution environment for CUDA compilation/execution."""

    config: Any

    def run(
        self,
        source_code: str | None = None,
        command: str = "",
        args: list[str] | None = None,
        work_dir: str | None = None,
    ) -> Any: ...

    def cleanup(self) -> None: ...

    @property
    def sandbox_root(self) -> str: ...


# ── Approval ─────────────────────────────────────────────────────────


@runtime_checkable
class CanApprove(Protocol):
    """Human-in-the-loop approval gate for tool execution."""

    def submit(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        permissions: list[str],
        mode: Any,
    ) -> Any: ...

    def respond(self, request_id: str, approved: bool, reason: str | None = None) -> None: ...


# ── Permission Checking ──────────────────────────────────────────────


@runtime_checkable
class CanCheckPermission(Protocol):
    """Permission gate — P2 (故障关闭)."""

    def is_allowed(self, permission: str) -> bool: ...

    def requires_approval(self, permission: str) -> bool: ...


# ── Context Management ───────────────────────────────────────────────


@runtime_checkable
class CanManageContext(Protocol):
    """Dynamic context assembly and compression — P3."""

    @property
    def total_tokens(self) -> int: ...

    def add_entry(self, role: Any, content: str, token_count: int = 0) -> None: ...

    def update_system_entry(self, content: str, token_count: int = 0) -> None: ...

    def get_entries(self) -> list[Any]: ...

    def is_over_budget(self) -> bool: ...

    def compress(self) -> int: ...

    def clear(self) -> None: ...

    def to_messages(self) -> list[dict[str, Any]]: ...


# ── Tool Registry ────────────────────────────────────────────────────


@runtime_checkable
class CanLookupTool(Protocol):
    """Tool contract registry — P1 + P2."""

    def get(self, name: str) -> Any: ...

    def has_tool(self, name: str) -> bool: ...

    def list_tools(self) -> list[str]: ...


# ── Event Emission ───────────────────────────────────────────────────


@runtime_checkable
class CanEmitEvents(Protocol):
    """Observable event source for agent loop lifecycle."""

    def on_event(self, handler: Any) -> None: ...

    def _emit(self, kind: Any, payload: dict[str, Any] | None = None) -> None: ...
