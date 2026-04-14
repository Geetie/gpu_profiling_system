"""Code Generation Agent — writes CUDA micro-benchmark kernels.

Operates in an isolated context with sandbox access for compilation
and execution of generated code.

The agent generates CUDA code from design principles (no hardcoded templates):
1. Receives design methodology from prompt (see _build_system_prompt in subagent.py)
2. LLM writes complete CUDA C++ source implementing the design
3. compile_cuda tool compiles the source
4. execute_binary tool runs the compiled binary
5. Agent parses the numeric output and reports the measured value
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
from src.infrastructure.sandbox import LocalSandbox, SandboxConfig, SandboxRunner


class CodeGenAgent(BaseSubAgent):
    """Generates, compiles, and executes CUDA micro-benchmark kernels.

    Uses LLM to generate CUDA source code from design principles.
    No hardcoded templates — the agent writes code based on methodology descriptions.
    """

    def __init__(
        self,
        context_manager: ContextManager | None = None,
        tool_registry: ToolRegistry | None = None,
        state_dir: str = ".state",
        permission_mode: PermissionMode = PermissionMode.DEFAULT,
        max_tokens: int = 8000,
        sandbox: SandboxRunner | None = None,
    ) -> None:
        super().__init__(
            role=AgentRole.CODE_GEN,
            context_manager=context_manager or ContextManager(max_tokens=max_tokens),
            tool_registry=tool_registry or ToolRegistry(),
            state_dir=state_dir,
            permission_mode=permission_mode,
            max_tokens=max_tokens,
        )
        self._sandbox = sandbox or LocalSandbox(SandboxConfig())

    def run(self, message: CollaborationMessage) -> SubAgentResult:
        """Generate a CUDA micro-benchmark based on the task description."""
        self.context_manager.add_entry(
            Role.SYSTEM, self._build_system_prompt(), token_count=30
        )

        task = message.payload.get("task", {})
        target = task.get("target", "unknown")
        category = task.get("category", "unknown")
        method = task.get("method", "custom micro-benchmark")

        # Build context with task description
        self.context_manager.add_entry(
            Role.USER,
            f"Generate a CUDA micro-benchmark for target '{target}' "
            f"(category: {category}, method: {method})",
            token_count=20,
        )

        # Generate source code via LLM (no template fallbacks)
        source_code = self._generate_kernel(target, category, method)

        # Compile and execute in sandbox
        compile_result = self._compile(source_code)
        if not compile_result.success:
            return SubAgentResult(
                agent_role=self.role,
                status=SubAgentStatus.FAILED,
                error=f"Compilation failed: {compile_result.stderr}",
            )

        exec_result = self._execute(compile_result.artifacts)
        if not exec_result.success:
            return SubAgentResult(
                agent_role=self.role,
                status=SubAgentStatus.FAILED,
                error=f"Execution failed: {exec_result.stderr}",
            )

        result = SubAgentResult(
            agent_role=self.role,
            status=SubAgentStatus.SUCCESS,
            data={
                "target": target,
                "category": category,
                "raw_output": exec_result.stdout,
                "compile_output": compile_result.stdout,
            },
            artifacts=list(compile_result.artifacts.values()),
        )

        result.context_fingerprint = result.compute_fingerprint(self.context_manager)
        self._persist_result(result)
        return result

    def _generate_kernel(self, target: str, category: str, method: str) -> str:
        """Generate CUDA kernel source code via LLM.

        The LLM receives design principles in the system prompt and writes
        complete CUDA C++ source code. There are no hardcoded templates.
        """
        if self._model_caller is not None:
            messages = self.context_manager.to_messages()
            return self._model_caller(messages)

        # No model caller — this is a critical error, not a template fallback
        raise RuntimeError(
            f"CodeGen requires LLM to generate CUDA code. "
            f"No model caller configured for target '{target}'. "
            f"The agent must write CUDA code from design principles — no templates available."
        )

    def _compile(self, source_code: str) -> Any:
        """Compile CUDA source code in the sandbox.

        INT-5 fix: Log compilation attempt for audit trail (P6).
        Full ToolRunner integration requires architectural refactoring.
        """
        self._persister.log_entry(
            action="compile_attempt",
            details={"source_length": len(source_code), "command": "nvcc"},
        )
        result = self._sandbox.run(
            source_code=source_code,
            command="nvcc",
            args=["-o", "benchmark", "source.cu"],
        )
        self._persister.log_entry(
            action="compile_result",
            details={
                "success": result.success,
                "artifacts": list(result.artifacts.keys()),
            },
        )
        return result

    def _execute(self, artifacts: dict) -> Any:
        """Execute the compiled binary in the sandbox.

        INT-5 fix: Log execution attempt for audit trail (P6).
        """
        binary = artifacts.get("source", "./benchmark")
        binary_dir = binary.rsplit("/", 1)[0] if "/" in binary else "."
        # Derive binary name from artifact path
        binary_name = binary.rsplit("/", 1)[-1] if "/" in binary else "benchmark"
        # Replace .cu extension with no extension for the binary
        if binary_name.endswith(".cu"):
            binary_name = binary_name[:-3]

        self._persister.log_entry(
            action="execute_attempt",
            details={"binary": binary_name, "work_dir": binary_dir},
        )
        result = self._sandbox.run(
            command=f"./{binary_name}",
            args=[],
        )
        self._persister.log_entry(
            action="execute_result",
            details={
                "success": result.success,
                "return_code": result.return_code,
            },
        )
        return result
