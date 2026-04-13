"""Code Generation Agent — writes CUDA micro-benchmark kernels.

Operates in an isolated context with sandbox access for compilation
and execution of generated code.
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
    """Generates, compiles, and executes CUDA micro-benchmark kernels."""

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

        # Generate source code (via model caller or template)
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
        """Generate CUDA kernel source code.

        Uses model caller if available, otherwise falls back to templates.
        """
        if self._model_caller is not None:
            messages = self.context_manager.to_messages()
            return self._model_caller(messages)

        # Fallback: generate a basic template kernel
        return self._template_kernel(target, category, method)

    def _template_kernel(self, target: str, category: str, method: str) -> str:
        """Generate a template CUDA kernel when no model caller is available."""
        templates = {
            "latency_measurement": self._pointer_chasing_kernel,
            "capacity_measurement": self._working_set_kernel,
            "clock_measurement": self._timing_loop_kernel,
            "bandwidth_measurement": self._stream_kernel,
        }
        generator = templates.get(category, self._generic_kernel)
        return generator(target, method)

    def _pointer_chasing_kernel(self, target: str, method: str) -> str:
        return f"""// Pointer-chasing kernel for {target}
// Method: {method}
#include <cuda_runtime.h>

__global__ void pointer_chase(uint32_t* next, uint32_t* latency) {{
    uint32_t idx = threadIdx.x;
    uint32_t start = idx;
    uint32_t iterations = 100000;

    clock_t t0 = clock();
    for (uint32_t i = 0; i < iterations; i++) {{
        idx = next[idx];
    }}
    clock_t t1 = clock();

    latency[start] = t1 - t0;
}}
"""

    def _working_set_kernel(self, target: str, method: str) -> str:
        return f"""// Working-set sweep kernel for {target}
// Method: {method}
#include <cuda_runtime.h>

__global__ void working_set(int* data, int* result, int size) {{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int sum = 0;
    for (int stride = 1; stride < size; stride *= 2) {{
        sum += data[(idx + stride) % size];
    }}
    result[idx] = sum;
}}
"""

    def _timing_loop_kernel(self, target: str, method: str) -> str:
        return f"""// Timing loop kernel for {target}
// Method: {method}
#include <cuda_runtime.h>

__global__ void timing_loop(uint64_t* out) {{
    clock_t t0 = clock();
    for (volatile int i = 0; i < 1000000; i++) {{}}
    clock_t t1 = clock();
    out[threadIdx.x] = t1 - t0;
}}
"""

    def _stream_kernel(self, target: str, method: str) -> str:
        return f"""// Stream kernel for {target}
// Method: {method}
#include <cuda_runtime.h>

__global__ void stream_copy(float* out, const float* in, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {{
        out[i] = in[i] * 2.0f + 1.0f;
    }}
}}
"""

    def _generic_kernel(self, target: str, method: str) -> str:
        return f"""// Generic kernel for {target}
// Method: {method}
#include <cuda_runtime.h>

__global__ void generic_kernel(int* out) {{
    out[threadIdx.x] = threadIdx.x;
}}
"""

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
