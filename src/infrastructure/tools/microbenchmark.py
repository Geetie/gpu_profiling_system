"""Micro-benchmark generation handler — infrastructure layer.

DISABLED: The CodeGen agent now writes CUDA code directly from design
principles (injected via system prompt) and passes source to compile_cuda.
This tool existed when code generation was template-based (方案 #1).
Now the agent generates code from design methodology (方案 #2).

If you need auto-generation, use the CodeGen agent's compile_cuda +
execute_binary tool chain directly.
"""
from __future__ import annotations

from typing import Any


def generate_microbenchmark_handler(arguments: dict[str, Any]) -> dict[str, Any]:
    """This tool is disabled.

    The CodeGen agent now writes CUDA code directly from design principles
    in its system prompt and uses compile_cuda + execute_binary.

    See subagent.py _build_system_prompt(AgentRole.CODE_GEN) for the
    design principles injected into the agent.
    """
    return {
        "source_code": "",
        "file_path": "",
        "error": (
            "generate_microbenchmark is disabled. "
            "The CodeGen agent writes CUDA code from design principles "
            "(see system prompt) and uses compile_cuda + execute_binary directly. "
            "Write the CUDA .cu source code string and pass it to compile_cuda."
        ),
    }
