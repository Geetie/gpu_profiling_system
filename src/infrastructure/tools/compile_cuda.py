"""CUDA compilation handler — infrastructure layer.

Compiles CUDA source code via nvcc through the sandbox for isolation.
"""
from __future__ import annotations

import os
import shutil
from typing import Any

from src.infrastructure.sandbox import LocalSandbox, SandboxConfig, SandboxRunner


def compile_cuda_handler(
    arguments: dict[str, Any],
    sandbox: SandboxRunner | None = None,
) -> dict[str, Any]:
    """Compile CUDA source code via nvcc.

    INT-9 fix: Executes through SandboxRunner so compilation occurs
    inside the sandbox, ensuring the output binary is in a sandbox-accessible
    directory for subsequent execute_binary calls.

    Args (from input_schema):
        source: str — CUDA source code
        flags: list[str] — compiler flags (e.g. ["-O3", "-arch=sm_80"])

    Returns (from output_schema):
        success: bool — whether compilation succeeded
        output: str — compiler stdout
        errors: str — compiler stderr
        binary_path: str — path to the compiled binary (on success)
    """
    source = arguments.get("source", "")
    flags = arguments.get("flags", [])

    if not source:
        return {
            "success": False,
            "output": "",
            "errors": "No source code provided",
            "binary_path": "",
        }

    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        return {
            "success": False,
            "output": "",
            "errors": "nvcc not found in PATH",
            "binary_path": "",
        }

    # Use provided sandbox or fall back to LocalSandbox (dev only)
    runner = sandbox or LocalSandbox(SandboxConfig())

    # Sanitize flags: only allow safe characters
    _SAFE_FLAG_CHARS = set("-_./+=:,")
    safe_flags = []
    for f in flags:
        if not all(c.isalnum() or c in _SAFE_FLAG_CHARS for c in f):
            return {
                "success": False,
                "output": "",
                "errors": f"Invalid compiler flag: {f!r}",
                "binary_path": "",
            }
        safe_flags.append(f)

    # INT-9 fix: compile inside sandbox so output binary is in sandbox root
    binary_name = "benchmark"
    cmd_args = ["-o", binary_name, "source.cu"] + safe_flags

    result = runner.run(
        source_code=source,
        command=nvcc_path,
        args=cmd_args,
        work_dir=runner.sandbox_root,
    )

    binary_path = ""
    if result.success:
        binary_path = os.path.join(runner.sandbox_root, binary_name)

    return {
        "success": result.success,
        "output": result.stdout,
        "errors": result.stderr if not result.success else "",
        "binary_path": binary_path,
    }
