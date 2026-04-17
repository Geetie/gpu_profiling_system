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
            "status": "error",
            "success": False,
            "output": "",
            "errors": "No source code provided",
            "binary_path": "",
        }

    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        return {
            "status": "error",
            "success": False,
            "output": "",
            "errors": "nvcc not found in PATH",
            "binary_path": "",
        }

    # Use provided sandbox or fall back to LocalSandbox (dev only)
    runner = sandbox or LocalSandbox(SandboxConfig())

    # Sanitize flags: only allow safe characters
    _SAFE_FLAG_CHARS = set("-_./+=:,\n")
    safe_flags = []
    for f in flags:
        # Skip empty flags
        if not f or not f.strip():
            continue
        if not all(c.isalnum() or c in _SAFE_FLAG_CHARS for c in f):
            return {
                    "status": "error",
                    "success": False,
                    "output": "",
                    "errors": f"Invalid compiler flag: {f!r}",
                    "binary_path": "",
                }
        # Filter out invalid architecture flags (e.g., sm_0)
        if f.startswith("-arch=sm_") and f.replace("-arch=sm_", "").isdigit():
            arch_num = int(f.replace("-arch=sm_", ""))
            if arch_num < 75:
                # Auto-correct to sm_75 for CUDA 12.x compatibility
                f = "-arch=sm_75"
        safe_flags.append(f)

    # INT-9 fix: compile inside sandbox so output binary is in sandbox root
    # Use src/bin subdirectories to avoid polluting sandbox root
    import os
    source_dir = os.path.join(runner.sandbox_root, "src")
    binary_dir = os.path.join(runner.sandbox_root, "bin")
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(binary_dir, exist_ok=True)
    
    binary_name = "benchmark"
    cmd_args = ["-o", os.path.join(binary_dir, binary_name), "source.cu"] + safe_flags

    result = runner.run(
        source_code=source,
        command=nvcc_path,
        args=cmd_args,
        work_dir=source_dir,
    )

    binary_path = ""
    if result.success:
        binary_path = os.path.join(binary_dir, binary_name)

    # Bug fix: Properly handle warnings vs errors
    # If compilation succeeded but has warnings, still return success
    # but include the warning in the response for visibility
    has_warning = result.error_type == "warning" or (
        result.returncode == 0 and "warning" in result.stderr.lower() and 
        "error:" not in result.stderr.lower() and "fatal" not in result.stderr.lower()
    )
    
    status = "success" if result.success else "error"
    if has_warning and result.success:
        status = "success_with_warning"

    return {
        "status": status,
        "success": result.success,
        "output": result.stdout,
        "errors": result.stderr if not result.success else (result.stderr if has_warning else ""),
        "binary_path": binary_path,
        "source_path": os.path.join(source_dir, "source.cu") if result.success else "",
        "has_warning": has_warning,
    }
