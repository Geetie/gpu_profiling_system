"""Compile CUDA source code via nvcc.

This tool compiles CUDA code submitted by the agent and returns the path to the
compiled binary, along with any compiler output or errors.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.infrastructure.sandbox import SandboxRunner


def compile_cuda_handler(
    arguments: dict[str, Any],
    sandbox: SandboxRunner | None = None,
) -> dict[str, Any]:
    """Compile CUDA source code via nvcc.

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

    output_hash = hashlib.md5(source.encode()).hexdigest()[:8]
    temp_cu_path = f"/tmp/{output_hash}.cu"
    binary_path = f"/workspace/.sandbox/bin/benchmark_{output_hash}"

    os.makedirs(os.path.dirname(binary_path), exist_ok=True)

    try:
        with open(temp_cu_path, "w", encoding="utf-8") as f:
            f.write(source)

        cmd = [nvcc_path, temp_cu_path, "-o", binary_path]
        cmd.extend(["-w"])
        if flags:
            cmd.extend(flags)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        try:
            os.remove(temp_cu_path)
        except OSError:
            pass

        if result.returncode == 0:
            return {
                "status": "ok",
                "success": True,
                "output": result.stdout,
                "errors": "",
                "binary_path": binary_path,
            }
        else:
            return {
                "status": "error",
                "success": False,
                "output": result.stdout,
                "errors": result.stderr,
                "binary_path": "",
            }
    except subprocess.TimeoutExpired:
        try:
            os.remove(temp_cu_path)
        except OSError:
            pass
        return {
            "status": "error",
            "success": False,
            "output": "",
            "errors": "Compilation timed out after 60 seconds",
            "binary_path": "",
        }
    except Exception as e:
        try:
            os.remove(temp_cu_path)
        except OSError:
            pass
        return {
            "status": "error",
            "success": False,
            "output": "",
            "errors": f"Compilation failed: {str(e)}",
            "binary_path": "",
        }
