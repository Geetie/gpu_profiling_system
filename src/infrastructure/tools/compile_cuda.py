"""CUDA compilation handler — infrastructure layer.

Compiles CUDA source code via nvcc through the sandbox for isolation.
"""
from __future__ import annotations

import os
import re
import shutil
from typing import Any

from src.infrastructure.sandbox import LocalSandbox, SandboxConfig, SandboxRunner


def _correct_arch_flag(flag: str) -> str:
    """Auto-correct architecture flags below sm_75 to sm_75 for CUDA 12.x.

    Supports multiple flag formats:
    - -arch=sm_XX
    - -arch=XX (bare number, e.g. -arch=0 which is invalid)
    - -gencode=arch=compute_XX,code=sm_XX
    - --gpu-architecture=compute_XX
    - -code=sm_XX

    Returns corrected flag if arch < 75, otherwise original flag.
    """
    if not flag or not flag.strip():
        return flag

    flag_lower = flag.lower()

    # Pattern 0: -arch=<bare_number> (e.g. -arch=0, -arch=60)
    # This is the most common LLM mistake - passing just a number
    if flag_lower.startswith("-arch="):
        value = flag.split("=", 1)[1].strip()
        if value.isdigit():
            arch_num = int(value)
            if arch_num < 75:
                return "-arch=sm_75"
            return f"-arch=sm_{arch_num}"

    # Pattern 1: -arch=sm_XX
    if flag_lower.startswith("-arch=sm_"):
        try:
            arch_num = int(flag.split("=")[1].replace("sm_", ""))
            if arch_num < 75:
                return "-arch=sm_75"
        except (ValueError, IndexError):
            pass

    # Pattern 2: -gencode=arch=compute_XX,code=sm_XX
    elif "arch=compute_" in flag_lower and "code=sm_" in flag_lower:
        try:
            match = re.search(r"compute_(\d+)", flag_lower)
            if match:
                arch_num = int(match.group(1))
                if arch_num < 75:
                    flag = re.sub(r"compute_\d+", "compute_75", flag, count=1)
                    flag = re.sub(r"code=sm_\d+", "code=sm_75", flag, count=1)
                    return flag
        except ValueError:
            pass

    # Pattern 3: --gpu-architecture=compute_XX or --gpu-architecture=sm_XX
    elif flag_lower.startswith("--gpu-architecture="):
        try:
            arch_part = flag.split("=", 1)[1]
            if "compute_" in arch_part.lower():
                arch_num = int(re.search(r"compute_(\d+)", arch_part.lower()).group(1))
            elif "sm_" in arch_part.lower():
                arch_num = int(arch_part.lower().split("sm_")[1])
            else:
                return flag
            
            if arch_num < 75:
                if "compute_" in arch_part.lower():
                    return "--gpu-architecture=compute_75"
                else:
                    return "--gpu-architecture=sm_75"
        except (ValueError, IndexError, AttributeError):
            pass

    # Pattern 4: -code=sm_XX (standalone)
    elif flag_lower.startswith("-code=sm_"):
        try:
            arch_num = int(flag.split("=")[1].replace("sm_", ""))
            if arch_num < 75:
                return "-code=sm_75"
        except (ValueError, IndexError):
            pass

    return flag


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
    has_arch_flag = False
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
        # Auto-correct architecture flags to sm_75+ for CUDA 12.x compatibility
        f = _correct_arch_flag(f)
        # Track if any architecture flag was provided
        if any(f.lower().startswith(p) for p in ["-arch=", "-gencode=", "--gpu-architecture=", "-code="]):
            has_arch_flag = True
        safe_flags.append(f)
    
    # Auto-inject architecture flag if LLM didn't provide one
    if not has_arch_flag:
        try:
            from src.infrastructure.probing.arch_detection import detect_gpu_arch
            detected_arch = detect_gpu_arch(runner)
            safe_flags.append(f"-arch={detected_arch}")
        except Exception:
            safe_flags.append("-arch=sm_75")

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
    # Must use same patterns as sandbox.py for consistency
    has_warning = result.error_type == "warning" or (
        result.return_code == 0 and (
            "warning" in result.stderr.lower() or 
            "deprecated" in result.stderr.lower() or
            "will be removed" in result.stderr.lower()
        ) and not (
            "error: " in result.stderr.lower() or
            "fatal error:" in result.stderr.lower() or
            "undefined reference to" in result.stderr.lower() or
            "cannot open" in result.stderr.lower() or
            ("invalid" in result.stderr.lower() and "option" in result.stderr.lower())
        )
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
        "next_action": "call execute_binary with binary_path" if result.success else "fix source code and retry compile_cuda",
    }
