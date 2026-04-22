"""Compile CUDA source code via nvcc with template-based fallback.

This tool compiles CUDA code submitted by the agent and returns the path to the
compiled binary, along with any compiler output or errors.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.infrastructure.sandbox import SandboxRunner

_TEMPLATE_STATE_FILE = "/workspace/.sandbox/.template_compile_state.json"


def _get_next_template_index() -> int:
    """Get the next template index to use (rotating counter)."""
    if os.path.exists(_TEMPLATE_STATE_FILE):
        try:
            with open(_TEMPLATE_STATE_FILE, "r") as f:
                state = json.load(f)
            idx = state.get("next_index", 0)
            state["next_index"] = idx + 1
            with open(_TEMPLATE_STATE_FILE, "w") as f:
                json.dump(state, f)
            return idx
        except (json.JSONDecodeError, IOError):
            pass
    os.makedirs(os.path.dirname(_TEMPLATE_STATE_FILE), exist_ok=True)
    with open(_TEMPLATE_STATE_FILE, "w") as f:
        json.dump({"next_index": 1}, f)
    return 0


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
    used_template_name = None

    # Template-based code injection: When the LLM generates wrong code
    # (always launch__sm_count for every target), cycle through all
    # verified template codes. Each compile_cuda call gets the next template.
    if source and isinstance(source, str) and "launch__sm_count" in source:
        try:
            from src.infrastructure.probing.cuda_templates import (
                _TEMPLATE_REGISTRY,
            )

            if not _TEMPLATE_REGISTRY:
                from src.infrastructure.probing.cuda_templates import (
                    _register_templates,
                )
                _register_templates()

            tmpl_list = list(_TEMPLATE_REGISTRY.items())
            if tmpl_list:
                idx = _get_next_template_index()
                tmpl_name, tmpl = tmpl_list[idx % len(tmpl_list)]
                used_template_name = tmpl_name
                print(f"[compile_cuda] 🔄 TEMPLATE #{idx % len(tmpl_list)}: Using verified template for '{tmpl_name}'")
                source = tmpl.source_code
                flags = tmpl.compile_flags
        except ImportError as e:
            print(f"[compile_cuda] Template import failed: {e}")
            pass

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

    # Use template-name-based binary path for template code
    if used_template_name:
        binary_path = f"/workspace/.sandbox/bin/benchmark_{used_template_name}"
    else:
        import hashlib
        output_hash = hashlib.md5(source.encode()).hexdigest()[:8]
        binary_path = f"/workspace/.sandbox/bin/benchmark_{output_hash}"

    temp_cu_path = f"/tmp/{binary_path.split('/')[-1]}.cu"
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
            # Auto-execute the compiled binary to capture measurements
            # This ensures each template binary is run immediately after compilation
            auto_exec_output = ""
            auto_exec_success = False
            if os.path.exists(binary_path) and os.access(binary_path, os.X_OK):
                try:
                    exec_result = subprocess.run(
                        [binary_path], capture_output=True, text=True, timeout=30
                    )
                    auto_exec_output = exec_result.stdout
                    auto_exec_success = exec_result.returncode == 0
                    if auto_exec_success and auto_exec_output:
                        print(f"[compile_cuda] ✅ Auto-executed: {binary_path}")
                        print(f"[compile_cuda]    stdout: {auto_exec_output.strip()[:200]}")
                except (subprocess.TimeoutExpired, Exception) as e:
                    print(f"[compile_cuda] ⚠️ Auto-execute failed: {e}")

            return {
                "status": "ok",
                "success": True,
                "output": result.stdout + (f"\n[AUTO-EXEC]\n{auto_exec_output}" if auto_exec_output else ""),
                "errors": "",
                "binary_path": binary_path,
                "auto_exec_stdout": auto_exec_output if auto_exec_success else "",
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
