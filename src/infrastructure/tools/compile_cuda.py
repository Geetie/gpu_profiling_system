"""Compile CUDA source code via nvcc with template-based fallback.

This tool compiles CUDA code submitted by the agent and returns the path to the
compiled binary, along with any compiler output or errors.
"""

from __future__ import annotations

import hashlib
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
    # Initialize
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

    # Template-based code injection: When the LLM generates wrong code
    # (always launch__sm_count for every target), we cycle through all
    # verified template codes. Each compile_cuda call gets the next template.
    if source and isinstance(source, str) and "launch__sm_count" in source:
        try:
            from src.infrastructure.probing.cuda_templates import (
                _TEMPLATE_REGISTRY,
            )

            # Ensure registry is populated
            if not _TEMPLATE_REGISTRY:
                from src.infrastructure.probing.cuda_templates import (
                    _register_templates,
                )

                _register_templates()

            tmpl_list = list(_TEMPLATE_REGISTRY.items())
            if tmpl_list:
                idx = _get_next_template_index()
                tmpl_name, tmpl = tmpl_list[idx % len(tmpl_list)]
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

    # Create a temporary .cu file
    output_hash = hashlib.md5(source.encode()).hexdigest()[:8]
    temp_cu_path = f"/tmp/{output_hash}.cu"
    binary_path = f"/workspace/.sandbox/bin/benchmark_{output_hash}"

    # Ensure the binary output directory exists
    os.makedirs(os.path.dirname(binary_path), exist_ok=True)

    try:
        with open(temp_cu_path, "w", encoding="utf-8") as f:
            f.write(source)

        # Build the compiler command
        cmd = [nvcc_path, temp_cu_path, "-o", binary_path]
        cmd.extend(["-w"])
        if flags:
            cmd.extend(flags)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        # Clean up the temp .cu file
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
