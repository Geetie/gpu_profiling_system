"""Binary execution handler — infrastructure layer.

Runs a compiled binary through the sandbox for isolation.
"""
from __future__ import annotations

import os
import stat
from typing import Any

from src.infrastructure.sandbox import LocalSandbox, SandboxConfig, SandboxRunner


def execute_binary_handler(
    arguments: dict[str, Any],
    sandbox: SandboxRunner | None = None,
) -> dict[str, Any]:
    """Run a compiled binary and capture output.

    VULN-P4-3 fix: Executes through SandboxRunner for isolation.
    If no sandbox is provided, falls back to LocalSandbox (dev only).

    Args (from input_schema):
        binary_path: str — path to the binary to execute
        args: list[str] — command-line arguments

    Returns (from output_schema):
        stdout: str — standard output
        stderr: str — standard error
        return_code: int — process exit code
    """
    binary_path = arguments.get("binary_path", "")
    args = arguments.get("args", [])

    if not binary_path:
        return {
            "status": "error",
            "stdout": "",
            "stderr": "No binary path specified",
            "return_code": -1,
        }

    if not os.path.isfile(binary_path):
        return {
            "status": "error",
            "stdout": "",
            "stderr": f"Binary not found: {binary_path}",
            "return_code": -1,
        }

    # Ensure execute permission on the binary
    try:
        if not os.access(binary_path, os.X_OK):
            current_mode = os.stat(binary_path).st_mode
            os.chmod(binary_path, current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError as e:
        return {
            "status": "error",
            "stdout": "",
            "stderr": f"Cannot set execute permission on {binary_path}: {e}",
            "return_code": -1,
        }

    # Use provided sandbox or fall back to LocalSandbox (dev only)
    runner = sandbox or LocalSandbox(SandboxConfig())

    # Extract binary name for the sandbox command
    binary_name = os.path.basename(binary_path)
    sandbox_dir = os.path.dirname(binary_path)

    # Use ./ prefix to ensure subprocess finds the binary in cwd
    # This is critical for environments where '.' is not in PATH
    if not os.path.isabs(binary_name):
        binary_name = f"./{binary_name}"

    try:
        result = runner.run(
            command=binary_name,
            args=list(args),
            work_dir=sandbox_dir,
        )
    except PermissionError as e:
        return {
            "status": "error",
            "stdout": "",
            "stderr": f"Permission denied: {e}",
            "return_code": -1,
        }

    # Enhanced error feedback: when command not found, list directory contents
    if result.return_code == -1 and "Command not found" in result.stderr:
        try:
            dir_contents = os.listdir(sandbox_dir) if os.path.isdir(sandbox_dir) else []
            files_hint = f"\nAvailable files in {sandbox_dir}: {dir_contents[:10]}"
        except OSError:
            files_hint = ""
        return {
            "status": "error",
            "stdout": result.stdout,
            "stderr": result.stderr + files_hint,
            "return_code": result.return_code,
        }

    stdout = result.stdout
    if len(stdout) > 4000:
        measurement_lines = [l for l in stdout.splitlines()
                             if l.strip() and ":" in l and not l.strip().startswith("//")]
        if measurement_lines:
            stdout = "\n".join(measurement_lines) + f"\n...[{len(result.stdout.splitlines()) - len(measurement_lines)} other lines truncated]"
        else:
            stdout = stdout[:4000] + f"\n...[truncated, {len(result.stdout)} total chars]"

    return {
        "status": "success" if result.return_code == 0 else "error",
        "stdout": stdout,
        "stderr": result.stderr[:2000] if len(result.stderr) > 2000 else result.stderr,
        "return_code": result.return_code,
    }
