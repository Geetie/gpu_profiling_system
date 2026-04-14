"""Binary execution handler — infrastructure layer.

Runs a compiled binary through the sandbox for isolation.
"""
from __future__ import annotations

import os
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

    # Use provided sandbox or fall back to LocalSandbox (dev only)
    runner = sandbox or LocalSandbox(SandboxConfig())

    # Extract binary name for the sandbox command
    binary_name = os.path.basename(binary_path)
    sandbox_dir = os.path.dirname(binary_path)

    try:
        result = runner.run(
            command=binary_name,
            args=list(args),
            work_dir=sandbox_dir,
        )
    except PermissionError as e:
        # INT-4 fix: sandbox path validation failed — return error dict
        return {
            "status": "error",
            "stdout": "",
            "stderr": f"Permission denied: {e}",
            "return_code": -1,
        }

    return {
        "status": "success" if result.return_code == 0 else "error",
        "stdout": result.stdout,
        "stderr": result.stderr,
        "return_code": result.return_code,
    }
