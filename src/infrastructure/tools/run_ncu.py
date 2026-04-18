"""Run NVIDIA Nsight Compute handler — infrastructure layer.

Executes ncu on a target binary through the sandbox for isolation.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

from src.infrastructure.sandbox import LocalSandbox, SandboxConfig, SandboxRunner


def run_ncu_handler(
    arguments: dict[str, Any],
    sandbox: SandboxRunner | None = None,
) -> dict[str, Any]:
    """Execute NVIDIA Nsight Compute analysis on a target binary.

    VULN-P4-2 fix: Executes through SandboxRunner for isolation.
    If no sandbox is provided, falls back to LocalSandbox (dev only).

    Args (from input_schema):
        executable: str — path to the CUDA binary to profile
        metrics: list[str] — list of metrics to collect

    Returns (from output_schema):
        raw_output: str — raw ncu output
        parsed_metrics: dict — key-value pairs extracted from output
    """
    executable = arguments.get("executable", "")
    metrics = arguments.get("metrics", [])

    if not executable:
        return {
            "raw_output": "",
            "parsed_metrics": {"error": "No executable specified"},
        }

    if not os.path.isfile(executable):
        return {
            "raw_output": "",
            "parsed_metrics": {"error": f"Executable not found: {executable}"},
        }

    ncu_path = shutil.which("ncu")
    if ncu_path is None:
        return {
            "raw_output": "",
            "parsed_metrics": {"error": "ncu (Nsight Compute) not found in PATH"},
        }

    # Use provided sandbox or fall back to LocalSandbox (dev only)
    runner = sandbox or LocalSandbox(SandboxConfig())

    # Build ncu command — metrics are passed as a comma-separated list
    safe_metrics = []
    for m in metrics:
        if not isinstance(m, str):
            continue
        # Skip empty or whitespace-only metrics
        if not m or not m.strip():
            continue
        # Sanitize: only allow alphanumeric, underscore, double underscore, tilde, dot
        if not all(c.isalnum() or c in ("_", "~", ".") for c in m):
            return {
                "status": "error",
                "success": False,
                "raw_output": "",
                "parsed_metrics": {
                    "error": f"Invalid metric name: {m!r}",
                    "hint": "Metric names must contain only alphanumeric chars, underscores, dots, or tildes. "
                            "Examples: 'sm__cycles', 'dram__throughput', 'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum', "
                            "'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed'",
                },
            }
        # Reject single-dot metric (common LLM mistake)
        if m.strip() == ".":
            return {
                "status": "error",
                "success": False,
                "raw_output": "",
                "parsed_metrics": {
                    "error": f"Invalid metric name: {m!r} — '.' is not a valid metric",
                    "hint": "You must provide real ncu metric names like: 'sm__cycles', 'dram__throughput', "
                            "'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum'. "
                            "Do NOT use '.' as a metric name. "
                            "If you're unsure which metrics to use, try: "
                            "'sm__cycles', 'dram__throughput', 'lts__t_sectors_op_read.sum', "
                            "'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum', "
                            "'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed'",
                },
            }
        safe_metrics.append(m)

    cmd_args = []
    if safe_metrics:
        cmd_args = ["--metrics", ",".join(safe_metrics), executable]
    else:
        cmd_args = [executable]

    result = runner.run(
        command=ncu_path,
        args=cmd_args,
        work_dir=os.path.dirname(executable) or os.getcwd(),
    )

    raw = result.stdout + result.stderr
    parsed = _parse_ncu_output(raw)
    return {
        "raw_output": raw,
        "parsed_metrics": parsed,
    }


def _parse_ncu_output(raw: str) -> dict[str, Any]:
    """Parse ncu output into structured metrics."""
    metrics: dict[str, Any] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("---"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            try:
                metrics[key] = float(value)
            except ValueError:
                metrics[key] = value
    return metrics
