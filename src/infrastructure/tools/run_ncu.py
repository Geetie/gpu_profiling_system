"""Run NVIDIA Nsight Compute handler — infrastructure layer.

Executes ncu on a target binary through the sandbox for isolation.

Phase 2 Optimization (OPT-001): NCU Permission Pre-check Mechanism
---------------------------------------------------------------
Problem: In environments like Kaggle, NCU lacks GPU counter permissions (ERR_NVGPUCTRPERM).
         Without caching, each NCU call wastes ~30s waiting for permission errors,
         causing MetricAnalysis to take 135s+ instead of <30s.

Solution: Implement a global permission cache that:
  1. Checks NCU permission status on first call
  2. Caches the result (allowed/denied) for all subsequent calls
  3. Returns immediately (<1ms) when NCU is known to be unavailable
  4. Reduces MetricAnalysis time from 135s to <30s in restricted environments

Architecture:
  - _ncu_permission_cache: Global dict storing permission state
  - check_ncu_permission_fast(): Lightweight pre-check (only runs once)
  - mark_ncu_unavailable(): Marks NCU as unavailable after first error
  - run_ncu_handler(): Enhanced with pre-check and error caching

Backward Compatibility:
  - Works normally in environments WITH NCU permission
  - Graceful degradation in environments WITHOUT NCU permission
  - No changes to function signature or return value format
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Any

from src.infrastructure.sandbox import LocalSandbox, SandboxConfig, SandboxRunner

# Configure module-level logger for NCU pre-check mechanism
logger = logging.getLogger(__name__)

# =============================================================================
# OPT-001: Global NCU Permission Cache
# =============================================================================
# This cache stores the result of the first NCU permission check.
# Once set, all subsequent calls to run_ncu_handler() will use this cached
# result instead of actually executing NCU, saving ~30s per call in restricted
# environments like Kaggle.
#
# State transitions:
#   Initial:  {"checked": False, "allowed": None, "error_message": ""}
#   Success:  {"checked": True,  "allowed": True,  "error_message": ""}
#   Denied:   {"checked": True,  "allowed": False, "error_message": "..."}
# =============================================================================

_ncu_permission_cache = {
    "checked": False,
    "allowed": None,  # None=not checked yet, True=available, False=unavailable
    "error_message": "",
}


def check_ncu_permission_fast() -> bool:
    """Quick check if NCU has GPU counter permissions (only executes once).

    This function implements a lightweight pre-check mechanism that:
    1. Returns immediately if already checked (cache hit)
    2. On first call, attempts to verify NCU availability
    3. Caches the result for future calls
    4. Assumes available by default (optimistic approach)

    The check is intentionally lightweight to minimize overhead:
    - Does NOT attempt a full profile operation (too slow)
    - Only verifies ncu binary exists and is executable
    - Actual permission verification happens on first real profile call

    Returns:
        bool: True if NCU is potentially available, False if definitely unavailable
    """
    global _ncu_permission_cache

    # Cache hit: return immediately (<1ms)
    if _ncu_permission_cache["checked"]:
        logger.debug(f"[NCU Pre-check] Cache hit: allowed={_ncu_permission_cache['allowed']}")
        return _ncu_permission_cache["allowed"]

    # Cache miss: perform initial check
    logger.info("[NCU Pre-check] First call — performing initial availability check...")

    try:
        # Step 1: Check if ncu binary exists in PATH
        ncu_path = shutil.which("ncu")
        if ncu_path is None:
            logger.warning("[NCU Pre-check] ncu binary not found in PATH")
            _ncu_permission_cache["checked"] = True
            _ncu_permission_cache["allowed"] = False
            _ncu_permission_cache["error_message"] = "ncu not found in PATH"
            return False

        # Step 2: Quick version check (verifies ncu is executable)
        # Using timeout of 5 seconds to avoid hanging
        result = subprocess.run(
            ["ncu", "--version"],
            capture_output=True,
            timeout=5,
        )

        if result.returncode == 0:
            # ncu exists and runs successfully
            # Note: This does NOT guarantee GPU counter permissions
            # Those are verified on first actual profile call
            logger.info(f"[NCU Pre-check] ncu is available at {ncu_path}")
            _ncu_permission_cache["checked"] = True
            _ncu_permission_cache["allowed"] = True  # Optimistic: assume available
            _ncu_permission_cache["error_message"] = ""
            return True
        else:
            # ncu binary exists but failed to run
            stderr_text = result.stderr.decode('utf-8', errors='replace')[:200]
            logger.warning(f"[NCU Pre-check] ncu --version failed (rc={result.returncode}): {stderr_text}")
            _ncu_permission_cache["checked"] = True
            _ncu_permission_cache["allowed"] = False
            _ncu_permission_cache["error_message"] = f"ncu --version failed: {stderr_text}"
            return False

    except subprocess.TimeoutExpired:
        logger.error("[NCU Pre-check] ncu --version timed out after 5 seconds")
        _ncu_permission_cache["checked"] = True
        _ncu_permission_cache["allowed"] = False
        _ncu_permission_cache["error_message"] = "ncu --version timed out"
        return False

    except FileNotFoundError:
        logger.warning("[NCU Pre-check] ncu binary not found (FileNotFoundError)")
        _ncu_permission_cache["checked"] = True
        _ncu_permission_cache["allowed"] = False
        _ncu_permission_cache["error_message"] = "ncu binary not found"
        return False

    except Exception as e:
        # Unexpected error during pre-check
        # Log but don't crash — allow actual profile call to determine status
        logger.warning(f"[NCU Pre-check] Unexpected error during pre-check: {e}")
        _ncu_permission_cache["checked"] = True
        _ncu_permission_cache["allowed"] = True  # Allow on uncertainty
        _ncu_permission_cache["error_message"] = str(e)[:200]
        return True


def mark_ncu_unavailable(error_msg: str) -> None:
    """Mark NCU as permanently unavailable after detecting a permission error.

    This function should be called when:
    - ERR_NVGPUCTRPERM error is detected in NCU output
    - Any other permission-related error occurs
    - NCU is confirmed to be unusable in the current environment

    After calling this function:
    - All subsequent run_ncu_handler() calls will return immediately
    - No more time will be wasted attempting NCU profiling
    - The error message will be included in returned results

    Args:
        error_msg: Human-readable description of why NCU is unavailable
                   (e.g., "ERR_NVGPUCTRPERM: Permission denied")
    """
    global _ncu_permission_cache

    previous_state = _ncu_permission_cache.copy()

    _ncu_permission_cache["checked"] = True
    _ncu_permission_cache["allowed"] = False
    _ncu_permission_cache["error_message"] = error_msg

    logger.warning(
        f"[NCU Pre-check] Marked as UNAVAILABLE: {error_msg}\n"
        f"  Previous state: checked={previous_state['checked']}, "
        f"allowed={previous_state['allowed']}\n"
        f"  All future NCU calls will be skipped (returns <1ms)"
    )


def get_ncu_permission_status() -> dict[str, Any]:
    """Get current NCU permission cache status (for debugging/monitoring).

    Returns:
        dict: Copy of the current permission cache state
              Useful for logging, debugging, and UI display
    """
    return _ncu_permission_cache.copy()


def reset_ncu_permission_cache() -> None:
    """Reset NCU permission cache (for testing or forced re-check).

    WARNING: This should only be used in testing scenarios.
    Production code should never reset the cache, as it would
    cause redundant NCU permission checks.

    After resetting, the next run_ncu_handler() call will perform
    a fresh permission check.
    """
    global _ncu_permission_cache

    logger.info("[NCU Pre-check] Permission cache RESET (testing/debug mode)")

    _ncu_permission_cache = {
        "checked": False,
        "allowed": None,
        "error_message": "",
    }


def run_ncu_handler(
    arguments: dict[str, Any],
    sandbox: SandboxRunner | None = None,
) -> dict[str, Any]:
    """Execute NVIDIA Nsight Compute analysis on a target binary.

    VULN-P4-2 fix: Executes through SandboxRunner for isolation.
    If no sandbox is provided, falls back to LocalSandbox (dev only).

    OPT-001 Enhancement: NCU Permission Pre-check
    ------------------------------------------------
    Before executing NCU, this function now:
    1. Checks the global permission cache (if previously marked unavailable, returns <1ms)
    2. Executes NCU normally if available
    3. Detects ERR_NVGPUCTRPERM and other permission errors in output
    4. Marks NCU as permanently unavailable on first permission error

    This reduces MetricAnalysis time from 135s to <30s in restricted environments.

    Args (from input_schema):
        executable: str — path to the CUDA binary to profile
        metrics: list[str] — list of metrics to collect

    Returns (from output_schema):
        raw_output: str — raw ncu output
        parsed_metrics: dict — key-value pairs extracted from output
    """
    # =====================================================================
    # OPT-001: Pre-check NCU permission status (cache hit = <1ms return)
    # =====================================================================
    if _ncu_permission_cache["checked"] and not _ncu_permission_cache["allowed"]:
        # NCU is known to be unavailable — return immediately without execution
        logger.info(
            f"[NCU Pre-check] SKIPPING execution (cached as unavailable)\n"
            f"  Reason: {_ncu_permission_cache['error_message']}\n"
            f"  This call completed in <1ms instead of ~30s"
        )
        return {
            "raw_output": "",
            "parsed_metrics": {
                "error": f"NCU unavailable (cached): {_ncu_permission_cache['error_message']}",
                "hint": (
                    "NCU permission denied in this environment. "
                    "Use text analysis instead of hardware profiling."
                ),
                "cached_result": True,  # Flag indicating this is a cached response
            },
        }

    # Perform initial availability check if not done yet
    # This only runs ONCE per session (subsequent calls use cache)
    if not _ncu_permission_cache["checked"]:
        check_ncu_permission_fast()

        # If pre-check determined NCU is unavailable, return immediately
        if _ncu_permission_cache["checked"] and not _ncu_permission_cache["allowed"]:
            logger.info(
                "[NCU Pre-check] Initial check failed — returning cached unavailable status"
            )
            return {
                "raw_output": "",
                "parsed_metrics": {
                    "error": f"NCU unavailable (pre-check failed): {_ncu_permission_cache['error_message']}",
                    "hint": (
                        "NCU is not available in this environment after initial check. "
                        "Use text analysis mode for performance measurements."
                    ),
                    "cached_result": True,
                },
            }

    # =====================================================================
    # Original logic: Validate inputs
    # =====================================================================
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
        # Mark as unavailable in cache
        mark_ncu_unavailable("ncu (Nsight Compute) not found in PATH")
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

    logger.debug(f"[NCU] Executing: {ncu_path} {' '.join(cmd_args)}")

    result = runner.run(
        command=ncu_path,
        args=cmd_args,
        work_dir=os.path.dirname(executable) or os.getcwd(),
    )

    raw = result.stdout + result.stderr

    # =====================================================================
    # OPT-001: Detect permission errors and cache them for future calls
    # =====================================================================
    raw_lower = raw.lower()
    if "err_nvgpuctrperm" in raw_lower or "permission denied" in raw_lower:
        # Critical: NCU permission error detected!
        # Mark as permanently unavailable to prevent future wasted calls
        error_context = raw[:500] if len(raw) > 500 else raw
        mark_ncu_unavailable("ERR_NVGPUCTRPERM: Permission denied by GPU driver")

        logger.error(
            f"[NCU Pre-check] PERMISSION ERROR DETECTED!\n"
            f"  Error context: {error_context}\n"
            f"  Action: Caching 'unavailable' status for ALL future calls\n"
            f"  Impact: Subsequent run_ncu_handler() calls will return <1ms"
        )

        return {
            "raw_output": raw,
            "parsed_metrics": {
                "error": "NCU permission denied (ERR_NVGPUCTRPERM)",
                "hint": (
                    "NCU is not available in this environment due to GPU driver permissions. "
                    "This is a security restriction that cannot be bypassed. "
                    "All subsequent NCU calls will be automatically skipped. "
                    "Please switch to text-based analysis using CodeGen measurements."
                ),
                "permission_error": True,  # Flag for agent_loop.py to detect
                "cached_result": False,   # First occurrence, not yet cached
            },
        }

    # Check for other common errors that indicate NCU issues
    if "no devices detected" in raw_lower or "cuda error" in raw_lower:
        logger.warning(f"[NCU] CUDA device error detected: {raw[:300]}")
        # Don't cache these — they might be transient

    parsed = _parse_ncu_output(raw)

    # Log success for debugging
    if parsed and not any(k.startswith("error") for k in parsed.keys()):
        logger.info(f"[NCU] Successfully collected {len(parsed)} metrics")

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
