"""Shared memory capacity probe — binary search for max shmem per block.

Uses cudaOccupancyMaxActiveBlocksPerMultiprocessor to probe the
maximum shared memory allocation per block. The kernel requests
increasing amounts of shared memory, and we check whether the
occupancy drops to zero.

This directly queries the CUDA runtime — no timing measurements
needed — making it the most reliable probe.
"""
from __future__ import annotations

from typing import Any

from src.infrastructure.probing.kernel_templates import shmem_capacity_kernel
from src.infrastructure.probing.probe_helpers import compile_and_run, parse_nvcc_output


def probe_shmem_capacity(
    sandbox=None,
) -> dict[str, Any] | None:
    """Measure maximum shared memory per block.

    Runs a probe binary that:
    1. Queries cudaOccupancyMaxActiveBlocksPerMP for various shmem sizes
    2. Reports device properties (sharedMemPerBlock, sharedMemPerSM)
    3. Identifies the maximum shared memory per block that still allows
       at least 1 block per SM.

    Returns dict with:
        max_shmem_per_block_kb: float — max shared memory per block in KB
        max_shmem_per_block_bytes: int — in bytes
        max_shmem_per_sm_kb: float — max shared memory per SM in KB
        compute_capability: str — e.g. "75" for T4
        method: str — methodology
    """
    kernel = shmem_capacity_kernel()
    result = compile_and_run(kernel.source, sandbox=sandbox)

    if not result or not result.success:
        return None

    parsed = parse_nvcc_output(result.stdout)

    results: dict[str, Any] = {
        "method": "cuda_occupancy_api_with_shmem_sweep",
    }

    # Parse device properties
    if "device_max_shared_mem_per_block" in parsed:
        block_bytes = int(parsed["device_max_shared_mem_per_block"])
        results["max_shmem_per_block_bytes"] = block_bytes
        results["max_shmem_per_block_kb"] = round(block_bytes / 1024, 2)

    if "device_max_shared_mem_per_sm" in parsed:
        sm_bytes = int(parsed["device_max_shared_mem_per_sm"])
        results["max_shmem_per_sm_bytes"] = sm_bytes
        results["max_shmem_per_sm_kb"] = round(sm_bytes / 1024, 2)

    if "compute_capability" in parsed:
        results["compute_capability"] = str(parsed["compute_capability"])

    # Parse occupancy sweep results
    occupancy_data = _parse_occupancy_sweep(result.stdout)
    if occupancy_data:
        results["occupancy_sweep"] = occupancy_data
        # Find max shmem where blocks > 0
        for entry in reversed(occupancy_data):
            if entry["blocks"] > 0:
                results["max_shmem_per_block_bytes"] = entry["shmem_bytes"]
                results["max_shmem_per_block_kb"] = round(
                    entry["shmem_bytes"] / 1024, 2
                )
                break

    # Confidence: occupancy sweep is very reliable (direct API query)
    # High confidence if we have a clear sweep with multiple data points
    if occupancy_data and len(occupancy_data) >= 4:
        results["_confidence"] = 0.9
    elif results.get("max_shmem_per_block_kb"):
        results["_confidence"] = 0.7
    else:
        results["_confidence"] = 0.0

    return results if results.get("max_shmem_per_block_kb") else None


def _parse_occupancy_sweep(stdout: str) -> list[dict[str, Any]]:
    """Parse shmem_X: blocks=N active_warps=M lines."""
    data = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("shmem_") and ": blocks=" in line:
            parts = line.split(":")
            shmem_part = parts[0]  # "shmem_1024"
            try:
                shmem_bytes = int(shmem_part.split("_")[1])
            except (IndexError, ValueError):
                continue

            blocks = 0
            warps = 0
            for item in parts[1].split():
                if item.startswith("blocks="):
                    try:
                        blocks = int(item.split("=")[1])
                    except (IndexError, ValueError):
                        pass
                elif item.startswith("active_warps="):
                    try:
                        warps = int(item.split("=")[1])
                    except (IndexError, ValueError):
                        pass

            data.append({
                "shmem_bytes": shmem_bytes,
                "blocks": blocks,
                "active_warps": warps,
            })

    return data
