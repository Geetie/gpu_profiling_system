"""L2 cache capacity probe — working-set sweep with cliff detection.

Measures L2 cache capacity by sweeping the working set size and
detecting the latency "cliff" — the point where latency jumps sharply
because the working set no longer fits in L2 cache.

The sweep runs the pointer chasing kernel at progressively larger
sizes and analyzes the latency curve for discontinuities.
"""
from __future__ import annotations

from typing import Any

from src.infrastructure.probing.kernel_templates import working_set_sweep_kernel
from src.infrastructure.probing.probe_helpers import (
    compile_and_run,
    parse_nvcc_output,
    _assess_from_ratio,
)


def probe_l2_cache_capacity(
    sandbox=None,
) -> dict[str, Any] | None:
    """Measure L2 cache capacity via working-set sweep.

    Sweeps working set sizes from 1 MB to 128 MB, measuring latency
    at each point. The capacity is identified at the "cliff" — where
    latency jumps by >3x, indicating cache overflow.

    Returns dict with:
        l2_cache_size_mb: float — estimated L2 capacity in MB
        l2_cache_size_kb: float — estimated L2 capacity in KB
        sweep_data: dict — latency at each working set size
        cliff_index: int — index where capacity cliff was detected
        method: str — methodology
    """
    # Working set sizes to sweep (in elements, each element = 4 bytes = int)
    # Coverage: 1 MB to 128 MB working sets, with intermediate points
    # for better resolution around common L2 capacities (25-64 MB range)
    sizes_elements = [
        256 * 1024,       # 1 MB
        512 * 1024,       # 2 MB
        1024 * 1024,      # 4 MB
        2 * 1024 * 1024,  # 8 MB
        3 * 1024 * 1024,  # 12 MB
        4 * 1024 * 1024,  # 16 MB
        5 * 1024 * 1024,  # 20 MB
        6 * 1024 * 1024,  # 24 MB
        8 * 1024 * 1024,  # 32 MB
        10 * 1024 * 1024, # 40 MB
        12 * 1024 * 1024, # 48 MB
        16 * 1024 * 1024, # 64 MB
        24 * 1024 * 1024, # 96 MB
        32 * 1024 * 1024, # 128 MB
    ]

    sweep_data: list[dict[str, float]] = []

    for size_elem in sizes_elements:
        kernel = working_set_sweep_kernel(
            max_size=size_elem,
            iterations=5000,
        )
        result = compile_and_run(kernel.source, sandbox=sandbox)
        if result and result.success:
            parsed = parse_nvcc_output(result.stdout)
            cpa = parsed.get("cycles_per_access", 0)
            size_bytes = parsed.get("size_bytes", size_elem * 4)
            if cpa > 0:
                sweep_data.append({
                    "size_bytes": int(size_bytes),
                    "size_mb": round(size_bytes / (1024 * 1024), 2),
                    "cycles_per_access": round(cpa, 2),
                })

    if not sweep_data:
        return None

    # Find the capacity cliff: largest latency jump
    cliff_index = _find_capacity_cliff(sweep_data)
    l2_size_bytes = 0
    cliff_ratio = 0.0

    if cliff_index > 0 and cliff_index < len(sweep_data):
        # The last size before the cliff is the cache capacity
        l2_size_bytes = int(sweep_data[cliff_index - 1]["size_bytes"])
        cliff_ratio = sweep_data[cliff_index]["cycles_per_access"] / \
                      sweep_data[cliff_index - 1]["cycles_per_access"] \
                      if sweep_data[cliff_index - 1]["cycles_per_access"] > 0 else 0.0
    elif sweep_data:
        # If no clear cliff, use the size where latency first exceeds
        # 2x the minimum latency
        min_cpa = min(d["cycles_per_access"] for d in sweep_data)
        for i, d in enumerate(sweep_data):
            if d["cycles_per_access"] > min_cpa * 2.5:
                l2_size_bytes = int(sweep_data[i - 1]["size_bytes"]) if i > 0 else 0
                cliff_index = i
                cliff_ratio = d["cycles_per_access"] / min_cpa if min_cpa > 0 else 0.0
                break

    results: dict[str, Any] = {
        "sweep_data": sweep_data,
        "cliff_index": cliff_index,
        "method": "working_set_sweep_with_cliff_detection",
    }

    # Confidence: based on how sharp the latency cliff is
    # A sharp cliff (3x+ jump) gives high confidence; a gradual slope is low confidence
    if cliff_ratio > 0:
        results["_confidence"] = round(
            _assess_from_ratio(cliff_ratio, ideal=6.0, tolerance=0.6), 2
        )
    elif sweep_data:
        results["_confidence"] = 0.3  # No cliff detected — fallback estimate

    if l2_size_bytes > 0:
        results["l2_cache_size_mb"] = round(l2_size_bytes / (1024 * 1024), 4)
        results["l2_cache_size_kb"] = round(l2_size_bytes / 1024, 2)

    return results


def _find_capacity_cliff(sweep_data: list[dict[str, float]]) -> int:
    """Find the index where latency jumps most dramatically."""
    if len(sweep_data) < 2:
        return 0

    max_ratio = 0.0
    cliff_idx = 0

    for i in range(1, len(sweep_data)):
        prev_cpa = sweep_data[i - 1]["cycles_per_access"]
        curr_cpa = sweep_data[i]["cycles_per_access"]
        if prev_cpa > 0:
            ratio = curr_cpa / prev_cpa
            if ratio > max_ratio:
                max_ratio = ratio
                cliff_idx = i

    return cliff_idx
