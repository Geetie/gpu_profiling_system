"""DRAM latency probe — measures memory hierarchy latency in hardware cycles.

Uses pointer chasing with different working set sizes to isolate:
- L1 cache latency (working set < 8 KB)
- L2 cache latency (working set < L2 capacity)
- DRAM latency (working set >> L2 capacity, ≥ 64 MB to exceed H100 L2)

All measurements use clock() cycle counts — completely independent of
the SM clock frequency. The actual frequency is only needed when
converting to nanoseconds for reporting.
"""
from __future__ import annotations

from typing import Any

from src.infrastructure.probing.kernel_templates import pointer_chase_kernel
from src.infrastructure.probing.probe_helpers import (
    compile_and_run,
    parse_nvcc_output,
    _assess_confidence,
)


def probe_dram_latency_cycles(
    sandbox=None,
    actual_freq_mhz: float | None = None,
) -> dict[str, Any] | None:
    """Measure DRAM latency in SM clock cycles.

    Strategy: run pointer chasing with a large working set that
    exceeds L2 cache capacity, ensuring all accesses hit DRAM.

    Returns dict with:
        dram_latency_cycles: float — latency in SM clock cycles
        dram_latency_ns: float — latency in nanoseconds (if freq known)
        l2_latency_cycles: float — L2 latency in cycles
        l1_latency_cycles: float — L1 latency in cycles
        method: str — measurement methodology
    """
    results: dict[str, Any] = {}

    # Probe at multiple sizes to identify cache hierarchy levels
    # Size progression: small (L1) -> medium (L2) -> large (DRAM)
    # L1: 2K ints = 8 KB — fits in ALL GPU L1 caches (T4: 24 KB data, A100: 128 KB)
    # L2: 512K ints = 2 MB — fits in all known GPU L2 caches
    # DRAM: 32M ints = 128 MB — well exceeds H100 L2 (50-60 MB) and A100-80GB (40-64 MB)
    #       with safe margin for memory compression or future larger-L2 GPUs
    probe_sizes = [
        (2 * 1024, "l1"),         # 2K ints = 8 KB
        (512 * 1024, "l2"),       # 512K ints = 2 MB
        (32 * 1024 * 1024, "dram"),  # 32M ints = 128 MB
    ]

    latencies: dict[str, float] = {}

    for size, level in probe_sizes:
        # Iteration counts scale with latency level:
        #   L1: fast (~20 cycles) → 50K iters for stable measurement
        #   L2: medium (~100 cycles) → 20K iters
        #   DRAM: slow (~400 cycles) → 100K iters for <0.1% noise
        iter_map = {"l1": 50000, "l2": 20000, "dram": 100000}
        iters = iter_map.get(level, 10000)
        kernel = pointer_chase_kernel(
            array_size=size,
            iterations=iters,
            output_name=f"latency_{level}",
        )
        result = compile_and_run(kernel.source, sandbox=sandbox)
        if result and result.success:
            parsed = parse_nvcc_output(result.stdout)
            cpa = parsed.get("cycles_per_access", 0)
            if cpa > 0:
                latencies[level] = cpa

    # Use the large-working-set measurement as DRAM latency
    if "dram" in latencies:
        dram_cycles = latencies["dram"]
        results["dram_latency_cycles"] = round(dram_cycles)

        if actual_freq_mhz:
            # Convert cycles to nanoseconds: ns = cycles / (freq_MHz)
            results["dram_latency_ns"] = round(dram_cycles / actual_freq_mhz * 1000, 2)

    if "l2" in latencies:
        results["l2_latency_cycles"] = round(latencies["l2"])
    if "l1" in latencies:
        results["l1_latency_cycles"] = round(latencies["l1"])

    # Latency ratio: DRAM/L2 — useful for cross-validation
    if "dram" in latencies and "l2" in latencies:
        results["dram_to_l2_latency_ratio"] = round(
            latencies["dram"] / latencies["l2"], 2
        )

    if "l1" in latencies and "l2" in latencies:
        results["l2_to_l1_latency_ratio"] = round(
            latencies["l2"] / latencies["l1"], 2
        )

    results["latency_profile"] = latencies
    results["method"] = "pointer_chasing_with_working_set_sweep"

    # P0: DRAM working set contamination check
    # 128 MB well exceeds all known GPU L2 sizes (H100: 50-60 MB, A100: 40-64 MB,
    # Ada: up to 72 MB). Still flag for LLM-as-a-Judge visibility.
    dram_size_mb = 128  # 32M ints × 4 bytes
    dram_contaminated = False
    if "dram" in latencies and "l2" in latencies:
        dram_l2_ratio = latencies["dram"] / latencies["l2"] if latencies["l2"] > 0 else 0
        if dram_l2_ratio > 0 and dram_l2_ratio < 1.5:
            # DRAM latency suspiciously close to L2 — working set may still fit in L2
            results["_dram_may_be_l2_contaminated"] = True
            dram_contaminated = True

    results["_dram_working_set_mb"] = dram_size_mb

    # Confidence: based on how many hierarchy levels detected and hierarchy validity
    levels_detected = sum(1 for l in ["l1", "l2", "dram"] if l in latencies)
    hierarchy_valid = (
        latencies.get("dram", 0) > latencies.get("l2", 0) > latencies.get("l1", 0)
        if levels_detected == 3 else False
    )
    if hierarchy_valid:
        results["_confidence"] = 0.9
    elif levels_detected >= 2:
        results["_confidence"] = 0.7
    elif levels_detected >= 1:
        results["_confidence"] = 0.5
    else:
        results["_confidence"] = 0.0

    # P0: After hierarchy confidence is set, cap it if contamination detected.
    # Hierarchy check may pass even with contaminated data (e.g. dram=200 vs l2=180),
    # so contamination must be checked AFTER hierarchy to ensure the cap applies.
    if dram_contaminated:
        results["_confidence"] = min(results["_confidence"], 0.4)

    return results if results.get("dram_latency_cycles") else None


def probe_latency_profile(
    sandbox=None,
    sizes: list[tuple[int, str]] | None = None,
) -> dict[str, Any] | None:
    """Full latency profile — runs pointer chasing at many sizes.

    The caller can specify custom sizes or use the default progression
    that sweeps from L1-sized to DRAM-sized working sets.

    Returns the full latency curve for cache capacity cliff detection.
    """
    if sizes is None:
        sizes = [
            (8 * 1024, "8k"),          # 32 KB
            (32 * 1024, "32k"),        # 128 KB
            (128 * 1024, "128k"),      # 512 KB
            (512 * 1024, "512k"),      # 2 MB
            (1024 * 1024, "1m"),       # 4 MB
            (2 * 1024 * 1024, "2m"),   # 8 MB
            (4 * 1024 * 1024, "4m"),   # 16 MB
            (8 * 1024 * 1024, "8m"),   # 32 MB
            (16 * 1024 * 1024, "16m"), # 64 MB
        ]

    curve: dict[str, float] = {}
    for size, label in sizes:
        kernel = pointer_chase_kernel(
            array_size=size,
            iterations=50000,
            output_name=f"latency_{label}",
        )
        result = compile_and_run(kernel.source, sandbox=sandbox)
        if result and result.success:
            parsed = parse_nvcc_output(result.stdout)
            cpa = parsed.get("cycles_per_access", 0)
            if cpa > 0:
                curve[label] = cpa

    return {"latency_curve": curve} if curve else None
