"""Shared memory bank conflict probe.

Measures the latency penalty of shared memory bank conflicts by
comparing strided access (conflicts) vs sequential access (no conflicts)
within the same kernel.

The ratio of strided_cycles / sequential_cycles gives the bank
conflict penalty factor — typically 16x-32x for 32-bank GPUs
when all 32 threads in a warp hit different banks.
"""
from __future__ import annotations

from typing import Any

from src.infrastructure.probing.kernel_templates import bank_conflict_kernel
from src.infrastructure.probing.probe_helpers import (
    compile_and_run,
    parse_nvcc_output,
    _assess_from_ratio,
)


def probe_bank_conflict_latency(
    sandbox=None,
) -> dict[str, Any] | None:
    """Measure bank conflict latency penalty.

    Runs a kernel with two access patterns in the same execution:
    1. Strided (bank-conflicting): thread t accesses t * 32
    2. Sequential (conflict-free): thread t accesses t + offset

    The ratio reveals the bank conflict cost multiplier.

    Returns dict with:
        bank_conflict_ratio: float — strided/sequential cycle ratio
        strided_cycles: int — cycles for strided access
        sequential_cycles: int — cycles for sequential access
        method: str — methodology
    """
    kernel = bank_conflict_kernel(size=32768)
    result = compile_and_run(kernel.source, sandbox=sandbox)

    if not result or not result.success:
        return None

    parsed = parse_nvcc_output(result.stdout)

    results: dict[str, Any] = {
        "method": "strided_vs_sequential_shmem_comparison",
    }

    if "strided_cycles" in parsed:
        results["strided_cycles"] = int(parsed["strided_cycles"])
    if "sequential_cycles" in parsed:
        results["sequential_cycles"] = int(parsed["sequential_cycles"])
    if "bank_conflict_ratio" in parsed:
        results["bank_conflict_ratio"] = float(parsed["bank_conflict_ratio"])
    if "stride" in parsed:
        results["stride"] = int(parsed["stride"])

    # Confidence: bank conflict ratio should be > 1.0 and typically < 32×
    # T4 32-way bank conflict theoretical max ~25-30×,实测约 15-20×
    bc_ratio = results.get("bank_conflict_ratio")
    if bc_ratio and bc_ratio > 1.0:
        results["_confidence"] = round(
            _assess_from_ratio(bc_ratio, ideal=16.0, tolerance=0.6), 2
        )
    else:
        results["_confidence"] = 0.2

    return results if results.get("bank_conflict_ratio") else None
