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
        print(f"[bank_conflict] compile_and_run failed")
        if result:
            print(f"  stdout: {result.stdout[:500]}")
            print(f"  stderr: {result.stderr[:500]}")
        return None

    parsed = parse_nvcc_output(result.stdout)
    print(f"[bank_conflict] parsed: {parsed}")

    results: dict[str, Any] = {
        "method": "strided_vs_sequential_shmem_comparison",
    }

    strided = parsed.get("strided_cycles", 0)
    sequential = parsed.get("sequential_cycles", 0)
    ratio = parsed.get("bank_conflict_ratio", 0)

    if strided:
        results["strided_cycles"] = int(strided)
    if sequential:
        results["sequential_cycles"] = int(sequential)
    if ratio:
        results["bank_conflict_ratio"] = float(ratio)
    if "stride" in parsed:
        results["stride"] = int(parsed["stride"])

    # If ratio is 0 but we have cycle data, compute it ourselves
    if not results.get("bank_conflict_ratio") and strided and sequential:
        results["bank_conflict_ratio"] = round(float(strided) / float(sequential), 2)
        print(f"[bank_conflict] computed ratio from cycles: {results['bank_conflict_ratio']}")

    # Confidence: bank conflict ratio should be > 1.0 and typically < 32×
    # T4 32-way bank conflict theoretical max ~25-30×
    bc_ratio = results.get("bank_conflict_ratio")
    if bc_ratio and bc_ratio > 1.0:
        results["_confidence"] = round(
            _assess_from_ratio(bc_ratio, ideal=16.0, tolerance=0.6), 2
        )
    elif bc_ratio and bc_ratio > 0:
        results["_confidence"] = 0.2

    if not results.get("bank_conflict_ratio"):
        print(f"[bank_conflict] no ratio found, returning None")
    return results if results.get("bank_conflict_ratio") else None
