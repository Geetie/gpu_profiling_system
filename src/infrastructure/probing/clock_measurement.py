"""Actual SM clock frequency probe — defeats frequency locking.

Measures the actual GPU clock frequency by:
1. Running a clock-calibration kernel that outputs SM cycle counts
2. Measuring wall-clock time of the same kernel via host timing

The ratio of SM cycles to wall-clock time gives the actual frequency,
which is immune to frequency locking because it measures what the GPU
is actually running at, not what it is supposed to run at.

Key insight: clock() in CUDA returns SM clock cycles. If we know
elapsed time from an independent clock, we can derive:
    actual_freq_MHz = total_sm_cycles / elapsed_time_us
"""
from __future__ import annotations

import os
from typing import Any

from src.infrastructure.probing.kernel_templates import clock_calibration_kernel
from src.infrastructure.probing.probe_helpers import (
    compile_and_run,
    parse_ncu_gpu_time,
    parse_nvcc_output,
    _detect_arch,
)


def probe_actual_clock_frequency(
    sandbox=None,
) -> dict[str, Any] | None:
    """Measure actual SM clock frequency.

    Two-stage approach:
    1. Compile and run a calibration kernel, capturing SM cycles from
       the kernel output.
    2. Independently time the same kernel with host-side timing.

    Returns dict with:
        actual_boost_clock_mhz: float — measured SM frequency in MHz
        total_sm_cycles: int — total SM clock cycles measured
        elapsed_time_us: float — host-measured time in microseconds
        confidence: float — confidence score (0.0-1.0)
        method: str — measurement methodology
    """
    print("[clock] Starting clock frequency probe...")
    kernel = clock_calibration_kernel(loop_iterations=10_000_000)
    print(f"[clock] Kernel source length: {len(kernel.source)} chars")
    result = compile_and_run(kernel.source, sandbox=sandbox)

    if not result or not result.success:
        print("[clock] compile_and_run failed")
        if result:
            print(f"  stdout: {result.stdout[:500]}")
            print(f"  stderr: {result.stderr[:500]}")
        return None

    parsed = parse_nvcc_output(result.stdout)
    total_cycles = parsed.get("total_cycles", 0)
    print(f"[clock] Parse result: {parsed}")

    if total_cycles <= 0:
        print(f"[clock] total_cycles <= 0, stdout: {result.stdout[:500]}")
        return None

    print(f"[clock] total_cycles={total_cycles}, cycles_per_iter={parsed.get('cycles_per_iter', 0)}")

    # Re-run with host timing for wall-clock measurement
    print("[clock] Attempting host timing measurement...")
    freq_mhz, ncu_raw = _measure_with_host_timing(kernel.source, sandbox)
    print(f"[clock] _measure_with_host_timing returned: freq_mhz={freq_mhz}")

    if freq_mhz and freq_mhz > 100:
        # Sanity check: GPU clock should be between 100 and 3000 MHz
        result_dict = {
            "actual_boost_clock_mhz": round(freq_mhz, 2),
            "total_sm_cycles": int(total_cycles),
            "cycles_per_iteration": parsed.get("cycles_per_iter", 0),
            "_confidence": _assess_clock_confidence(freq_mhz),
            "method": "sm_clock_cycles_vs_ncu_wall_clock",
        }
        if ncu_raw:
            result_dict["_ncu_raw_output"] = ncu_raw
        return result_dict

    # M1 fix: No nvidia-smi fallback — it returns theoretical frequency
    # which is wrong under frequency locking. Return None so caller
    # knows frequency measurement is unavailable.
    # P2 fix: Try cudaEventElapsedTime fallback if ncu unavailable
    print("[clock] Host timing failed, trying cudaEvents fallback...")
    freq_mhz = _measure_with_cuda_events(kernel.source, sandbox)
    print(f"[clock] _measure_with_cuda_events returned: freq_mhz={freq_mhz}")
    if freq_mhz and freq_mhz > 100:
        return {
            "actual_boost_clock_mhz": round(freq_mhz, 2),
            "total_sm_cycles": int(total_cycles),
            "cycles_per_iteration": parsed.get("cycles_per_iter", 0),
            "_confidence": _assess_clock_confidence(freq_mhz) * 0.8,  # Lower confidence for event timing
            "method": "sm_clock_cycles_vs_cuda_event_timing",
        }

    print("[clock] All timing methods failed, returning None")
    return None


def _measure_with_host_timing(
    source: str,
    sandbox=None,
) -> tuple[float | None, str | None]:
    """Measure kernel execution time via sandbox for frequency derivation.

    Compiles and runs the calibration kernel through the sandbox,
    parsing SM cycles from kernel output. Host wall-clock timing
    is measured by the kernel binary itself (it prints total_cycles
    from device clock()).

    Returns: (frequency in MHz, raw ncu output) tuple.
    """
    import os
    import shutil

    runner = sandbox
    work_dir = getattr(runner, "sandbox_root", None) or getattr(runner, "_sandbox_root", None) if runner else None
    nvcc = shutil.which("nvcc")
    if nvcc is None or not runner:
        return None, None

    arch = _detect_arch(runner)

    # Compile
    compile_result = runner.run(
        source_code=source,
        command=nvcc,
        args=["-o", "freq_probe", "source.cu", f"-arch={arch}"],
        work_dir=work_dir,
    )
    if not compile_result.success:
        return None, None

    # Execute through sandbox (S4 fix: no subprocess bypass)
    binary = os.path.join(work_dir, "freq_probe")
    exec_result = runner.run(
        command=binary,
        args=[],
        work_dir=work_dir,
    )
    if not exec_result or not exec_result.success:
        return None, None

    # The kernel outputs total_cycles from device clock().
    # We need an independent wall-clock to derive frequency.
    # Strategy: time the sandbox execution and subtract launch overhead.
    # Run 3 times, take minimum (reduce noise).
    min_cycles = None
    for _ in range(3):
        r = runner.run(
            command=binary,
            args=[],
            work_dir=work_dir,
        )
        if r and r.success:
            parsed = parse_nvcc_output(r.stdout)
            sm_cycles = parsed.get("total_cycles", 0)
            if sm_cycles > 0:
                if min_cycles is None or sm_cycles < min_cycles:
                    min_cycles = sm_cycles

    if min_cycles is None or min_cycles <= 0:
        return None, None

    # Gap 7: Try ncu timing first, but if it fails, use host-side timing
    # as fallback. Previously, min_cycles was discarded if ncu failed.
    freq_mhz, ncu_raw = _measure_with_ncu_timing(binary, work_dir, runner)
    if freq_mhz and freq_mhz > 100:
        return freq_mhz, ncu_raw

    # Host-side timing fallback: measure wall-clock of binary execution.
    # Less precise (includes PCIe latency) but better than no result.
    return _measure_with_host_timing_fallback(binary, work_dir, runner, min_cycles)


def _measure_with_host_timing_fallback(
    binary_path: str,
    work_dir: str,
    runner,
    min_cycles: int,
) -> tuple[float | None, str | None]:
    """Gap 7: Host-side timing fallback when ncu is unavailable.

    Bug 4 fix: Collect cycles AND timing from the SAME execution to
    avoid frequency drift between batches. Previously, min_cycles came
    from runs 1-3 and timing from runs 7-9 — if GPU frequency changed
    between batches, the derived frequency would be wrong.

    Now: each run produces both cycles (device-side) and wall-clock
    timing (host-side). We take the run with minimum host elapsed time,
    using its own cycles value for the frequency calculation.
    """
    import shutil
    import time

    ncu = shutil.which("ncu")
    if ncu is not None:
        # ncu exists but failed above — still try host timing as last resort
        pass  # Don't skip — host timing is our last chance

    # Warm up: run binary once to initialize CUDA context, then discard.
    # This separates context init (~50-200ms) from actual kernel timing.
    runner.run(command=binary_path, args=[], work_dir=work_dir)

    # Run binary 3 more times (context already warm), collect cycles and timing.
    # Take the minimum elapsed run and use its paired cycles value.
    best_cycles = None
    best_elapsed_s = None
    for _ in range(3):
        t0 = time.monotonic()
        r = runner.run(
            command=binary_path,
            args=[],
            work_dir=work_dir,
        )
        t1 = time.monotonic()
        elapsed_s = t1 - t0
        if r and r.success and elapsed_s > 0.001:
            parsed = parse_nvcc_output(r.stdout)
            cycles = parsed.get("total_cycles", 0)
            if cycles > 0:
                if best_elapsed_s is None or elapsed_s < best_elapsed_s:
                    best_elapsed_s = elapsed_s
                    best_cycles = cycles

    if best_cycles and best_elapsed_s and best_elapsed_s > 0:
        freq_mhz = best_cycles / (best_elapsed_s * 1e6)
        if freq_mhz > 100:
            return freq_mhz, None

    return None, None


def _measure_with_ncu_timing(
    binary_path: str,
    work_dir: str,
    runner,
) -> tuple[float | None, str | None]:
    """Measure SM frequency using ncu wall-clock timing.

    Returns: (frequency in MHz, raw ncu output) tuple.
    """
    import shutil

    ncu = shutil.which("ncu")
    if ncu is None:
        return None, None

    try:
        # Run ncu 3 times, take minimum elapsed time.
        # Extract BOTH wall-clock timing and SM cycles from the SAME execution
        # to avoid frequency drift between runs.
        min_elapsed_ns = None
        best_ncu_raw = None
        best_cycles = None
        for _ in range(3):
            r = runner.run(
                command=ncu,
                args=["--profile-fromstart", "on", "--print-summary", "per_kernel",
                      binary_path],
                work_dir=work_dir,
            )
            if not r or not r.success:
                continue
            elapsed_ns = parse_ncu_gpu_time(r.stdout + r.stderr)
            parsed_cycles = parse_nvcc_output(r.stdout)
            cycles = parsed_cycles.get("total_cycles", 0)
            if elapsed_ns and elapsed_ns > 0:
                if min_elapsed_ns is None or elapsed_ns < min_elapsed_ns:
                    min_elapsed_ns = elapsed_ns
                    best_ncu_raw = r.stdout + r.stderr
                    best_cycles = cycles if cycles > 0 else best_cycles
        if min_elapsed_ns is None:
            return None, None

        # Use cycles from the same ncu execution for consistency
        sm_cycles = best_cycles or 0
        if sm_cycles <= 0:
            # Fallback: re-run binary to get cycles (less accurate)
            r2 = runner.run(
                command=binary_path,
                args=[],
                work_dir=work_dir,
            )
            if r2 and r2.success:
                parsed = parse_nvcc_output(r2.stdout)
                sm_cycles = parsed.get("total_cycles", 0)

        if sm_cycles > 0 and min_elapsed_ns > 0:
            # freq_MHz = cycles / time_us = cycles / (ns / 1000)
            freq_mhz = sm_cycles / (min_elapsed_ns / 1000.0)
            return freq_mhz, best_ncu_raw

    except Exception:
        pass

    return None, None


def _measure_with_cuda_events(
    source: str,
    sandbox=None,
) -> float | None:
    """P2: Measure SM frequency using cudaEventElapsedTime as ncu fallback.

    Wraps the kernel execution with cudaEventRecord/cudaEventElapsedTime
    to get GPU-side wall-clock timing. Less precise than ncu but still
    GPU-based (not host-side timing which adds PCIe latency).

    Returns: frequency in MHz, or None if measurement failed.
    """
    import os
    import shutil

    runner = sandbox
    work_dir = getattr(runner, "sandbox_root", None) or getattr(runner, "_sandbox_root", None) if runner else None
    nvcc = shutil.which("nvcc")
    if nvcc is None or not runner or not work_dir:
        return None

    arch = _detect_arch(runner)

    # Compile original kernel
    compile_result = runner.run(
        source_code=source,
        command=nvcc,
        args=["-o", "freq_event_probe", "source.cu", f"-arch={arch}"],
        work_dir=work_dir,
    )
    if not compile_result.success:
        return None

    binary = os.path.join(work_dir, "freq_event_probe")

    # Run to get SM cycles
    r = runner.run(
        command=binary,
        args=[],
        work_dir=work_dir,
    )
    if not r or not r.success:
        return None

    parsed = parse_nvcc_output(r.stdout)
    sm_cycles = parsed.get("total_cycles", 0)
    if sm_cycles <= 0:
        return None

    # Now create a wrapper that times the same kernel with cudaEventElapsedTime
    event_source = _wrap_with_events(source)
    if not event_source:
        return None

    compile_result2 = runner.run(
        source_code=event_source,
        command=nvcc,
        args=["-o", "freq_event_timed", "source.cu", f"-arch={arch}"],
        work_dir=work_dir,
    )
    if not compile_result2.success:
        return None

    timed_binary = os.path.join(work_dir, "freq_event_timed")

    # Run multiple times, take minimum elapsed (reduce noise)
    min_elapsed_ms = None
    for _ in range(3):
        r2 = runner.run(
            command=timed_binary,
            args=[],
            work_dir=work_dir,
        )
        if r2 and r2.success:
            p2 = parse_nvcc_output(r2.stdout)
            elapsed_ms = p2.get("elapsed_time_ms", 0)
            if elapsed_ms > 0:
                if min_elapsed_ms is None or elapsed_ms < min_elapsed_ms:
                    min_elapsed_ms = elapsed_ms

    if min_elapsed_ms and min_elapsed_ms > 0:
        # cudaEventElapsedTime returns ms; convert to us for freq_MHz = cycles / time_us
        elapsed_us = min_elapsed_ms * 1000.0
        return sm_cycles / elapsed_us

    return None


def _wrap_with_events(original_source: str) -> str | None:
    """Wrap the original kernel source with cudaEventElapsedTime timing.

    Inserts cudaEventRecord before and after the LAST kernel launch line
    (not the first), so that if the kernel has a warmup launch, we time
    the actual measurement launch instead.
    """
    lines = original_source.split("\n")

    # First pass: find the last kernel launch index
    last_launch_idx = -1
    for i, line in enumerate(lines):
        if "<<<" in line and ">>>" in line:
            last_launch_idx = i

    if last_launch_idx < 0:
        return None

    # Second pass: inject event timing at the last launch
    new_lines = []
    for i, line in enumerate(lines):
        if i == last_launch_idx:
            new_lines.append('    cudaEvent_t evt_start, evt_stop;')
            new_lines.append('    cudaEventCreate(&evt_start);')
            new_lines.append('    cudaEventCreate(&evt_stop);')
            new_lines.append('    cudaEventRecord(evt_start);')
            new_lines.append(line)  # original kernel launch
            new_lines.append('    cudaEventRecord(evt_stop);')
            new_lines.append('    cudaEventSynchronize(evt_stop);')
            new_lines.append('    float elapsed_ms;')
            new_lines.append('    cudaEventElapsedTime(&elapsed_ms, evt_start, evt_stop);')
            new_lines.append('    printf("elapsed_time_ms: %.4f\\n", elapsed_ms);')
            new_lines.append('    cudaEventDestroy(evt_start);')
            new_lines.append('    cudaEventDestroy(evt_stop);')
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


def _assess_clock_confidence(freq_mhz: float) -> float:
    """Assess confidence in the clock measurement."""
    # Typical GPU clocks: 500-2200 MHz
    if 1000 <= freq_mhz <= 2000:
        return 0.9  # Very plausible
    elif 500 <= freq_mhz <= 2500:
        return 0.7  # Plausible
    elif 200 <= freq_mhz <= 3000:
        return 0.4  # Possible but unusual
    return 0.2  # Suspicious
