"""Helper utilities for hardware probing.

Provides compile_and_run() — a reusable function that compiles CUDA
source via nvcc and executes the binary, returning stdout/stderr.
"""
from __future__ import annotations

import os
import shutil
from typing import Any

from src.infrastructure.sandbox import LocalSandbox, SandboxConfig, SandboxRunner, SandboxResult


def compile_and_run(
    source: str,
    sandbox: SandboxRunner | None = None,
    nvcc_flags: list[str] | None = None,
    timeout: int = 60,
) -> SandboxResult | None:
    """Compile CUDA source and execute the binary.

    Args:
        source: CUDA source code.
        sandbox: SandboxRunner to use. If None, creates a LocalSandbox.
        nvcc_flags: Extra flags for nvcc (e.g. ["-arch=sm_80"]).
        timeout: Execution timeout in seconds.

    Returns:
        SandboxResult from binary execution, or None if compilation failed.
    """
    runner = sandbox or LocalSandbox(SandboxConfig())
    work_dir = getattr(runner, "sandbox_root", None) or getattr(runner, "_sandbox_root", None)

    # Find nvcc
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        print(f"[compile_and_run] nvcc not found")
        return None

    # Determine GPU architecture
    arch = _detect_arch(runner)
    flags = [f"-arch={arch}"] + (nvcc_flags or [])

    # Sanitize flags
    safe_flags = _sanitize_flags(flags)

    # Compile
    compile_args = ["-o", "probe_binary", "source.cu"] + safe_flags
    compile_result = runner.run(
        source_code=source,
        command=nvcc,
        args=compile_args,
        work_dir=work_dir,
    )

    if not compile_result.success:
        print(f"[compile_and_run] compile failed with arch={arch}")
        print(f"  stderr: {compile_result.stderr[:300]}")
        return None

    # Execute
    binary_path = os.path.join(work_dir or ".", "probe_binary")
    exec_result = _run_binary(binary_path, runner, work_dir, timeout)
    if not exec_result or not exec_result.success:
        print(f"[compile_and_run] binary execution failed: {binary_path}")
        if exec_result:
            print(f"  stderr: {exec_result.stderr[:300]}")
    return exec_result


def _detect_arch(runner: SandboxRunner) -> str:
    """Detect GPU compute capability via direct CUDA API query.

    Uses cudaDeviceGetAttribute to get the actual hardware compute
    capability, which is more reliable than compilation+run testing.
    Compilation testing can mislead because nvcc can compile for any
    arch and the driver JIT-compiles PTX at runtime — but clock()
    and other timing functions may behave incorrectly under JIT.

    Falls back to nvidia-smi parsing, then compilation testing.
    """
    import subprocess as _sub

    # Method 1: Direct CUDA API query — most reliable
    work_dir = getattr(runner, "sandbox_root", None) or getattr(runner, "_sandbox_root", None)
    nvcc = shutil.which("nvcc")

    if nvcc and work_dir:
        cc_source = """
#include <stdio.h>
#include <cuda_runtime.h>
int main() {
    int dev;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) { printf("no_device\\n"); return 1; }
    int major, minor;
    err = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    if (err != cudaSuccess) { printf("no_major\\n"); return 1; }
    err = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    if (err != cudaSuccess) { printf("no_minor\\n"); return 1; }
    printf("sm_%d%d\\n", major, minor);
    return 0;
}
"""
        cc_result = runner.run(
            source_code=cc_source,
            command=nvcc,
            args=["-o", "cc_detect", "source.cu"],
            work_dir=work_dir,
        )
        if cc_result and cc_result.success:
            cc_binary = os.path.join(work_dir, "cc_detect")
            run_result = runner.run(command=cc_binary, args=[], work_dir=work_dir)
            if run_result and run_result.success:
                # cc_detect outputs "sm_75" without colon — parse directly, not via parse_nvcc_output
                cc_line = run_result.stdout.strip().splitlines()[-1] if run_result.stdout.strip() else ""
                if cc_line.startswith("sm_"):
                    cc_val = cc_line.strip()
                    print(f"[_detect_arch] cc={cc_val} (from cudaDeviceGetAttribute)")
                    return cc_val

    # Method 2: Parse nvidia-smi output (via sandbox runner if available)
    try:
        if runner is not None:
            # Use sandbox runner to respect container environment
            r = runner.run(
                command="nvidia-smi",
                args=["--query-gpu=name", "--format=csv,noheader"],
                work_dir=getattr(runner, "sandbox_root", None) or os.getcwd(),
            )
            if r and r.success and r.stdout.strip():
                gpu_name = r.stdout.strip().lower()
                gpu_to_cc = {
                    "a100": "sm_80", "a800": "sm_80",
                    "h100": "sm_90", "h800": "sm_90",
                    "v100": "sm_70",
                    "t4": "sm_75",
                    "p100": "sm_60", "p40": "sm_61", "p4": "sm_61",
                    "k80": "sm_37", "k40": "sm_35",
                    "rtx 20": "sm_75", "rtx 30": "sm_86",
                    "rtx 40": "sm_89", "rtx 50": "sm_120",
                }
                for pattern, cc in gpu_to_cc.items():
                    if pattern in gpu_name:
                        print(f"[_detect_arch] cc={cc} (from nvidia-smi: {r.stdout.strip()})")
                        return cc
        else:
            # No sandbox runner — use direct subprocess as fallback
            r = _sub.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                gpu_name = r.stdout.strip().lower()
                gpu_to_cc = {
                    "a100": "sm_80", "a800": "sm_80",
                    "h100": "sm_90", "h800": "sm_90",
                    "v100": "sm_70",
                    "t4": "sm_75",
                    "p100": "sm_60", "p40": "sm_61", "p4": "sm_61",
                    "k80": "sm_37", "k40": "sm_35",
                    "rtx 20": "sm_75", "rtx 30": "sm_86",
                    "rtx 40": "sm_89", "rtx 50": "sm_120",
                }
                for pattern, cc in gpu_to_cc.items():
                    if pattern in gpu_name:
                        print(f"[_detect_arch] cc={cc} (from nvidia-smi: {r.stdout.strip()})")
                        return cc
    except Exception:
        pass

    # Method 3: Fallback — compilation + run testing with clock() kernel
    return _detect_arch_by_compilation(runner)


def _detect_arch_by_compilation(runner: SandboxRunner) -> str:
    """Fallback architecture detection via compilation testing.

    Uses a test kernel with clock() to ensure timing functions work
    correctly under the target architecture.
    """
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return "sm_50"

    test_source = """
#include <cuda_runtime.h>
__global__ void test_kernel() {
    volatile unsigned long long c = clock();
    (void)c;
}
int main() {
    test_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
"""
    work_dir = getattr(runner, "sandbox_root", None) or getattr(runner, "_sandbox_root", None)

    arch_list = ["sm_80", "sm_75", "sm_70", "sm_61", "sm_60", "sm_50"]
    for arch in arch_list:
        result = runner.run(
            source_code=test_source,
            command=nvcc,
            args=["-o", "arch_test", "source.cu", f"-arch={arch}"],
            work_dir=work_dir,
        )
        if result and result.success:
            binary = os.path.join(work_dir, "arch_test")
            run_result = runner.run(command=binary, args=[], work_dir=work_dir)
            if run_result and run_result.success:
                print(f"[_detect_arch] arch={arch} (compiled and ran)")
                return arch
            else:
                print(f"[_detect_arch] arch={arch} compiled but failed to run")

    print(f"[_detect_arch] nothing compiled, falling back to sm_50")
    return "sm_50"


def _sanitize_flags(flags: list[str]) -> list[str]:
    """Remove unsafe characters from compiler flags."""
    safe = set("-_./+=:,")
    result = []
    for f in flags:
        if all(c.isalnum() or c in safe for c in f):
            result.append(f)
    return result


def _run_binary(
    binary_path: str,
    runner: SandboxRunner,
    work_dir: str | None,
    timeout: int = 60,
) -> SandboxResult | None:
    """Execute a compiled binary through the sandbox.

    Never bypasses sandbox — if work_dir is None, derives it from runner.
    """
    # Ensure work_dir is valid — derive from sandbox if not set
    if not work_dir:
        work_dir = getattr(runner, "sandbox_root", None) or getattr(runner, "_sandbox_root", None)
    if not work_dir:
        # No sandbox root available — cannot execute safely
        return SandboxResult(
            stdout="",
            stderr="No sandbox work directory available",
            return_code=-1,
            success=False,
        )

    try:
        result = runner.run(
            command=binary_path,
            args=[],
            work_dir=work_dir,
        )
        return result
    except Exception:
        return None


def parse_nvcc_output(stdout: str) -> dict[str, Any]:
    """Parse key: value output from probe binaries."""
    result: dict[str, Any] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "---" in line:
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            try:
                if "." in value:
                    result[key] = float(value)
                else:
                    result[key] = int(value)
            except ValueError:
                result[key] = value
    return result


def _assess_confidence(
    value: float,
    low: float,
    high: float,
    ideal_low: float | None = None,
    ideal_high: float | None = None,
) -> float:
    """Assess confidence that a measurement value is plausible.

    Args:
        value: The measured value.
        low: Minimum plausible value.
        high: Maximum plausible value.
        ideal_low: Lower bound of ideal range (optional).
        ideal_high: Upper bound of ideal range (optional).

    Returns:
        Confidence score 0.0-1.0.
    """
    if value <= 0:
        return 0.0
    if ideal_low is not None and ideal_high is not None:
        if ideal_low <= value <= ideal_high:
            return 0.9
    if low <= value <= high:
        return 0.7
    # Outside plausible range but not zero
    return 0.2


def _assess_from_ratio(ratio: float, ideal: float, tolerance: float) -> float:
    """Assess confidence from how close a ratio is to an ideal value.

    Args:
        ratio: Measured ratio.
        ideal: Ideal expected ratio.
        tolerance: Acceptable fractional deviation.

    Returns:
        Confidence score 0.0-1.0.
    """
    if ratio <= 0 or ideal <= 0:
        return 0.0
    deviation = abs(ratio - ideal) / ideal
    if deviation <= tolerance:
        return 0.9
    elif deviation <= tolerance * 2:
        return 0.7
    elif deviation <= tolerance * 4:
        return 0.4
    return 0.2


def parse_ncu_gpu_time(output: str) -> float | None:
    """Parse ncu output for GPU kernel execution time in nanoseconds.

    Unified function used by bandwidth, shmem_bandwidth, and clock_measurement
    probes. Handles multiple ncu output format variants.
    """
    import re
    patterns = [
        r"gpu_time_ns[^\d]*([\d,.]+)",
        r"Duration\s*\(ns\)\s*:\s*([\d,.]+)",
        r"gpu__time_duration[^\d]*([\d,.]+)",
        r"Elapsed\s+Cycles[^\d]*([\d,.]+)",
        r"([\d,.]+)\s*ns\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, output):
            try:
                val = float(match.group(1).replace(",", ""))
                if val > 0:
                    return val
            except (ValueError, IndexError):
                continue
    return None
