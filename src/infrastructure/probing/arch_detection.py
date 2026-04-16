"""GPU architecture detection utilities.

Provides unified architecture detection logic used by:
- CodeGen agent for correct nvcc -arch flag
- Probe helpers for compilation

Detection methods (in order of reliability):
1. cudaDeviceGetAttribute (most reliable, works on actual GPU)
2. nvidia-smi parsing (fallback for remote/headless)
3. Compilation testing (last resort)
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.infrastructure.sandbox import SandboxRunner, SandboxResult


def detect_gpu_arch(
    runner: "SandboxRunner | None" = None,
    prefer_method: str = "auto",
) -> str:
    """Detect GPU compute capability for correct nvcc -arch flag.

    Args:
        runner: SandboxRunner for compilation/execution tests.
        prefer_method: "auto", "cuda_api", "nvidia_smi", or "compilation".

    Returns:
        Architecture string like 'sm_60', 'sm_80', etc.
    """
    if prefer_method == "cuda_api" or prefer_method == "auto":
        arch = _detect_arch_via_cuda_api(runner)
        if arch:
            return arch

    if prefer_method == "nvidia_smi" or prefer_method == "auto":
        arch = _detect_arch_via_nvidia_smi()
        if arch:
            return arch

    if prefer_method == "compilation" or prefer_method == "auto":
        if runner:
            arch = _detect_arch_by_compilation(runner)
            if arch:
                return arch

    return "sm_50"


def _detect_arch_via_cuda_api(runner: "SandboxRunner | None") -> str | None:
    """Detect architecture using cudaDeviceGetAttribute.

    Most reliable method when running on actual GPU hardware.
    """
    if not runner:
        return None

    nvcc = shutil.which("nvcc")
    if not nvcc:
        return None

    work_dir = getattr(runner, "sandbox_root", None) or getattr(runner, "_sandbox_root", None)
    if not work_dir:
        return None

    source = """
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int device;
    if (cudaGetDevice(&device) != cudaSuccess) {
        printf("error: no device\\n");
        return 1;
    }

    int major, minor;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) {
        printf("error: no attribute\\n");
        return 1;
    }
    if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
        printf("error: no attribute\\n");
        return 1;
    }

    printf("sm_%d%d\\n", major, minor);
    return 0;
}
"""
    result = runner.run(
        source_code=source,
        command=nvcc,
        args=["-o", "detect_arch", "source.cu"],
        work_dir=work_dir,
    )

    if not result or not result.success:
        return None

    binary = os.path.join(work_dir, "detect_arch")
    run_result = runner.run(command=binary, args=[], work_dir=work_dir)

    if not run_result or not run_result.success:
        return None

    output = run_result.stdout.strip()
    if output.startswith("sm_"):
        print(f"[detect_gpu_arch] Detected via CUDA API: {output}")
        return output

    return None


def _detect_arch_via_nvidia_smi() -> str | None:
    """Detect architecture by parsing nvidia-smi GPU name.

    Fallback for remote/headless environments.
    """
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return None

        gpu_name = r.stdout.strip().lower()

        gpu_to_cc = {
            "a100": "sm_80",
            "a800": "sm_80",
            "a10": "sm_86",
            "a30": "sm_80",
            "a40": "sm_86",
            "h100": "sm_90",
            "h800": "sm_90",
            "h200": "sm_90",
            "v100": "sm_70",
            "t4": "sm_75",
            "l4": "sm_89",
            "l40": "sm_90",
            "p100": "sm_60",
            "p40": "sm_61",
            "p4": "sm_61",
            "k80": "sm_37",
            "k40": "sm_35",
            "rtx 20": "sm_75",
            "rtx 30": "sm_86",
            "rtx 40": "sm_89",
            "rtx 50": "sm_120",
            "gtx 10": "sm_61",
            "gtx 16": "sm_75",
            "quadro rtx": "sm_75",
            "tesla": "sm_75",
        }

        for pattern, cc in gpu_to_cc.items():
            if pattern in gpu_name:
                print(f"[detect_gpu_arch] Detected via nvidia-smi: {cc} ({r.stdout.strip()})")
                return cc

    except Exception as e:
        print(f"[detect_gpu_arch] nvidia-smi failed: {e}")

    return None


def _detect_arch_by_compilation(runner: "SandboxRunner") -> str | None:
    """Fallback architecture detection via compilation testing.

    Tests compilation with different architectures to find one that works.
    """
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return None

    test_source = """
#include <cuda_runtime.h>
__global__ void test_kernel() {
    volatile unsigned long long c = clock64();
    (void)c;
}
int main() {
    test_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
"""
    work_dir = getattr(runner, "sandbox_root", None) or getattr(runner, "_sandbox_root", None)
    if not work_dir:
        return None

    arch_list = ["sm_90", "sm_89", "sm_86", "sm_80", "sm_75", "sm_70", "sm_61", "sm_60", "sm_50"]

    for arch in arch_list:
        result = runner.run(
            source_code=test_source,
            command=nvcc,
            args=["-o", "arch_test", "source.cu", f"-arch={arch}"],
            work_dir=work_dir,
        )
        if not result or not result.success:
            continue

        binary = os.path.join(work_dir, "arch_test")
        run_result = runner.run(command=binary, args=[], work_dir=work_dir)
        if run_result and run_result.success:
            print(f"[detect_gpu_arch] Detected via compilation: {arch}")
            return arch

    return None


def get_arch_fallback() -> str:
    """Return the default fallback architecture.

    sm_50 is the most widely compatible, supporting GPUs from Kepler onward.
    """
    return "sm_50"
