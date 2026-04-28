from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build"
BENCH_DIR = ROOT / "benchmarks"


def run_cmd(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
    )


def compile_benchmark(source: Path, output: Path) -> None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc not found in PATH. Please install CUDA toolkit or add nvcc to PATH.")

    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[compile] Compiling {source.name}...")
    cmd = [
        nvcc,
        str(source),
        "-O3",
        "-arch=native",
        "-std=c++17",
        "-lineinfo",
        "-o",
        str(output),
    ]
    result = run_cmd(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Compilation failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    print(f"[compile] Successfully compiled to {output.name}")


def run_binary(binary: Path, program_args: list[str]) -> str:
    print(f"[run] Executing {binary.name}...")
    cmd = [str(binary), *program_args]
    result = run_cmd(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Program failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    output = result.stdout.strip()
    print(f"[run] Output: {output}")
    return result.stdout


def profile_with_ncu(binary: Path, program_args: list[str], output_base: Path, metrics: str | None = None) -> str:
    ncu = shutil.which("ncu")
    if ncu is None:
        raise RuntimeError("ncu not found in PATH. Please install Nsight Compute or add it to PATH.")

    if metrics is None:
        metrics = ",".join([
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__pipe_tensor_op_hmma_cycle_active.avg.pct_of_peak_sustained_active",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "l2__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "sm__maximum_warps_per_active_cycle_pct",
        ])

    print(f"[ncu] Profiling with metrics: {metrics}")
    cmd = [
        ncu,
        "-f",
        "--target-processes", "all",
        "--metrics", metrics,
        "--csv",
        str(binary),
        *program_args,
    ]
    
    start_time = time.time()
    result = run_cmd(cmd)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        raise RuntimeError(f"NCU profiling failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

    print(f"[ncu] Profiling completed in {elapsed:.1f}s")
    output = "=== NCU CSV REPORT ===\n"
    output += result.stdout
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="gemm")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--metrics", default=None)
    parser.add_argument("program_args", nargs="*")
    args = parser.parse_args()

    source = BENCH_DIR / f"{args.benchmark}.cu"
    if not source.exists():
        print(f"[error] Benchmark source not found: {source}")
        return

    binary = BUILD_DIR / args.benchmark

    compile_benchmark(source, binary)
    stdout = run_binary(binary, args.program_args)

    print("=== PROGRAM OUTPUT ===")
    print(stdout.strip())

    if args.profile:
        profile_dir = BUILD_DIR / "profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        output_base = profile_dir / args.benchmark
        profile_output = profile_with_ncu(binary, args.program_args, output_base, args.metrics)

        print("\n=== NCU OUTPUT ===")
        print(profile_output.strip())


if __name__ == "__main__":
    main()
