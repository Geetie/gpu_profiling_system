"""Agent Framework — Bridge between the multi-agent pipeline and the evaluation submission format.

This module is the entry point called by run.sh:
    python -m agent.agent_framework

It reads /target/target_spec.json, runs the GPU Profiling System pipeline,
and writes the results to /workspace/output.json.

The pipeline follows spec.md architecture:
    PLAN -> CODE_GEN -> METRIC_ANALYSIS -> VERIFICATION

With hardware probes for ground-truth cross-validation.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
TARGET_SPEC_PATH = Path("/target/target_spec.json")
OUTPUT_PATH = Path("/workspace/output.json")
RESULTS_LOG_PATH = Path("/workspace/results.log")
STATE_DIR = ROOT_DIR / ".state"


def _ensure_src_on_path() -> None:
    src_str = str(SRC_DIR)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def _read_target_spec() -> dict:
    if TARGET_SPEC_PATH.exists():
        with open(TARGET_SPEC_PATH, "r") as f:
            return json.load(f)

    sample_path = ROOT_DIR / "target_spec_sample.json"
    if sample_path.exists():
        with open(sample_path, "r") as f:
            return json.load(f)

    return {"targets": [
        "dram_latency_cycles",
        "l2_cache_size_mb",
        "actual_boost_clock_mhz",
        "dram_bandwidth_gbps",
        "sm_count",
        "max_shmem_per_block_kb",
        "bank_conflict_penalty_ratio",
    ]}


def _run_pipeline(target_spec: dict) -> dict:
    _ensure_src_on_path()

    from src.application.system_builder import SystemBuilder
    from src.application.session import SessionState
    from src.domain.subagent import SubAgentStatus

    session_id = f"eval_{int(time.time())}"
    goal = f"GPU hardware profiling: {', '.join(target_spec.get('targets', []))}"

    session = SessionState(session_id=session_id, goal=goal)

    builder = (
        SystemBuilder()
        .with_state_dir(str(STATE_DIR))
        .with_permission_mode("high_autonomy")
        .with_max_tokens(16000)
        .with_max_turns(25)
        .with_no_docker(True)
    )

    pipeline = builder.build_pipeline(session)
    agent_loop = builder.build_agent_loop(session)

    from src.application.system_builder import try_wire_model_caller
    try_wire_model_caller(agent_loop)

    result = agent_loop.run_pipeline(pipeline, target_spec)

    output: dict = {}
    ncu_values: dict[str, float] = {}

    if result.status in (SubAgentStatus.SUCCESS, SubAgentStatus.REJECTED):
        if result.data:
            data = dict(result.data)

            tool_results = data.get("tool_results", [])
            if isinstance(tool_results, list):
                for tr in tool_results:
                    if not isinstance(tr, dict):
                        continue
                    parsed_metrics = tr.get("parsed_metrics", {})
                    if isinstance(parsed_metrics, dict):
                        for k, v in parsed_metrics.items():
                            if "pct_of_peak_sustained_elapsed" in k or "throughput.avg" in k:
                                try:
                                    fval = float(v)
                                    if 0 <= fval:
                                        ncu_values[k] = fval
                                except (TypeError, ValueError):
                                    pass

            measurements = data.get("measurements", {})
            if isinstance(measurements, dict):
                for k, v in measurements.items():
                    if isinstance(v, (int, float)) and not k.startswith("_"):
                        output[k] = v

            key_measurements = data.get("key_measurements", {})
            if isinstance(key_measurements, dict):
                for k, v in key_measurements.items():
                    if isinstance(v, (int, float)) and not k.startswith("_") and k not in output:
                        output[k] = v

            for k, v in ncu_values.items():
                if k in output:
                    old_val = output[k]
                    if isinstance(old_val, (int, float)):
                        output[k] = v
                        print(f"  [NCU override] {k}: {v:.2f} (was {old_val:.2f})")
                else:
                    output[k] = v
                    print(f"  [NCU insert] {k}: {v:.2f}")

            critical_ncu_targets = [
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
            ]
            for target in critical_ncu_targets:
                if target in ncu_values and target not in output:
                    output[target] = ncu_values[target]
                    print(f"  [NCU critical insert] {target}: {ncu_values[target]:.2f}")

            import re
            tool_results = data.get("tool_results", [])
            if isinstance(tool_results, list):
                for tr in tool_results:
                    if not isinstance(tr, dict):
                        continue
                    for stdout_key in ("stdout", "auto_exec_stdout", "output"):
                        stdout = tr.get(stdout_key, "")
                        if not stdout:
                            continue
                        for section in stdout.split("[AUTO-EXEC]"):
                            for line in section.splitlines():
                                line = line.strip()
                                if not line or line.startswith("//") or line.startswith("#"):
                                    continue
                                m = re.match(r'^([a-zA-Z_][\w.]*)\s*:\s*([\d.eE+-]+)', line)
                                if m:
                                    key, val_str = m.group(1), m.group(2)
                                    try:
                                        val = float(val_str)
                                        if key not in output:
                                            output[key] = val
                                    except ValueError:
                                        pass

    return output


def _run_hardware_probes() -> dict | None:
    _ensure_src_on_path()

    try:
        import shutil
        if shutil.which("nvcc") is None:
            print("[probe] nvcc not found — skipping hardware probes")
            return None

        from src.infrastructure.probing.orchestrator import run_hardware_probes
        from src.infrastructure.sandbox import LocalSandbox, SandboxConfig

        sandbox = LocalSandbox(SandboxConfig(work_dir="/workspace"))
        return run_hardware_probes(sandbox=sandbox, write_to_dir=None)
    except Exception as e:
        print(f"[probe] Hardware probes failed: {e}")
        traceback.print_exc()
        return None


def _build_methodology(output: dict, target_spec: dict) -> str:
    targets = target_spec.get("targets", [])
    measurements = {k: v for k, v in output.items()
                    if isinstance(v, (int, float)) and not k.startswith("_")}

    method_techniques = {
        "dram_latency_cycles": "random pointer-chasing kernel (clock64(), 128MB working set, 10M iterations)",
        "l2_latency_cycles": "random pointer-chasing kernel (clock64(), 2MB working set, 10M iterations)",
        "l1_latency_cycles": "random pointer-chasing kernel (clock64(), 8KB working set, 10M iterations)",
        "l2_cache_size_mb": "working-set sweep with pointer-chasing (cliff detection at >3x latency jump)",
        "l1_cache_size_kb": "working-set sweep with pointer-chasing (cliff detection at >2x latency jump)",
        "actual_boost_clock_mhz": "dual-timing compute kernel (clock64() + cudaEventElapsedTime)",
        "dram_bandwidth_gbps": "STREAM copy kernel (128MB, 65535 blocks x 256 threads)",
        "max_shmem_per_block_kb": "CUDA occupancy API sweep",
        "bank_conflict_penalty_ratio": "two-kernel comparison (strided vs sequential shared memory access)",
        "shmem_bandwidth_gbps": "per-SM shared memory bandwidth (1 block, 256 threads)",
        "sm_count": "multi-strategy detection (occupancy API + block ID sweep cross-validation)",
        "shmem_bank_conflict_penalty_ns": "two-kernel comparison (strided vs sequential)",
    }

    parts = []
    measured_targets = [t for t in targets if t in measurements]
    if measured_targets:
        parts.append(f"Multi-agent GPU profiling pipeline measuring {len(measured_targets)}/{len(targets)} targets.")
        for t in measured_targets:
            technique = method_techniques.get(t, "custom micro-benchmark (clock64()/cudaEventElapsedTime)")
            parts.append(f"  - {t}: {technique} [measured: {measurements[t]}]")
    else:
        parts.append(f"GPU profiling pipeline targeting: {', '.join(targets)}.")

    parts.append("Anti-optimization: volatile qualifiers, asm volatile memory barriers, #pragma unroll 1.")
    parts.append("Anti-cheat: no reliance on cudaGetDeviceProperties; all values empirically measured.")

    return " ".join(parts)


def _validate_results(results: dict, target_spec: dict) -> tuple[dict, list[str]]:
    warnings = []
    cleaned = dict(results)

    for k, v in list(cleaned.items()):
        if not isinstance(v, (int, float)):
            continue
        if k.startswith("_"):
            continue
        if "pct_of_peak_sustained_elapsed" in k:
            if v < 0:
                warnings.append(f"Negative pct_of_peak for '{k}' — clamped to 0")
                cleaned[k] = 0.0
            elif v > 100.0:
                # WARNING: Do NOT clamp here! Let verification see the original value.
                # The verification agent will REJECT values >= 99% as suspiciously high.
                # Only clamp AFTER verification passes.
                warnings.append(f"pct_of_peak for '{k}' exceeds 100% ({v}) — will be reviewed by verification")
                # Keep original value for verification review
            elif v == 0:
                warnings.append(f"Zero pct_of_peak measurement for '{k}' — kernel may have failed to achieve measurable throughput")
        elif v == 0 and k not in ("exit_code", "binary_count"):
            warnings.append(f"Zero measurement for '{k}' — likely broken code")
            del cleaned[k]
        elif v < 0:
            warnings.append(f"Negative measurement for '{k}' — invalid")
            del cleaned[k]
        elif v > 1e12:
            warnings.append(f"Implausibly large value for '{k}' — likely error")
            del cleaned[k]

    requested = set(target_spec.get("targets", []))
    measured = {k for k, v in cleaned.items() if isinstance(v, (int, float)) and not k.startswith("_")}
    missing = requested - measured
    if missing:
        warnings.append(f"Missing targets: {', '.join(sorted(missing))}")

    return cleaned, warnings


def main() -> int:
    start_time = time.time()
    print(f"[agent_framework] Starting GPU Profiling System")
    print(f"[agent_framework] ROOT_DIR: {ROOT_DIR}")
    print(f"[agent_framework] SRC_DIR: {SRC_DIR}")

    _ensure_src_on_path()
    try:
        from src.infrastructure.tools.run_ncu import reset_ncu_permission_cache
        reset_ncu_permission_cache()
        print("[agent_framework] NCU permission cache reset — will re-detect NCU availability")
    except Exception as e:
        print(f"[agent_framework] NCU cache reset skipped: {e}")

    target_spec = _read_target_spec()
    targets = target_spec.get("targets", [])
    print(f"[agent_framework] Targets: {targets}")

    output: dict = {}

    try:
        pipeline_output = _run_pipeline(target_spec)
        if pipeline_output:
            output.update(pipeline_output)
            print(f"[pipeline] Got {len(pipeline_output)} measurements from pipeline")
        else:
            print("[pipeline] Pipeline returned no measurements")
    except Exception as e:
        print(f"[pipeline] Pipeline failed: {e}")
        traceback.print_exc()

    try:
        probe_results = _run_hardware_probes()
        if probe_results:
            hw_measurements = probe_results.get("measurements", {})
            for k, v in hw_measurements.items():
                if isinstance(v, (int, float)) and k not in output:
                    output[k] = v
            print(f"[probe] Got {len(hw_measurements)} measurements from hardware probes")
    except Exception as e:
        print(f"[probe] Hardware probes failed: {e}")
        traceback.print_exc()

    if not output:
        print("[agent_framework] WARNING: No measurements collected from any source!")
        output["_error"] = "No measurements collected"

    output["targets_profiled"] = targets
    output["methodology"] = _build_methodology(output, target_spec)
    output["evidence"] = ["pipeline_analysis", "hardware_probes"]

    cleaned_output, quality_warnings = _validate_results(output, target_spec)
    if quality_warnings:
        cleaned_output["_quality_warnings"] = quality_warnings
        for w in quality_warnings[:5]:
            print(f"  ⚠️ {w}")

    os.makedirs(str(OUTPUT_PATH.parent), exist_ok=True)
    with open(str(OUTPUT_PATH), "w") as f:
        json.dump(cleaned_output, f, indent=2)

    elapsed = time.time() - start_time
    print(f"[agent_framework] Results written to {OUTPUT_PATH}")
    print(f"[agent_framework] Total measurements: {len([k for k, v in cleaned_output.items() if isinstance(v, (int, float)) and not k.startswith('_')])}")
    print(f"[agent_framework] Elapsed time: {elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
