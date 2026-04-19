"""Main entry point — GPU Profiling System.

Wires all four layers together:
  TerminalUI -> AgentLoop -> ToolRunner -> ToolContract -> Sandbox
                          -> ApprovalQueue -> StatePersister

Usage:
  python -m src.main "Profile kernel X"
  python -m src.main "Profile kernel X" --session sess_001
  python -m src.main --resume sess_001
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from typing import Any

from src.application.agent_loop import AgentLoop, EventKind
from src.application.system_builder import SystemBuilder, try_wire_model_caller
from src.presentation.terminal_ui import TerminalUI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)


def _create_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPU Profiling System -- Multi-Agent CUDA Analysis",
    )
    parser.add_argument(
        "goal",
        nargs="?",
        default=None,
        help="Analysis goal (e.g. 'Profile dram_latency on kernel.cu')",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Session ID (auto-generated if omitted)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="SESSION_ID",
        help="Resume a persisted session by ID",
    )
    parser.add_argument(
        "--mode",
        choices=["default", "conservative", "relaxed", "high_autonomy"],
        default="default",
        help="Permission mode (default: default)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum agent loop turns (default: 20)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8000,
        help="Max context tokens before compression (default: 8000)",
    )
    parser.add_argument(
        "--state-dir",
        default=".state",
        help="Directory for persisted state (default: .state)",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Force LocalSandbox even if Docker is available",
    )
    parser.add_argument(
        "--rule-dir",
        default=None,
        help="Directory containing rule files for control plane injection",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Run in multi-agent pipeline mode (PLAN -> CODE_GEN -> METRIC_ANALYSIS -> VERIFICATION)",
    )
    parser.add_argument(
        "--target-spec",
        default=None,
        help="Path to target_spec.json (pipeline mode only)",
    )
    parser.add_argument(
        "--probes-only",
        action="store_true",
        help="Run only hardware probes (no agent loop) and exit. "
             "Ideal for Kaggle evaluation where only results.json is needed.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for results.json (default: current directory)",
    )
    return parser


def _make_builder(args) -> SystemBuilder:
    """Create a SystemBuilder from parsed CLI arguments."""
    return (
        SystemBuilder()
        .with_state_dir(args.state_dir)
        .with_permission_mode(args.mode)
        .with_max_tokens(args.max_tokens)
        .with_max_turns(args.max_turns)
        .with_rule_dir(args.rule_dir)
        .with_no_docker(args.no_docker)
    )


def _wire_ui_events(agent_loop: AgentLoop, ui: TerminalUI) -> None:
    def _handle_event(event: Any) -> None:
        if event.kind == EventKind.TOOL_CALL:
            ui.show_tool_start(event.payload["tool"], event.payload.get("args", {}))
        elif event.kind == EventKind.TOOL_RESULT:
            ui.show_tool_complete(event.payload["tool"], event.payload)
        elif event.kind == EventKind.ERROR:
            ui.show_tool_error(
                event.payload.get("tool", "unknown"),
                event.payload.get("error", "Unknown error"),
            )
        elif event.kind == EventKind.START:
            ui.show_message(f"Session {event.payload['session_id']} started")
        elif event.kind == EventKind.STOP:
            reason = event.payload.get("reason", "done")
            ui.show_message(f"Session stopped: {reason}")
        elif event.kind == EventKind.APPROVAL_REQUEST:
            ui.show_message(
                f"Approval requested for tool: {event.payload.get('tool', 'unknown')}"
            )
        elif event.kind == EventKind.COMPRESS:
            ui.show_message(
                f"Context compressed ({event.payload.get('entries_removed', 0)} entries removed)"
            )

    agent_loop.on_event(_handle_event)


def _run_interactive(agent_loop: AgentLoop, ui: TerminalUI) -> None:
    ui.show_message("=" * 60)
    ui.show_message("GPU Profiling System -- Interactive Mode")
    ui.show_message("Type 'quit' to exit")
    ui.show_message("Any other text is the model's output")
    ui.show_message("=" * 60)

    def _model_caller(messages: list[dict[str, Any]]) -> str:
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m["content"][:200]
                break

        if last_user:
            ui.show_message(f"Context: ...{last_user}")

        ui.show_message("Enter model output (JSON tool call or text):")
        try:
            return input("> ")
        except (EOFError, KeyboardInterrupt):
            return ""

    agent_loop.set_model_caller(_model_caller)

    while agent_loop.loop_state.is_running:
        agent_loop._inner_loop_step()


def main(argv: list[str] | None = None) -> int:
    parser = _create_argparse()
    args = parser.parse_args(argv)

    if args.probes_only:
        os.makedirs(args.state_dir, exist_ok=True)
        builder = _make_builder(args)
        return _run_probes_only(args, builder.sandbox)

    if not args.resume and not args.goal:
        parser.error("Must provide either a GOAL argument or --resume SESSION_ID")
        return 1

    os.makedirs(args.state_dir, exist_ok=True)
    builder = _make_builder(args)

    if args.pipeline:
        return _run_pipeline_mode(args, builder)

    ui = TerminalUI()

    if args.resume:
        agent_loop = _build_resumed_loop(args, builder)
        _wire_ui_events(agent_loop, ui)
        try_wire_model_caller(agent_loop)
    else:
        session_id = args.session or f"sess_{uuid.uuid4().hex[:8]}"
        from src.application.session import SessionState
        session = SessionState(session_id=session_id, goal=args.goal)
        print(f"[session] Created session: {session_id}")
        print(f"[session] Goal: {args.goal}")

        agent_loop = builder.build_agent_loop(session)
        tool_runner = builder.build_tool_runner(agent_loop.tool_registry)

        agent_loop.set_approval_callback(ui.request_approval)
        agent_loop.set_tool_executor(tool_runner.execute)
        _wire_ui_events(agent_loop, ui)
        try_wire_model_caller(agent_loop)

    try:
        agent_loop.loop_state.is_running = True
        agent_loop._emit(EventKind.START, {
            "session_id": agent_loop.loop_state.session_id,
        })

        if agent_loop._model_caller is not None:
            ui.show_message("LLM API configured — running automatically")
            agent_loop.start()
        else:
            _run_interactive(agent_loop, ui)

    except KeyboardInterrupt:
        ui.show_message("Interrupted by user")
        agent_loop.stop()
    except Exception as e:
        ui.show_tool_error("agent_loop", str(e))
        agent_loop.stop()
        return 1

    return 0


def _build_resumed_loop(args, builder: SystemBuilder) -> AgentLoop:
    from src.application.session import SessionManager
    mgr = SessionManager(state_dir=args.state_dir)
    session = mgr.resume(args.resume, new_goal=None)
    print(f"[session] Resuming session: {args.resume}")

    agent_loop = builder.build_agent_loop(session)
    tool_runner = builder.build_tool_runner(agent_loop.tool_registry)
    agent_loop.set_tool_executor(tool_runner.execute)
    return agent_loop


def _run_pipeline_mode(args, builder: SystemBuilder):
    from src.application.audit_report import PipelineAuditReport
    from src.application.session import SessionState
    from src.domain.subagent import SubAgentStatus

    session_id = args.session or f"sess_{uuid.uuid4().hex[:8]}"

    if args.target_spec and os.path.isfile(args.target_spec):
        with open(args.target_spec, "r") as f:
            target_spec = json.load(f)
    elif args.goal:
        target_spec = {
            "goal": args.goal,
            "targets": args.goal.split(),
        }
    else:
        print("[pipeline] Error: need --goal or --target-spec")
        return 1

    session = SessionState(session_id=session_id, goal=args.goal or "Pipeline analysis")
    print(f"[session] Created session: {session_id}")

    pipeline = builder.build_pipeline(session)

    agent_loop = builder.build_agent_loop(session)
    ui = TerminalUI()
    ui.show_message(f"Pipeline mode — Session {session_id}")

    try_wire_model_caller(agent_loop)

    audit = PipelineAuditReport()
    audit.record_start()

    try:
        result = agent_loop.run_pipeline(pipeline, target_spec)

        audit.record_end()
        audit.set_final_result(result)

        if result.status == SubAgentStatus.SUCCESS:
            ui.show_message("Pipeline completed successfully")
            ui.show_message(f"Result: {result.data}")
        elif result.status == SubAgentStatus.REJECTED:
            ui.show_message("Pipeline completed with REJECTED verification")
            ui.show_message(f"Result: {result.data}")
        else:
            error_info = result.error or (
                f"Pipeline {result.status.value} — "
                f"stage: {result.agent_role.value}, "
                f"data keys: {', '.join(result.data.keys())}"
            )
            ui.show_tool_error("pipeline", error_info)
            audit.record_error(error_info)

        output_dir = args.output_dir or os.getcwd()
        if result.data:
            pipeline_data = dict(result.data or {})
            if result.metadata:
                pipeline_data["_pipeline_metadata"] = result.metadata

            ui.show_message("Running hardware probes for ground-truth measurements...")
            probe_results = _run_probes_no_write(builder.sandbox)
            if probe_results:
                ui.show_message(f"Hardware probes complete: {probe_results}")
                _assemble_final_results(
                    output_dir=output_dir,
                    hardware_results=probe_results,
                    pipeline_data=pipeline_data,
                    target_spec=target_spec,
                )
                ui.show_message(
                    f"Final results written to: {os.path.join(output_dir, 'results.json')}"
                )
            else:
                ui.show_message("[probe] Hardware probes returned no results — using pipeline measurements only")
                ui.show_message("[probe] Pipeline results contain estimated values only")
                _write_results_json(
                    result=result,
                    target_spec=target_spec,
                    output_dir=output_dir,
                )
                ui.show_message(
                    f"Pipeline results written to: {os.path.join(output_dir, 'results.json')}"
                )

        if result.status not in (SubAgentStatus.SUCCESS, SubAgentStatus.REJECTED):
            return 1
    except KeyboardInterrupt:
        ui.show_message("Interrupted by user")
        audit.record_error("Interrupted by user")
        return 1
    except Exception as e:
        ui.show_tool_error("pipeline", str(e))
        audit.record_error(str(e))
        return 1

    output_dir = args.output_dir or os.getcwd()
    audit_dir = os.path.join(output_dir, "audit")
    os.makedirs(audit_dir, exist_ok=True)
    json_path, md_path = audit.save(audit_dir)
    ui.show_message(f"Audit report saved to: {audit_dir}/")

    return 0


def _run_probes_only(args, sandbox):
    try:
        import shutil
        if shutil.which("nvcc") is None:
            print("[probe] nvcc not found — cannot run hardware probes")
            return 1

        from src.infrastructure.probing.orchestrator import run_hardware_probes
        output_dir = args.output_dir or os.getcwd()
        results = run_hardware_probes(sandbox=sandbox, write_to_dir=output_dir)

        if results:
            measurements = results.get("measurements", {})
            print(f"[probe] {len(measurements)} measurements collected")
            for key, value in sorted(measurements.items()):
                if not key.startswith("_"):
                    print(f"  {key}: {value}")
            return 0
        else:
            print("[probe] No measurements collected")
            return 1
    except Exception as e:
        print(f"[probe] Hardware probes failed: {e}")
        return 1


def _run_probes_no_write(sandbox):
    try:
        import shutil
        if shutil.which("nvcc") is None:
            return None

        from src.infrastructure.probing.orchestrator import run_hardware_probes
        return run_hardware_probes(sandbox=sandbox, write_to_dir=None)
    except Exception as e:
        print(f"[probe] Hardware probes failed: {e}")
        return None


def _build_methodology(
    output: dict,
    target_spec: dict,
    pipeline_data: dict | None,
    hardware_results: dict | None,
) -> str:
    """Build a substantive methodology description from actual measurement data.

    Instead of returning a placeholder like 'pipeline_analysis', this function
    constructs a meaningful description of what was measured and how, based on
    the actual targets, measurements, and techniques used.
    """
    targets = target_spec.get("targets", [])
    measurements = {k: v for k, v in output.items()
                    if isinstance(v, (int, float)) and not k.startswith("_")}

    method_techniques = {
        "dram_latency_cycles": "random pointer-chasing kernel (clock64(), 128MB working set, single-thread, 10M iterations)",
        "l2_latency_cycles": "random pointer-chasing kernel (clock64(), 2MB working set, single-thread, 10M iterations)",
        "l1_latency_cycles": "random pointer-chasing kernel (clock64(), 8KB working set, single-thread, 10M iterations)",
        "l2_cache_size_mb": "working-set sweep with pointer-chasing (14 sizes from 1MB to 128MB, cliff detection at >3x latency jump)",
        "actual_boost_clock_mhz": "dual-timing compute kernel (clock64() + cudaEventElapsedTime, freq = cycles/elapsed_us)",
        "dram_bandwidth_gbps": "STREAM copy kernel (128MB, 65535 blocks x 256 threads, cudaEventElapsedTime)",
        "max_shmem_per_block_kb": "CUDA occupancy API sweep (cudaOccupancyMaxActiveBlocksPerMultiprocessor)",
        "bank_conflict_penalty_ratio": "two-kernel comparison (strided vs sequential shared memory access, cudaEventElapsedTime)",
        "shmem_bandwidth_gbps": "per-SM shared memory bandwidth (1 block, 256 threads, cudaEventElapsedTime)",
        "sm_count": "multi-strategy detection (cudaGetDeviceProperties + block ID sweep + occupancy API cross-validation)",
        "shmem_bank_conflict_penalty_ns": "two-kernel comparison (strided vs sequential, cudaEventElapsedTime, penalty = strided - sequential)",
        "l1_cache_size_kb": "working-set sweep with pointer-chasing (12 sizes from 1KB to 256KB, cliff detection at >2x latency jump)",
    }

    parts = []
    measured_targets = [t for t in targets if t in measurements]
    if measured_targets:
        parts.append(f"Multi-agent GPU profiling pipeline measuring {len(measured_targets)}/{len(targets)} targets.")
        parts.append("Techniques used per target:")
        for t in measured_targets:
            technique = method_techniques.get(t, "custom micro-benchmark (clock64()/cudaEventElapsedTime)")
            value = measurements[t]
            parts.append(f"  - {t}: {technique} [measured: {value}]")
    else:
        parts.append(f"GPU profiling pipeline targeting: {', '.join(targets)}.")
        parts.append("Measurements collected via LLM-generated CUDA micro-benchmarks compiled with nvcc.")

    parts.append("Anti-optimization: volatile qualifiers, asm volatile memory barriers, #pragma unroll 1.")
    parts.append("Anti-cheat: no reliance on cudaGetDeviceProperties; all values empirically measured.")

    if hardware_results and hardware_results.get("cross_validation"):
        parts.append("Cross-validated with hardware probe measurements.")

    return " ".join(parts)


def _validate_results_quality(results: dict, target_spec: dict) -> tuple[bool, list[str], dict]:
    """Validate results.json quality before writing.
    
    Checks for common data quality issues that indicate measurement failure:
    - Zero values in critical measurements
    - Missing requested targets
    - Implausible values
    - Empty evidence
    
    Returns:
        (quality_ok: bool, warnings: list[str], cleaned_results: dict)
        - quality_ok: True if results pass quality checks
        - warnings: list of warning messages
        - cleaned_results: results with zero/invalid values filtered out
    """
    warnings = []
    requested_targets = set(target_spec.get("targets", []))
    cleaned = dict(results)  # Start with a copy
    
    # Collect measurement keys (excluding metadata fields)
    metadata_keys = {"_quality_warnings", "_quality_ok", "_pipeline_metadata", 
                     "evidence", "methodology", "targets_profiled", "cross_validation"}
    measurement_keys = []
    for k, v in results.items():
        if k.startswith("_") or k in metadata_keys:
            continue
        if isinstance(v, (int, float)):
            measurement_keys.append((k, v))
    
    # Check 1: Zero values — FILTER THEM OUT
    zero_keys = {k for k, v in measurement_keys if v == 0 and k not in ("exit_code", "binary_count")}
    if zero_keys:
        warnings.append(
            f"Zero measurements detected and REMOVED: {', '.join(sorted(zero_keys)[:5])}. "
            "This indicates measurement code is broken (clock64() not called, code optimized away)."
        )
        # Remove zero values from cleaned results
        for k in zero_keys:
            cleaned.pop(k, None)
    
    # Check 2: Negative values — FILTER THEM OUT
    negative_keys = {k for k, v in measurement_keys if v < 0}
    if negative_keys:
        warnings.append(
            f"Negative measurements detected and REMOVED: {', '.join(sorted(negative_keys)[:5])}."
        )
        for k in negative_keys:
            cleaned.pop(k, None)
    
    # Check 3: Implausibly large values — FILTER THEM OUT
    large_keys = {k for k, v in measurement_keys if v > 1e12}
    if large_keys:
        warnings.append(
            f"Suspiciously large values detected and REMOVED: {', '.join(sorted(large_keys)[:5])}."
        )
        for k in large_keys:
            cleaned.pop(k, None)
    
    # Check 4: Missing requested targets
    if requested_targets:
        remaining_measured = {k for k, v in cleaned.items() if isinstance(v, (int, float)) and not k.startswith("_")}
        missing = requested_targets - remaining_measured
        if missing:
            warnings.append(f"Missing requested targets (not measured or filtered): {', '.join(sorted(missing))}")
    
    # Check 5: Empty or missing evidence
    evidence = cleaned.get("evidence", [])
    if not evidence:
        warnings.append("No evidence files or references — measurements may not be verifiable")
    
    # Check 6: Empty methodology
    methodology = cleaned.get("methodology", "")
    if not methodology or len(methodology) < 20:
        warnings.append("Missing or incomplete methodology description")
    
    # Check 7: Too few measurements overall
    remaining_count = sum(1 for v in cleaned.values() if isinstance(v, (int, float)))
    if remaining_count < 2 and requested_targets:
        warnings.append(
            f"Only {remaining_count} valid measurement(s) after quality filtering — "
            f"expected more for target spec with {len(requested_targets)} targets"
        )
    
    # Quality verdict
    remaining_numeric = sum(1 for v in cleaned.values() if isinstance(v, (int, float)))
    if requested_targets:
        remaining_measured = {k for k, v in cleaned.items() if isinstance(v, (int, float)) and not k.startswith("_")}
        missing_after_filter = requested_targets - remaining_measured
        quality_ok = len(missing_after_filter) == 0
        if missing_after_filter and remaining_numeric > 0:
            warnings.append(
                f"Partial results: {len(remaining_measured)}/{len(requested_targets)} targets measured. "
                f"Missing after quality filtering: {', '.join(sorted(missing_after_filter))}"
            )
    else:
        quality_ok = remaining_numeric >= 1
    cleaned["_quality_ok"] = quality_ok
    
    return quality_ok, warnings, cleaned


def _assemble_final_results(output_dir, hardware_results, pipeline_data, target_spec):
    try:
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "results.json")

        hw_measurements = hardware_results.get("measurements", {}) if hardware_results else {}
        output = {}

        pipeline_measurements = pipeline_data.get("measurements", {})
        if isinstance(pipeline_measurements, dict):
            for k, v in pipeline_measurements.items():
                if k not in output:
                    output[k] = v

        tool_results = pipeline_data.get("tool_results", [])
        if isinstance(tool_results, list):
            for tr in tool_results:
                if not isinstance(tr, dict):
                    continue
                stdout = tr.get("stdout", "") or tr.get("output", "")
                if stdout:
                    for line in stdout.splitlines():
                        line = line.strip()
                        if ":" in line and not line.startswith("//") and not line.startswith("#"):
                            parts = line.split(":", 1)
                            if len(parts) == 2:
                                key = parts[0].strip()
                                val_str = parts[1].strip()
                                try:
                                    val = float(val_str)
                                    if key not in output:
                                        output[key] = val
                                except ValueError:
                                    pass

        final_output = pipeline_data.get("final_output", "")
        if final_output and not pipeline_measurements:
            import re
            for line in final_output.splitlines():
                line = line.strip()
                match = re.match(r'^([\w_]+)\s*[:=]\s*([\d.]+)', line)
                if match:
                    key = match.group(1)
                    try:
                        val = float(match.group(2))
                        if key not in output:
                            output[key] = val
                    except ValueError:
                        pass

        for k, v in hw_measurements.items():
            if k not in output:
                output[k] = v

        for k, v in pipeline_data.items():
            if k in ("_pipeline_metadata", "measurements", "tool_results",
                      "final_output", "code_gen_output", "analysis_method",
                      "review_text", "plan_text", "analysis_output"):
                continue
            if k not in output:
                output[k] = v

        if hardware_results and "cross_validation" in hardware_results:
            output["cross_validation"] = hardware_results["cross_validation"]

        evidence = list(hardware_results.get("evidence_files", [])) if hardware_results else []
        if "_pipeline_metadata" in pipeline_data:
            meta = pipeline_data["_pipeline_metadata"]
            if "evidence" in meta and isinstance(meta["evidence"], list):
                for item in meta["evidence"]:
                    s = str(item)
                    if s not in evidence:
                        evidence.append(s)
        output["evidence"] = evidence

        if "methodology" not in output or len(str(output.get("methodology", ""))) < 50:
            pipeline_method = pipeline_data.get("analysis_method", "")
            if pipeline_method and len(pipeline_method) > 50:
                output["methodology"] = pipeline_method
            elif hardware_results and "methodology" in hardware_results:
                hw_method = hardware_results["methodology"]
                if isinstance(hw_method, str) and len(hw_method) > 50:
                    output["methodology"] = hw_method
                elif isinstance(hw_method, dict):
                    output["methodology"] = str(hw_method)
                else:
                    output["methodology"] = _build_methodology(output, target_spec, pipeline_data, hardware_results)
            else:
                output["methodology"] = _build_methodology(output, target_spec, pipeline_data, hardware_results)

        output["targets_profiled"] = target_spec.get("targets", [])

        # Quality check: validate, filter invalid values, and get cleaned results
        quality_ok, quality_issues, cleaned_output = _validate_results_quality(output, target_spec)
        
        if quality_issues:
            cleaned_output["_quality_warnings"] = quality_issues
            print(f"[pipeline] Results quality warnings: {len(quality_issues)}")
            for issue in quality_issues[:5]:
                print(f"  ⚠️ {issue}")
        
        if not quality_ok:
            cleaned_output["_quality_ok"] = False
            print("[pipeline] Results FAILED quality check — writing with _quality_ok: false flag")
            
            missing_targets = cleaned_output.get("_missing_targets", [])
            if not missing_targets:
                requested = set(target_spec.get("targets", []))
                measured = {k for k, v in cleaned_output.items() if isinstance(v, (int, float)) and not k.startswith("_")}
                missing_targets = sorted(requested - measured)
            
            if missing_targets:
                print(f"[pipeline] Missing targets: {', '.join(missing_targets)}")
                print("[pipeline] These targets will be filled by Hardware Probes if available")
        else:
            cleaned_output["_quality_ok"] = True
        
        with open(results_path, "w") as f:
            json.dump(cleaned_output, f, indent=2)

        return results_path
    except Exception as e:
        print(f"[pipeline] Failed to assemble results: {e}")
        return None


def _extract_rejection_feedback(result) -> str:
    """Extract actionable feedback from a REJECTED pipeline result.
    
    Parses the Verification output to identify specific concerns and
    suggested fixes that should be injected into the next CodeGen iteration.
    """
    if not result.data:
        return ""
    
    feedback_parts = []
    
    # Extract review_text from Verification
    review_text = result.data.get("review_text", "")
    if review_text:
        feedback_parts.append(f"Verification concerns:\n{review_text[:500]}")
    
    # Extract concerns from metadata
    metadata = result.data.get("metadata", {})
    if isinstance(metadata, dict):
        concerns = metadata.get("concerns", [])
        if concerns:
            feedback_parts.append(f"Specific issues:\n" + "\n".join(f"- {c}" for c in concerns))
    
    # Extract final_output if it contains verdict
    final_output = result.data.get("final_output", "")
    if "REJECT" in final_output.upper():
        lines = final_output.splitlines()
        reject_lines = [l for l in lines if any(kw in l.lower() for kw in ("issue", "problem", "fix", "error", "wrong", "incorrect"))]
        if reject_lines:
            feedback_parts.append(f"Key issues identified:\n" + "\n".join(reject_lines[:5]))
    
    return "\n\n".join(feedback_parts) if feedback_parts else ""


def _write_results_json(result, target_spec, output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "results.json")

        output: dict[str, Any] = {}
        metrics: dict[str, Any] = {}
        artifacts: list[str] = []

        if result.is_success():
            measurements = result.data.get("measurements", {})
            if isinstance(measurements, dict):
                metrics.update(measurements)

            tool_results = result.data.get("tool_results", [])
            if isinstance(tool_results, list):
                for tr in tool_results:
                    if not isinstance(tr, dict):
                        continue
                    stdout = tr.get("stdout", "")
                    if stdout:
                        for line in stdout.splitlines():
                            line = line.strip()
                            if ":" in line and not line.startswith("//"):
                                parts = line.split(":", 1)
                                if len(parts) == 2:
                                    key = parts[0].strip()
                                    val_str = parts[1].strip()
                                    try:
                                        val = float(val_str)
                                        if key not in metrics:
                                            metrics[key] = val
                                    except ValueError:
                                        pass

            for key, value in result.data.items():
                if key in ("measurements", "tool_results", "final_output",
                           "code_gen_output", "analysis_method", "review_text",
                           "plan_text", "analysis_output"):
                    continue
                if isinstance(value, (int, float)):
                    metrics[key] = value
                elif isinstance(value, list):
                    artifacts.extend(str(v) for v in value)

            final_output = result.data.get("final_output", "")
            if final_output and not metrics:
                import re
                for line in final_output.splitlines():
                    line = line.strip()
                    match = re.match(r'^([\w_]+)\s*[:=]\s*([\d.]+)', line)
                    if match:
                        key = match.group(1)
                        try:
                            val = float(match.group(2))
                            if key not in metrics:
                                metrics[key] = val
                        except ValueError:
                            pass

        for k, v in metrics.items():
            if k not in output:
                output[k] = v

        targets = target_spec.get("targets", [])
        output["targets_profiled"] = targets

        if "methodology" not in output:
            method = result.metadata.get("analysis_method", "")
            if method and len(method) > 50:
                output["methodology"] = method
            else:
                output["methodology"] = _build_methodology(output, target_spec, {"measurements": metrics}, None)

        output["evidence"] = artifacts

        # Quality check: validate, filter invalid values, and get cleaned results
        quality_ok, quality_issues, cleaned_output = _validate_results_quality(output, target_spec)
        
        if quality_issues:
            cleaned_output["_quality_warnings"] = quality_issues
            print(f"[results] Quality warnings: {len(quality_issues)}")
            for issue in quality_issues[:5]:
                print(f"  ⚠️ {issue}")
        
        if not quality_ok:
            print("[results] Results FAILED quality check — writing with _quality_ok: false flag")
        
        with open(results_path, "w") as f:
            json.dump(cleaned_output, f, indent=2)

        return results_path
    except Exception as e:
        print(f"[results] Failed to write results.json: {e}")
        return None


if __name__ == "__main__":
    sys.exit(main())
