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
import os
import sys
import uuid
from typing import Any

from src.application.agent_loop import AgentLoop, EventKind
from src.application.system_builder import SystemBuilder, try_wire_model_caller
from src.presentation.terminal_ui import TerminalUI


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
                ui.show_message("[probe] No GPU available — skipping hardware probes")
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


def _assemble_final_results(output_dir, hardware_results, pipeline_data, target_spec):
    try:
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "results.json")

        measurements = hardware_results.get("measurements", {})
        output = dict(measurements)

        pipeline_measurements = pipeline_data.get("measurements", {})
        for k, v in pipeline_measurements.items():
            if k not in output:
                output[k] = v

        for k, v in pipeline_data.items():
            if k in ("_pipeline_metadata", "measurements"):
                continue
            if k not in output:
                output[k] = v

        if "cross_validation" in hardware_results:
            output["cross_validation"] = hardware_results["cross_validation"]

        evidence = list(hardware_results.get("evidence_files", []))
        if "_pipeline_metadata" in pipeline_data:
            meta = pipeline_data["_pipeline_metadata"]
            if "evidence" in meta and isinstance(meta["evidence"], list):
                for item in meta["evidence"]:
                    s = str(item)
                    if s not in evidence:
                        evidence.append(s)
        output["evidence"] = evidence

        if "methodology" not in output:
            output["methodology"] = pipeline_data.get(
                "analysis_method", "pipeline_analysis"
            )

        output["targets_profiled"] = target_spec.get("targets", [])

        with open(results_path, "w") as f:
            json.dump(output, f, indent=2)

        return results_path
    except Exception as e:
        print(f"[pipeline] Failed to assemble results: {e}")
        return None


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
            output["methodology"] = result.metadata.get("analysis_method", "pipeline_analysis")

        output["evidence"] = artifacts

        with open(results_path, "w") as f:
            json.dump(output, f, indent=2)

        return results_path
    except Exception as e:
        print(f"[results] Failed to write results.json: {e}")
        return None


if __name__ == "__main__":
    sys.exit(main())
