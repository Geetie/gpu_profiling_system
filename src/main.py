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


def _map_permission_mode(mode_str: str):
    from src.domain.permission import PermissionMode

    return {
        "default": PermissionMode.DEFAULT,
        "conservative": PermissionMode.CONSERVATIVE,
        "relaxed": PermissionMode.RELAXED,
        "high_autonomy": PermissionMode.HIGH_AUTONOMY,
    }[mode_str]


def _build_sandbox(no_docker: bool):
    """Create the best available sandbox."""
    from src.infrastructure.sandbox import (
        DockerSandbox,
        LocalSandbox,
        SandboxConfig,
        docker_available,
    )

    config = SandboxConfig()
    if not no_docker and docker_available():
        print("[sandbox] Using DockerSandbox")
        return DockerSandbox(config=config)
    else:
        print("[sandbox] Using LocalSandbox (dev mode)")
        return LocalSandbox(config=config)


def _try_wire_model_caller(agent_loop) -> bool:
    """Try to wire the LLM model caller. Silently skip if config missing."""
    try:
        from src.infrastructure.model_caller import make_model_caller
        caller = make_model_caller()
        agent_loop.set_model_caller(caller)
        print("[llm] Model caller configured from config/api_config.json")
        return True
    except (FileNotFoundError, ValueError) as e:
        print(f"[llm] No LLM API configured: {e}")
        print("[llm] Falling back to interactive REPL mode")
        return False


def _wire_subagent_model_caller(subagent) -> bool:
    """Wire LLM model caller to a subagent."""
    try:
        from src.infrastructure.model_caller import make_subagent_model_caller
        caller = make_subagent_model_caller()
        subagent.set_model_caller(caller)
        return True
    except (FileNotFoundError, ValueError):
        return False


def _wire_all_subagents(planner, code_gen, metric_analysis, verification) -> bool:
    from src.infrastructure.model_caller import make_model_caller, load_config

    config = None
    env = {}
    try:
        config = load_config()
        env = config["env"]
    except (FileNotFoundError, ValueError) as e:
        print(f"[llm] No api_config.json: {e} — using provider_manager")

    provider_name = "unknown"
    try:
        from src.infrastructure.provider_manager import get_provider_manager
        provider_manager = get_provider_manager()
        print(f"[llm] Provider manager loaded: {list(provider_manager.providers.keys())}")

        if provider_manager.providers:
            provider_priority = ["longcat", "aliyun_bailian", "anthropic"]
            for pn in provider_priority:
                provider = provider_manager.providers.get(pn)
                if not provider:
                    print(f"[llm] Provider '{pn}' not in config, skipping")
                    continue
                api_key = provider.get_api_key()
                if not api_key:
                    print(f"[llm] Provider '{pn}' has no API key, skipping")
                    continue
                print(f"[llm] Selected provider: {pn} (model: {provider.get_model('default')})")
                provider_manager.current_provider = provider
                provider_name = pn
                break

            if provider_name == "unknown" and provider_manager.providers:
                first_pn = next(iter(provider_manager.providers))
                first_provider = provider_manager.providers[first_pn]
                if first_provider.get_api_key():
                    print(f"[llm] Fallback to first available provider: {first_pn}")
                    provider_manager.current_provider = first_provider
                    provider_name = first_pn
    except Exception as e:
        print(f"[llm] Provider manager error: {e}")

    if provider_name == "longcat":
        main_model = "longcat-flash-chat"
        code_model = "longcat-flash-chat"
        reasoning_model = "longcat-flash-chat"
    elif provider_name == "aliyun_bailian":
        main_model = "qwen-plus"
        code_model = "qwen-plus"
        reasoning_model = "qwen-max"
    else:
        main_model = env.get("ANTHROPIC_MODEL", "qwen3.6-plus")
        code_model = env.get("ANTHROPIC_DEFAULT_SONNET_MODEL", main_model)
        reasoning_model = env.get("ANTHROPIC_REASONING_MODEL", main_model)

    print(f"[llm] Models: main={main_model}, code={code_model}, reasoning={reasoning_model}")

    planner.set_model_caller(make_model_caller(model_name=main_model))
    print(f"[llm] PlannerAgent -> {main_model} (Provider: {provider_name})")

    code_gen.set_model_caller(make_model_caller(model_name=code_model))
    print(f"[llm] CodeGenAgent -> {code_model} (Provider: {provider_name})")

    metric_analysis.set_model_caller(make_model_caller(model_name=main_model))
    print(f"[llm] MetricAnalysisAgent -> {main_model} (Provider: {provider_name})")

    verification.set_model_caller(make_model_caller(
        model_name=reasoning_model,
        max_tokens=8192,
    ))
    print(f"[llm] VerificationAgent -> {reasoning_model} (Provider: {provider_name})")

    return True


def _build_tool_handlers(sandbox):
    """Create all tool handler functions bound to the sandbox."""
    from src.infrastructure.file_ops import FileOperations
    from src.infrastructure.tools.compile_cuda import compile_cuda_handler
    from src.infrastructure.tools.execute_binary import execute_binary_handler
    from src.infrastructure.tools.file_tools import (
        make_read_file_handler,
        make_write_file_handler,
    )
    from src.infrastructure.tools.kaggle_push import kaggle_push_handler
    from src.infrastructure.tools.microbenchmark import generate_microbenchmark_handler
    from src.infrastructure.tools.run_ncu import run_ncu_handler

    file_ops = FileOperations(sandbox_root=sandbox.sandbox_root)

    return {
        "run_ncu": lambda args: run_ncu_handler(args, sandbox=sandbox),
        "compile_cuda": lambda args: compile_cuda_handler(args, sandbox=sandbox),
        "execute_binary": lambda args: execute_binary_handler(args, sandbox=sandbox),
        "read_file": make_read_file_handler(file_ops),
        "write_file": make_write_file_handler(file_ops),
        "generate_microbenchmark": generate_microbenchmark_handler,
        "kaggle_push": kaggle_push_handler,
    }


def _build_loop_components(args, session, sandbox):
    """Create and wire all components needed for the agent loop.

    Returns (AgentLoop, TerminalUI, ToolRunner).
    """
    from src.application.approval_queue import ApprovalQueue
    from src.application.context import ContextManager
    from src.application.control_plane import ControlPlane
    from src.application.tool_runner import ToolRunner
    from src.domain.permission import PermissionChecker
    from src.domain.schema_validator import SchemaValidator
    from src.domain.tool_contract import build_standard_registry
    from src.infrastructure.state_persist import StatePersister

    # Shared persister (single instance -- INT-10 fix)
    persister = StatePersister(log_dir=args.state_dir)

    # Tool registry
    registry = build_standard_registry()
    print(f"[registry] {len(registry.list_tools())} tools registered: "
          f"{', '.join(registry.list_tools())}")

    # Tool handlers
    handlers = _build_tool_handlers(sandbox)

    # Approval queue (shares persister)
    approval_queue = ApprovalQueue(
        state_dir=args.state_dir,
        persister=persister,
    )

    # Tool runner
    permission_checker = PermissionChecker(mode=_map_permission_mode(args.mode))
    validator = SchemaValidator()
    tool_runner = ToolRunner(
        registry=registry,
        tool_handlers=handlers,
        approval_queue=approval_queue,
        permission_checker=permission_checker,
        persister=persister,
        validator=validator,
    )

    # Control plane
    control_plane = ControlPlane(rule_dir=args.rule_dir)

    # Context manager
    context_manager = ContextManager(max_tokens=args.max_tokens)

    # Agent loop
    agent_loop = AgentLoop(
        session=session,
        context_manager=context_manager,
        control_plane=control_plane,
        tool_registry=registry,
        max_turns=args.max_turns,
        state_dir=args.state_dir,
        permission_mode=_map_permission_mode(args.mode),
    )

    # Presentation layer
    ui = TerminalUI()

    # Wire: approval callback
    agent_loop.set_approval_callback(ui.request_approval)

    # Wire: tool executor
    agent_loop.set_tool_executor(tool_runner.execute)

    # Wire: event handling
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

    return agent_loop, ui, tool_runner


def _run_interactive(agent_loop: AgentLoop, ui: TerminalUI) -> None:
    """Run the agent loop in interactive REPL mode.

    The user types tool-call JSON or text responses as the "model".
    """
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

    # Gap-1 fix: --probes-only is self-contained — no goal/resume needed
    if args.probes_only:
        os.makedirs(args.state_dir, exist_ok=True)
        sandbox = _build_sandbox(args.no_docker)
        return _run_probes_only(args, sandbox)

    # Validate: need either goal or resume
    if not args.resume and not args.goal:
        parser.error("Must provide either a GOAL argument or --resume SESSION_ID")
        return 1

    # Create state directory
    os.makedirs(args.state_dir, exist_ok=True)

    # Build sandbox
    sandbox = _build_sandbox(args.no_docker)

    # Pipeline mode
    if args.pipeline:
        return _run_pipeline_mode(args, sandbox)

    ui = TerminalUI()

    if args.resume:
        # Resume mode
        print(f"[session] Resuming session: {args.resume}")
        from src.application.context import ContextManager
        from src.application.control_plane import ControlPlane
        from src.domain.tool_contract import build_standard_registry

        context_manager = ContextManager(max_tokens=args.max_tokens)
        control_plane = ControlPlane(rule_dir=args.rule_dir)
        registry = build_standard_registry()

        agent_loop = AgentLoop.from_resume(
            session_id=args.resume,
            control_plane=control_plane,
            context_manager=context_manager,
            tool_registry=registry,
            state_dir=args.state_dir,
            max_turns=args.max_turns,
            permission_mode=_map_permission_mode(args.mode),
        )

        # Wire tool executor and handlers for resumed session
        handlers = _build_tool_handlers(sandbox)
        from src.application.approval_queue import ApprovalQueue
        from src.application.tool_runner import ToolRunner
        from src.domain.permission import PermissionChecker
        from src.domain.schema_validator import SchemaValidator
        from src.infrastructure.state_persist import StatePersister

        persister = StatePersister(log_dir=args.state_dir)
        approval_queue = ApprovalQueue(state_dir=args.state_dir, persister=persister)
        permission_checker = PermissionChecker(mode=_map_permission_mode(args.mode))
        tool_runner = ToolRunner(
            registry=registry,
            tool_handlers=handlers,
            approval_queue=approval_queue,
            permission_checker=permission_checker,
            persister=persister,
            validator=SchemaValidator(),
        )

        agent_loop.set_tool_executor(tool_runner.execute)
        agent_loop.set_approval_callback(ui.request_approval)
        _try_wire_model_caller(agent_loop)
    else:
        # New session
        session_id = args.session or f"sess_{uuid.uuid4().hex[:8]}"
        from src.application.session import SessionState

        session = SessionState(session_id=session_id, goal=args.goal)
        print(f"[session] Created session: {session_id}")
        print(f"[session] Goal: {args.goal}")

        agent_loop, ui, _tool_runner = _build_loop_components(args, session, sandbox)

        # Wire LLM model caller for batch mode (non-interactive)
        if not sys.stdin.isatty() or True:  # Always try to wire API caller
            _try_wire_model_caller(agent_loop)

    # Run the loop
    try:
        agent_loop.loop_state.is_running = True
        agent_loop._emit(EventKind.START, {
            "session_id": agent_loop.loop_state.session_id,
        })

        # If model caller is wired (API configured), run automatically.
        # Otherwise fall back to interactive REPL.
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


def _build_handoff_validator():
    """Build handoff validator for pipeline stage boundaries."""
    from src.application.handoff_validation import HandoffValidator
    return HandoffValidator()


def _build_circuit_breaker():
    """Build circuit breaker for progressive degradation detection."""
    from src.application.circuit_breaker import CircuitBreaker
    return CircuitBreaker(
        degradation_threshold=3,
        min_quality_threshold=0.3,
    )


def _build_pipeline(args, sandbox, registry, session):
    """Build a Pipeline with all 4 subagents wired to the sandbox.

    P2 (fail-closed): each agent only gets tools for its specific role.
    P1 (tools define boundaries): an agent cannot perform operations
    outside its registered tool contracts.
    """
    from src.application.context import ContextManager
    from src.domain.pipeline import Pipeline, PipelineStep
    from src.domain.subagent import AgentRole, PipelineStage
    from src.domain.tool_contract import build_agent_registry
    from src.application.subagents.codegen import CodeGenAgent
    from src.application.subagents.metric_analysis import MetricAnalysisAgent
    from src.application.subagents.planner import PlannerAgent
    from src.application.subagents.verification import VerificationAgent

    perm_mode = _map_permission_mode(args.mode)

    # P2: per-agent tool registries — agents can ONLY use their role's tools
    planner_tools = {"read_file", "write_file"}
    codegen_tools = {"compile_cuda", "execute_binary", "write_file", "read_file"}
    metric_tools = {"run_ncu", "read_file"}
    verification_tools = {"read_file"}  # read-only evidence review (P7)

    planner_reg = build_agent_registry(planner_tools)
    codegen_reg = build_agent_registry(codegen_tools)
    metric_reg = build_agent_registry(metric_tools)
    verification_reg = build_agent_registry(verification_tools)

    # Planner — global coordinator, task decomposition only
    planner = PlannerAgent(
        context_manager=ContextManager(max_tokens=args.max_tokens),
        tool_registry=planner_reg,
        state_dir=args.state_dir,
        permission_mode=perm_mode,
    )

    # CodeGen — write CUDA, compile, execute, parse output
    code_gen = CodeGenAgent(
        context_manager=ContextManager(max_tokens=args.max_tokens),
        tool_registry=codegen_reg,
        state_dir=args.state_dir,
        permission_mode=perm_mode,
        sandbox=sandbox,
    )

    # MetricAnalysis — NCU profiling, bottleneck identification
    metric_analysis = MetricAnalysisAgent(
        context_manager=ContextManager(max_tokens=args.max_tokens),
        tool_registry=metric_reg,
        state_dir=args.state_dir,
        permission_mode=perm_mode,
    )

    # Verification — independent review, read-only (P7)
    verification = VerificationAgent(
        tool_registry=verification_reg,
        state_dir=args.state_dir,
        permission_mode=perm_mode,
        max_tokens=args.max_tokens,
    )

    pipeline = Pipeline.build_default(
        planner=planner,
        code_gen=code_gen,
        metric_analysis=metric_analysis,
        verification=verification,
        state_dir=args.state_dir,
        sandbox=sandbox,
        tool_handlers=_build_tool_handlers(sandbox),
        max_turns_per_stage=50,
        handoff_validator=_build_handoff_validator(),
        circuit_breaker=_build_circuit_breaker(),
    )

    # Wire LLM model callers to all subagents
    _wire_all_subagents(planner, code_gen, metric_analysis, verification)

    return pipeline


def _run_probes(sandbox, output_dir):
    """Run hardware probes and return summary of measurements."""
    try:
        import shutil
        # Check if nvcc is available (needed for all probes)
        if shutil.which("nvcc") is None:
            return None

        from src.infrastructure.probing.orchestrator import run_hardware_probes
        results = run_hardware_probes(sandbox=sandbox, write_to_dir=output_dir)
        measurements = results.get("measurements", {})
        # Return summary of key measurements
        summary = {}
        for key in [
            "dram_latency_cycles", "l2_latency_cycles", "l1_latency_cycles",
            "l2_cache_size_mb", "actual_boost_clock_mhz", "max_shmem_per_block_kb",
            "dram_bandwidth_gbps", "shmem_bandwidth_gbps",
            "bank_conflict_penalty_ratio", "sm_count",
        ]:
            if key in measurements:
                summary[key] = measurements[key]
        # Include confidence scores
        for key in measurements:
            if key.startswith("_confidence_"):
                summary[key] = measurements[key]
        # Include cross-validation results
        if "cross_validation" in results:
            summary["cross_validation"] = results["cross_validation"]
        return summary
    except Exception as e:
        print(f"[probe] Hardware probes failed: {e}")
        return None


def _run_probes_only(args, sandbox) -> int:
    """Gap 9: Run hardware probes directly and exit.

    Bypasses the entire agent loop. Ideal for Kaggle evaluation where
    only results.json is needed. No goal, no pipeline, no session.
    """
    output_dir = args.output_dir or os.getcwd()
    print(f"[probes-only] Running hardware probes...")
    print(f"[probes-only] Output directory: {output_dir}")

    summary = _run_probes(sandbox, output_dir)

    if summary is None:
        print("[probes-only] Failed: nvcc not available or probes failed")
        return 1

    # Print summary to terminal
    print("[probes-only] Measurement summary:")
    for key, value in summary.items():
        if key == "cross_validation":
            pass_count = sum(1 for v in value.values() if v is True)
            total_count = len(value)
            print(f"  cross_validation: {pass_count}/{total_count} checks passed")
        else:
            print(f"  {key}: {value}")

    # Verify results.json was written
    results_path = os.path.join(output_dir, "results.json")
    if os.path.isfile(results_path):
        print(f"[probes-only] results.json written to: {results_path}")
    else:
        print("[probes-only] WARNING: results.json not found!")

    return 0


def _write_results_json(result, target_spec, output_dir):
    """Write results.json from pipeline output.

    Gap 1 fix: If orchestrator already wrote a complete results.json,
    load and enrich it with pipeline analysis data rather than overwriting.
    """
    try:
        results_path = os.path.join(output_dir, "results.json")

        # Start from existing orchestrator results if available
        output = {}
        if os.path.isfile(results_path):
            try:
                with open(results_path, "r") as f:
                    existing = json.load(f)
                output.update(existing)
            except (json.JSONDecodeError, IOError):
                pass

        # Extract numeric metrics from pipeline analysis
        metrics = {}
        artifacts = list(output.get("evidence", []))

        if result.data:
            # Priority 1: structured measurements dict (from CodeGen extraction)
            measurements = result.data.get("measurements", {})
            if isinstance(measurements, dict):
                for k, v in measurements.items():
                    if isinstance(v, (int, float)):
                        metrics[k] = v

            # Priority 2: top-level numeric keys in result.data (backward compat)
            for key, value in result.data.items():
                if key in ("measurements", "tool_results", "final_output",
                           "code_gen_output", "analysis_method", "review_text",
                           "plan_text", "analysis_output"):
                    continue  # skip structured fields
                if isinstance(value, (int, float)):
                    metrics[key] = value
                elif isinstance(value, list):
                    artifacts.extend(str(v) for v in value)

        # Merge pipeline metrics (don't overwrite orchestrator measurements)
        for k, v in metrics.items():
            if k not in output:
                output[k] = v

        # Include targets that were profiled
        targets = target_spec.get("targets", [])

        output["targets_profiled"] = targets

        # Preserve orchestrator methodology if present, else use pipeline's
        if "methodology" not in output:
            output["methodology"] = result.metadata.get("analysis_method", "pipeline_analysis")

        # Merge evidence lists
        output["evidence"] = artifacts

        with open(results_path, "w") as f:
            json.dump(output, f, indent=2)

        return results_path
    except Exception as e:
        print(f"[results] Failed to write results.json: {e}")
        return None


def _run_pipeline_mode(args, sandbox):
    """Execute in multi-agent pipeline mode."""
    from src.application.audit_report import PipelineAuditReport
    from src.application.context import ContextManager
    from src.application.control_plane import ControlPlane
    from src.application.session import SessionState
    from src.domain.tool_contract import build_standard_registry

    session_id = args.session or f"sess_{uuid.uuid4().hex[:8]}"

    # Load target spec
    if args.target_spec and os.path.isfile(args.target_spec):
        with open(args.target_spec, "r") as f:
            target_spec = json.load(f)
    elif args.goal:
        # Build a minimal target spec from the goal
        target_spec = {
            "goal": args.goal,
            "targets": args.goal.split(),
        }
    else:
        print("[pipeline] Error: need --goal or --target-spec")
        return 1

    session = SessionState(session_id=session_id, goal=args.goal or "Pipeline analysis")
    print(f"[session] Created session: {session_id}")

    registry = build_standard_registry()
    pipeline = _build_pipeline(args, sandbox, registry, session)

    # Build agent loop for pipeline integration
    control_plane = ControlPlane(rule_dir=args.rule_dir)
    context_manager = ContextManager(max_tokens=args.max_tokens)
    agent_loop = AgentLoop(
        session=session,
        context_manager=context_manager,
        control_plane=control_plane,
        tool_registry=registry,
        max_turns=args.max_turns,
        state_dir=args.state_dir,
        permission_mode=_map_permission_mode(args.mode),
    )

    ui = TerminalUI()
    ui.show_message(f"Pipeline mode — Session {session_id}")

    # Wire LLM model caller to AgentLoop (fallback if subagents don't handle it)
    _try_wire_model_caller(agent_loop)

    # Initialize audit report
    audit = PipelineAuditReport()
    audit.record_start()

    try:
        result = agent_loop.run_pipeline(pipeline, target_spec)
        from src.domain.subagent import SubAgentStatus

        audit.record_end()
        audit.set_final_result(result)

        if result.status == SubAgentStatus.SUCCESS:
            ui.show_message("Pipeline completed successfully")
            ui.show_message(f"Result: {result.data}")

            # Risk 1 fix: Single-write results.json.
            # Pipeline collects hardware probe data (no file written), then
            # assembles the final results.json itself. Eliminates the fragile
            # three-step coupling (write → overwrite → merge back).
            output_dir = args.output_dir or os.getcwd()
            pipeline_data = dict(result.data or {})
            if result.metadata:
                pipeline_data["_pipeline_metadata"] = result.metadata

            # Run hardware probes — no file written, returns data only
            ui.show_message("Running hardware probes for ground-truth measurements...")
            probe_results = _run_probes_no_write(sandbox)
            if probe_results:
                ui.show_message(f"Hardware probes complete: {probe_results}")

                # Assemble final results.json: hardware measurements take priority
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
                # Write pipeline-only results
                _write_results_json(
                    result=result,
                    target_spec=target_spec,
                    output_dir=output_dir,
                )
        else:
            error_info = result.error or ""
            if not error_info:
                error_info = (
                    f"Pipeline {result.status.value} — "
                    f"stage: {result.agent_role.value}, "
                    f"data keys: {', '.join(result.data.keys())}"
                )
            ui.show_tool_error("pipeline", error_info)
            audit.record_error(error_info)
            return 1
    except KeyboardInterrupt:
        ui.show_message("Interrupted by user")
        audit.record_error("Interrupted by user")
        return 1
    except Exception as e:
        ui.show_tool_error("pipeline", str(e))
        audit.record_error(str(e))
        return 1

    # Save audit report
    output_dir = args.output_dir or os.getcwd()
    audit_dir = os.path.join(output_dir, "audit")
    os.makedirs(audit_dir, exist_ok=True)
    json_path, md_path = audit.save(audit_dir)
    ui.show_message(f"Audit report saved to: {audit_dir}/")

    return 0


def _run_probes_no_write(sandbox):
    """Risk 1 fix: Run hardware probes without writing files.

    Returns the full results dict. Caller is responsible for assembling
    and writing the final output.
    """
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
    """Assemble the final results.json from hardware probes + pipeline data.

    Risk 1 fix: Single-write function that replaces the fragile three-step
    coupling (Pipeline writes → orchestrator overwrites → merge back).

    Hardware measurements take priority over pipeline estimates.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "results.json")

        # Start from hardware measurements (ground truth)
        measurements = hardware_results.get("measurements", {})
        output = dict(measurements)

        # Extract pipeline measurements (nested in pipeline_data["measurements"])
        pipeline_measurements = pipeline_data.get("measurements", {})

        # Add pipeline measurements for keys NOT already in hardware output
        for k, v in pipeline_measurements.items():
            if k not in output:
                output[k] = v

        # Add other pipeline data (metadata, outputs)
        for k, v in pipeline_data.items():
            if k in ("_pipeline_metadata", "measurements"):
                continue
            if k not in output:
                output[k] = v

        # Add cross-validation from hardware probes
        if "cross_validation" in hardware_results:
            output["cross_validation"] = hardware_results["cross_validation"]

        # Merge evidence from both sources
        evidence = list(hardware_results.get("evidence_files", []))
        if "_pipeline_metadata" in pipeline_data:
            meta = pipeline_data["_pipeline_metadata"]
            if "evidence" in meta and isinstance(meta["evidence"], list):
                for item in meta["evidence"]:
                    s = str(item)
                    if s not in evidence:
                        evidence.append(s)
        output["evidence"] = evidence

        # Add pipeline methodology if hardware didn't override it
        if "methodology" not in output:
            output["methodology"] = pipeline_data.get(
                "analysis_method", "pipeline_analysis"
            )

        # Add targets profiled
        output["targets_profiled"] = target_spec.get("targets", [])

        with open(results_path, "w") as f:
            json.dump(output, f, indent=2)

        return results_path
    except Exception as e:
        print(f"[pipeline] Failed to assemble results: {e}")
        return None


if __name__ == "__main__":
    sys.exit(main())
