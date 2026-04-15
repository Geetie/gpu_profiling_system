"""System builder — assembles all components for the GPU Profiling System.

Builder pattern: step-by-step construction of the complex object graph.
Factory pattern: creates sub-agents, tool registries, and pipelines
with proper P2 isolation guarantees.

Extracted from main.py to:
1. Eliminate duplicated wiring logic (resume vs new session vs pipeline)
2. Make component assembly testable in isolation
3. Provide a single place to modify when adding new components
"""
from __future__ import annotations

from typing import Any

from src.application.agent_loop import AgentLoop
from src.application.context import ContextManager
from src.application.control_plane import ControlPlane
from src.application.session import SessionState
from src.domain.permission import PermissionMode
from src.domain.tool_contract import ToolRegistry
from src.infrastructure.state_persist import StatePersister


class SystemBuilder:
    """Step-by-step builder for the GPU Profiling System.

    Usage:
        builder = SystemBuilder()
        builder.with_state_dir(".state")
        builder.with_permission_mode("default")
        builder.with_max_tokens(8000)
        ...
        agent_loop = builder.build_agent_loop(session)
    """

    def __init__(self) -> None:
        self._state_dir: str = ".state"
        self._permission_mode: PermissionMode = PermissionMode.DEFAULT
        self._max_tokens: int = 8000
        self._max_turns: int = 20
        self._rule_dir: str | None = None
        self._no_docker: bool = False
        self._sandbox: Any = None
        self._persister: StatePersister | None = None

    def with_state_dir(self, state_dir: str) -> SystemBuilder:
        self._state_dir = state_dir
        return self

    def with_permission_mode(self, mode: str | PermissionMode) -> SystemBuilder:
        if isinstance(mode, PermissionMode):
            self._permission_mode = mode
        else:
            self._permission_mode = _map_permission_mode(mode)
        return self

    def with_max_tokens(self, max_tokens: int) -> SystemBuilder:
        self._max_tokens = max_tokens
        return self

    def with_max_turns(self, max_turns: int) -> SystemBuilder:
        self._max_turns = max_turns
        return self

    def with_rule_dir(self, rule_dir: str | None) -> SystemBuilder:
        self._rule_dir = rule_dir
        return self

    def with_no_docker(self, no_docker: bool) -> SystemBuilder:
        self._no_docker = no_docker
        return self

    def with_sandbox(self, sandbox: Any) -> SystemBuilder:
        self._sandbox = sandbox
        return self

    @property
    def persister(self) -> StatePersister:
        if self._persister is None:
            self._persister = StatePersister(log_dir=self._state_dir)
        return self._persister

    @property
    def sandbox(self) -> Any:
        if self._sandbox is None:
            self._sandbox = _build_sandbox(self._no_docker)
        return self._sandbox

    def build_context_manager(self) -> ContextManager:
        return ContextManager(max_tokens=self._max_tokens)

    def build_control_plane(self) -> ControlPlane:
        return ControlPlane(rule_dir=self._rule_dir)

    def build_tool_registry(self) -> ToolRegistry:
        from src.domain.tool_contract import build_standard_registry
        return build_standard_registry()

    def build_agent_loop(self, session: SessionState) -> AgentLoop:
        return AgentLoop(
            session=session,
            context_manager=self.build_context_manager(),
            control_plane=self.build_control_plane(),
            tool_registry=self.build_tool_registry(),
            max_turns=self._max_turns,
            state_dir=self._state_dir,
            permission_mode=self._permission_mode,
        )

    def build_tool_runner(self, registry: ToolRegistry | None = None):
        from src.application.approval_queue import ApprovalQueue
        from src.application.tool_runner import ToolRunner
        from src.domain.permission import PermissionChecker
        from src.domain.schema_validator import SchemaValidator

        registry = registry or self.build_tool_registry()
        handlers = _build_tool_handlers(self.sandbox)
        approval_queue = ApprovalQueue(
            state_dir=self._state_dir,
            persister=self.persister,
        )
        return ToolRunner(
            registry=registry,
            tool_handlers=handlers,
            approval_queue=approval_queue,
            permission_checker=PermissionChecker(mode=self._permission_mode),
            persister=self.persister,
            validator=SchemaValidator(),
        )

    def build_pipeline(self, session: SessionState):
        from src.domain.pipeline import Pipeline

        perm_mode = self._permission_mode
        agents = SubAgentFactory.create_all(
            max_tokens=self._max_tokens,
            state_dir=self._state_dir,
            permission_mode=perm_mode,
            sandbox=self.sandbox,
        )

        pipeline = Pipeline.build_default(
            planner=agents["planner"],
            code_gen=agents["code_gen"],
            metric_analysis=agents["metric_analysis"],
            verification=agents["verification"],
            state_dir=self._state_dir,
            sandbox=self.sandbox,
            tool_handlers=_build_tool_handlers(self.sandbox),
            max_turns_per_stage=50,
            handoff_validator=_build_handoff_validator(),
            circuit_breaker=_build_circuit_breaker(),
        )

        _wire_all_subagents(**agents)
        return pipeline


class SubAgentFactory:
    """Factory for creating sub-agents with P2 isolation.

    Each agent gets its own ContextManager and a restricted
    ToolRegistry containing only the tools for its role.
    """

    _ROLE_TOOLS = {
        "planner": {"read_file", "write_file"},
        "code_gen": {"compile_cuda", "execute_binary", "write_file", "read_file"},
        "metric_analysis": {"run_ncu", "read_file"},
        "verification": {"read_file"},
    }

    @classmethod
    def create_all(
        cls,
        max_tokens: int = 8000,
        state_dir: str = ".state",
        permission_mode: PermissionMode = PermissionMode.DEFAULT,
        sandbox: Any = None,
    ) -> dict[str, Any]:
        from src.application.subagents.codegen import CodeGenAgent
        from src.application.subagents.metric_analysis import MetricAnalysisAgent
        from src.application.subagents.planner import PlannerAgent
        from src.application.subagents.verification import VerificationAgent
        from src.domain.tool_contract import build_agent_registry

        planner_reg = build_agent_registry(cls._ROLE_TOOLS["planner"])
        codegen_reg = build_agent_registry(cls._ROLE_TOOLS["code_gen"])
        metric_reg = build_agent_registry(cls._ROLE_TOOLS["metric_analysis"])
        verification_reg = build_agent_registry(cls._ROLE_TOOLS["verification"])

        planner = PlannerAgent(
            context_manager=ContextManager(max_tokens=max_tokens),
            tool_registry=planner_reg,
            state_dir=state_dir,
            permission_mode=permission_mode,
        )

        code_gen = CodeGenAgent(
            context_manager=ContextManager(max_tokens=max_tokens),
            tool_registry=codegen_reg,
            state_dir=state_dir,
            permission_mode=permission_mode,
            sandbox=sandbox,
        )

        metric_analysis = MetricAnalysisAgent(
            context_manager=ContextManager(max_tokens=max_tokens),
            tool_registry=metric_reg,
            state_dir=state_dir,
            permission_mode=permission_mode,
        )

        verification = VerificationAgent(
            tool_registry=verification_reg,
            state_dir=state_dir,
            permission_mode=permission_mode,
            max_tokens=max_tokens,
        )

        return {
            "planner": planner,
            "code_gen": code_gen,
            "metric_analysis": metric_analysis,
            "verification": verification,
        }


def _map_permission_mode(mode_str: str) -> PermissionMode:
    return {
        "default": PermissionMode.DEFAULT,
        "conservative": PermissionMode.CONSERVATIVE,
        "relaxed": PermissionMode.RELAXED,
        "high_autonomy": PermissionMode.HIGH_AUTONOMY,
    }[mode_str]


def _build_sandbox(no_docker: bool):
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


def _build_tool_handlers(sandbox):
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


def _build_handoff_validator():
    from src.application.handoff_validation import HandoffValidator
    return HandoffValidator()


def _build_circuit_breaker():
    from src.application.circuit_breaker import CircuitBreaker
    return CircuitBreaker(
        degradation_threshold=3,
        min_quality_threshold=0.3,
    )


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


def try_wire_model_caller(agent_loop) -> bool:
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
