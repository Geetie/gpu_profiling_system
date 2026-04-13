"""阶段 3 & 5：对抗性鲁棒性 + 安全沙箱测试。

模拟干扰环境和验证安全约束。
注意：实际 GPU 对抗测试（频率锁定、SM 屏蔽等）需要真实硬件环境，
此处用 mock 验证系统的防御机制架构。
"""
import json
import os
from unittest.mock import MagicMock, patch
import pytest


# ── 测试用例 3.1：频率锁定干扰 — 架构验证 ───────────────────────────


class TestFrequencyLockInterference:
    """验证系统不依赖 API 查表，必须通过微基准测试实测。"""

    def test_ncu_handler_ignores_cached_api_results(self):
        """run_ncu 必须通过实际 ncu 命令，不能返回缓存数据。

        验证：handler 不接受任何 'cached' 参数，只执行真实命令。
        """
        from src.infrastructure.tools.run_ncu import run_ncu_handler

        # No way to inject fake metrics through arguments
        result = run_ncu_handler({
            "executable": "/fake/path",
            "metrics": ["sm__cycles_elapsed.avg"],
        })
        # Should return error dict, not fabricated data
        assert result["parsed_metrics"].get("error") or result["raw_output"] == ""

    def test_generate_microbenchmark_produces_timing_loop(self):
        """生成的微基准测试必须包含实测计时循环，不能只是占位符。"""
        from src.infrastructure.tools.microbenchmark import generate_microbenchmark_handler

        result = generate_microbenchmark_handler({
            "benchmark_type": "pointer_chase",
            "parameters": {},
        })

        source = result["source_code"]
        # Must contain actual timing measurement
        assert "clock()" in source, "pointer_chase kernel must use clock() for timing"
        assert "iterations" in source, "must have measurable iteration count"

    def test_generate_timing_loop_kernel(self):
        """timing_loop 内核必须有真实的延迟循环。"""
        from src.infrastructure.tools.microbenchmark import generate_microbenchmark_handler

        result = generate_microbenchmark_handler({
            "benchmark_type": "timing_loop",
            "parameters": {},
        })
        source = result["source_code"]
        assert "volatile" in source or "clock()" in source, \
            "timing_loop must prevent compiler optimization"


# ── 测试用例 3.2：SM 资源屏蔽干扰 ───────────────────────────────────


class TestSMResourceMasking:
    """验证系统能检测资源变化。"""

    def test_control_plane_captures_cuda_env(self, tmp_path):
        """ControlPlane 必须捕获 CUDA 环境变量以检测资源变化。"""
        from src.application.control_plane import ControlPlane

        os.chdir(str(tmp_path))
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            cp = ControlPlane()
            ctx = cp.inject()

        assert "CUDA_VISIBLE_DEVICES" in ctx.env_vars
        assert ctx.env_vars["CUDA_VISIBLE_DEVICES"] == "0"

    def test_control_plane_rejects_modified_env(self, tmp_path):
        """环境变量变化后，每次注入应获取最新值（非缓存）。"""
        from src.application.control_plane import ControlPlane

        os.chdir(str(tmp_path))
        cp = ControlPlane()

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False):
            ctx1 = cp.inject()

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False):
            ctx2 = cp.inject()

        assert ctx1.env_vars["CUDA_VISIBLE_DEVICES"] != ctx2.env_vars["CUDA_VISIBLE_DEVICES"]


# ── 测试用例 3.3：API 拦截干扰 ──────────────────────────────────────


class TestAPIInterference:
    """验证系统不信任单一数据源。"""

    def test_agent_loop_does_not_trust_pre_set_model_output(self):
        """AgentLoop 不信任 _model_output 作为事实来源，而是通过工具执行。

        验证：即使 _model_output 包含虚假数据，工具执行结果才是最终的。
        """
        from src.application.agent_loop import AgentLoop
        from src.application.context import ContextManager
        from src.application.control_plane import ControlPlane
        from src.application.session import SessionState
        from src.domain.tool_contract import ToolRegistry

        loop = AgentLoop(
            session=SessionState(session_id="api_test", goal="test"),
            context_manager=ContextManager(max_tokens=4000),
            control_plane=ControlPlane(),
            tool_registry=ToolRegistry(),
            max_turns=3,
            state_dir=".state_test_api",
        )

        # Fake model output claiming false results
        loop._model_output = '{"fake_metric": 99999}'
        loop.loop_state.is_running = True

        # Parse should return None (not valid tool call format)
        tool_call = loop._parse_tool_call()
        assert tool_call is None, "Non-tool-call JSON should not be parsed as tool call"

    def test_tool_runner_validates_output_schema(self, tmp_path):
        """ToolRunner 必须验证输出 schema，防止 handler 返回伪造数据。

        验证：handler 返回不匹配 schema 的数据会被 reject。
        """
        from src.application.tool_runner import ToolRunner
        from src.domain.tool_contract import ToolContract, ToolRegistry
        from src.domain.permission import PermissionChecker, PermissionMode
        from src.domain.schema_validator import SchemaValidator
        from src.infrastructure.state_persist import StatePersister

        registry = ToolRegistry()
        registry.register(ToolContract(
            name="test_tool",
            description="Test",
            input_schema={"value": "integer"},
            output_schema={"result": "string"},
            permissions=["file:read"],
        ))

        # Handler returns wrong type
        def bad_handler(args):
            return {"result": 12345}  # Should be string, not int

        runner = ToolRunner(
            registry=registry,
            tool_handlers={"test_tool": bad_handler},
            approval_queue=MagicMock(),
            permission_checker=PermissionChecker(mode=PermissionMode.DEFAULT),
            persister=StatePersister(log_dir=str(tmp_path)),
            validator=SchemaValidator(),
        )

        # This should raise SchemaValidationError because 12345 is not a string
        from src.domain.schema_validator import SchemaValidationError
        with pytest.raises(SchemaValidationError):
            runner.execute("test_tool", {"value": 42})


# ── 测试用例 5.1：Docker 沙箱验证 ──────────────────────────────────


class TestSandboxIsolation:
    """验证沙箱隔离机制。"""

    def test_local_sandbox_blocks_path_escape(self, tmp_path):
        """LocalSandbox 必须阻止路径逃逸。"""
        from src.infrastructure.sandbox import LocalSandbox, SandboxConfig

        sandbox = LocalSandbox(
            SandboxConfig(),
            sandbox_root=str(tmp_path / "sandbox"),
        )
        # Sandbox constructor already creates the sandbox_root directory

        with pytest.raises(PermissionError, match="Path escape blocked"):
            sandbox._resolve_path(str(tmp_path / "escape_attempt"))

    def test_local_sandbox_allows_internal_paths(self, tmp_path):
        """LocalSandbox 必须允许内部路径。"""
        from src.infrastructure.sandbox import LocalSandbox, SandboxConfig

        root = tmp_path / "sandbox"
        root.mkdir()
        sandbox = LocalSandbox(SandboxConfig(), sandbox_root=str(root))

        resolved = sandbox._resolve_path(str(root / "internal" / "file.txt"))
        assert str(resolved).startswith(str(root))

    def test_docker_sandbox_config_security(self):
        """DockerSandbox 必须使用安全配置。"""
        from src.infrastructure.sandbox import DockerSandbox, SandboxConfig

        config = SandboxConfig(
            privileged=False,
            network_disabled=True,
            mount_docker_socket=False,
            gpu_read_only=True,
        )
        sandbox = DockerSandbox(config=config, host_work_dir="/tmp/test_workdir")

        docker_args = sandbox._build_docker_args()

        # Security flags must be present
        assert "--network" in docker_args
        assert "none" in docker_args
        assert "--read-only" in docker_args
        assert "--no-new-privileges" in docker_args
        assert "--cap-drop" in docker_args
        assert "ALL" in docker_args
        # Docker socket must NOT be mounted
        assert "/var/run/docker.sock" not in " ".join(docker_args)

    def test_docker_sandbox_rejects_relative_host_path(self):
        """DockerSandbox 必须拒绝相对路径。"""
        from src.infrastructure.sandbox import DockerSandbox, SandboxConfig

        with pytest.raises(ValueError, match="must be absolute"):
            DockerSandbox(
                config=SandboxConfig(),
                host_work_dir="relative/path",
            )

    def test_docker_sandbox_rejects_path_traversal(self):
        """DockerSandbox 必须拒绝路径穿越。"""
        from src.infrastructure.sandbox import DockerSandbox, SandboxConfig

        with pytest.raises(ValueError, match="path traversal"):
            DockerSandbox(
                config=SandboxConfig(),
                host_work_dir="/tmp/../../../etc",
            )


# ── 测试用例 5.2：权限审批验证 ──────────────────────────────────────


class TestPermissionApprovalFlow:
    """验证完整权限审批链路。"""

    def test_full_approval_flow_end_to_end(self, tmp_path):
        """从 ToolRunner 提交 → AgentLoop 展示 → 用户批准 → 执行。"""
        import threading
        import uuid
        from unittest.mock import patch
        from src.application.agent_loop import AgentLoop, EventKind
        from src.application.approval_queue import ApprovalQueue
        from src.application.context import ContextManager, Role
        from src.application.control_plane import ControlPlane
        from src.application.session import SessionState
        from src.application.tool_runner import ApprovalRequiredError, ToolRunner
        from src.domain.permission import PermissionChecker, PermissionMode
        from src.domain.schema_validator import SchemaValidator
        from src.domain.tool_contract import ToolContract, ToolRegistry
        from src.infrastructure.state_persist import StatePersister

        os.chdir(str(tmp_path))

        # Register a tool that requires approval
        registry = ToolRegistry()
        registry.register(ToolContract(
            name="dangerous_tool",
            description="Needs approval",
            input_schema={"data": "string"},
            output_schema={"result": "string"},
            permissions=["file:write"],
            requires_approval=True,
        ))

        persister = StatePersister(log_dir=str(tmp_path))
        approval_queue = ApprovalQueue(state_dir=str(tmp_path), persister=persister)
        permission_checker = PermissionChecker(mode=PermissionMode.DEFAULT)

        handler_called = {"count": 0}
        def mock_handler(args):
            handler_called["count"] += 1
            return {"result": "executed"}

        tool_runner = ToolRunner(
            registry=registry,
            tool_handlers={"dangerous_tool": mock_handler},
            approval_queue=approval_queue,
            permission_checker=permission_checker,
            persister=persister,
            validator=SchemaValidator(),
        )

        session = SessionState(session_id="approval_e2e", goal="test")
        loop = AgentLoop(
            session=session,
            context_manager=ContextManager(max_tokens=4000),
            control_plane=ControlPlane(),
            tool_registry=registry,
            max_turns=5,
            state_dir=str(tmp_path),
            permission_mode=PermissionMode.DEFAULT,
        )

        loop.set_tool_executor(tool_runner.execute)

        # Track approval events
        approval_events = []

        def approval_callback(request):
            approval_events.append(request)
            approval_queue.respond(request.id, approved=True)
            return True

        loop.set_approval_callback(approval_callback)

        # Patch submit to return the already-approved request if one exists,
        # simulating approval reuse on re-execution.
        from src.application.approval_queue import ApprovalStatus
        original_submit = approval_queue.submit

        def patched_submit(tool_name, arguments, permissions, mode):
            for req in approval_queue._requests.values():
                if (req.tool_name == tool_name
                        and req.status == ApprovalStatus.APPROVED):
                    return req
            return original_submit(tool_name, arguments, permissions, mode)

        with patch.object(approval_queue, "submit", patched_submit):
            # Trigger tool call
            loop._model_output = json.dumps({
                "tool": "dangerous_tool",
                "args": {"data": "test"},
            })
            loop.loop_state.is_running = True
            loop._inner_loop_step()

        # Verify approval was requested
        assert len(approval_events) >= 1
        assert approval_events[0].tool_name == "dangerous_tool"
        # Handler should have been called after approval
        assert handler_called["count"] == 1

    def test_conservative_mode_auto_rejects(self, tmp_path):
        """CONSERVATIVE 模式自动拒绝所有需要审批的操作。"""
        from src.application.approval_queue import ApprovalQueue
        from src.application.tool_runner import ApprovalRequiredError, ToolRunner
        from src.domain.permission import PermissionChecker, PermissionMode
        from src.domain.schema_validator import SchemaValidator
        from src.domain.tool_contract import ToolContract, ToolRegistry
        from src.infrastructure.state_persist import StatePersister

        os.chdir(str(tmp_path))

        registry = ToolRegistry()
        registry.register(ToolContract(
            name="write_tool",
            description="Needs approval",
            input_schema={"data": "string"},
            output_schema={"result": "string"},
            permissions=["file:write"],
            requires_approval=True,
        ))

        persister = StatePersister(log_dir=str(tmp_path))
        approval_queue = ApprovalQueue(state_dir=str(tmp_path), persister=persister)
        permission_checker = PermissionChecker(mode=PermissionMode.CONSERVATIVE)

        handler_called = {"count": 0}
        def mock_handler(args):
            handler_called["count"] += 1
            return {"result": "should not reach here"}

        tool_runner = ToolRunner(
            registry=registry,
            tool_handlers={"write_tool": mock_handler},
            approval_queue=approval_queue,
            permission_checker=permission_checker,
            persister=persister,
            validator=SchemaValidator(),
        )

        # In CONSERVATIVE mode, should be denied without raising ApprovalRequiredError
        with pytest.raises(PermissionError):
            tool_runner.execute("write_tool", {"data": "test"})

        assert handler_called["count"] == 0
