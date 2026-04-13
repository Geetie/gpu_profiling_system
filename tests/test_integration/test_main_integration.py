"""阶段 1 补充：main.py 入口点集成测试。

验证 CLI 入口能正确组装所有层。
"""
import os
import pytest


class TestMainEntryPoint:
    """验证 main.py 入口点能正确接线所有组件。"""

    def test_main_builds_all_components(self, tmp_path):
        """_build_loop_components 应创建并接线所有层。"""
        from src.main import _build_loop_components, _build_sandbox, _map_permission_mode
        from src.application.session import SessionState
        from argparse import Namespace

        os.chdir(str(tmp_path))
        sandbox = _build_sandbox(no_docker=True)

        args = Namespace(
            mode="default",
            state_dir=str(tmp_path),
            max_tokens=4000,
            max_turns=20,
            rule_dir=None,
        )

        session = SessionState(session_id="test", goal="verify wiring")
        agent_loop, ui, tool_runner = _build_loop_components(args, session, sandbox)

        # Verify all layers connected
        assert agent_loop._tool_executor is not None
        assert agent_loop._approval_callback is not None
        assert len(agent_loop._event_handlers) == 1

    def test_main_cli_parses_goal(self):
        """CLI 应能解析 goal 参数。"""
        from src.main import main

        # Should not raise — just parse and validate
        # We test with --help to avoid full execution
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0

    def test_main_cli_requires_goal_or_resume(self):
        """CLI 必须提供 goal 或 --resume。"""
        from src.main import main

        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 2

    def test_main_no_docker_mode(self, tmp_path):
        """--no-docker 应强制使用 LocalSandbox。"""
        from src.main import _build_sandbox
        from src.infrastructure.sandbox import LocalSandbox

        sandbox = _build_sandbox(no_docker=True)
        assert isinstance(sandbox, LocalSandbox)

    def test_main_pipeline_mode_builds_pipeline(self, tmp_path):
        """--pipeline 模式应能构建 Pipeline。"""
        from src.main import _build_pipeline, _build_sandbox
        from argparse import Namespace
        from src.domain.tool_contract import build_standard_registry
        from src.application.session import SessionState

        os.chdir(str(tmp_path))
        sandbox = _build_sandbox(no_docker=True)
        registry = build_standard_registry()
        session = SessionState(session_id="pipeline_test", goal="test")

        args = Namespace(
            mode="default",
            state_dir=str(tmp_path),
            max_tokens=4000,
        )

        pipeline = _build_pipeline(args, sandbox, registry, session)
        assert pipeline is not None
        assert len(pipeline._stages) == 4
