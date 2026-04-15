"""阶段 1 补充：main.py 入口点集成测试。

验证 CLI 入口能正确组装所有层。
"""
import os
import pytest


class TestMainEntryPoint:
    """验证 main.py 入口点能正确接线所有组件。"""

    def test_main_builds_all_components(self, tmp_path):
        """SystemBuilder 应创建并接线所有层。"""
        from src.application.system_builder import SystemBuilder
        from src.application.session import SessionState
        from src.domain.permission import PermissionMode

        os.chdir(str(tmp_path))
        builder = (
            SystemBuilder()
            .with_state_dir(str(tmp_path))
            .with_permission_mode(PermissionMode.DEFAULT)
            .with_max_tokens(4000)
            .with_max_turns(20)
            .with_no_docker(True)
        )

        session = SessionState(session_id="test", goal="verify wiring")
        agent_loop = builder.build_agent_loop(session)

        assert agent_loop is not None
        assert agent_loop.tool_registry is not None

    def test_main_cli_parses_goal(self):
        """CLI 应能解析 goal 参数。"""
        from src.main import main

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
        from src.application.system_builder import SystemBuilder
        from src.infrastructure.sandbox import LocalSandbox

        builder = SystemBuilder().with_no_docker(True)
        sandbox = builder.sandbox
        assert isinstance(sandbox, LocalSandbox)

    def test_main_pipeline_mode_builds_pipeline(self, tmp_path):
        """--pipeline 模式应能构建 Pipeline。"""
        from src.application.system_builder import SystemBuilder
        from src.application.session import SessionState
        from src.domain.permission import PermissionMode

        os.chdir(str(tmp_path))
        builder = (
            SystemBuilder()
            .with_state_dir(str(tmp_path))
            .with_permission_mode(PermissionMode.DEFAULT)
            .with_max_tokens(4000)
            .with_no_docker(True)
        )

        session = SessionState(session_id="pipeline_test", goal="test")
        pipeline = builder.build_pipeline(session)

        assert pipeline is not None
        assert len(pipeline._stages) == 4
