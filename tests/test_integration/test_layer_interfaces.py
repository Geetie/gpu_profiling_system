"""阶段 1：模块接口集成测试。

验证各层之间的接口是否匹配，没有调用断裂。
"""
import importlib
import os
import sys
import pytest


# ── 测试用例 1.1：工具契约验证 ──────────────────────────────────────


class TestToolContracts:
    """所有注册的工具必须有完整的契约定义。"""

    def test_build_standard_registry_returns_all_six_tools(self):
        from src.domain.tool_contract import build_standard_registry
        registry = build_standard_registry()
        expected = {
            "run_ncu", "compile_cuda", "execute_binary",
            "write_file", "read_file", "generate_microbenchmark",
        }
        assert set(registry.list_tools()) == expected

    def test_every_contract_has_complete_schema(self):
        from src.domain.tool_contract import build_standard_registry
        registry = build_standard_registry()
        for name in registry.list_tools():
            contract = registry.get(name)
            assert contract.input_schema, f"{name}: input_schema is empty"
            assert contract.output_schema, f"{name}: output_schema is empty"
            assert contract.permissions, f"{name}: permissions is empty"

    def test_no_unregistered_tools(self):
        """P1：工具定义能力边界 — 只有注册的工具才能被调用。"""
        from src.domain.tool_contract import ToolRegistry
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent_tool")

    def test_disabled_tools_eliminated_at_build_time(self):
        """P5：编译时消除 — 禁用工具不会出现在注册表中。"""
        from src.domain.tool_contract import build_standard_registry
        registry = build_standard_registry(disabled_tools={"compile_cuda"})
        assert "compile_cuda" not in registry.list_tools()
        assert len(registry.list_tools()) == 5

    def test_output_schema_matches_handler_return(self):
        """工具 handler 的返回值必须匹配 output_schema 中的字段。"""
        from src.domain.tool_contract import build_standard_registry
        from src.infrastructure.tools.run_ncu import run_ncu_handler

        registry = build_standard_registry()
        contract = registry.get("run_ncu")

        # Handler with missing executable should return error dict
        result = run_ncu_handler({"executable": "", "metrics": []})
        for field in contract.output_schema:
            assert field in result, (
                f"run_ncu handler missing output field: {field}"
            )


# ── 测试用例 1.2：分层依赖验证 ──────────────────────────────────────


class TestLayerDependencies:
    """验证四层架构依赖方向正确。"""

    def _get_imports(self, module_name):
        """获取模块的直接导入依赖。"""
        mod = importlib.import_module(module_name)
        imports = set()
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if hasattr(attr, "__module__"):
                imports.add(attr.__module__)
        # Also scan __dict__ for module-level references
        for key, val in vars(mod).items():
            if isinstance(val, type) and hasattr(val, "__module__"):
                imports.add(val.__module__)
        return imports

    def test_domain_layer_has_no_infrastructure_dependency(self):
        """领域层不能依赖基础设施层。

        例外：StatePersister (src.infrastructure.state_persist) 是 P6
        持久化的共享工具，不是沙箱/工具处理器，允许领域层使用。
        """
        domain_modules = [
            "src.domain.tool_contract",
            "src.domain.permission",
            "src.domain.subagent",
            "src.domain.pipeline",
            "src.domain.schema_validator",
        ]
        infra_prefix = "src.infrastructure"
        excluded = {"src.infrastructure.state_persist"}
        for mod_name in domain_modules:
            mod = importlib.import_module(mod_name)
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if hasattr(attr, "__module__"):
                    dep = attr.__module__
                    if dep in excluded:
                        continue
                    assert not dep.startswith(infra_prefix), (
                        f"领域层 {mod_name} 违规依赖基础设施层: {dep}"
                    )

    def test_application_layer_imports_infrastructure_only_indirectly(self):
        """应用层通过领域层间接使用基础设施，不直接执行副作用。

        注意：subagent 模块可以 import sandbox（代码生成需要沙箱），
        但不应该直接 import 工具 handler 等基础设施执行代码。
        """
        app_modules = [
            "src.application.agent_loop",
            "src.application.context",
            "src.application.control_plane",
            "src.application.session",
            "src.application.approval_queue",
            "src.application.tool_runner",
        ]
        infra_handlers = {
            "src.infrastructure.tools.run_ncu",
            "src.infrastructure.tools.compile_cuda",
            "src.infrastructure.tools.execute_binary",
            "src.infrastructure.tools.file_tools",
            "src.infrastructure.tools.microbenchmark",
        }
        for mod_name in app_modules:
            mod = importlib.import_module(mod_name)
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if hasattr(attr, "__module__"):
                    dep = attr.__module__
                    assert dep not in infra_handlers, (
                        f"应用层 {mod_name} 直接依赖工具 handler: {dep}"
                    )

    def test_presentation_layer_has_no_execution_dependency(self):
        """呈现层不能直接调用基础设施层（不能执行命令）。"""
        pres_modules = [
            "src.presentation.terminal_ui",
            "src.presentation.diff_renderer",
            "src.presentation.progress",
            "src.presentation.permission_prompt",
            "src.presentation.result_display",
        ]
        infra_prefix = "src.infrastructure"
        for mod_name in pres_modules:
            mod = importlib.import_module(mod_name)
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if hasattr(attr, "__module__"):
                    dep = attr.__module__
                    assert not dep.startswith(infra_prefix), (
                        f"呈现层 {mod_name} 违规依赖基础设施层: {dep}"
                    )


# ── 测试用例 1.3：机械不变式验证 ────────────────────────────────────


class TestMechanicalInvariants:
    """验证底层安全机制在集成后依然生效。"""

    def test_m1_read_before_write_enforced(self, tmp_path):
        """M1：未读取过的文件不能被写入。"""
        from src.infrastructure.file_ops import FileOperations
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        f = sandbox / "target.txt"
        f.write_text("existing")

        ops = FileOperations(sandbox_root=str(sandbox))
        with pytest.raises(PermissionError, match="M1 violation"):
            ops.write(str(f), "modified without read")

    def test_m1_read_then_write_succeeds(self, tmp_path):
        """M1：先读后写应该成功。"""
        from src.infrastructure.file_ops import FileOperations
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        f = sandbox / "target.txt"
        f.write_text("original")

        ops = FileOperations(sandbox_root=str(sandbox))
        content = ops.read(str(f))
        assert content == "original"
        ops.write(str(f), "modified")
        assert f.read_text() == "modified"

    def test_m1_create_bypass_for_generation_tools(self, tmp_path):
        """M1 例外：生成工具创建新文件应被允许并记录。"""
        from src.infrastructure.file_ops import FileOperations
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        f = sandbox / "new_file.cu"

        ops = FileOperations(sandbox_root=str(sandbox))
        # File doesn't exist yet — create should work
        ops.create(str(f), "new content")
        assert f.read_text() == "new content"

    def test_m4_anti_loop_terminates_after_3_failures(self, tmp_path):
        """M4：同一失败模式重复 3 次应强制终止。"""
        from src.domain.permission import InvariantTracker
        tracker = InvariantTracker()
        pattern = "tool_error:run_ncu"
        assert tracker.should_terminate(pattern) is False
        tracker.record_failure(pattern)
        assert tracker.should_terminate(pattern) is False
        tracker.record_failure(pattern)
        assert tracker.should_terminate(pattern) is False
        tracker.record_failure(pattern)
        assert tracker.should_terminate(pattern) is True

    def test_p6_state_persisted_to_disk(self, tmp_path):
        """P6：状态必须落盘。"""
        from src.infrastructure.state_persist import StatePersister
        persister = StatePersister(log_dir=str(tmp_path))
        persister.log_tool_execution("test_tool", {"input": "data"}, "success")

        history = persister.load_history()
        assert len(history) >= 1
        assert history[0]["tool_name"] == "test_tool"

    def test_approval_decision_persisted(self, tmp_path):
        """P6：审批决策必须落盘。"""
        from src.application.approval_queue import ApprovalQueue
        from src.domain.permission import PermissionMode
        from src.infrastructure.state_persist import StatePersister

        persister = StatePersister(log_dir=str(tmp_path))
        queue = ApprovalQueue(state_dir=str(tmp_path), persister=persister)

        request = queue.submit(
            tool_name="test_tool",
            arguments={},
            permissions=["file:write"],
            mode=PermissionMode.DEFAULT,
        )
        queue.respond(request.id, approved=True)

        history = persister.load_history()
        decisions = [h for h in history if h.get("action") == "approval_decision"]
        assert len(decisions) >= 1
        assert decisions[0]["details"]["status"] == "approved"
