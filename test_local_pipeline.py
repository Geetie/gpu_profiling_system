#!/usr/bin/env python3
"""本地 Pipeline 测试 — 验证重构后的多 Agent 框架能正常生成代码。

测试目标：
1. Pipeline 能正常启动
2. Planner 能分解任务
3. CodeGen 能生成 CUDA 代码文件
4. 代码文件实际写入磁盘

使用方法：
    python test_local_pipeline.py

环境要求：
    - 需要配置 LLM API (api_config.json)
    - 不需要 GPU（使用 LocalSandbox）
    - 不需要 Docker
"""
import json
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.application.system_builder import SystemBuilder
from src.application.session import SessionState
from src.domain.permission import PermissionMode
from src.domain.subagent import SubAgentStatus


def test_pipeline_generates_code():
    """验证 Pipeline 能让 CodeGen 生成 CUDA 代码文件。"""
    print("=" * 60)
    print("本地 Pipeline 测试 — 代码生成验证")
    print("=" * 60)
    
    # 创建临时工作目录
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n工作目录：{tmpdir}")
        
        # 创建 target_spec.json
        target_spec = {
            "goal": "Profile GPU L2 cache size",
            "targets": ["l2_cache_size_kb"],
        }
        target_spec_path = os.path.join(tmpdir, "target_spec.json")
        with open(target_spec_path, "w") as f:
            json.dump(target_spec, f, indent=2)
        print(f"[OK] 创建 target_spec: {target_spec_path}")
        
        # 配置 SystemBuilder
        builder = (
            SystemBuilder()
            .with_state_dir(os.path.join(tmpdir, ".state"))
            .with_permission_mode(PermissionMode.HIGH_AUTONOMY)
            .with_max_tokens(8000)
            .with_max_turns(50)
            .with_no_docker(True)
        )
        
        # 创建 session
        session = SessionState(
            session_id="test_local_pipeline",
            goal="Profile GPU L2 cache size"
        )
        
        # 构建 Pipeline 和 AgentLoop
        print("\n构建 Pipeline 和 AgentLoop...")
        pipeline = builder.build_pipeline(session)
        agent_loop = builder.build_agent_loop(session)
        
        # 配置 LLM caller（如果已配置 API）
        try:
            from src.application.system_builder import try_wire_model_caller
            if try_wire_model_caller(agent_loop):
                print("[OK] LLM API 已配置")
            else:
                print("[WARN] 未配置 LLM API，将使用 Mock 模式")
                # 使用 Mock model caller 进行测试
                def mock_caller(messages):
                    return "Task completed successfully."
                agent_loop.set_model_caller(mock_caller)
        except Exception as e:
            print(f"[WARN] LLM 配置失败：{e}")
            def mock_caller(messages):
                return "Task completed successfully."
            agent_loop.set_model_caller(mock_caller)
        
        # 执行 Pipeline
        print("\n开始执行 Pipeline...")
        print("Planner → CodeGen → MetricAnalysis → Verification")
        print("-" * 60)
        
        try:
            result = agent_loop.run_pipeline(pipeline, target_spec)
            
            print("-" * 60)
            print("\n[OK] Pipeline 执行完成")
            print(f"状态：{result.status.value}")
            print(f"Agent 角色：{result.agent_role.value}")
            
            if result.data:
                print(f"\n结果数据:")
                for key, value in list(result.data.items())[:5]:
                    if isinstance(value, str) and len(value) > 200:
                        print(f"  {key}: {value[:200]}...")
                    else:
                        print(f"  {key}: {value}")
            
            if result.error:
                print(f"\n错误信息：{result.error}")
            
            # 检查 CodeGen 是否生成了代码文件
            print("\n" + "=" * 60)
            print("检查生成的代码文件...")
            print("=" * 60)
            
            # 查找所有 .cu 文件
            cu_files = list(Path(tmpdir).rglob("*.cu"))
            if cu_files:
                print(f"[OK] 找到 {len(cu_files)} 个 CUDA 源文件:")
                for cu_file in cu_files:
                    print(f"  - {cu_file.relative_to(tmpdir)}")
                    # 显示文件大小和前几行
                    try:
                        size = cu_file.stat().st_size
                        with open(cu_file, "r", encoding="utf-8") as f:
                            first_lines = "".join([f.readline() for _ in range(3)])
                        print(f"    大小：{size} 字节")
                        print(f"    开头：{first_lines.strip()[:80]}...")
                    except Exception as e:
                        print(f"    读取失败：{e}")
                
                print("\n[OK] 测试通过：Pipeline 成功生成 CUDA 代码文件")
                return True
            else:
                print("\n⚠ 未找到 .cu 文件")
                
                # 检查 agent 日志
                state_dir = os.path.join(tmpdir, ".state")
                if os.path.isdir(state_dir):
                    agent_logs = list(Path(state_dir).rglob("agent_*_log.jsonl"))
                    if agent_logs:
                        print(f"\n找到 {len(agent_logs)} 个 Agent 日志文件:")
                        for log_file in agent_logs:
                            print(f"  - {log_file.relative_to(state_dir)}")
                            # 读取最后一条记录
                            try:
                                with open(log_file, "r", encoding="utf-8") as f:
                                    lines = f.readlines()
                                    if lines:
                                        last_entry = json.loads(lines[-1])
                                        print(f"    最后状态：{last_entry.get('details', {}).get('status', 'unknown')}")
                            except Exception as e:
                                print(f"    读取失败：{e}")
                
                print("\n⚠ 测试未完成：未生成 CUDA 代码文件")
                return False
                
        except Exception as e:
            print(f"\n✗ Pipeline 执行失败：{e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    success = test_pipeline_generates_code()
    sys.exit(0 if success else 1)
