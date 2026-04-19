"""
仓库清理脚本 - 清理临时文件、测试结果和不必要的调试文件

运行此脚本将:
1. 删除 kaggle_results 中的编译产物和临时文件
2. 删除根目录的调试 JSON 文件
3. 删除临时测试文件
4. 保留核心日志文件用于错误分析

安全提示:
- 只会删除临时生成的文件，不会删除源代码和配置文件
- 核心日志文件 (execution.log, pipeline_log.jsonl 等) 会被保留
"""
import os
import shutil
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 需要删除的文件模式
FILES_TO_DELETE = {
    # kaggle_results 目录中的临时文件
    "kaggle_results/source.cu": "临时 CUDA 源文件",
    "kaggle_results/freq_probe": "编译产物（二进制文件）",
    "kaggle_results/freq_event_probe": "编译产物（二进制文件）",
    "kaggle_results/freq_event_timed": "编译产物（二进制文件）",
    "kaggle_results/probe_binary": "编译产物（二进制文件）",
    "kaggle_results/stream_event": "编译产物（二进制文件）",
    
    # 根目录的调试文件（保留 kaggle_results 中的）
    "debug_messages_*.json": "调试消息文件（大量冗余）",
}

# 需要删除的目录
DIRS_TO_DELETE = [
    # Python 缓存
    "__pycache__",
    
    # 临时测试目录
    "temp_tests",
    
    # 调试目录
    "debug",
]

# 保留的核心文件（即使匹配模式也不删除）
PRESERVE_FILES = [
    "execution.log",
    "pipeline_log.jsonl",
    "session_log.jsonl",
    "audit_report.md",
]

def delete_file(file_path: Path, reason: str) -> bool:
    """安全删除单个文件"""
    try:
        if file_path.exists():
            file_size = file_path.stat().st_size
            file_path.unlink()
            print(f"✅ 已删除：{file_path.relative_to(PROJECT_ROOT)}")
            print(f"   原因：{reason}")
            print(f"   大小：{file_size:,} 字节")
            return True
    except Exception as e:
        print(f"❌ 删除失败：{file_path.relative_to(PROJECT_ROOT)}")
        print(f"   错误：{e}")
    return False

def delete_directory(dir_path: Path) -> bool:
    """安全删除目录"""
    try:
        if dir_path.exists() and dir_path.is_dir():
            shutil.rmtree(dir_path)
            print(f"✅ 已删除目录：{dir_path.relative_to(PROJECT_ROOT)}")
            return True
    except Exception as e:
        print(f"❌ 删除目录失败：{dir_path.relative_to(PROJECT_ROOT)}")
        print(f"   错误：{e}")
    return False

def cleanup_kaggle_results():
    """清理 kaggle_results 目录"""
    kaggle_dir = PROJECT_ROOT / "kaggle_results"
    if not kaggle_dir.exists():
        print("ℹ️  kaggle_results 目录不存在，跳过")
        return
    
    print("\n" + "="*60)
    print("清理 kaggle_results 目录")
    print("="*60)
    
    deleted_count = 0
    preserved_count = 0
    
    # 删除编译产物和临时文件
    binary_files = [
        "source.cu",
        "freq_probe",
        "freq_event_probe",
        "freq_event_timed",
        "probe_binary",
        "stream_event",
        "results.json",
    ]
    
    # 删除多余的 cmd_*.log（保留一个即可）
    cmd_logs = list(kaggle_dir.glob("cmd_*.log"))
    if len(cmd_logs) > 1:
        # 只保留最新的 1 个
        cmd_logs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        for cmd_log in cmd_logs[1:]:
            delete_file(cmd_log, "冗余的命令日志")
    
    # 删除多余的 debug_messages（只保留 9msg_3tool）
    debug_files = list(kaggle_dir.glob("debug_messages_*.json"))
    for debug_file in debug_files:
        if debug_file.name != "debug_messages_longcat_9msg_3tool.json":
            delete_file(debug_file, "冗余的调试文件")
    
    for filename in binary_files:
        file_path = kaggle_dir / filename
        if file_path.exists() and file_path.name not in PRESERVE_FILES:
            if delete_file(file_path, "编译产物/临时文件"):
                deleted_count += 1
    
    # 保留核心日志文件
    for filename in PRESERVE_FILES:
        file_path = kaggle_dir / filename
        if file_path.exists():
            preserved_count += 1
            print(f"📌 保留核心文件：{filename}")
    
    print(f"\n✅ kaggle_results 清理完成")
    print(f"   删除文件数：{deleted_count}")
    print(f"   保留核心文件：{preserved_count}")

def cleanup_debug_files():
    """清理根目录的调试文件"""
    print("\n" + "="*60)
    print("清理根目录调试文件")
    print("="*60)
    
    deleted_count = 0
    
    # 删除 debug_messages_*.json 文件
    for file_path in PROJECT_ROOT.glob("debug_messages_*.json"):
        if file_path.is_file():
            file_size = file_path.stat().st_size
            file_path.unlink()
            print(f"✅ 已删除：{file_path.name}")
            print(f"   大小：{file_size:,} 字节")
            deleted_count += 1
    
    print(f"\n✅ 调试文件清理完成")
    print(f"   删除文件数：{deleted_count}")

def cleanup_temp_dirs():
    """清理临时目录"""
    print("\n" + "="*60)
    print("清理临时目录")
    print("="*60)
    
    deleted_count = 0
    
    for dir_name in DIRS_TO_DELETE:
        dir_path = PROJECT_ROOT / dir_name
        if delete_directory(dir_path):
            deleted_count += 1
    
    # 递归查找所有 __pycache__ 目录
    for pycache_dir in PROJECT_ROOT.rglob("__pycache__"):
        if delete_directory(pycache_dir):
            deleted_count += 1
    
    print(f"\n✅ 临时目录清理完成")
    print(f"   删除目录数：{deleted_count}")

def cleanup_test_output():
    """清理 test_output 目录（可选）"""
    test_output_dir = PROJECT_ROOT / "test_output"
    if not test_output_dir.exists():
        return
    
    print("\n" + "="*60)
    print("清理 test_output 目录")
    print("="*60)
    
    # 保留重要的测试结果
    preserve_patterns = [
        "*_summary.json",
        "*_report.json",
    ]
    
    # 删除临时日志
    deleted_count = 0
    for file_path in test_output_dir.iterdir():
        if file_path.suffix in [".log", ".jsonl"]:
            file_path.unlink()
            print(f"✅ 已删除：{file_path.name}")
            deleted_count += 1
    
    print(f"\n✅ test_output 清理完成")
    print(f"   删除文件数：{deleted_count}")

def show_cleanup_summary():
    """显示清理摘要"""
    print("\n" + "="*60)
    print("清理完成摘要")
    print("="*60)
    
    # 计算释放的空间
    print("\n📊 清理统计:")
    print(f"   ✅ 已删除 kaggle_results 中的编译产物")
    print(f"   ✅ 已删除根目录调试文件")
    print(f"   ✅ 已删除临时目录")
    print(f"   ✅ 已删除 test_output 临时文件")
    
    print("\n📌 保留的核心文件（kaggle_results）:")
    print(f"   - execution.log（执行日志）")
    print(f"   - pipeline_log.jsonl（Pipeline 状态）")
    print(f"   - session_log.jsonl（AgentLoop 状态）")
    print(f"   - audit_report.md（审计报告）")
    
    print("\n💡 提示:")
    print(f"   - 从 Kaggle 下载结果时，只需下载保留的核心文件")
    print(f"   - 编译产物和临时文件无需下载（占用空间大、无分析价值）")
    print(f"   - 详细的下载指南请查看：KAGGLE 文件下载指南.md")
    
    print("\n✨ 仓库已清理完成!")

def main():
    """主清理函数"""
    print("="*60)
    print("GPU Profiling System 仓库清理工具")
    print("="*60)
    print(f"\n项目根目录：{PROJECT_ROOT}\n")
    
    # 执行清理
    cleanup_kaggle_results()
    cleanup_debug_files()
    cleanup_temp_dirs()
    cleanup_test_output()
    
    # 显示摘要
    show_cleanup_summary()

if __name__ == "__main__":
    main()
