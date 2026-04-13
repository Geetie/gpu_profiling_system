#!/usr/bin/env python3
"""错误处理和日志记录模块"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class ErrorHandler:
    """错误处理器"""

    def __init__(self, log_dir: str = ".state"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # 设置日志
        self.setup_logging()

    def setup_logging(self):
        """设置日志记录"""
        log_file = self.log_dir / "error_log.jsonl"

        # 配置JSON日志处理器
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        self.logger = logging.getLogger('GPUProfiling')

        # 添加JSON文件处理器
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(file_handler)

    def log_error(self, error_type: str, error_message: str,
                  context: Dict[str, Any] = None, stack_trace: str = None):
        """记录错误"""
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {},
            "stack_trace": stack_trace,
            "python_version": sys.version,
            "platform": os.name
        }

        # 写入JSONL日志文件
        log_file = self.log_dir / "error_log.jsonl"
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(error_record, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Failed to write error log: {e}")

        # 也输出到标准错误
        print(f"ERROR [{error_type}]: {error_message}", file=sys.stderr)

    def log_operation(self, operation: str, status: str,
                     details: Dict[str, Any] = None):
        """记录操作"""
        operation_record = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "status": status,
            "details": details or {}
        }

        log_file = self.log_dir / "operation_log.jsonl"
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(operation_record, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Failed to write operation log: {e}")

    def handle_exception(self, exception: Exception, context: Dict[str, Any] = None):
        """处理异常"""
        error_type = type(exception).__name__
        error_message = str(exception)
        stack_trace = traceback.format_exc()

        self.log_error(error_type, error_message, context, stack_trace)

        # 根据异常类型提供建议
        suggestions = self.get_error_suggestions(error_type, error_message)
        if suggestions:
            print(f"建议: {suggestions}")

    def get_error_suggestions(self, error_type: str, error_message: str) -> str:
        """获取错误建议"""
        suggestions = {
            "FileNotFoundError": "检查文件路径是否正确，确保文件存在",
            "PermissionError": "检查文件权限，确保有读写权限",
            "ImportError": "检查模块是否已安装，PYTHONPATH是否正确",
            "subprocess.CalledProcessError": "检查命令参数和外部环境",
            "json.JSONDecodeError": "检查JSON文件格式是否正确",
            "KeyError": "检查配置文件中是否包含所需的键",
            "ValueError": "检查输入值是否在有效范围内",
            "TypeError": "检查参数类型是否正确"
        }

        return suggestions.get(error_type, "请检查相关配置和参数")

    def log_session_start(self, session_id: str, config: Dict[str, Any]):
        """记录会话开始"""
        self.log_operation("session_start", "started", {
            "session_id": session_id,
            "config": config
        })

    def log_session_end(self, session_id: str, success: bool, summary: Dict[str, Any]):
        """记录会话结束"""
        self.log_operation("session_end", "success" if success else "failed", {
            "session_id": session_id,
            "summary": summary
        })


def safe_file_operation(operation_func, file_path: str, *args, **kwargs):
    """安全的文件操作包装器"""
    error_handler = ErrorHandler()

    try:
        return operation_func(file_path, *args, **kwargs)
    except Exception as e:
        error_handler.handle_exception(e, {
            "file_path": file_path,
            "operation": operation_func.__name__,
            "args": args,
            "kwargs": kwargs
        })
        return None


def safe_subprocess_call(command: list, timeout: int = 300, **kwargs):
    """安全的子进程调用包装器"""
    import subprocess
    error_handler = ErrorHandler()

    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, **kwargs)
        error_handler.log_operation("subprocess_call", "completed", {
            "command": command,
            "returncode": result.returncode,
            "stdout_length": len(result.stdout),
            "stderr_length": len(result.stderr)
        })
        return result
    except Exception as e:
        error_handler.handle_exception(e, {
            "command": command,
            "timeout": timeout,
            "kwargs": kwargs
        })
        return None


# 全局错误处理器实例
error_handler = ErrorHandler()


def get_error_summary() -> Dict[str, Any]:
    """获取错误摘要"""
    log_file = Path(".state") / "error_log.jsonl"
    if not log_file.exists():
        return {"total_errors": 0, "error_types": {}}

    error_types = {}
    total_errors = 0

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    error_type = record.get("error_type", "Unknown")
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                    total_errors += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Failed to read error log: {e}")

    return {
        "total_errors": total_errors,
        "error_types": error_types,
        "log_file_size": log_file.stat().st_size if log_file.exists() else 0
    }


if __name__ == "__main__":
    # 测试错误处理器
    print("Testing ErrorHandler...")

    try:
        # 故意引发一个错误
        result = 1 / 0
    except Exception as e:
        error_handler.handle_exception(e, {"test_context": "division_by_zero"})

    # 获取错误摘要
    summary = get_error_summary()
    print(f"Error summary: {summary}")

    print("ErrorHandler tests completed!")