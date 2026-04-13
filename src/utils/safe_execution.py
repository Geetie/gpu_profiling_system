#!/usr/bin/env python3
"""安全执行工具 - 防止命令注入和动态代码执行风险"""

import os
import sys
import json
import subprocess
import shlex
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


class SafeExecutor:
    """安全执行器 - 防止命令注入"""

    @staticmethod
    def validate_path(path: str) -> bool:
        """验证路径安全性"""
        if not path or not isinstance(path, str):
            return False

        # 移除首尾空白
        path = path.strip()
        if not path:
            return False

        # 禁止危险字符和模式
        dangerous_patterns = ['..', '~', '|', '&', ';', '$', '`', '>', '<', '\0', '\\']
        for pattern in dangerous_patterns:
            if pattern in path:
                return False

        # 检查绝对路径（只允许相对路径）
        if path.startswith('/') or path.startswith('\\'):
            return False

        # 只允许字母、数字、下划线、连字符、点、斜杠
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-./')
        if not all(c in allowed_chars for c in path):
            return False

        # 防止路径遍历攻击
        normalized_path = os.path.normpath(path)
        if normalized_path.startswith('..') or '/../' in normalized_path:
            return False

        return True

    @staticmethod
    def validate_query(query: str) -> bool:
        """验证查询文本安全性"""
        if not query or not isinstance(query, str):
            return False

        query = query.strip()
        if not query or len(query) > 1000:  # 限制长度
            return False

        # 禁止危险字符
        dangerous_chars = ['|', '&', ';', '$', '`', '>', '<', '\0', '\\', '\n', '\r']
        for char in dangerous_chars:
            if char in query:
                return False

        return True

    @staticmethod
    def validate_integer(value: Any, min_val: int = 1, max_val: int = 1000) -> bool:
        """验证整数参数"""
        try:
            if isinstance(value, str):
                value = int(value)
            elif not isinstance(value, int):
                return False

            return min_val <= value <= max_val
        except (ValueError, TypeError):
            return False

    @staticmethod
    def safe_run_command(cmd_parts: List[str], timeout: int = 300) -> Tuple[bool, str, str]:
        """安全运行命令"""
        try:
            # 验证命令参数
            for part in cmd_parts:
                if not SafeExecutor.validate_path(part):
                    return False, "", f"Invalid command argument: {part}"

            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            return result.returncode == 0, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            return False, "", "Command timeout"
        except Exception as e:
            return False, "", str(e)

    @staticmethod
    def safe_file_read(file_path: str) -> Tuple[bool, str]:
        """安全读取文件"""
        try:
            if not SafeExecutor.validate_path(file_path):
                return False, f"Invalid file path: {file_path}"

            with open(file_path, 'r', encoding='utf-8') as f:
                return True, f.read()

        except Exception as e:
            return False, str(e)

    @staticmethod
    def safe_file_write(file_path: str, content: str) -> Tuple[bool, str]:
        """安全写入文件"""
        try:
            if not SafeExecutor.validate_path(file_path):
                return False, f"Invalid file path: {file_path}"

            # 确保目录存在
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return True, "Success"

        except Exception as e:
            return False, str(e)


class SafeImporter:
    """安全导入器 - 替代exec()"""

    @staticmethod
    def import_module_safely(module_name: str):
        """安全导入模块"""
        try:
            if not module_name.replace('.', '').replace('_', '').isalnum():
                raise ValueError(f"Invalid module name: {module_name}")

            return __import__(module_name)

        except Exception as e:
            raise ImportError(f"Failed to import {module_name}: {e}")

    @staticmethod
    def run_module_function(module_name: str, function_name: str, *args, **kwargs):
        """安全运行模块函数"""
        try:
            module = SafeImporter.import_module_safely(module_name)

            if not hasattr(module, function_name):
                raise AttributeError(f"Function {function_name} not found in {module_name}")

            func = getattr(module, function_name)
            return func(*args, **kwargs)

        except Exception as e:
            raise RuntimeError(f"Failed to run {module_name}.{function_name}: {e}")


# 全局安全执行函数
def safe_run_probes(output_dir: str = ".") -> Tuple[bool, str, str]:
    """安全运行探针测试"""
    if not SafeExecutor.validate_path(output_dir):
        return False, "", f"Invalid output directory: {output_dir}"

    return SafeExecutor.safe_run_command([
        sys.executable, "-m", "src.main",
        "--probes-only",
        "--output-dir", output_dir,
        "--no-docker"
    ], timeout=300)


def safe_run_pipeline(query: str, target_spec: str = "config/target_spec.json",
                     max_turns: int = 10, output_dir: str = ".") -> Tuple[bool, str, str]:
    """安全运行管道模式"""
    # 验证所有参数
    if not SafeExecutor.validate_path(target_spec):
        return False, "", f"Invalid target_spec path: {target_spec}"

    if not SafeExecutor.validate_path(output_dir):
        return False, "", f"Invalid output_dir path: {output_dir}"

    if not SafeExecutor.validate_integer(max_turns, 1, 100):
        return False, "", f"Invalid max_turns value: {max_turns} (must be 1-100)"

    if not SafeExecutor.validate_query(query):
        return False, "", f"Invalid query: contains dangerous characters or too long"

    # 转义参数
    safe_query = shlex.quote(query)
    safe_target_spec = shlex.quote(target_spec)
    safe_output_dir = shlex.quote(output_dir)

    return SafeExecutor.safe_run_command([
        sys.executable, "-m", "src.main",
        safe_query,
        "--pipeline",
        "--target-spec", safe_target_spec,
        "--no-docker",
        "--max-turns", str(max_turns),
        "--output-dir", safe_output_dir
    ], timeout=600)


def safe_load_json(file_path: str) -> Tuple[bool, Dict[str, Any]]:
    """安全加载JSON文件"""
    success, content = SafeExecutor.safe_file_read(file_path)
    if not success:
        return False, {}

    try:
        data = json.loads(content)
        return True, data
    except json.JSONDecodeError as e:
        return False, {}


# 替换exec()的安全函数
def safe_run_kaggle_kernel() -> bool:
    """安全运行Kaggle内核 - 替代exec(open().read())"""
    try:
        # 直接调用main模块而不是exec
        from src.main import main

        # 模拟探针模式
        return main([
            "--probes-only",
            "--output-dir", ".",
            "--no-docker"
        ]) == 0

    except Exception as e:
        print(f"Error running kaggle kernel: {e}")
        return False


if __name__ == "__main__":
    # 测试安全执行器
    print("Testing SafeExecutor...")

    # 测试路径验证
    assert SafeExecutor.validate_path("config/target_spec.json") == True
    assert SafeExecutor.validate_path("../dangerous/path") == False
    assert SafeExecutor.validate_path("valid_path_123") == True

    print("SafeExecutor tests passed!")