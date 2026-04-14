"""File operation tool handlers — infrastructure layer.

Delegates to FileOperations which enforces M1 (read-before-write)
and sandbox confinement.

VULN-P4-1 fix: write_file strictly enforces M1. The create() method
is only used by generate_microbenchmark, not by the generic write_file tool.
"""
from __future__ import annotations

from typing import Any, Callable

from src.infrastructure.file_ops import FileOperations


def make_read_file_handler(file_ops: FileOperations) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a read_file handler bound to the given FileOperations instance.

    Returns (from output_schema):
        content: str — file contents
        lines: int — number of lines
    """

    def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        file_path = arguments.get("file_path", "")
        if not file_path:
            return {
                "content": "",
                "lines": 0,
            }
        try:
            content = file_ops.read(file_path)
            return {
                "content": content,
                "lines": content.count("\n") + (1 if not content.endswith("\n") else 0) if content else 0,
            }
        except FileNotFoundError:
            return {
                "content": "",
                "lines": 0,
            }
        except PermissionError:
            return {
                "content": "",
                "lines": 0,
            }

    return handler


def make_write_file_handler(file_ops: FileOperations) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a write_file handler bound to the given FileOperations instance.

    M1 handling:
    - If the target file doesn't exist on disk, use create() to create it
      (new file creation doesn't require prior read).
    - If the target file exists, use write() which requires prior read (M1).

    Returns (from output_schema):
        bytes_written: int — size of written content in bytes
    """

    def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        import os
        file_path = arguments.get("file_path", "")
        content = arguments.get("content", "")

        if not file_path:
            return {"bytes_written": 0}

        try:
            resolved = os.path.abspath(file_path)
            # New file: use create() which bypasses M1 (nothing to read)
            if not os.path.exists(resolved):
                return {"bytes_written": file_ops.create(file_path, content)}
            # Existing file: use write() which requires prior read (M1)
            return {"bytes_written": file_ops.write(file_path, content)}
        except PermissionError as e:
            # M1 violation — re-raise so ToolRunner can log and propagate
            if "M1 violation" in str(e):
                raise
            return {"bytes_written": 0, "error": str(e)}
        except Exception as e:
            return {"bytes_written": 0, "error": str(e)}

    return handler


def make_create_file_handler(file_ops: FileOperations) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a file creation handler for generation tools.

    This is the ONLY handler that can create new files without prior read.
    It should only be exposed to tools like generate_microbenchmark.

    Returns (from output_schema):
        bytes_written: int — size of written content in bytes
    """

    def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        file_path = arguments.get("file_path", "")
        content = arguments.get("content", "")

        if not file_path:
            return {"bytes_written": 0}

        try:
            return {"bytes_written": file_ops.create(file_path, content)}
        except Exception as e:
            return {"bytes_written": 0, "error": str(e)}

    return handler
