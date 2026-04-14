"""File operation tool handlers — infrastructure layer.

Delegates to FileOperations which enforces M1 (read-before-write)
and sandbox confinement.

M1 fix (v3): write_file handler always tries create() first for new file
creation (bypasses M1 since nothing to read), falls back to write() for
existing files (requires prior read). This avoids path resolution mismatches
between the handler's os.path.abspath() and FileOps._resolve().
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

    M1 handling (delegated to FileOps.write()):
    - New files: created directly (bypasses M1 — nothing to read)
    - Existing files: requires prior read (M1 enforced)

    The FileOps.write() method handles both cases — the handler just
    calls it and propagates results/errors.

    Returns (from output_schema):
        bytes_written: int — size of written content in bytes
    """

    def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        file_path = arguments.get("file_path", "")
        content = arguments.get("content", "")

        if not file_path:
            return {"bytes_written": 0}

        try:
            return {"bytes_written": file_ops.write(file_path, content)}
        except PermissionError as e:
            # M1 violation or sandbox escape — re-raise for AgentLoop handling
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
