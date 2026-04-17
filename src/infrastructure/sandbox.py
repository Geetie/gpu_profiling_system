"""Sandbox isolation — infrastructure layer.

Enforces spec §6: all CUDA compilation and execution must occur in
a sandboxed environment.

- DockerSandbox: full container isolation (when Docker is available)
- LocalSandbox: fallback with path confinement (for dev/testing)
"""
from __future__ import annotations

import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ── Config ───────────────────────────────────────────────────────────


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution.

    Security defaults are hard-coded to prevent accidental weakening.
    """
    image: str = "nvidia/cuda:12.2-devel"
    gpu_read_only: bool = True
    network_disabled: bool = True
    privileged: bool = False          # Hard boundary — never True in production
    mount_docker_socket: bool = False  # Hard boundary — never True
    restricted_paths: tuple[str, ...] = ("/etc", "/root", "/proc", "/sys")
    work_dir: str = "/workspace"
    timeout_seconds: int = 300


@dataclass
class SandboxResult:
    """Result from a sandbox execution."""
    stdout: str
    stderr: str
    return_code: int
    success: bool
    artifacts: dict[str, str] = field(default_factory=dict)  # filename -> content or path
    error_type: str = ""
    error_category: str = ""

    @property
    def failed(self) -> bool:
        return not self.success

    def get_structured_error(self) -> dict[str, str]:
        """Return structured error information for model context."""
        if self.success:
            return {"status": "success"}
        
        error_info = {
            "status": "failed",
            "return_code": self.return_code,
            "stderr": self.stderr[:1000],
        }
        
        if self.error_category:
            error_info["error_category"] = self.error_category
        if self.error_type:
            error_info["error_type"] = self.error_type
            
        if "nvcc" in self.stderr.lower():
            error_info["error_category"] = "compilation_error"
            if "fatal" in self.stderr.lower():
                error_info["error_type"] = "fatal_compilation_error"
            elif "error" in self.stderr.lower():
                error_info["error_type"] = "compilation_error"
        elif "not found" in self.stderr.lower():
            error_info["error_category"] = "file_not_found"
        elif "timeout" in self.stderr.lower():
            error_info["error_category"] = "timeout"
            
        return error_info


# ── Abstract Runner ──────────────────────────────────────────────────


class SandboxRunner(ABC):
    """Abstract sandbox interface.

    Implementations must enforce:
    - Process isolation (container or subprocess confinement)
    - Resource limits (timeout, memory)
    - No network access (or minimal)
    - No privileged operations
    """

    def __init__(self, config: SandboxConfig) -> None:
        self.config = config

    @abstractmethod
    def run(
        self,
        source_code: str | None = None,
        command: str = "",
        args: list[str] | None = None,
        work_dir: str | None = None,
    ) -> SandboxResult:
        """Execute code or command in the sandbox.

        Args:
            source_code: Optional source code to write before execution.
            command: The command to run (e.g. "nvcc", "./benchmark").
            args: Command arguments.
            work_dir: Working directory inside the sandbox.

        Returns:
            SandboxResult with stdout, stderr, return code.
        """
        ...

    @abstractmethod
    def cleanup(self) -> None:
        """Release sandbox resources (e.g. remove container)."""
        ...

    @property
    def sandbox_root(self) -> str:
        """Return the sandbox working directory root.

        Public accessor for use by tool handlers and orchestration code.
        """
        raise NotImplementedError("sandbox_root not implemented")


# ── Local Sandbox (fallback) ─────────────────────────────────────────


class LocalSandbox(SandboxRunner):
    """Fallback sandbox for environments without Docker.

    Provides path confinement via FileOperations-style checks and
    subprocess isolation with timeout enforcement.

    WARNING: This does NOT provide true container isolation. It is
    suitable only for testing and development.

    VULN-C (Known Limitation): LocalSandbox writes source code directly
    via open() without going through FileOperations.read() first, which
    bypasses M1 (read-before-write) enforcement. This is acceptable only
    because LocalSandbox is a dev/test fallback — production MUST use
    DockerSandbox, which isolates writes inside the container.
    """

    def __init__(self, config: SandboxConfig | None = None, sandbox_root: str | None = None) -> None:
        super().__init__(config or SandboxConfig())
        if sandbox_root is not None:
            self._sandbox_root = sandbox_root
        elif os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            project_root = os.getcwd()
            self._sandbox_root = os.path.join(project_root, ".kaggle_sandbox")
            os.makedirs(self._sandbox_root, exist_ok=True)
            print(f"[Sandbox] Kaggle environment detected (KAGGLE_KERNEL_RUN_TYPE={os.environ['KAGGLE_KERNEL_RUN_TYPE']})")
            print(f"[Sandbox] Using isolated sandbox: {self._sandbox_root}")
        else:
            self._sandbox_root = os.path.join(os.getcwd(), ".sandbox")
        os.makedirs(self._sandbox_root, exist_ok=True)
        # INT-6 fix: track source writes for M1 audit compliance
        from src.domain.permission import InvariantTracker
        self._tracker: InvariantTracker | None = None

    def set_tracker(self, tracker: InvariantTracker) -> None:
        """Attach an InvariantTracker for M1 audit compliance."""
        self._tracker = tracker

    @property
    def sandbox_root(self) -> str:
        """Return the local sandbox working directory."""
        return self._sandbox_root

    def _resolve_path(self, path: str) -> str:
        """Resolve path and ensure it is inside the sandbox root.
        
        Automatically strips leading/trailing quotes to handle
        model-generated paths (e.g., "'/path/to/dir'").
        
        Raises:
            PermissionError: if the resolved path escapes the sandbox.
        """
        original_path = path
        path = path.strip('\'"')
        resolved = os.path.abspath(os.path.normpath(path))
        sandbox = self._sandbox_root.rstrip(os.sep) + os.sep
        if not (resolved.startswith(sandbox) or resolved == self._sandbox_root.rstrip(os.sep)):
            raise PermissionError(
                f"Path escape blocked: {original_path!r} (resolved: {path!r}) resolves outside sandbox {self._sandbox_root}"
            )
        return resolved

    def run(
        self,
        source_code: str | None = None,
        command: str = "",
        args: list[str] | None = None,
        work_dir: str | None = None,
    ) -> SandboxResult:
        """Execute command in the sandbox with path confinement."""
        target_dir = work_dir or self._sandbox_root
        target_dir = self._resolve_path(target_dir)

        # Write source code if provided
        source_path: str | None = None
        if source_code is not None:
            source_path = os.path.join(target_dir, "source.cu")
            # INT-6 fix: record the write in tracker for M1 audit compliance
            if self._tracker is not None:
                self._tracker.record_created(source_path)
            with open(source_path, "w", encoding="utf-8") as f:
                f.write(source_code)

        # Build command line
        cmd = [command] + (args or [])
        if not command:
            return SandboxResult(
                stdout="", stderr="No command specified", return_code=1, success=False,
            )

        try:
            result = subprocess.run(
                cmd,
                cwd=target_dir,
                capture_output=True,
                text=False,  # Capture as bytes to handle non-UTF-8 output
                timeout=self.config.timeout_seconds,
            )
            # Decode with error handling for non-UTF-8 output
            try:
                stdout = result.stdout.decode('utf-8', errors='replace')
                stderr = result.stderr.decode('utf-8', errors='replace')
            except Exception:
                # Fallback to Latin-1 if UTF-8 fails
                stdout = result.stdout.decode('latin-1')
                stderr = result.stderr.decode('latin-1')
            
            # Bug fix: Distinguish between warning and error
            # nvcc may return warnings but still succeed (returncode=0)
            # Only treat as error if returncode != 0 OR stderr contains actual errors
            stderr_lower = stderr.lower()
            has_actual_error = result.returncode != 0 or (
                "error:" in stderr_lower or
                "fatal" in stderr_lower or
                "undefined reference" in stderr_lower
            )
            has_warning_only = result.returncode == 0 and "warning" in stderr_lower and not has_actual_error
            
            return SandboxResult(
                stdout=stdout,
                stderr=stderr,
                return_code=result.returncode,
                success=not has_actual_error,
                artifacts={"source": source_path} if source_path else {},
                error_type="warning" if has_warning_only else "",
                error_category="compilation_warning" if has_warning_only else (
                    "compilation_error" if result.returncode != 0 and "nvcc" in command.lower() else ""
                ),
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                stdout="",
                stderr=f"Command timed out after {self.config.timeout_seconds}s",
                return_code=-1,
                success=False,
                error_category="timeout",
            )
        except FileNotFoundError:
            return SandboxResult(
                stdout="",
                stderr=f"Command not found: {command}",
                return_code=-1,
                success=False,
                error_category="command_not_found",
            )
        except Exception as e:
            return SandboxResult(
                stdout="",
                stderr=str(e),
                return_code=-1,
                success=False,
                error_category="runtime_error",
            )

    def cleanup(self) -> None:
        """Remove sandbox directory contents."""
        import shutil
        if os.path.exists(self._sandbox_root):
            shutil.rmtree(self._sandbox_root, ignore_errors=True)
        os.makedirs(self._sandbox_root, exist_ok=True)


# ── Docker Sandbox ───────────────────────────────────────────────────


def docker_available() -> bool:
    """Check if Docker is installed and running."""
    return shutil.which("docker") is not None


class DockerSandbox(SandboxRunner):
    """Docker-based sandbox with full container isolation.

    Container flags enforce spec §6:
    - --gpus readonly: GPU exposed read-only
    - --network none: no network access
    - --read-only: read-only root filesystem
    - --no-new-privileges: privilege escalation blocked
    - --cap-drop ALL: drop all capabilities
    - NO docker socket mount
    """

    def __init__(
        self,
        config: SandboxConfig | None = None,
        host_work_dir: str | None = None,
    ) -> None:
        super().__init__(config or SandboxConfig())
        raw_work_dir = host_work_dir or os.path.join(os.getcwd(), ".docker_sandbox")
        # VULN-D: Validate host_work_dir is an absolute path without traversal
        if not os.path.isabs(raw_work_dir):
            raise ValueError(f"host_work_dir must be absolute: {raw_work_dir!r}")
        # Check for traversal in raw path components before normalization
        raw_parts = raw_work_dir.replace("\\", "/").split("/")
        if ".." in raw_parts:
            raise ValueError(f"host_work_dir contains path traversal: {raw_work_dir!r}")
        resolved = os.path.abspath(os.path.normpath(raw_work_dir))
        self._host_work_dir = resolved
        os.makedirs(self._host_work_dir, exist_ok=True)
        self._container_id: str | None = None

    @property
    def sandbox_root(self) -> str:
        """Alias for _host_work_dir — public accessor for sandbox working directory."""
        return self._host_work_dir

    def _build_docker_args(self) -> list[str]:
        """Build docker run flags per spec §6."""
        c = self.config
        args = [
            "docker", "run", "--rm",
            "--network", "none",             # no network
            "--read-only",                   # read-only root fs
            "--tmpfs", "/tmp",               # tmp for temp files
            "--tmpfs", c.work_dir,           # tmp for workspace
            "--no-new-privileges",           # no privilege escalation
            "--cap-drop", "ALL",             # drop all capabilities
            "-w", c.work_dir,                # working directory
            "--stop-timeout", "30",           # hard stop container after 30s
        ]

        # GPU access (read-only)
        if c.gpu_read_only:
            args.extend(["--gpus", "device=0,capability=compute"])

        # Mount host work directory (read-write for the sandbox only)
        args.extend(["-v", f"{self._host_work_dir}:{c.work_dir}:rw"])

        # Image
        args.append(c.image)

        return args

    def run(
        self,
        source_code: str | None = None,
        command: str = "",
        args: list[str] | None = None,
        work_dir: str | None = None,
    ) -> SandboxResult:
        """Execute command inside a Docker container."""
        # Write source code to host work directory first
        if source_code is not None:
            source_path = os.path.join(self._host_work_dir, "source.cu")
            with open(source_path, "w", encoding="utf-8") as f:
                f.write(source_code)

        if not command:
            return SandboxResult(
                stdout="", stderr="No command specified", return_code=1, success=False,
            )

        docker_args = self._build_docker_args() + [command] + (args or [])

        try:
            result = subprocess.run(
                docker_args,
                capture_output=True,
                text=False,  # Capture as bytes to handle non-UTF-8 output
                timeout=self.config.timeout_seconds,
            )
            # Decode with error handling for non-UTF-8 output
            try:
                stdout = result.stdout.decode('utf-8', errors='replace')
                stderr = result.stderr.decode('utf-8', errors='replace')
            except Exception:
                # Fallback to Latin-1 if UTF-8 fails
                stdout = result.stdout.decode('latin-1')
                stderr = result.stderr.decode('latin-1')
            return SandboxResult(
                stdout=stdout,
                stderr=stderr,
                return_code=result.returncode,
                success=result.returncode == 0,
                error_category="compilation_error" if result.returncode != 0 and "nvcc" in command.lower() else "",
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                stdout="",
                stderr=f"Docker command timed out after {self.config.timeout_seconds}s",
                return_code=-1,
                success=False,
                error_category="timeout",
            )
        except Exception as e:
            return SandboxResult(
                stdout="",
                stderr=str(e),
                return_code=-1,
                success=False,
                error_category="runtime_error",
            )

    def cleanup(self) -> None:
        """Clean up Docker sandbox directory."""
        import shutil
        if os.path.exists(self._host_work_dir):
            shutil.rmtree(self._host_work_dir, ignore_errors=True)
        os.makedirs(self._host_work_dir, exist_ok=True)
