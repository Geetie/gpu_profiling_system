"""Tests for sandbox infrastructure (infrastructure/sandbox.py)."""
import os
import pytest
from src.infrastructure.sandbox import (
    SandboxConfig,
    SandboxResult,
    SandboxRunner,
    LocalSandbox,
    DockerSandbox,
    docker_available,
)


# ── SandboxConfig Tests ──────────────────────────────────────────────


class TestSandboxConfig:
    def test_secure_defaults(self):
        """Config defaults should be security-hardened."""
        config = SandboxConfig()
        assert config.privileged is False
        assert config.mount_docker_socket is False
        assert config.network_disabled is True
        assert config.gpu_read_only is True
        assert config.timeout_seconds == 300

    def test_custom_config(self):
        config = SandboxConfig(
            image="nvidia/cuda:12.0-devel",
            timeout_seconds=60,
            work_dir="/tmp/work",
        )
        assert config.image == "nvidia/cuda:12.0-devel"
        assert config.timeout_seconds == 60
        assert config.work_dir == "/tmp/work"


# ── SandboxResult Tests ──────────────────────────────────────────────


class TestSandboxResult:
    def test_success(self):
        result = SandboxResult(stdout="ok", stderr="", return_code=0, success=True)
        assert result.success is True
        assert result.failed is False

    def test_failure(self):
        result = SandboxResult(stdout="", stderr="error", return_code=1, success=False)
        assert result.success is False
        assert result.failed is True

    def test_artifacts(self):
        result = SandboxResult(
            stdout="", stderr="", return_code=0, success=True,
            artifacts={"binary": "/path/to/binary"},
        )
        assert result.artifacts == {"binary": "/path/to/binary"}


# ── LocalSandbox Tests ───────────────────────────────────────────────


class TestLocalSandbox:
    def test_execute_echo(self, tmp_path):
        """Basic command execution should work."""
        sandbox = LocalSandbox(sandbox_root=str(tmp_path))
        result = sandbox.run(
            command="echo",
            args=["hello"],
        )
        assert result.success is True
        assert "hello" in result.stdout

    def test_execute_nonexistent_command(self, tmp_path):
        """Non-existent command should fail gracefully."""
        sandbox = LocalSandbox(sandbox_root=str(tmp_path))
        result = sandbox.run(command="nonexistent_command_xyz_123")
        assert result.success is False
        assert "not found" in result.stderr.lower()

    def test_execute_no_command(self, tmp_path):
        """Empty command should return error."""
        sandbox = LocalSandbox(sandbox_root=str(tmp_path))
        result = sandbox.run(command="")
        assert result.success is False
        assert "No command specified" in result.stderr

    def test_write_source_code(self, tmp_path):
        """Source code should be written to sandbox directory."""
        sandbox = LocalSandbox(sandbox_root=str(tmp_path))
        result = sandbox.run(
            source_code="int main() { return 0; }",
            command="echo",
            args=["done"],
        )
        source_path = os.path.join(str(tmp_path), "source.cu")
        assert os.path.exists(source_path)
        with open(source_path) as f:
            assert "int main()" in f.read()

    def test_path_escape_blocked(self, tmp_path):
        """Path escape outside sandbox should be blocked."""
        # LocalSandbox auto-creates the sandbox root
        sandbox = LocalSandbox(sandbox_root=str(tmp_path / "sandbox"))
        # Create a sibling directory to test escape detection
        (tmp_path / "outside").mkdir(exist_ok=True)

        with pytest.raises(PermissionError, match="Path escape blocked"):
            sandbox._resolve_path(str(tmp_path / "outside" / "file.txt"))

    def test_sibling_path_blocked(self, tmp_path):
        """Sibling directory should not pass sandbox check."""
        sandbox = LocalSandbox(sandbox_root=str(tmp_path / "sandbox"))
        # Create a sibling directory
        (tmp_path / "sandbox_evil").mkdir(exist_ok=True)

        with pytest.raises(PermissionError, match="Path escape blocked"):
            sandbox._resolve_path(str(tmp_path / "sandbox_evil" / "file.txt"))

    def test_cleanup(self, tmp_path):
        """Cleanup should remove sandbox contents."""
        sandbox = LocalSandbox(sandbox_root=str(tmp_path))
        sandbox.run(source_code="test", command="echo", args=["cleanup test"])
        sandbox.cleanup()
        # The directory should still exist (recreated), but contents cleared
        source_path = os.path.join(str(tmp_path), "source.cu")
        assert not os.path.exists(source_path)

    def test_timeout(self, tmp_path):
        """Command should time out after timeout_seconds."""
        from src.infrastructure.sandbox import SandboxConfig
        config = SandboxConfig(timeout_seconds=1)
        sandbox = LocalSandbox(config=config, sandbox_root=str(tmp_path))
        result = sandbox.run(command="sleep", args=["10"])
        assert result.success is False
        assert "timed out" in result.stderr.lower()


class TestLocalSandboxQuotedPaths:
    """Test handling of quoted paths in LocalSandbox (model-generated path fix)."""
    
    def test_quoted_absolute_path(self, tmp_path):
        """Test that quoted absolute paths are handled correctly."""
        sandbox = LocalSandbox(sandbox_root=str(tmp_path))
        
        # Simulate model-generated path with quotes
        absolute_path = str(tmp_path / "workdir")
        quoted_path = f"'{absolute_path}'"
        
        # Should resolve correctly
        resolved = sandbox._resolve_path(quoted_path)
        assert resolved == absolute_path
    
    def test_double_quoted_absolute_path(self, tmp_path):
        """Test that double-quoted absolute paths are handled correctly."""
        sandbox = LocalSandbox(sandbox_root=str(tmp_path))
        
        absolute_path = str(tmp_path / "workdir")
        quoted_path = f'"{absolute_path}"'
        
        resolved = sandbox._resolve_path(quoted_path)
        assert resolved == absolute_path
    
    def test_quoted_path_escape_blocked(self, tmp_path):
        """Test that path escape attempts with quotes are still blocked."""
        sandbox = LocalSandbox(sandbox_root=str(tmp_path / "sandbox"))
        
        # Create a sibling directory to test escape
        (tmp_path / "outside").mkdir(exist_ok=True)
        outside_path = str(tmp_path / "outside" / "file.txt")
        
        # Test with single quotes
        quoted_path = f"'{outside_path}'"
        with pytest.raises(PermissionError, match="Path escape blocked"):
            sandbox._resolve_path(quoted_path)
        
        # Test with double quotes
        quoted_path2 = f'"{outside_path}"'
        with pytest.raises(PermissionError, match="Path escape blocked"):
            sandbox._resolve_path(quoted_path2)


# ── DockerSandbox Tests ─────────────────────────────────────────────


class TestDockerSandbox:
    def test_build_docker_args(self, tmp_path):
        """Docker args should include security flags."""
        sandbox = DockerSandbox(host_work_dir=str(tmp_path))
        args = sandbox._build_docker_args()

        # Security flags must be present
        assert "--network" in args
        assert "none" in args
        assert "--read-only" in args
        assert "--no-new-privileges" in args
        assert "--cap-drop" in args
        assert "ALL" in args
        assert "--gpus" in args
        # Docker socket should NOT be mounted
        assert "docker.sock" not in " ".join(args)

    def test_no_command_returns_error(self, tmp_path):
        sandbox = DockerSandbox(host_work_dir=str(tmp_path))
        result = sandbox.run(command="")
        assert result.success is False
        assert "No command specified" in result.stderr

    def test_write_source_before_run(self, tmp_path):
        """Source code should be written to host work dir before docker run."""
        sandbox = DockerSandbox(host_work_dir=str(tmp_path))
        sandbox.run(
            source_code="test source",
            command="echo",
        )
        source_path = os.path.join(str(tmp_path), "source.cu")
        assert os.path.exists(source_path)

    def test_docker_available_check(self):
        """docker_available() should return a boolean."""
        result = docker_available()
        assert isinstance(result, bool)

    def test_cleanup(self, tmp_path):
        sandbox = DockerSandbox(host_work_dir=str(tmp_path))
        sandbox.run(source_code="test", command="echo")
        sandbox.cleanup()
        # Host work dir should be clean
        assert not os.path.exists(os.path.join(str(tmp_path), "source.cu"))

    def test_config_image_used(self, tmp_path):
        from src.infrastructure.sandbox import SandboxConfig
        config = SandboxConfig(image="nvidia/cuda:11.8-devel")
        sandbox = DockerSandbox(config=config, host_work_dir=str(tmp_path))
        args = sandbox._build_docker_args()
        assert "nvidia/cuda:11.8-devel" in args

    def test_gpus_arg_no_quotes(self, tmp_path):
        """VULN-A: --gpus arg must not have surrounding quotes."""
        sandbox = DockerSandbox(host_work_dir=str(tmp_path))
        args = sandbox._build_docker_args()
        gpus_idx = args.index("--gpus")
        gpus_value = args[gpus_idx + 1]
        # Must NOT have quotes around device=0
        assert gpus_value == "device=0,capability=compute"
        assert '"' not in gpus_value

    def test_host_work_dir_relative_path_rejected(self):
        """VULN-D: Relative host_work_dir must be rejected."""
        with pytest.raises(ValueError, match="must be absolute"):
            DockerSandbox(host_work_dir="relative/path")

    def test_host_work_dir_traversal_rejected(self, tmp_path):
        """VULN-D: Path traversal via .. must be rejected."""
        with pytest.raises(ValueError, match="path traversal"):
            DockerSandbox(host_work_dir=str(tmp_path / ".." / "evil"))

    def test_host_work_dir_normalized(self, tmp_path):
        """VULN-D: host_work_dir should be normalized to absolute path."""
        sandbox = DockerSandbox(host_work_dir=str(tmp_path))
        assert os.path.isabs(sandbox._host_work_dir)
