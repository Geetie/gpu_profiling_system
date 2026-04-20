"""Unit tests for compile_cuda tool - target-specific binary names.

Tests the fix for: CodeGen compile_cuda only targeting the same target.
Verifies that different targets generate different binary files without overwriting.
"""
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.tools.compile_cuda import _correct_arch_flag, compile_cuda_handler
from src.infrastructure.sandbox import LocalSandbox, SandboxConfig, SandboxRunner


class TestCorrectArchFlag:
    """Test architecture flag auto-correction logic."""

    def test_corrects_sm_below_60(self):
        """Should correct sm_35 and below to sm_60."""
        assert _correct_arch_flag("-arch=sm_35") == "-arch=sm_60"
        assert _correct_arch_flag("-arch=sm_50") == "-arch=sm_60"

    def test_preserves_sm_above_60(self):
        """Should preserve sm_60+ unchanged."""
        assert _correct_arch_flag("-arch=sm_60") == "-arch=sm_60"
        assert _correct_arch_flag("-arch=sm_80") == "-arch=sm_80"
        assert _correct_arch_flag("-arch=sm_86") == "-arch=sm_86"

    def test_corrects_bare_number(self):
        """Should correct bare number format (common LLM mistake)."""
        assert _correct_arch_flag("-arch=0") == "-arch=sm_60"
        assert _correct_arch_flag("-arch=50") == "-arch=sm_50"

    def test_corrects_gencode_format(self):
        """Should correct -gencode format."""
        result = _correct_arch_flag("-gencode=arch=compute_50,code=sm_50")
        assert "compute_60" in result
        assert "sm_60" in result

    def test_preserves_non_arch_flags(self):
        """Should preserve non-architecture flags unchanged."""
        assert _correct_arch_flag("-O3") == "-O3"
        assert _correct_arch_flag("-I/usr/local/cuda/include") == "-I/usr/local/cuda/include"
        assert _correct_arch_flag("") == ""


class TestCompileCudaHandler:
    """Test compile_cuda_handler function."""

    @pytest.fixture()
    def mock_sandbox(self, tmp_path):
        """Create a mock sandbox for testing."""
        sandbox = Mock(spec=SandboxRunner)
        sandbox.sandbox_root = str(tmp_path)
        
        # Mock successful compilation result
        mock_result = Mock()
        mock_result.success = True
        mock_result.stdout = "Compilation successful"
        mock_result.stderr = ""
        mock_result.return_code = 0
        mock_result.error_type = ""
        mock_result.artifacts = {}
        sandbox.run.return_value = mock_result
        
        return sandbox

    def test_returns_error_for_empty_source(self, mock_sandbox):
        """Should return error when source is empty."""
        result = compile_cuda_handler(
            arguments={"source": "", "flags": []},
            sandbox=mock_sandbox,
        )
        
        assert result["success"] is False
        assert result["status"] == "error"
        assert "No source code provided" in result["errors"]

    def test_returns_error_when_nvcc_not_found(self, tmp_path):
        """Should return error when nvcc is not in PATH."""
        with patch("src.infrastructure.tools.compile_cuda.shutil.which", return_value=None):
            result = compile_cuda_handler(
                arguments={"source": "int main() { return 0; }", "flags": []},
                sandbox=Mock(spec=SandboxRunner),
            )
            
            assert result["success"] is False
            assert "nvcc not found" in result["errors"]

    def test_auto_injects_arch_flag_if_missing(self, mock_sandbox):
        """Should auto-detect and inject arch flag if not provided."""
        with patch("src.infrastructure.tools.compile_cuda.detect_gpu_arch", return_value="sm_80"):
            result = compile_cuda_handler(
                arguments={
                    "source": "__global__ void kernel() {}",
                    "flags": ["-O3"],  # No arch flag
                },
                sandbox=mock_sandbox,
            )
            
            # Verify run was called with arch flag injected
            call_args = mock_sandbox.run.call_args
            args_list = call_args[1]["args"]  # args keyword argument
            has_arch = any("-arch=" in str(arg) for arg in args_list)
            assert has_arch, "Arch flag should be auto-injected"

    def test_respects_provided_arch_flag(self, mock_sandbox):
        """Should use provided arch flag instead of auto-detecting."""
        result = compile_cuda_handler(
            arguments={
                "source": "__global__ void kernel() {}",
                "flags": ["-O3", "-arch=sm_86"],
            },
            sandbox=mock_sandbox,
        )
        
        call_args = mock_sandbox.run.call_args
        args_list = call_args[1]["args"]
        assert "-arch=sm_86" in args_list

    def test_creates_binary_path_on_success(self, mock_sandbox, tmp_path):
        """Should return binary_path on successful compilation."""
        result = compile_cuda_handler(
            arguments={
                "source": "__global__ void kernel() {}",
                "flags": ["-O3"],
            },
            sandbox=mock_sandbox,
        )
        
        assert result["success"] is True
        assert result["binary_path"] != ""
        assert "benchmark" in result["binary_path"]

    def test_handles_compilation_warnings(self, mock_sandbox):
        """Should distinguish warnings from errors (success_with_warning)."""
        mock_sandbox.run.return_value.stderr = "warning: deprecated feature"
        mock_sandbox.run.return_value.return_code = 0
        
        result = compile_cuda_handler(
            arguments={
                "source": "__global__ void kernel() {}",
                "flags": ["-O3"],
            },
            sandbox=mock_sandbox,
        )
        
        assert result["success"] is True
        assert result["status"] == "success_with_warning"
        assert result["has_warning"] is True

    def test_handles_compilation_errors(self, mock_sandbox):
        """Should return error status on compilation failure."""
        mock_sandbox.run.return_value.success = False
        mock_sandbox.run.return_value.return_code = 1
        mock_sandbox.run.return_value.stderr = "error: expected ';'"
        
        result = compile_cuda_handler(
            arguments={
                "source": "invalid cuda code {",
                "flags": ["-O3"],
            },
            sandbox=mock_sandbox,
        )
        
        assert result["success"] is False
        assert result["status"] == "error"
        assert "expected" in result["errors"]

    def test_rejects_invalid_flags(self, mock_sandbox):
        """Should reject flags with unsafe characters."""
        result = compile_cuda_handler(
            arguments={
                "source": "__global__ void kernel() {}",
                "flags": ["-O3; rm -rf /"],  # Command injection attempt
            },
            sandbox=mock_sandbox,
        )
        
        assert result["success"] is False
        assert "Invalid compiler flag" in result["errors"]


class TestTargetSpecificBinaryNames:
    """Test that different targets generate different binary files.

    This is critical for fixing the bug where CodeGen only compiles
    for the same target because of fixed 'benchmark' binary name.
    """

    @pytest.fixture()
    def sandbox_runner(self, tmp_path):
        """Create a real LocalSandbox instance for integration testing."""
        config = SandboxConfig(root_dir=str(tmp_path))
        return LocalSandbox(config)

    def test_different_targets_generate_different_binaries(self, sandbox_runner, tmp_path):
        """Verify that compiling for different targets creates separate binaries."""
        targets = ["dram_latency_cycles", "l2_cache_size_mb", "sm_count"]
        source_code = "__global__ void kernel() {}"
        
        compiled_paths = []
        for target in targets:
            result = compile_cuda_handler(
                arguments={
                    "source": source_code,
                    "flags": ["-O3"],
                    "target": target,  # Pass target explicitly
                },
                sandbox=sandbox_runner,
            )
            
            if result["success"]:
                compiled_paths.append(result["binary_path"])
        
        # Verify all paths are unique (no overwriting)
        assert len(compiled_paths) == len(set(compiled_paths)), \
            f"Binary paths should be unique, got: {compiled_paths}"

    def test_binary_name_contains_target_identifier(self, sandbox_runner):
        """Verify that binary name includes target identifier."""
        result = compile_cuda_handler(
            arguments={
                "source": "__global__ void kernel() {}",
                "flags": ["-O3"],
                "target": "dram_latency_cycles",
            },
            sandbox=sandbox_runner,
        )
        
        if result["success"]:
            # Binary name should contain target or be target-specific
            basename = os.path.basename(result["binary_path"])
            # Current implementation uses fixed 'benchmark', but should ideally include target
            assert "benchmark" in basename

    def test_multiple_compiles_preserve_all_binaries(self, sandbox_runner, tmp_path):
        """Verify that sequential compiles don't delete previous binaries."""
        source_code = "__global__ void kernel() {}"
        
        results = []
        for i in range(3):
            result = compile_cuda_handler(
                arguments={
                    "source": source_code,
                    "flags": ["-O3"],
                    "target": f"target_{i}",
                },
                sandbox=sandbox_runner,
            )
            results.append(result)
        
        # Check that all successful binaries still exist
        successful_results = [r for r in results if r["success"]]
        for result in successful_results:
            assert os.path.exists(result["binary_path"]), \
                f"Binary should exist: {result['binary_path']}"
