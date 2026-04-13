"""Tests for tool handlers (infrastructure/tools/)."""
import os
from unittest.mock import patch, MagicMock
import pytest


# ── run_ncu Tests ────────────────────────────────────────────────────


class TestRunNcuHandler:
    def test_empty_executable(self):
        from src.infrastructure.tools.run_ncu import run_ncu_handler
        result = run_ncu_handler({"executable": "", "metrics": []})
        assert "error" in result["parsed_metrics"]
        assert "No executable" in result["parsed_metrics"]["error"]

    def test_nonexistent_executable(self, tmp_path):
        from src.infrastructure.tools.run_ncu import run_ncu_handler
        fake_path = str(tmp_path / "nonexistent_binary")
        result = run_ncu_handler({"executable": fake_path, "metrics": []})
        assert "error" in result["parsed_metrics"]

    def test_ncu_not_found(self, tmp_path):
        from src.infrastructure.tools.run_ncu import run_ncu_handler
        fake_bin = str(tmp_path / "fake_binary")
        with open(fake_bin, "w") as f:
            f.write("fake")
        with patch("src.infrastructure.tools.run_ncu.shutil.which", return_value=None):
            result = run_ncu_handler({"executable": fake_bin, "metrics": ["sm__cycles"]})
        assert "error" in result["parsed_metrics"]
        assert "not found" in result["parsed_metrics"]["error"].lower()

    def test_ncu_success(self, tmp_path):
        from src.infrastructure.tools.run_ncu import run_ncu_handler
        fake_bin = str(tmp_path / "fake_binary")
        with open(fake_bin, "w") as f:
            f.write("fake")
        # Use a LocalSandbox mock so it goes through sandbox.run()
        mock_sandbox = MagicMock()
        mock_sandbox.run.return_value = MagicMock(
            stdout="SM Cycles: 12345\nL2 Hit Rate: 85.5\n",
            stderr="",
        )
        with patch("src.infrastructure.tools.run_ncu.shutil.which", return_value="/usr/bin/ncu"):
            result = run_ncu_handler({"executable": fake_bin, "metrics": ["sm__cycles"]}, sandbox=mock_sandbox)
        assert "SM Cycles: 12345" in result["raw_output"]
        assert result["parsed_metrics"]["SM Cycles"] == 12345.0

    def test_ncu_timeout(self, tmp_path):
        from src.infrastructure.tools.run_ncu import run_ncu_handler
        fake_bin = str(tmp_path / "fake_binary")
        with open(fake_bin, "w") as f:
            f.write("fake")
        from src.infrastructure.sandbox import SandboxResult
        mock_sandbox = MagicMock()
        mock_sandbox.run.return_value = SandboxResult(
            stdout="", stderr="ncu timed out after 300s", return_code=-1, success=False,
        )
        with patch("src.infrastructure.tools.run_ncu.shutil.which", return_value="/usr/bin/ncu"):
            result = run_ncu_handler({"executable": fake_bin, "metrics": []}, sandbox=mock_sandbox)
        # Even on sandbox timeout, handler parses output
        assert "timed out" in result["raw_output"] or "timed out" in str(result["parsed_metrics"])

    def test_ncu_invalid_metric_rejected(self, tmp_path):
        from src.infrastructure.tools.run_ncu import run_ncu_handler
        fake_bin = str(tmp_path / "fake_binary")
        with open(fake_bin, "w") as f:
            f.write("fake")
        # Metric validation happens before subprocess, so mock ncu as available
        with patch("src.infrastructure.tools.run_ncu.shutil.which", return_value="/usr/bin/ncu"):
            result = run_ncu_handler({"executable": fake_bin, "metrics": ["valid_metric", "; rm -rf /"]})
        assert "error" in result["parsed_metrics"]
        assert "Invalid metric" in result["parsed_metrics"]["error"]


# ── compile_cuda Tests ───────────────────────────────────────────────


class TestCompileCudaHandler:
    def test_empty_source(self):
        from src.infrastructure.tools.compile_cuda import compile_cuda_handler
        result = compile_cuda_handler({"source": "", "flags": []})
        assert result["success"] is False
        assert "No source" in result["errors"]

    def test_nvcc_not_found(self):
        from src.infrastructure.tools.compile_cuda import compile_cuda_handler
        with patch("src.infrastructure.tools.compile_cuda.shutil.which", return_value=None):
            result = compile_cuda_handler({"source": "int main(){}", "flags": []})
        assert result["success"] is False
        assert "not found" in result["errors"].lower()

    def test_compilation_success(self):
        from src.infrastructure.tools.compile_cuda import compile_cuda_handler
        sandbox = MagicMock()
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_result.returncode = 0
        mock_result.success = True
        sandbox.run.return_value = mock_result
        sandbox._sandbox_root = "/tmp/.sandbox"
        with patch("src.infrastructure.tools.compile_cuda.shutil.which", return_value="/usr/bin/nvcc"):
            result = compile_cuda_handler(
                {"source": "__global__ void k(){}", "flags": ["-O3"]},
                sandbox=sandbox,
            )
        assert result["success"] is True
        assert "benchmark" in result["binary_path"]

    def test_compilation_failure(self):
        from src.infrastructure.tools.compile_cuda import compile_cuda_handler
        sandbox = MagicMock()
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = "error: expected ';'"
        mock_result.returncode = 1
        mock_result.success = False
        sandbox.run.return_value = mock_result
        sandbox._sandbox_root = "/tmp/.sandbox"
        with patch("src.infrastructure.tools.compile_cuda.shutil.which", return_value="/usr/bin/nvcc"):
            result = compile_cuda_handler(
                {"source": "bad code", "flags": []},
                sandbox=sandbox,
            )
        assert result["success"] is False


# ── execute_binary Tests ─────────────────────────────────────────────


class TestExecuteBinaryHandler:
    def test_empty_binary_path(self):
        from src.infrastructure.tools.execute_binary import execute_binary_handler
        result = execute_binary_handler({"binary_path": "", "args": []})
        assert result["return_code"] == -1
        assert "No binary" in result["stderr"]

    def test_binary_not_found(self):
        from src.infrastructure.tools.execute_binary import execute_binary_handler
        result = execute_binary_handler({"binary_path": "/nonexistent/binary", "args": []})
        assert result["return_code"] == -1
        assert "not found" in result["stderr"]

    def test_execution_success(self, tmp_path):
        from src.infrastructure.tools.execute_binary import execute_binary_handler
        fake_bin = str(tmp_path / "fake_binary")
        with open(fake_bin, "w") as f:
            f.write("fake")
        from src.infrastructure.sandbox import SandboxResult
        mock_sandbox = MagicMock()
        mock_sandbox.run.return_value = SandboxResult(
            stdout="output data",
            stderr="",
            return_code=0,
            success=True,
        )
        result = execute_binary_handler({"binary_path": fake_bin, "args": ["--flag"]}, sandbox=mock_sandbox)
        assert result["stdout"] == "output data"
        assert result["return_code"] == 0


# ── File Tools Tests ─────────────────────────────────────────────────


class TestFileTools:
    def test_read_file_handler(self, tmp_path):
        from src.infrastructure.file_ops import FileOperations
        from src.infrastructure.tools.file_tools import make_read_file_handler
        sandbox = str(tmp_path / "sandbox")
        os.makedirs(sandbox)
        test_file = str(tmp_path / "sandbox" / "test.txt")
        with open(test_file, "w") as f:
            f.write("line1\nline2\nline3\n")
        file_ops = FileOperations(sandbox_root=sandbox)
        # Use absolute path inside sandbox
        file_ops.read(test_file)
        handler = make_read_file_handler(file_ops)
        result = handler({"file_path": test_file})
        assert "line1" in result["content"]
        assert result["lines"] == 3

    def test_write_file_handler_m1_compliant(self, tmp_path):
        from src.infrastructure.file_ops import FileOperations
        from src.infrastructure.tools.file_tools import make_write_file_handler
        sandbox = str(tmp_path / "sandbox")
        os.makedirs(sandbox)
        test_file = str(tmp_path / "sandbox" / "test.txt")
        with open(test_file, "w") as f:
            f.write("original\n")
        file_ops = FileOperations(sandbox_root=sandbox)
        file_ops.read(test_file)
        handler = make_write_file_handler(file_ops)
        result = handler({"file_path": test_file, "content": "new content"})
        assert result["bytes_written"] > 0
        assert "error" not in result

    def test_write_file_handler_m1_violation_reports_error(self, tmp_path):
        """VULN-P4-1 fix: M1 violation should raise PermissionError, not silently create."""
        from src.infrastructure.file_ops import FileOperations
        from src.infrastructure.tools.file_tools import make_write_file_handler
        sandbox = str(tmp_path / "sandbox")
        os.makedirs(sandbox)
        # Create file so it exists on disk (path won't escape sandbox)
        test_file = str(tmp_path / "sandbox" / "existing.txt")
        with open(test_file, "w") as f:
            f.write("original\n")
        file_ops = FileOperations(sandbox_root=sandbox)
        # Do NOT read the file — try to write directly (M1 violation)
        handler = make_write_file_handler(file_ops)
        with pytest.raises(PermissionError, match="M1 violation"):
            handler({"file_path": test_file, "content": "should fail"})


# ── Microbenchmark Tests ─────────────────────────────────────────────


class TestMicrobenchmarkHandler:
    def test_pointer_chase_kernel(self):
        from src.infrastructure.tools.microbenchmark import generate_microbenchmark_handler
        result = generate_microbenchmark_handler({
            "benchmark_type": "pointer_chase",
            "parameters": {"iterations": 50000},
        })
        assert "pointer_chase" in result["source_code"]
        assert "50000" in result["source_code"]
        assert "__global__" in result["source_code"]

    def test_working_set_kernel(self):
        from src.infrastructure.tools.microbenchmark import generate_microbenchmark_handler
        result = generate_microbenchmark_handler({
            "benchmark_type": "working_set",
            "parameters": {"size": 256},
        })
        assert "working_set" in result["source_code"]

    def test_timing_loop_kernel(self):
        from src.infrastructure.tools.microbenchmark import generate_microbenchmark_handler
        result = generate_microbenchmark_handler({
            "benchmark_type": "timing_loop",
            "parameters": {"iterations": 999999},
        })
        assert "timing_loop" in result["source_code"]

    def test_stream_kernel(self):
        from src.infrastructure.tools.microbenchmark import generate_microbenchmark_handler
        result = generate_microbenchmark_handler({
            "benchmark_type": "stream",
            "parameters": {},
        })
        assert "stream_copy" in result["source_code"]

    def test_unknown_type_falls_back_to_generic(self):
        from src.infrastructure.tools.microbenchmark import generate_microbenchmark_handler
        result = generate_microbenchmark_handler({
            "benchmark_type": "unknown_type",
            "parameters": {},
        })
        assert "generic_kernel" in result["source_code"]

    def test_source_code_is_nonempty(self):
        from src.infrastructure.tools.microbenchmark import generate_microbenchmark_handler
        result = generate_microbenchmark_handler({
            "benchmark_type": "pointer_chase",
            "parameters": {},
        })
        assert len(result["source_code"]) > 0
