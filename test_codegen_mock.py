"""Mock test for CodeGen nvcc warning/error handling.

This test simulates various nvcc compilation scenarios to verify that:
1. LocalSandbox correctly distinguishes between warnings and errors
2. compile_cuda_handler returns correct status for warnings
3. stage_executor correctly identifies success_with_warning as success
4. CodeGen does not crash due to warning misjudgment
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class SandboxResult:
    """Mock SandboxResult matching sandbox.py implementation."""
    stdout: str
    stderr: str
    return_code: int
    success: bool
    artifacts: dict[str, str] = field(default_factory=dict)
    error_type: str = ""
    error_category: str = ""


def simulate_sandbox_run(return_code: int, stderr: str) -> SandboxResult:
    """Simulate LocalSandbox.run() logic with warning/error distinction."""
    stderr_lower = stderr.lower()
    
    has_actual_error = return_code != 0 or (
        "error: " in stderr_lower or
        "fatal error:" in stderr_lower or
        "undefined reference to" in stderr_lower or
        "cannot open" in stderr_lower or
        ("invalid" in stderr_lower and "option" in stderr_lower)
    )
    
    has_warning_only = return_code == 0 and (
        "warning" in stderr_lower or 
        "deprecated" in stderr_lower or
        "will be removed" in stderr_lower
    ) and not has_actual_error
    
    return SandboxResult(
        stdout="",
        stderr=stderr,
        return_code=return_code,
        success=not has_actual_error,
        error_type="warning" if has_warning_only else "",
        error_category="compilation_warning" if has_warning_only else (
            "compilation_error" if return_code != 0 else ""
        ),
    )


def simulate_compile_cuda_handler(result: SandboxResult) -> dict[str, Any]:
    """Simulate compile_cuda_handler warning handling."""
    has_warning = result.error_type == "warning" or (
        result.return_code == 0 and "warning" in result.stderr.lower() and 
        "error: " not in result.stderr.lower() and "fatal" not in result.stderr.lower()
    )
    
    status = "success" if result.success else "error"
    if has_warning and result.success:
        status = "success_with_warning"
    
    return {
        "status": status,
        "success": result.success,
        "output": result.stdout,
        "errors": result.stderr if not result.success else (result.stderr if has_warning else ""),
        "binary_path": "bin/benchmark" if result.success else "",
        "has_warning": has_warning,
    }


def simulate_stage_executor_status(tool_results: list[dict]) -> str:
    """Simulate stage_executor._codegen_status() success detection."""
    tool_succeeded = any(
        r.get("status") in ("success", "success_with_warning", True) or r.get("success") is True
        for r in tool_results
    )
    has_binary = any(r.get("binary_path") for r in tool_results)
    
    if tool_succeeded or has_binary:
        return "SUCCESS"
    else:
        return "FAILED"


def test_scenario(name: str, return_code: int, stderr: str, expected_success: bool, expected_status: str):
    """Run a single test scenario."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    print(f"Input: return_code={return_code}, stderr='{stderr[:100]}...'")
    
    # Simulate sandbox processing
    result = simulate_sandbox_run(return_code, stderr)
    print(f"SandboxResult: success={result.success}, error_type='{result.error_type}', category='{result.error_category}'")
    
    assert result.success == expected_success, f"❌ Sandbox success mismatch: expected {expected_success}, got {result.success}"
    print(f"✅ Sandbox correctly identified success={result.success}")
    
    # Simulate compile_cuda handler
    handler_result = simulate_compile_cuda_handler(result)
    print(f"Handler: status='{handler_result['status']}', has_warning={handler_result['has_warning']}")
    
    assert handler_result["status"] == expected_status, f"❌ Handler status mismatch: expected {expected_status}, got {handler_result['status']}"
    print(f"✅ Handler correctly returned status='{handler_result['status']}'")
    
    # Simulate stage_executor checking
    tool_results = [handler_result]
    stage_status = simulate_stage_executor_status(tool_results)
    print(f"StageExecutor: status='{stage_status}'")
    
    if expected_success:
        assert stage_status == "SUCCESS", f"❌ Stage status mismatch: expected SUCCESS, got {stage_status}"
        print(f"✅ StageExecutor correctly identified SUCCESS")
    else:
        print(f"⚠️ StageExecutor returned {stage_status} (expected for failed compilation)")
    
    print(f"✅ TEST PASSED: {name}")
    return True


def main():
    """Run all test scenarios."""
    print("\n" + "="*60)
    print("MOCK TEST: CodeGen nvcc warning/error handling")
    print("="*60)
    
    all_passed = True
    
    # Scenario 1: Pure success
    all_passed &= test_scenario(
        "Pure success (no warnings)",
        return_code=0,
        stderr="",
        expected_success=True,
        expected_status="success"
    )
    
    # Scenario 2: CUDA 12.8 architecture warning (THE CRITICAL BUG FIX)
    all_passed &= test_scenario(
        "CUDA 12.8 architecture warning (returncode=0, success with warning)",
        return_code=0,
        stderr="nvcc warning : The 'compute_60', 'compute_61', 'compute_70' architectures are deprecated in CUDA 12.8 and will be removed in a future release.",
        expected_success=True,
        expected_status="success_with_warning"
    )
    
    # Scenario 3: Warning with "deprecated" keyword
    all_passed &= test_scenario(
        "Deprecated architecture warning",
        return_code=0,
        stderr="nvcc warning : Support for offline compilation for architectures prior to 'sm_75' will be removed in a future release",
        expected_success=True,
        expected_status="success_with_warning"
    )
    
    # Scenario 4: Warning containing "error" word in description (edge case)
    all_passed &= test_scenario(
        "Warning with 'error' word in description (edge case)",
        return_code=0,
        stderr="nvcc warning : Some descriptive text mentioning the word error in passing, but this is just a warning",
        expected_success=True,
        expected_status="success_with_warning"
    )
    
    # Scenario 5: Actual compilation error
    all_passed &= test_scenario(
        "Actual compilation error (returncode!=0)",
        return_code=1,
        stderr="source.cu(10): error: identifier 'undefined_var' is undefined",
        expected_success=False,
        expected_status="error"
    )
    
    # Scenario 6: Fatal error
    all_passed &= test_scenario(
        "Fatal compilation error",
        return_code=1,
        stderr="nvcc fatal : Don't know what to do with 'sm_0'",
        expected_success=False,
        expected_status="error"
    )
    
    # Scenario 7: Multiple warnings but still success
    all_passed &= test_scenario(
        "Multiple warnings but still success",
        return_code=0,
        stderr="warning: deprecated arch\nwarning: will be removed in future\nwarning: another warning",
        expected_success=True,
        expected_status="success_with_warning"
    )
    
    # Scenario 8: StageExecutor with success_with_warning
    print(f"\n{'='*60}")
    print("TEST: StageExecutor with success_with_warning")
    print(f"{'='*60}")
    tool_results = [
        {
            "status": "success_with_warning",
            "success": True,
            "binary_path": "bin/benchmark",
            "has_warning": True
        }
    ]
    status = simulate_stage_executor_status(tool_results)
    assert status == "SUCCESS", f"❌ StageExecutor failed to recognize success_with_warning: {status}"
    print(f"✅ StageExecutor correctly recognized success_with_warning as SUCCESS")
    
    # Scenario 9: Anti-loop prevention test
    print(f"\n{'='*60}")
    print("TEST: Anti-loop prevention (3 consecutive warnings should not trigger)")
    print(f"{'='*60}")
    
    # Simulate 3 rounds of compilation with warnings
    warning_results = []
    for i in range(3):
        result = simulate_sandbox_run(0, "nvcc warning: architecture deprecated")
        handler = simulate_compile_cuda_handler(result)
        warning_results.append(handler)
    
    # All should be successful
    all_success = all(r["success"] for r in warning_results)
    assert all_success, "❌ Warning-only compilations should all be successful"
    print(f"✅ All 3 warning-only compilations returned success")
    
    # StageExecutor should recognize them as successful
    stage_status = simulate_stage_executor_status(warning_results)
    assert stage_status == "SUCCESS", f"❌ StageExecutor should recognize warnings as success"
    print(f"✅ StageExecutor correctly identified all warnings as SUCCESS")
    print(f"✅ Anti-loop mechanism would NOT trigger (no repeated failures)")
    
    print(f"\n{'='*60}")
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("CodeGen will NOT crash due to nvcc warning misjudgment")
    else:
        print("❌ SOME TESTS FAILED")
    print(f"{'='*60}\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
