"""
BUG#8 Completion Threshold Fix — Unit Tests

验证StageExecutor引入80%完成度阈值后，部分完成的工作不再被错误标记为SUCCESS。
这是解决67%完成度被接受为SUCCESS的关键修复。

运行: pytest tests/test_bug8_completion_threshold.py -v
"""
import pytest
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.domain.stage_executor import StageExecutor
from src.domain.enums import SubAgentStatus


class TestBug8CompletionThreshold:
    """
    BUG#8: Verify 80% completion threshold enforcement.

    OLD BEHAVIOR (BUG):
      if len(measured_keys) > 0:
          status = SubAgentStatus.SUCCESS  # ← Accepts ANY positive count!

    NEW BEHAVIOR (FIXED):
      completion_rate = len(measured_keys) / len(requested_targets)
      if completion_rate >= 0.8:
          status = SubAgentStatus.SUCCESS
      elif completion_rate > 0:
          status = SubAgentStatus.PARTIAL  # ← New status for partial work
      else:
          status = SubAgentStatus.FAILED
    """

    def test_full_completion_returns_success(self):
        """
        TC-8.1: 100% of targets measured → SUCCESS.

        This is the ideal case where all requested targets have measurements.
        Should always return SUCCESS regardless of threshold.
        """
        target_spec = {
            "targets": ["dram_latency_cycles", "sm_count", "l2_cache_size_mb"]
        }
        tool_results = [
            {
                "tool": "execute_binary",
                "stdout": "dram_latency_cycles: 450.5\nsm_count: 56.0\nl2_cache_size_mb: 4.0",
                "return_code": 0,
                "binary_path": "/tmp/binary"
            }
        ]
        data = {}

        status = StageExecutor._codegen_status(
            final_text="✅ All measurements complete",
            tool_results=tool_results,
            data=data,
            target_spec=target_spec
        )

        assert status == SubAgentStatus.SUCCESS, \
            f"Expected SUCCESS for 100% completion, got {status}"

        print("✅ TC-8.1 PASSED: 100% completion → SUCCESS")

    def test_67_percent_returns_partial_core_fix(self):
        """
        TC-8.2: 67% measured → PARTIAL (THE CORE FIX).

        THIS IS THE MOST CRITICAL TEST CASE.

        Original bug scenario from incident report:
          - Requested: [dram_latency_cycles, sm_count, actual_boost_clock_mhz] (3 targets)
          - Measured:   [dram_latency_cycles: 450.5, sm_count: 56.0] (2 targets)
          - Missing:    actual_boost_clock_mhz
          - Old result: SUCCESS ❌ (wrong!)
          - New result: PARTIAL ✅ (correct!)

        The old code accepted any len(measured_keys) > 0 as SUCCESS,
        which allowed 67% completion to pass validation.
        """
        target_spec = {
            "targets": ["dram_latency_cycles", "sm_count", "actual_boost_clock_mhz"]
        }

        # Only 2 out of 3 targets have measurements (missing actual_boost_clock_mhz!)
        tool_results = [
            {
                "tool": "execute_binary",
                "stdout": "dram_latency_cycles: 450.5\nsm_count: 56.0",
                "return_code": 0,
                "binary_path": "/tmp/binary"
            }
        ]
        data = {}

        status = StageExecutor._codegen_status(
            final_text="Partial results available",
            tool_results=tool_results,
            data=data,
            target_spec=target_spec
        )

        # ===== THE CRITICAL ASSERTION =====
        # Must be PARTIAL, NOT SUCCESS!
        assert status == SubAgentStatus.PARTIAL, \
            f"❌ FAIL: Expected PARTIAL but got {status}. " \
            f"Old bug would incorrectly return SUCCESS here!"

        # Verify completion rate is recorded correctly
        assert "completion_rate" in data, "completion_rate should be in data dict"
        expected_rate = 2 / 3  # ≈ 0.6667
        actual_rate = data["completion_rate"]
        assert abs(actual_rate - expected_rate) < 0.01, \
            f"Expected completion_rate≈{expected_rate:.4f}, got {actual_rate:.4f}"

        # Verify error detail explains the issue
        assert "error_detail" in data, "error_detail should explain partial completion"
        error_detail = data["error_detail"]
        assert "67%" in error_detail or "66" in error_detail, \
            f"error_detail should mention ~67%, got: {error_detail}"
        assert "actual_boost_clock_mhz" in error_detail, \
            f"error_detail should list missing target, got: {error_detail}"

        print(f"\n✅ TC-8.2 PASSED: 67% completion → PARTIAL (was SUCCESS before fix!)")
        print(f"   Completion rate: {actual_rate*100:.1f}%")
        print(f"   Error detail: {error_detail[:100]}...")

    def test_83_percent_returns_success_boundary(self):
        """
        TC-8.3: 83.3% measured → SUCCESS (boundary test just above threshold).

        Tests the boundary condition: 5 out of 6 targets = 83.3% ≥ 80%.
        This should be accepted as SUCCESS since it meets the minimum threshold.
        """
        num_targets = 6
        num_measured = 5  # 5/6 = 83.3%

        target_spec = {
            "targets": [f"target_{i}" for i in range(num_targets)]
        }

        # Generate stdout with 5 measurements (missing target_5)
        measured_lines = "\n".join([f"target_{i}: {i * 10.0}" for i in range(num_measured)])
        # execute_binary result should NOT have binary_path (only compile_cuda does)
        tool_results = [
            {
                "tool": "execute_binary",
                "stdout": measured_lines,
                "return_code": 0,
                # NOTE: No binary_path here - this is execution result, not compilation
            }
        ]
        data = {}

        status = StageExecutor._codegen_status(
            final_text="Most measurements complete",
            tool_results=tool_results,
            data=data,
            target_spec=target_spec
        )

        # Should be SUCCESS because 83.3% ≥ 80%
        assert status == SubAgentStatus.SUCCESS, \
            f"Expected SUCCESS for 83.3% completion, got {status}"

        # Verify completion rate is recorded
        assert "completion_rate" in data
        expected_rate = num_measured / num_targets  # 0.8333...
        actual_rate = data["completion_rate"]
        assert abs(actual_rate - expected_rate) < 0.01

        print(f"✅ TC-8.3 PASSED: {num_measured}/{num_targets} ({actual_rate*100:.1f}%) → SUCCESS")

    def test_79_percent_returns_partial_boundary(self):
        """
        TC-8.3b: 79% measured → PARTIAL (boundary test just below threshold).

        Tests the other side of the boundary: 7 out of 9 targets ≈ 77.8% < 80%.
        This should be rejected as PARTIAL since it's below the threshold.
        """
        num_targets = 9
        num_measured = 7  # 7/9 ≈ 77.8%

        target_spec = {
            "targets": [f"target_{i}" for i in range(num_targets)]
        }

        measured_lines = "\n".join([f"target_{i}: {i * 10.0}" for i in range(num_measured)])
        # execute_binary result format (no binary_path)
        tool_results = [
            {
                "tool": "execute_binary",
                "stdout": measured_lines,
                "return_code": 0,
            }
        ]
        data = {}

        status = StageExecutor._codegen_status(
            final_text="Some measurements missing",
            tool_results=tool_results,
            data=data,
            target_spec=target_spec
        )

        # Should be PARTIAL because 77.8% < 80%
        assert status == SubAgentStatus.PARTIAL, \
            f"Expected PARTIAL for {num_measured/num_targets*100:.1f}% completion, got {status}"

        assert "completion_rate" in data
        actual_rate = data["completion_rate"]
        assert actual_rate < 0.8, "Should be below 80% threshold"

        print(f"✅ TC-8.3b PASSED: {num_measured}/{num_targets} ({actual_rate*100:.1f}%) → PARTIAL")

    def test_zero_completion_returns_failed(self):
        """
        TC-8.4: 0% measured → FAILED.

        When no measurements exist at all, this is a complete failure,
        not even partial success.
        """
        target_spec = {
            "targets": ["target_x", "target_y", "target_z"]
        }
        # No measurements in stdout - compilation succeeded but no execution
        tool_results = [
            {
                "tool": "compile_cuda",
                "success": True,
                "binary_path": "/tmp/binary",
                # NOTE: No stdout with measurements!
            }
        ]
        data = {}

        status = StageExecutor._codegen_status(
            final_text="No measurements obtained",
            tool_results=tool_results,
            data=data,
            target_spec=target_spec
        )

        assert status == SubAgentStatus.FAILED, \
            f"Expected FAILED for 0% completion, got {status}"

        print("✅ TC-8.4 PASSED: 0% completion → FAILED")

    def test_exact_80_percent_returns_success(self):
        """
        TC-8.5: Exactly 80% measured → SUCCESS (exact threshold boundary).

        Test that exactly 80% (4 out of 5) is accepted as SUCCESS.
        """
        num_targets = 5
        num_measured = 4  # 4/5 = 80.0%

        target_spec = {
            "targets": [f"t{i}" for i in range(num_targets)]
        }

        measured_lines = "\n".join([f"t{i}: {i}" for i in range(num_measured)])
        # execute_binary result format (no binary_path)
        tool_results = [
            {
                "tool": "execute_binary",
                "stdout": measured_lines,
                "return_code": 0,
            }
        ]
        data = {}

        status = StageExecutor._codegen_status(
            final_text="At threshold",
            tool_results=tool_results,
            data=data,
            target_spec=target_spec
        )

        # Exactly 80% should be SUCCESS (>= not >)
        assert status == SubAgentStatus.SUCCESS, \
            f"Expected SUCCESS for exact 80%, got {status}"

        print("✅ TC-8.5 PASSED: Exactly 80% → SUCCESS (threshold inclusive)")

    def test_partial_status_exists_in_enum(self):
        """TC-8.6: Verify PARTIAL status was added to SubAgentStatus enum."""
        assert hasattr(SubAgentStatus, 'PARTIAL'), \
            "SubAgentStatus enum must have PARTIAL value"

        assert SubAgentStatus.PARTIAL.value == "partial", \
            f"PARTIAL value should be 'partial', got '{SubAgentStatus.PARTIAL.value}'"

        # Verify it's distinct from other statuses
        assert SubAgentStatus.PARTIAL != SubAgentStatus.SUCCESS
        assert SubAgentStatus.PARTIAL != SubAgentStatus.FAILED
        assert SubAgentStatus.PARTIAL != SubAgentStatus.REJECTED
        assert SubAgentStatus.PARTIAL != SubAgentStatus.PENDING
        assert SubAgentStatus.PARTIAL != SubAgentStatus.RUNNING

        print("✅ TC-8.6 PASSED: PARTIAL status exists and is unique in enum")


class TestBug8RealWorldScenarios:
    """Tests based on real-world pipeline scenarios."""

    def test_three_targets_two_measured_original_bug(self):
        """
        Reproduce EXACT original bug scenario:

        Incident report stated:
          "本次测试：只完成2/3目标（67%），仍被标记为SUCCESS"

        This test ensures that specific scenario now returns PARTIAL.
        """
        # Exact targets from incident report
        target_spec = {
            "targets": [
                "dram_latency_cycles",
                "sm_count",
                "actual_boost_clock_mhz"
            ]
        }

        # Exact measurements from incident (missing actual_boost_clock_mhz)
        tool_results = [
            {
                "tool": "execute_binary",
                "stdout": (
                    "dram_latency_cycles: 452.3\n"
                    "sm_count: 56.0\n"
                    # NOTE: actual_boost_clock_mhz is MISSING!
                    # This was the bug: 2/3 = 67% was marked SUCCESS
                ),
                "return_code": 0,
                "binary_path": "/tmp/binary"
            }
        ]
        data = {}

        status = StageExecutor._codegen_status(
            final_text="CodeGen completed with partial results",
            tool_results=tool_results,
            data=data,
            target_spec=target_spec
        )

        # THE FIX: Must be PARTIAL now, not SUCCESS like before!
        assert status == SubAgentStatus.PARTIAL, \
            f"\n{'='*60}\n" \
            f"❌ ORIGINAL BUG REPRODUCED!\n" \
            f"   Expected: PARTIAL (2/3 = 67% < 80%)\n" \
            f"   Got: {status}\n" \
            f"   The BUG#8 fix did NOT work correctly!\n" \
            f"{'='*60}"

        print("\n" + "="*60)
        print("✅ ORIGINAL BUG SCENARIO NOW FIXED!")
        print(f"   Targets: {target_spec['targets']}")
        print(f"   Measured: ['dram_latency_cycles', 'sm_count']")
        print(f"   Missing: ['actual_boost_clock_mhz']")
        print(f"   Status: {status} (was SUCCESS before fix)")
        print("="*60)

    def test_single_target_measured(self):
        """Edge case: only 1 target requested and measured → 100% success."""
        target_spec = {"targets": ["only_target"]}
        # execute_binary result format (no binary_path)
        tool_results = [
            {
                "tool": "execute_binary",
                "stdout": "only_target: 42.0",
                "return_code": 0,
            }
        ]
        data = {}

        status = StageExecutor._codegen_status(
            final_text="Single target done",
            tool_results=tool_results,
            data=data,
            target_spec=target_spec
        )

        assert status == SubAgentStatus.SUCCESS
        # completion_rate may or may not be present for 100% case (it's SUCCESS anyway)
        if "completion_rate" in data:
            assert data["completion_rate"] == 1.0

        print("✅ REAL-WORLD: Single target 100% → SUCCESS")


class TestBug8DataIntegrity:
    """Verify data dictionary integrity for downstream consumers."""

    def test_partial_status_includes_completion_rate(self):
        """When PARTIAL, data must include numeric completion_rate."""
        target_spec = {"targets": ["a", "b", "c"]}
        tool_results = [
            {
                "tool": "execute_binary",
                "stdout": "a: 1",
                "return_code": 0,
            }
        ]  # 33.3%
        data = {}

        status = StageExecutor._codegen_status(
            final_text="test",
            tool_results=tool_results,
            data=data,
            target_spec=target_spec
        )

        if status == SubAgentStatus.PARTIAL:
            assert "completion_rate" in data, \
                "PARTIAL status must include completion_rate in data"
            assert isinstance(data["completion_rate"], float), \
                "completion_rate must be float"
            assert 0 < data["completion_rate"] < 1, \
                f"completion_rate should be in (0,1), got {data['completion_rate']}"

            print(f"✅ DATA INTEGRITY: completion_rate={data['completion_rate']:.4f}")

    def test_partial_status_includes_error_detail(self):
        """When PARTIAL, data must include explanatory error_detail."""
        target_spec = {"targets": ["x", "y"]}
        tool_results = [
            {
                "tool": "execute_binary",
                "stdout": "x: 1",
                "return_code": 0,
            }
        ]  # 50%
        data = {}

        status = StageExecutor._codegen_status(
            final_text="test",
            tool_results=tool_results,
            data=data,
            target_spec=target_spec
        )

        if status == SubAgentStatus.PARTIAL:
            assert "error_detail" in data, \
                "PARTIAL status must include error_detail explaining the issue"
            assert len(data["error_detail"]) > 0, \
                "error_detail should not be empty"
            assert "50%" in data["error_detail"] or "y" in data["error_detail"], \
                f"error_detail should mention missing targets: {data['error_detail']}"

            print(f"✅ DATA INTEGRITY: error_detail present and informative")

    def test_success_status_no_error_detail_required(self):
        """When SUCCESS, error_detail may or may not be present (not required)."""
        target_spec = {"targets": ["a"]}
        tool_results = [{"stdout": "a: 1"}]
        data = {}

        status = StageExecutor._codegen_status(
            final_text="test",
            tool_results=tool_results,
            data=data,
            target_spec=target_spec
        )

        assert status == SubAgentStatus.SUCCESS
        # error_detail optional for SUCCESS
        print("✅ DATA INTEGRITY: SUCCESS status (error_detail optional)")


if __name__ == "__main__":
    # Run all tests with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-s"])
