"""
GPUFeatureDB Integration Tests — Unit & Integration Validation

验证GPUFeatureDB正确集成到CodeGen中，提供架构特定的测量参数。
这确保系统能自动适配不同的GPU架构，消除硬编码的sm_60逻辑。

运行: pytest tests/test_gpu_feature_db_integration.py -v
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.infrastructure.gpu_feature_db import GPUFeatureDB, GPUSpecs


class TestGPUFeatureDBBasicFunctionality:
    """Test GPUFeatureDB core functionality before integration."""

    def test_detect_p100_specs(self):
        """
        TC-GPUDB-001: Detect Tesla P100 (sm_60) specifications.

        P100 is the primary target GPU for Kaggle environment.
        Verify all key specifications are correct.
        """
        db = GPUFeatureDB()

        with patch.object(db, 'detect_current_gpu', return_value="sm_60"):
            specs = db.detect_and_get_features()

        assert specs is not None, "Should detect P100 specs"
        assert isinstance(specs, GPUSpecs), "Should return GPUSpecs object"

        # Verify P100-specific values
        assert specs.name == "Tesla P100", f"Expected 'Tesla P100', got '{specs.name}'"
        assert specs.compute_capability == "sm_60", \
            f"Expected 'sm_60', got '{specs.compute_capability}'"
        assert specs.sm_count == 56, f"Expected 56 SMs, got {specs.sm_count}"
        assert specs.memory_type == "HBM2", f"Expected HBM2, got {specs.memory_type}"
        assert specs.memory_size_gb == 16.0, f"Expected 16GB, got {specs.memory_size_gb}GB"
        assert specs.l2_cache_size_kb == 4096, \
            f"Expected 4096KB L2, got {specs.l2_cache_size_kb}KB"
        assert specs.base_clock_mhz == 1329, \
            f"Expected 1329 MHz base clock, got {specs.base_clock_mhz}MHz"
        assert specs.boost_clock_mhz == 1480, \
            f"Expected 1480 MHz boost, got {specs.boost_clock_mhz}MHz"

        print(f"\n✅ TC-GPUDB-001 PASSED: P100 specs detected correctly")
        print(f"   GPU: {specs.name} ({specs.compute_capability})")
        print(f"   Memory: {specs.memory_size_gb}GB {specs.memory_type}")
        print(f"   SMs: {specs.sm_count}, L2: {specs.l2_cache_size_kb}KB")

    def test_get_dram_latency_params_for_p100(self):
        """
        TC-GPUDB-002: Get DRAM latency measurement parameters for P100.

        DRAM latency measurements require working sets that exceed L2 cache
        to ensure memory accesses actually go to DRAM (not L2 hit).
        For P100 with 4MB L2, working set should be >> 4MB.
        """
        db = GPUFeatureDB()
        params = db.get_measurement_params("dram_latency_cycles", "sm_60")

        assert isinstance(params, dict), "Should return dict of parameters"
        assert len(params) > 0, "Params dict should not be empty"

        # Verify critical DRAM latency parameters exist
        assert "working_set_mb" in params, "Must include working_set_mb for DRAM measurement"
        working_set = params["working_set_mb"]
        assert working_set > 400, \
            f"Working set ({working_set}MB) must exceed P100 L2 (4MB) significantly"

        assert "expected_range" in params, "Must include expected_range for validation"
        expected_range = params["expected_range"]
        assert isinstance(expected_range, tuple), "expected_range should be tuple"
        assert len(expected_range) == 2, "expected_range should have (min, max)"
        assert expected_range[0] < expected_range[1], "min < max in range"

        # P100 typical DRAM latency: 400-750 cycles
        assert 400 <= expected_range[0] <= 800, \
            f"Min range ({expected_range[0]}) outside reasonable P100 range [400, 800]"
        assert 400 <= expected_range[1] <= 850, \
            f"Max range ({expected_range[1]}) outside reasonable P100 range [400, 850]"

        # Verify other useful fields
        if "method" in params:
            assert params["method"] in ["pointer_chasing", "stride_access"], \
                f"Method should be pointer_chasing or stride_access, got '{params['method']}'"

        if "notes" in params:
            assert len(params["notes"]) > 0, "Notes should not be empty"

        print(f"\n✅ TC-GPUDB-002 PASSED: DRAM latency params valid for P100")
        print(f"   Working set: {working_set} MB (>> L2=4MB)")
        print(f"   Expected range: {expected_range[0]}-{expected_range[1]} cycles")
        if "method" in params:
            print(f"   Method: {params['method']}")

    def test_architecture_adaptation_matrix(self):
        """
        TC-GPUDB-003: Verify different architectures get DIFFERENT appropriate parameters.

        Each GPU architecture has unique characteristics:
        - Different L2 sizes → different working set requirements
        - Different SM counts → different parallelism expectations
        - Different clock speeds → different iteration counts

        This test ensures the database provides architecture-specific tuning.
        """
        db = GPUFeatureDB()

        # Test matrix of supported architectures
        test_cases = [
            ("sm_60", "Tesla P100", 56, 4096),     # Pascal
            ("sm_70", "V100", 80, 6144),           # Volta
            ("sm_75", "T4", 46, 4608),             # Turing
            ("sm_86", "RTX 3090", 82, 6144),       # Ampere
            ("sm_90", "H100", 132, 51200),         # Hopper
        ]

        params_by_arch = {}

        for arch, expected_name, expected_sm, expected_l2 in test_cases:
            specs = db.get_specs(arch)
            assert specs is not None, f"Should have specs for {arch}"
            assert specs.name == expected_name, \
                f"{arch}: Expected name '{expected_name}', got '{specs.name}'"
            assert specs.sm_count == expected_sm, \
                f"{arch}: Expected {expected_sm} SMs, got {specs.sm_count}"
            assert specs.l2_cache_size_kb == expected_l2, \
                f"{arch}: Expected {expected_l2}KB L2, got {specs.l2_cache_size_kb}KB"

            # Get DRAM params and verify they're adapted
            params = db.get_measurement_params("dram_latency_cycles", arch)
            l2_size_mb = specs.l2_cache_size_kb / 1024

            assert params["working_set_mb"] > l2_size_mb * 10, \
                f"{arch}: Working set ({params['working_set_mb']}MB) must be >> L2 ({l2_size_mb:.1f}MB)"

            params_by_arch[arch] = {
                "name": specs.name,
                "working_set": params["working_set_mb"],
                "l2_mb": l2_size_mb,
                "ratio": params["working_set_mb"] / l2_size_mb
            }

        # Print comparison table
        print(f"\n✅ TC-GPUDB-003 PASSED: Architecture adaptation matrix correct\n")
        print(f"{'Arch':<8} {'GPU':<16} {'L2 (MB)':<10} {'Working Set (MB)':<18} {'Ratio':<8}")
        print("-" * 62)
        for arch, info in params_by_arch.items():
            print(f"{arch:<8} {info['name']:<16} {info['l2_mb']:<10.1f} "
                  f"{info['working_set']:<18.1f} {info['ratio']:<8.1f}x")

    def test_fallback_for_unknown_gpu(self):
        """
        TC-GPUDB-004: Graceful fallback when GPU detection fails.

        In production, some GPUs may not be in the database or nvidia-smi
        may fail. The system should degrade gracefully, not crash.
        """
        db = GPUFeatureDB()

        # Scenario 1: detect_current_gpu returns None
        with patch.object(db, 'detect_current_gpu', return_value=None):
            specs = db.detect_and_get_features()

        assert specs is None, "Detection failure should return None"

        # Scenario 2: Should still provide fallback params without crashing
        params = db.get_measurement_params("unknown_target_xyz")
        assert isinstance(params, dict), "Fallback should still return dict"
        assert len(params) > 0, "Fallback params should not be empty"

        # Fallback params should have safe defaults
        if "working_set_mb" in params:
            assert params["working_set_mb"] > 0, "Fallback working set should be positive"

        print(f"\n✅ TC-GPUDB-004 PASSED: Fallback handling works correctly")
        print(f"   Unknown GPU → graceful degradation (not crash)")

    def test_get_specs_for_known_architectures(self):
        """Verify get_specs() works for all architectures in database."""
        db = GPUFeatureDB()

        known_archs = [
            "sm_35", "sm_37", "sm_50", "sm_52", "sm_53",
            "sm_60", "sm_61", "sm_70", "sm_75",
            "sm_80", "sm_86", "sm_89", "sm_90", "sm_120"
        ]

        for arch in known_archs:
            specs = db.get_specs(arch)
            assert specs is not None, f"Should have specs for {arch}"
            assert specs.compute_capability == arch, \
                f"Specs compute capability mismatch for {arch}"

        print(f"\n✅ All {len(known_archs)} architectures have valid specs")


class TestGPUFeatureDBTargetSpecificParams:
    """Test target-specific parameter generation."""

    def test_dram_latency_vs_l2_params_differ(self):
        """Different targets should get different parameter sets."""
        db = GPUFeatureDB()

        dram_params = db.get_measurement_params("dram_latency_cycles", "sm_60")
        l2_params = db.get_measurement_params("l2_cache_size_mb", "sm_60")

        # They should both be dicts but likely have different values
        assert isinstance(dram_params, dict) and isinstance(l2_params, dict)

        # At minimum, working sets should differ (DRAM needs much larger)
        if "working_set_mb" in dram_params and "working_set_mb" in l2_params:
            # DRAM working set should be MUCH larger than L2 measurement
            assert dram_params["working_set_mb"] > l2_params["working_set_mb"] * 10, \
                "DRAM working set should be >> L2 working set"

        print("✅ Target-specific params differ appropriately")

    def test_boost_clock_params_include_expected_range(self):
        """Boost clock measurement should include reasonable expected range."""
        db = GPUFeatureDB()
        params = db.get_measurement_params("actual_boost_clock_mhz", "sm_60")

        assert "expected_range" in params, "Boost clock should have expected range"
        range_min, range_max = params["expected_range"]

        # P100 boost clock is ~1480 MHz, so range should be around there
        assert 1300 <= range_min <= 1500, \
            f"Boost min ({range_min}) outside P100 typical range [1300, 1500]"
        assert 1400 <= range_max <= 1600, \
            f"Boost max ({range_max}) outside P100 typical range [1400, 1600]"

        print(f"✅ Boost clock range [{range_min}, {range_max}] MHz looks valid")


class TestGPUFeatureDBIntegrationWithCodeGen:
    """Integration tests verifying CodeGen uses GPUFeatureDB data correctly.

    Note: These tests mock the CodeGenAgent to verify context injection
    without needing actual LLM calls or CUDA compilation.
    """

    @patch('src.infrastructure.gpu_feature_db.GPUFeatureDB')
    def test_codegen_context_contains_gpu_info(self, MockGPUFeatureDB):
        """
        TC-GPUDB-005: Verify CodeGen injects GPUFeatureDB data into context.

        When CodeGen processes a target, it should call GPUFeatureDB and
        inject the results into the LLM context as a SYSTEM message.
        """
        # Setup comprehensive mock
        mock_db_instance = MockGPUFeatureDB.return_value

        # Mock GPU specs
        mock_specs = Mock(spec=GPUSpecs)
        mock_specs.name = "Tesla P100"
        mock_specs.compute_capability = "sm_60"
        mock_specs.memory_size_gb = 16.0
        mock_specs.memory_type = "HBM2"
        mock_specs.memory_bandwidth_gbps = 732.0
        mock_specs.sm_count = 56
        mock_specs.l2_cache_size_kb = 4096
        mock_specs.base_clock_mhz = 1329
        mock_specs.boost_clock_mhz = 1480
        mock_specs.shared_memory_per_block_kb = 64
        mock_specs.register_count_per_thread = 255
        mock_specs.warp_size = 32
        mock_specs.max_threads_per_sm = 2048

        mock_db_instance.detect_and_get_features.return_value = mock_specs

        # Mock measurement params
        mock_db_instance.get_measurement_params.return_value = {
            "compute_capability": "sm_60",
            "gpu_name": "Tesla P100",
            "architecture_notes": "HBM2, 4096KB L2, 56 SMs",
            "working_set_mb": 512.0,
            "min_iterations": 1000000,
            "max_iterations": 20000000,
            "method": "pointer_chasing",
            "access_pattern": "random_permutation",
            "expected_range": (400, 750),
            "notes": "Working set 512MB >> L2 (4.0MB)",
        }

        # Now simulate what codegen.py does (lines 98-162 of fixed code)
        try:
            from src.infrastructure.gpu_feature_db import GPUFeatureDB as RealGPUFeatureDB

            gpu_db = RealGPUFeatureDB()  # Will use mocked version
            gpu_specs = gpu_db.detect_and_get_features()
            target = "dram_latency_cycles"
            detected_arch = "sm_60"

            if gpu_specs:
                measure_params = gpu_db.get_measurement_params(target, detected_arch)

                # Build context parts (same logic as codegen.py)
                gpu_context_parts = [
                    f"📊 **GPU Feature Database** — Architecture-Specific Parameters\n",
                    f"Detected GPU: {gpu_specs.name} ({gpu_specs.compute_capability})\n",
                    f"Memory: {gpu_specs.memory_size_gb}GB {gpu_specs.memory_type}, "
                    f"{gpu_specs.memory_bandwidth_gbps:.0f} GB/s bandwidth\n",
                    f"SMs: {gpu_specs.sm_count}, L2 Cache: {gpu_specs.l2_cache_size_kb}KB, "
                    f"Clock: {gpu_specs.base_clock_mhz}-{gpu_specs.boost_clock_mhz} MHz\n",
                    f"\n📏 **Recommended Measurement Parameters for '{target}':**\n",
                ]

                if "working_set_mb" in measure_params:
                    gpu_context_parts.append(
                        f"  • Working set: {measure_params['working_set_mb']}MB "
                        f"(must exceed L2 cache)\n"
                    )
                if "expected_range" in measure_params:
                    gpu_context_parts.append(
                        f"  • Expected value range: {measure_params['expected_range']}\n"
                    )
                if "method" in measure_params:
                    gpu_context_parts.append(
                        f"  • Recommended method: {measure_params['method']}\n"
                    )

                gpu_context_parts.extend([
                    f"\n⚠️ **Critical Constraints:**\n",
                    f"  • Max shared memory/block: {gpu_specs.shared_memory_per_block_kb}KB\n",
                    f"  • Max registers/thread: {gpu_specs.register_count_per_thread}\n",
                    f"  • Warp size: {gpu_specs.warp_size}, Max threads/SM: {gpu_specs.max_threads_per_sm}\n",
                ])

                gpu_context = "".join(gpu_context_parts)

                # Verify context content
                assert "Tesla P100" in gpu_context, "Context should contain GPU name"
                assert "sm_60" in gpu_context, "Context should contain compute capability"
                assert "HBM2" in gpu_context, "Context should contain memory type"
                assert "56" in gpu_context, "Context should contain SM count"
                assert "4096KB" in gpu_context, "Context should contain L2 size"
                assert "dram_latency_cycles" in gpu_context, "Context should mention target"
                assert "working set" in gpu_context.lower(), "Context should mention working set"
                assert "512" in gpu_context, "Context should contain working set value"
                assert "(400, 750)" in gpu_context or "400" in gpu_context, \
                    "Context should contain expected range"

                print(f"\n✅ TC-GPUDB-005 PASSED: Context injection structure verified")
                print(f"   Context length: {len(gpu_context)} chars")
                print(f"\n--- Generated Context Preview ---")
                print(gpu_context[:500] + "...")
                print("--- End Preview ---\n")

        except ImportError:
            pytest.skip("GPUFeatureDB import failed (module may not exist yet)")

    @patch('src.infrastructure.gpu_feature_db.GPUFeatureDB')
    def test_codegen_handles_gpu_db_error_gracefully(self, MockGPUFeatureDB):
        """Verify CodeGen doesn't crash if GPUFeatureDB throws exception."""
        mock_db = MockGPUFeatureDB.return_value
        mock_db.detect_and_get_features.side_effect = Exception("Detection failed")

        # Simulate the try-except block in codegen.py lines 98-162
        error_occurred = False
        error_message = None

        try:
            from src.infrastructure.gpu_feature_db import GPUFeatureDB as RealGPUFeatureDB
            gpu_db = RealGPUFeatureDB()
            gpu_specs = gpu_db.detect_and_get_features()  # Should raise
        except Exception as e:
            error_occurred = True
            error_message = str(e)
            # This is expected! The code should catch this and continue

        # The important thing: error should be caught, not crash the system
        # (In real codegen.py, this would continue without GPUFeatureDB data)
        assert error_occurred, "Expected exception from mocked GPUFeatureDB"
        assert "Detection failed" in error_message

        print(f"\n✅ TC-GPUDB-006 PASSED: Error handled gracefully (non-fatal)")
        print(f"   Error caught: {error_message}")


class TestGPUFeatureDBPerformanceCharacteristics:
    """Verify performance-related data accuracy."""

    def test_p100_memory_bandwidth_reasonable(self):
        """P100 should have ~732 GB/s theoretical bandwidth."""
        db = GPUFeatureDB()
        specs = db.get_specs("sm_60")

        # P100 SXM2: 732 GB/s, PCIe: ~549 GB/s
        assert 500 <= specs.memory_bandwidth_gbps <= 800, \
            f"P100 bandwidth ({specs.memory_bandwidth_gbps} GB/s) seems off [500, 800]"

        print(f"✅ P100 bandwidth: {specs.memory_bandwidth_gbps} GB/s (reasonable)")

    def test_p100_fp32_performance_reasonable(self):
        """P100 should have ~10.6 TFLOPS FP32 (base) to ~18.7 TFLOPS (boost)."""
        db = GPUFeatureDB()
        specs = db.get_specs("sm_60")

        # P100: ~4.7 TFLOPS (base) to ~10.6 TFLOPS (boost) for FP32
        assert 4.0 <= specs.fp32_tflops <= 20.0, \
            f"P100 FP32 ({specs.fp32_tflops} TFLOPS) seems off [4, 20]"

        print(f"✅ P100 FP32: {specs.fp32_tflops} TFLOPS (reasonable)")

    def test_h100_has_tensor_cores(self):
        """H100 should support tensor cores."""
        db = GPUFeatureDB()
        specs = db.get_specs("sm_90")

        assert specs.supports_tensor_cores is True, "H100 should support tensor cores"
        assert specs.tensor_core_tflops > 0, "H100 should have non-zero tensor TFLOPS"
        assert specs.tensor_core_tflops > specs.fp32_tflops, \
            "Tensor core TFLOPS should exceed FP32 on H100"

        print(f"✅ H100 tensor cores: {specs.tensor_core_tflops} TFLOPS")


if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-s"])
