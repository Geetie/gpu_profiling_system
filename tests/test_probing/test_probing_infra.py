"""Unit tests for hardware probing infrastructure.

Tests kernel generation, output parsing, and orchestrator logic
without requiring actual GPU hardware.
"""
import pytest


class TestKernelTemplates:
    """Verify kernel templates generate valid CUDA source code."""

    def test_pointer_chase_kernel_generates_source(self):
        from src.infrastructure.probing.kernel_templates import pointer_chase_kernel
        ks = pointer_chase_kernel(array_size=1024, iterations=1000)
        assert ks.name == "pointer_chase"
        assert "clock()" in ks.source
        assert "measure_latency_kernel" in ks.source
        assert "cycles_per_access" in ks.source
        assert len(ks.source) > 200

    def test_working_set_sweep_kernel_generates_source(self):
        from src.infrastructure.probing.kernel_templates import working_set_sweep_kernel
        ks = working_set_sweep_kernel(max_size=4096, iterations=500)
        assert "sweep_kernel" in ks.source or "sweep" in ks.source
        assert "clock()" in ks.source
        assert "cycles_per_access" in ks.source

    def test_clock_calibration_kernel_generates_source(self):
        from src.infrastructure.probing.kernel_templates import clock_calibration_kernel
        ks = clock_calibration_kernel(loop_iterations=1000)
        assert "clock_cal_kernel" in ks.source
        assert "clock()" in ks.source
        assert "total_cycles" in ks.source
        # P3 fix: uses random permutation chain, not fixed stride
        assert "d_chain" in ks.source
        assert "Knuth shuffle" in ks.source
        assert "measure_wrapper_kernel" not in ks.source  # No device-side launch

    def test_stream_copy_kernel_generates_source(self):
        from src.infrastructure.probing.kernel_templates import stream_copy_kernel
        ks = stream_copy_kernel(size_elements=1024)
        assert "stream_copy_kernel" in ks.source
        assert "dst" in ks.source
        assert "src" in ks.source
        assert "elements" in ks.source
        assert "bytes_copied" in ks.source
        # S2 fix: no device-side kernel launch (which needs -rdc=true)
        assert "measure_wrapper_kernel" not in ks.source
        assert "d_dev_cycles" not in ks.source

    def test_shmem_capacity_kernel_generates_source(self):
        from src.infrastructure.probing.kernel_templates import shmem_capacity_kernel
        ks = shmem_capacity_kernel()
        # S1 fix: uses occupancy API, not cudaGetDeviceProperties
        assert "cudaOccupancyMaxActiveBlocksPerMultiprocessor" in ks.source
        assert "cudaGetDeviceProperties" not in ks.source
        assert "shmem_max_tested" in ks.source

    def test_bank_conflict_kernel_generates_source(self):
        from src.infrastructure.probing.kernel_templates import bank_conflict_kernel
        ks = bank_conflict_kernel(size=1024)
        assert "bank_conflict_kernel" in ks.source
        assert "strided" in ks.source.lower()
        assert "sequential" in ks.source.lower() or "seq" in ks.source.lower()
        assert "bank_conflict_ratio" in ks.source


class TestParseNvccOutput:
    """Verify key:value parsing from probe binary output."""

    def test_parse_simple_key_value(self):
        from src.infrastructure.probing.probe_helpers import parse_nvcc_output
        output = "total_cycles: 12345\niterations: 1000\n"
        result = parse_nvcc_output(output)
        assert result["total_cycles"] == 12345
        assert result["iterations"] == 1000

    def test_parse_float_values(self):
        from src.infrastructure.probing.probe_helpers import parse_nvcc_output
        output = "cycles_per_access: 42.5\n"
        result = parse_nvcc_output(output)
        assert result["cycles_per_access"] == 42.5

    def test_parse_string_values(self):
        from src.infrastructure.probing.probe_helpers import parse_nvcc_output
        output = "method: pointer_chasing\n"
        result = parse_nvcc_output(output)
        assert result["method"] == "pointer_chasing"

    def test_parse_ignores_blank_and_comments(self):
        from src.infrastructure.probing.probe_helpers import parse_nvcc_output
        output = "\n# comment\nvalue: 42\n\n"
        result = parse_nvcc_output(output)
        assert result == {"value": 42}

    def test_parse_empty_output(self):
        from src.infrastructure.probing.probe_helpers import parse_nvcc_output
        result = parse_nvcc_output("")
        assert result == {}


class TestOrchestrator:
    """Test orchestrator logic without GPU."""

    def test_cross_validation_latency_hierarchy(self):
        from src.infrastructure.probing.orchestrator import _run_cross_validation
        results = {
            "measurements": {
                "dram_latency_cycles": 400,
                "l2_latency_cycles": 40,
                "l1_latency_cycles": 4,
            },
            "probe_status": {},
            "cross_validation": {},
        }
        _run_cross_validation(results)
        assert results["cross_validation"]["latency_hierarchy_valid"] is True

    def test_cross_validation_invalid_hierarchy(self):
        from src.infrastructure.probing.orchestrator import _run_cross_validation
        results = {
            "measurements": {
                "dram_latency_cycles": 10,
                "l2_latency_cycles": 100,  # L2 slower than DRAM — impossible
                "l1_latency_cycles": 4,
            },
            "probe_status": {},
            "cross_validation": {},
        }
        _run_cross_validation(results)
        assert results["cross_validation"]["latency_hierarchy_valid"] is False

    def test_cross_validation_dram_only(self):
        from src.infrastructure.probing.orchestrator import _run_cross_validation
        results = {
            "measurements": {
                "dram_latency_cycles": 442,
            },
            "probe_status": {},
            "cross_validation": {},
        }
        _run_cross_validation(results)
        # Should check DRAM latency plausibility
        assert results["cross_validation"]["dram_latency_plausible"] is True

    def test_cross_validation_implausible_dram_latency(self):
        from src.infrastructure.probing.orchestrator import _run_cross_validation
        results = {
            "measurements": {
                "dram_latency_cycles": 10,  # Way too low
            },
            "probe_status": {},
            "cross_validation": {},
        }
        _run_cross_validation(results)
        assert results["cross_validation"]["dram_latency_plausible"] is False

    def test_cross_validation_clock_frequency_plausible(self):
        from src.infrastructure.probing.orchestrator import _run_cross_validation
        results = {
            "measurements": {
                "actual_boost_clock_mhz": 1590,  # Typical T4 boost
            },
            "probe_status": {},
            "cross_validation": {},
        }
        _run_cross_validation(results)
        assert results["cross_validation"]["clock_frequency_plausible"] is True

    def test_cross_validation_l2_capacity_plausible(self):
        from src.infrastructure.probing.orchestrator import _run_cross_validation
        results = {
            "measurements": {
                "l2_cache_size_mb": 4.0,  # T4 has 4 MB L2
            },
            "probe_status": {},
            "cross_validation": {},
        }
        _run_cross_validation(results)
        assert results["cross_validation"]["l2_capacity_plausible"] is True

    def test_write_results_json_creates_file(self, tmp_path):
        from src.infrastructure.probing.orchestrator import _write_results_json
        results = {
            "measurements": {"dram_latency_cycles": 442},
            "probe_status": {"dram_latency": "success"},
            "cross_validation": {"dram_latency_plausible": True},
            "evidence_files": [str(tmp_path / "evidence_dram_latency.json")],
        }
        path = str(tmp_path / "results.json")
        _write_results_json(results, path)
        import json
        with open(path) as f:
            data = json.load(f)
        assert data["dram_latency_cycles"] == 442
        assert "methodology" in data
        assert "evidence" in data
        assert len(data["evidence"]) == 1
        assert "evidence_dram_latency" in data["evidence"][0]


class TestProbeHelpers:
    """Test helper functions."""

    def test_sanitize_flags_allows_safe(self):
        from src.infrastructure.probing.probe_helpers import _sanitize_flags
        flags = ["-arch=sm_80", "-O3", "-std=c++14"]
        result = _sanitize_flags(flags)
        assert len(result) == 3

    def test_sanitize_flags_removes_unsafe(self):
        from src.infrastructure.probing.probe_helpers import _sanitize_flags
        flags = ["-arch=sm_80", "`rm -rf /`", "-O3"]
        result = _sanitize_flags(flags)
        assert "-arch=sm_80" in result
        assert "`rm -rf /`" not in result
        assert "-O3" in result


class TestCacheCapacityCliffDetection:
    """Test cliff detection logic."""

    def test_find_cliff_with_clear_jump(self):
        from src.infrastructure.probing.cache_capacity import _find_capacity_cliff
        sweep_data = [
            {"size_bytes": 1_048_576, "size_mb": 1.0, "cycles_per_access": 30.0},
            {"size_bytes": 2_097_152, "size_mb": 2.0, "cycles_per_access": 32.0},
            {"size_bytes": 4_194_304, "size_mb": 4.0, "cycles_per_access": 35.0},
            {"size_bytes": 8_388_608, "size_mb": 8.0, "cycles_per_access": 280.0},  # cliff!
            {"size_bytes": 16_777_216, "size_mb": 16.0, "cycles_per_access": 300.0},
        ]
        cliff_idx = _find_capacity_cliff(sweep_data)
        # The jump at index 3 (280 vs 35) is 8x — should be detected
        assert cliff_idx == 3

    def test_find_cliff_no_jump(self):
        from src.infrastructure.probing.cache_capacity import _find_capacity_cliff
        sweep_data = [
            {"size_bytes": 1_048_576, "size_mb": 1.0, "cycles_per_access": 30.0},
            {"size_bytes": 2_097_152, "size_mb": 2.0, "cycles_per_access": 31.0},
        ]
        cliff_idx = _find_capacity_cliff(sweep_data)
        # No significant jump, but should return something
        assert cliff_idx >= 0


class TestSMDetectionKernel:
    """Verify SM detection kernel generates valid CUDA source."""

    def test_sm_detection_kernel_generates_source(self):
        from src.infrastructure.probing.sm_detection import _sm_count_kernel_final
        source = _sm_count_kernel_final()
        assert "cudaDeviceGetAttribute" in source
        assert "cudaDevAttrMultiProcessorCount" in source
        assert "sm_count" in source
        assert "cudaOccupancyMaxActiveBlocksPerMultiprocessor" in source
        assert "cudaGetDeviceProperties" not in source
        assert len(source) > 500

    def test_sm_kernel_has_no_cudaGetDeviceProperties(self):
        from src.infrastructure.probing.sm_detection import _sm_count_kernel_final
        source = _sm_count_kernel_final()
        # Anti-cheat: must NOT use cudaGetDeviceProperties
        assert "cudaGetDeviceProperties" not in source

    def test_sm_kernel_queries_max_threads_per_sm(self):
        from src.infrastructure.probing.sm_detection import _sm_count_kernel_final
        source = _sm_count_kernel_final()
        assert "cudaDevAttrMaxThreadsPerMultiProcessor" in source

    def test_sm_kernel_queries_max_shmem_per_sm(self):
        from src.infrastructure.probing.sm_detection import _sm_count_kernel_final
        source = _sm_count_kernel_final()
        assert "cudaDevAttrMaxSharedMemoryPerMultiprocessor" in source


class TestShmemBandwidthKernel:
    """Verify shmem bandwidth kernel generates valid CUDA source."""

    def test_shmem_bandwidth_kernel_generates_source(self):
        from src.infrastructure.probing.shmem_bandwidth import _shmem_bandwidth_kernel
        source = _shmem_bandwidth_kernel()
        assert "shmem_bw_kernel" in source
        assert "__shared__" in source
        assert "__syncthreads" in source
        assert "cudaGetDeviceProperties" not in source
        assert len(source) > 500

    def test_shmem_bandwidth_uses_cooperative_access(self):
        from src.infrastructure.probing.shmem_bandwidth import _shmem_bandwidth_kernel
        source = _shmem_bandwidth_kernel()
        # Should use threadIdx.x for cooperative access
        assert "threadIdx.x" in source
        assert "blockDim.x" in source

    def test_shmem_bandwidth_no_get_device_properties(self):
        from src.infrastructure.probing.shmem_bandwidth import _shmem_bandwidth_kernel
        source = _shmem_bandwidth_kernel()
        # Anti-cheat compliance
        assert "cudaGetDeviceProperties" not in source


class TestCrossValidationEnhancements:
    """P0: Enhanced cross-validation checks."""

    def test_sm_family_match_t4(self):
        from src.infrastructure.probing.orchestrator import _check_sm_family_match
        assert _check_sm_family_match(40, "turing_t4") is True

    def test_sm_family_match_t4_wrong_count(self):
        from src.infrastructure.probing.orchestrator import _check_sm_family_match
        assert _check_sm_family_match(20, "turing_t4") is False

    def test_sm_family_match_ampere(self):
        from src.infrastructure.probing.orchestrator import _check_sm_family_match
        assert _check_sm_family_match(108, "ampere_a100") is True

    def test_sm_family_match_unknown_family(self):
        from src.infrastructure.probing.orchestrator import _check_sm_family_match
        # Unknown family defaults to True (can't validate)
        assert _check_sm_family_match(50, "custom_gpu") is True

    def test_cross_validation_with_sm_check(self):
        from src.infrastructure.probing.orchestrator import _run_cross_validation
        results = {
            "measurements": {
                "sm_count": 40,
                "likely_gpu_family": "turing_t4",
            },
            "probe_status": {},
            "cross_validation": {},
        }
        _run_cross_validation(results)
        assert results["cross_validation"]["sm_count_matches_family"] is True

    def test_cross_validation_with_sm_mismatch(self):
        from src.infrastructure.probing.orchestrator import _run_cross_validation
        results = {
            "measurements": {
                "sm_count": 20,  # Half of T4's 40 — likely masked
                "likely_gpu_family": "turing_t4",
            },
            "probe_status": {},
            "cross_validation": {},
        }
        _run_cross_validation(results)
        assert results["cross_validation"]["sm_count_matches_family"] is False


class TestSMMicrobenchmarkKernel:
    """P1: Pure microbenchmark SM detection kernel."""

    def test_microbenchmark_kernel_generates_source(self):
        from src.infrastructure.probing.sm_detection import _sm_count_microbenchmark_kernel
        source = _sm_count_microbenchmark_kernel()
        assert "spin_kernel" in source
        assert "cudaEventElapsedTime" in source
        assert "cudaEventRecord" in source
        assert "sm_count_microbenchmark" in source
        assert "cudaDeviceGetAttribute" not in source
        assert "cudaGetDeviceProperties" not in source
        assert len(source) > 500

    def test_microbenchmark_uses_volatile_spin(self):
        from src.infrastructure.probing.sm_detection import _sm_count_microbenchmark_kernel
        source = _sm_count_microbenchmark_kernel()
        assert "volatile" in source

    def test_microbenchmark_tests_multiple_sm_counts(self):
        from src.infrastructure.probing.sm_detection import _sm_count_microbenchmark_kernel
        source = _sm_count_microbenchmark_kernel()
        # Should test multiple SM estimate values
        assert "sm_test_" in source


class TestEnhancedCrossValidation:
    """Test new cross-validation checks 8-12."""

    def test_sm_count_valid_known_config(self):
        from src.infrastructure.probing.orchestrator import _run_cross_validation
        results = {
            "measurements": {"sm_count": 40},
            "probe_status": {},
            "cross_validation": {},
        }
        _run_cross_validation(results)
        assert results["cross_validation"]["sm_count_is_known_config"] is True

    def test_sm_count_invalid_unknown_config(self):
        from src.infrastructure.probing.orchestrator import _run_cross_validation
        results = {
            "measurements": {"sm_count": 999},
            "probe_status": {},
            "cross_validation": {},
        }
        _run_cross_validation(results)
        assert results["cross_validation"]["sm_count_is_known_config"] is False

    def test_l2_capacity_power_of_two(self):
        from src.infrastructure.probing.orchestrator import _run_cross_validation
        results = {
            "measurements": {"l2_cache_size_mb": 4.0},
            "probe_status": {},
            "cross_validation": {},
        }
        _run_cross_validation(results)
        assert results["cross_validation"]["l2_capacity_is_power_of_two"] is True

    def test_l2_capacity_1_5x_power_of_two(self):
        from src.infrastructure.probing.orchestrator import _run_cross_validation
        results = {
            "measurements": {"l2_cache_size_mb": 6.0},  # 1.5 × 4
            "probe_status": {},
            "cross_validation": {},
        }
        _run_cross_validation(results)
        assert results["cross_validation"]["l2_capacity_is_power_of_two"] is True

    def test_bank_conflict_ratio_bounded(self):
        from src.infrastructure.probing.orchestrator import _run_cross_validation
        results = {
            "measurements": {"bank_conflict_penalty_ratio": 3.5},
            "probe_status": {},
            "cross_validation": {},
        }
        _run_cross_validation(results)
        assert results["cross_validation"]["bank_conflict_ratio_bounded"] is True

    def test_bank_conflict_ratio_exceeds_warp_size(self):
        from src.infrastructure.probing.orchestrator import _run_cross_validation
        results = {
            "measurements": {"bank_conflict_penalty_ratio": 35.0},  # > 32
            "probe_status": {},
            "cross_validation": {},
        }
        _run_cross_validation(results)
        assert results["cross_validation"]["bank_conflict_ratio_bounded"] is False

    def test_shmem_faster_than_dram(self):
        from src.infrastructure.probing.orchestrator import _run_cross_validation
        results = {
            "measurements": {
                "shmem_bandwidth_gbps": 1500.0,
                "dram_bandwidth_gbps": 300.0,
            },
            "probe_status": {},
            "cross_validation": {},
        }
        _run_cross_validation(results)
        assert results["cross_validation"]["shmem_faster_than_dram"] is True

    def test_shmem_slower_than_dram_fails(self):
        from src.infrastructure.probing.orchestrator import _run_cross_validation
        results = {
            "measurements": {
                "shmem_bandwidth_gbps": 100.0,
                "dram_bandwidth_gbps": 300.0,
            },
            "probe_status": {},
            "cross_validation": {},
        }
        _run_cross_validation(results)
        assert results["cross_validation"]["shmem_faster_than_dram"] is False


class TestUnifiedNcuParser:
    """Test unified parse_ncu_gpu_time function."""

    def test_parse_gpu_time_ns_pattern(self):
        from src.infrastructure.probing.probe_helpers import parse_ncu_gpu_time
        result = parse_ncu_gpu_time("gpu_time_ns  42.5")
        assert result == 42.5

    def test_parse_duration_ns_pattern(self):
        from src.infrastructure.probing.probe_helpers import parse_ncu_gpu_time
        result = parse_ncu_gpu_time("Duration (ns) : 1,234.56")
        assert result == 1234.56

    def test_parse_ns_suffix(self):
        from src.infrastructure.probing.probe_helpers import parse_ncu_gpu_time
        result = parse_ncu_gpu_time("some value 99.5ns")
        assert result == 99.5

    def test_parse_returns_none_on_no_match(self):
        from src.infrastructure.probing.probe_helpers import parse_ncu_gpu_time
        result = parse_ncu_gpu_time("no timing info here")
        assert result is None

    def test_sm_detection_redundant_kernels_removed(self):
        """P3-12: Verify redundant kernel generators are removed."""
        import src.infrastructure.probing.sm_detection as sm_mod
        assert not hasattr(sm_mod, "_sm_detection_kernel")
        assert not hasattr(sm_mod, "_sm_count_kernel_v2")
        assert hasattr(sm_mod, "_sm_count_kernel_final")


class TestClockCalibrationRandomChain:
    """P3: clock_calibration uses Knuth shuffle."""

    def test_clock_calibration_uses_knuth_shuffle(self):
        from src.infrastructure.probing.kernel_templates import clock_calibration_kernel
        source = clock_calibration_kernel().source
        assert "Knuth shuffle" in source
        assert "seed * 1103515245 + 12345" in source  # LCG seed
        assert "d_chain" in source

    def test_clock_calibration_no_fixed_stride(self):
        from src.infrastructure.probing.kernel_templates import clock_calibration_kernel
        source = clock_calibration_kernel().source
        # Should NOT use fixed (i + 1) % size pattern
        assert "(i + 1) % size" not in source


class TestWrapWithEvents:
    """Test _wrap_with_events wraps last kernel launch, not first."""

    def test_wrap_single_kernel_launch(self):
        from src.infrastructure.probing.clock_measurement import _wrap_with_events
        source = """
int main() {
    kernel<<<1, 256>>>(args);
    return 0;
}
"""
        wrapped = _wrap_with_events(source)
        assert wrapped is not None
        assert "cudaEventCreate" in wrapped
        assert "cudaEventRecord" in wrapped

    def test_wrap_uses_last_launch(self):
        from src.infrastructure.probing.clock_measurement import _wrap_with_events
        source = """
int main() {
    kernel<<<1, 256>>>(args);  // warmup
    kernel<<<1, 256>>>(args);  // measurement
    return 0;
}
"""
        wrapped = _wrap_with_events(source)
        assert wrapped is not None
        # Count event injections — should only wrap the last launch
        event_count = wrapped.count("cudaEventCreate")
        assert event_count == 2  # evt_start + evt_stop, only for last launch

    def test_wrap_no_launch_returns_none(self):
        from src.infrastructure.probing.clock_measurement import _wrap_with_events
        source = "int main() { return 0; }"
        assert _wrap_with_events(source) is None


class TestMethodologyCoverage:
    """Verify methodology string covers all techniques."""

    def test_methodology_mentions_cuda_event_fallback(self):
        """Methodology should mention cudaEventElapsedTime fallback."""
        from src.infrastructure.probing.orchestrator import _write_results_json
        # We need to inspect the methodology string indirectly
        import json
        import tempfile
        import os
        results = {
            "measurements": {},
            "probe_status": {},
            "cross_validation": {},
            "evidence_files": [],
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name
        try:
            _write_results_json(results, path)
            with open(path) as f:
                data = json.load(f)
            meth = data.get("methodology", "")
            assert "cudaEventElapsedTime" in meth
            assert "cudaDeviceGetAttribute" in meth
            assert "Knuth shuffle" in meth
            assert "cross-validation" in meth.lower()
            assert "SM count" in meth or "SM masking" in meth
            # Methodology should mention cross-validation count (≥15 checks)
            assert "18 checks" in meth or "18 cross-validation" in meth or "18:" in meth
        finally:
            os.unlink(path)
