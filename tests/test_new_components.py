#!/usr/bin/env python3
"""Quick validation test for new infrastructure components."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_cuda_version_manager():
    """Test CUDA Version Manager."""
    import tempfile
    import shutil
    
    # Use temporary directory for clean test isolation
    temp_dir = tempfile.mkdtemp(prefix="cuda_version_test_")
    
    try:
        from src.infrastructure.cuda_version_manager import CUDAVersionManager
        
        manager = CUDAVersionManager(state_dir=temp_dir)
        
        # Test version recording
        v1 = manager.record_generation(
            target="dram_latency_cycles",
            source_code="__global__ void kernel() { }",
            iteration=1,
            metadata={"method": "pointer_chasing"}
        )
        
        assert v1, "Version ID should be generated"
        print(f"  ✅ Version created: {v1}")
        
        # Test compilation record
        manager.record_compilation(v1, success=True, warnings=2)
        
        # Test execution record
        manager.record_execution(
            v1,
            success=True,
            measurements={"dram_latency_cycles": 485.0},
            execution_time_ms=12.5
        )
        
        # Test performance trend
        trend = manager.get_performance_trend("dram_latency_cycles")
        assert trend.total_versions == 1, f"Expected 1 version, got {trend.total_versions}"
        assert trend.latest_measurement == 485.0
        print(f"  ✅ Performance trend: {trend.latest_measurement} cycles ({trend.successful_versions} successful)")
        
        # Test summary stats
        stats = manager.get_summary_stats()
        assert stats["total_versions"] == 1
        print(f"  ✅ Summary: {stats['total_versions']} version(s)")
        
        print("✅ CUDA Version Manager: PASSED\n")
        
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_gpu_feature_db():
    """Test GPU Feature Database."""
    from src.infrastructure.gpu_feature_db import GPUFeatureDB
    
    db = GPUFeatureDB()
    
    # Test database size
    assert db.gpu_count > 0, "Database should have entries"
    print(f"  ✅ Database loaded: {db.gpu_count} architectures")
    
    # Test P100 specs (Kaggle GPU)
    p100 = db.get_specs("sm_60")
    assert p100 is not None, "P100 should be in database"
    assert p100.sm_count == 56, "P100 should have 56 SMs"
    assert p100.l2_cache_size_kb == 4096, "P100 L2 should be 4096 KB"
    print(f"  ✅ P100: {p100.name}, {p100.sm_count} SMs, {p100.l2_cache_size_kb/1024:.1f}MB L2")
    
    # Test measurement params for P100
    params = db.get_measurement_params("dram_latency_cycles", "sm_60")
    assert params["working_set_mb"] >= 128, "Working set should exceed L2"
    print(f"  ✅ DRAM params: {params['working_set_mb']}MB working set, "
          f"range {params['expected_range']}")
    
    # Test A100 specs
    a100 = db.get_specs("sm_80")
    assert a100 is not None
    assert a100.supports_tensor_cores == True
    print(f"  ✅ A100: {a100.name}, tensor cores={a100.supports_tensor_cores}")
    
    # Test architecture list
    archs = db.supported_architectures
    assert "sm_60" in archs
    assert "sm_90" in archs
    print(f"  ✅ Architectures: {len(archs)} supported ({archs[0]} to {archs[-1]})")
    
    print("✅ GPU Feature Database: PASSED\n")


def test_feedback_enhancer():
    """Test Feedback Enhancer."""
    from src.infrastructure.feedback_enhancer import FeedbackEnhancer
    
    enhancer = FeedbackEnhancer()
    
    sample_analysis = """
    Analysis of DRAM latency benchmark:
    - Memory-bound behavior detected (90% of peak bandwidth utilization)
    - Global memory throughput is the primary bottleneck
    - Consider using shared memory tiling to reduce DRAM traffic
    - L2 cache hit rate is low due to large 128MB working set
    - Coalescing looks good but could improve with struct-of-arrays layout
    Recommendation: Implement 2D tiling with 32x32 thread blocks.
    """
    
    report = enhancer.create_feedback_report(
        target="dram_latency_cycles",
        metric_analysis_output=sample_analysis,
        measurements={"dram_latency_cycles": 485.0},
        compute_capability="sm_60",
    )
    
    assert report is not None, "Report should be generated"
    assert len(report.suggestions) > 0, "Should generate suggestions"
    assert report.overall_verdict in ["ACCEPT", "REJECT", "CONDITIONAL_ACCEPT"]
    
    print(f"  ✅ Report created: {report.overall_verdict}")
    print(f"  ✅ Suggestions: {len(report.suggestions)} generated")
    print(f"  ✅ Top suggestion: {report.suggestions[0].title}")
    print(f"  ✅ Confidence: {report.confidence_score:.2f}")
    
    # Test formatting for CodeGen
    formatted = enhancer.format_for_codegen(report, max_suggestions=2)
    assert len(formatted) > 0, "Formatted output should not be empty"
    assert "Performance Analysis Feedback" in formatted
    print(f"  ✅ Formatted output: {len(formatted)} chars")
    
    # Test action items extraction
    action_items = enhancer.get_action_items_for_pipeline_context(report)
    assert "suggested_fixes" in action_items
    assert len(action_items["suggested_fixes"]) > 0
    print(f"  ✅ Action items: {len(action_items['suggested_fixes'])} fixes")
    
    print("✅ Feedback Enhancer: PASSED\n")


def test_optimization_plan():
    """Test Optimization Plan."""
    from src.infrastructure.optimization_plan import (
        Q2_2026_PLAN, 
        get_optimization_roadmap,
        OptimizationPriority,
    )
    
    plan = Q2_2026_PLAN
    
    # Test plan structure
    assert plan.plan_name, "Plan should have name"
    assert len(plan.tasks) > 0, "Plan should have tasks"
    print(f"  ✅ Plan: {plan.plan_name}")
    print(f"  ✅ Tasks: {len(plan.tasks)} total")
    
    # Test priority breakdown
    p0_tasks = plan.get_tasks_by_priority(OptimizationPriority.P0_CRITICAL)
    p1_tasks = plan.get_tasks_by_priority(OptimizationPriority.P1_HIGH)
    print(f"  ✅ P0 Critical: {len(p0_tasks)} tasks")
    print(f"  ✅ P1 High: {len(p1_tasks)} tasks")
    
    # Test roadmap generation
    roadmap = get_optimization_roadmap()
    assert "progress" in roadmap
    assert "tasks_by_priority" in roadmap
    assert "key_metrics" in roadmap
    print(f"  ✅ Roadmap: {roadmap['progress']['progress_pct']}% complete")
    
    # Verify key targets
    metrics = roadmap["key_metrics"]
    assert metrics["total_pipeline_time_target_s"] < metrics["total_pipeline_time_current_s"]
    print(f"  ✅ Pipeline time target: {metrics['total_pipeline_time_current_s']}s → "
          f"{metrics['total_pipeline_time_target_s']}s "
          f"({metrics['improvement_target_pct']}% improvement)")
    
    print("✅ Optimization Plan: PASSED\n")


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("🧪 Validating New Infrastructure Components")
    print("=" * 70 + "\n")
    
    try:
        test_cuda_version_manager()
        test_gpu_feature_db()
        test_feedback_enhancer()
        test_optimization_plan()
        
        print("=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
