#!/usr/bin/env python3
"""Phase 2 Optimization Verification Test Script

Tests the following Phase 2 enhancements:
1. NCU Permission Pre-check Mechanism (OPT-001)
2. GPUFeatureDB Integration Logging
3. Backward Compatibility

Run this script to verify all optimizations are working correctly.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_ncu_permission_cache():
    """Test 1: Verify NCU permission cache mechanism works correctly."""
    print("\n" + "="*70)
    print("TEST 1: NCU Permission Pre-check Mechanism (OPT-001)")
    print("="*70)

    try:
        from src.infrastructure.tools.run_ncu import (
            _ncu_permission_cache,
            check_ncu_permission_fast,
            mark_ncu_unavailable,
            get_ncu_permission_status,
            reset_ncu_permission_cache,
        )

        print("\n✅ Successfully imported all NCU pre-check functions")

        # Test initial state
        print(f"\n📋 Initial cache state:")
        status = get_ncu_permission_status()
        print(f"   checked={status['checked']}, allowed={status['allowed']}")

        # Test reset function
        reset_ncu_permission_cache()
        status = get_ncu_permission_status()
        assert status['checked'] == False, "Reset should set checked=False"
        assert status['allowed'] is None, "Reset should set allowed=None"
        print("\n✅ reset_ncu_permission_cache() works correctly")

        # Test mark_unavailable function
        mark_ncu_unavailable("Test error message")
        status = get_ncu_permission_status()
        assert status['checked'] == True, "mark_unavailable should set checked=True"
        assert status['allowed'] == False, "mark_unavailable should set allowed=False"
        assert "Test error message" in status['error_message'], "Error message should be stored"
        print("\n✅ mark_ncu_unavailable() works correctly")
        print(f"   Cached error: {status['error_message']}")

        # Reset for next test
        reset_ncu_permission_cache()

        # Test check_ncu_permission_fast (this will actually check ncu)
        print("\n🔄 Running check_ncu_permission_fast()...")
        result = check_ncu_permission_fast()
        status = get_ncu_permission_status()
        print(f"   Result: {result}")
        print(f"   Cache state after check: checked={status['checked']}, allowed={status['allowed']}")
        if status['error_message']:
            print(f"   Error message: {status['error_message'][:100]}")

        print("\n✅ All NCU pre-check mechanism tests PASSED")
        return True

    except Exception as e:
        print(f"\n❌ NCU pre-check test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_featuredb_logging():
    """Test 2: Verify GPUFeatureDB integration has proper logging."""
    print("\n" + "="*70)
    print("TEST 2: GPUFeatureDB Integration Logging")
    print("="*70)

    try:
        # Check if logging module is imported in codegen.py
        with open('src/application/subagents/codegen.py', 'r', encoding='utf-8') as f:
            codegen_content = f.read()

        # Check for logger import
        assert 'import logging' in codegen_content, "codegen.py should import logging module"
        assert 'logger = logging.getLogger' in codegen_content, "codegen.py should configure logger"

        # Check for GPUFeatureDB log messages
        gpu_featuredb_logs = [
            "[CodeGen] 🔄 Attempting to initialize GPUFeatureDB",
            "[CodeGen] ✅ GPUFeatureDB module imported successfully",
            "[CodeGen] ✅ GPUFeatureDB initialized successfully",
            "[CodeGen] 📊 GPUFeatureDB detected:",
            "[CodeGen] 📏 Measurement params for",
            "[CodeGen] ✅ GPUFeatureDB context injected successfully",
            "[CodeGen] ⚠️ GPUFeatureDB.detect_and_get_features() returned None",
            "[CodeGen] ❌ FAILED to import GPUFeatureDB module",
            "[CodeGen] ⚠️ GPUFeatureDB integration encountered an error",
        ]

        found_logs = []
        missing_logs = []
        for log_msg in gpu_featuredb_logs:
            if log_msg in codegen_content:
                found_logs.append(log_msg)
            else:
                missing_logs.append(log_msg)

        print(f"\n📊 Logging coverage:")
        print(f"   Found {len(found_logs)}/{len(gpu_featuredb_logs)} expected log messages")

        if found_logs:
            print("\n✅ Found log messages:")
            for log in found_logs:
                print(f"   • {log[:60]}...")

        if missing_logs:
            print("\n⚠️ Missing log messages:")
            for log in missing_logs:
                print(f"   • {log}")

        # Check for DEBUG-level logs
        debug_log_count = codegen_content.count('logger.debug(')
        info_log_count = codegen_content.count('logger.info(')
        warning_log_count = codegen_content.count('logger.warning(')
        error_log_count = codegen_content.count('logger.error(')

        print(f"\n📈 Log level distribution in codegen.py:")
        print(f"   DEBUG:    {debug_log_count} messages")
        print(f"   INFO:     {info_log_count} messages")
        print(f"   WARNING:  {warning_log_count} messages")
        print(f"   ERROR:    {error_log_count} messages")

        assert len(found_logs) >= len(gpu_featuredb_logs) * 0.8, \
            f"At least 80% of log messages should be present (found {len(found_logs)}/{len(gpu_featuredb_logs)})"

        print("\n✅ GPUFeatureDB logging test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ GPUFeatureDB logging test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """Test 3: Verify backward compatibility of modified functions."""
    print("\n" + "="*70)
    print("TEST 3: Backward Compatibility Check")
    print("="*70)

    try:
        from src.infrastructure.tools.run_ncu import run_ncu_handler

        # Test function signature hasn't changed
        import inspect
        sig = inspect.signature(run_ncu_handler)
        params = list(sig.parameters.keys())

        print(f"\n📋 run_ncu_handler signature: {sig}")
        assert 'arguments' in params, "'arguments' parameter should exist"
        assert 'sandbox' in params, "'sandbox' parameter should exist"

        # Test that it returns a dict with expected keys
        print("\n🔄 Testing run_ncu_handler with invalid input...")
        result = run_ncu_handler({"executable": ""})

        assert isinstance(result, dict), "Should return a dict"
        assert 'raw_output' in result, "Should have 'raw_output' key"
        assert 'parsed_metrics' in result, "Should have 'parsed_metrics' key"

        print(f"   Return type: dict ✓")
        print(f"   Keys present: raw_output, parsed_metrics ✓")
        print(f"   Error handling works: ✓")

        # Test agent_loop imports work
        print("\n🔄 Testing agent_loop.py imports...")
        from src.application.agent_loop import AgentLoop
        print("   AgentLoop import: ✓")

        # Test codegen imports work
        print("\n🔄 Testing codegen.py imports...")
        from src.application.subagents.codegen import CodeGenAgent
        print("   CodeGenAgent import: ✓")

        print("\n✅ Backward compatibility test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Backward compatibility test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_loop_enhancement():
    """Test 4: Verify agent_loop.py NCU error handling enhancement."""
    print("\n" + "="*70)
    print("TEST 4: Agent Loop NCU Error Handling Enhancement")
    print("="*70)

    try:
        with open('src/application/agent_loop.py', 'r', encoding='utf-8') as f:
            agent_loop_content = f.read()

        # Check for enhanced error message
        enhanced_markers = [
            "mark_ncu_unavailable",
            "get_ncu_permission_status",
            "PERMANENTLY DISABLED",
            "AUTOMATIC ACTION TAKEN",
            "SWITCH TO TEXT ANALYSIS MODE",
            "ABSOLUTELY FORBIDDEN",
            "ALTERNATIVE DATA SOURCES AVAILABLE",
        ]

        found_markers = []
        for marker in enhanced_markers:
            if marker in agent_loop_content:
                found_markers.append(marker)

        print(f"\n📊 Enhanced error handling markers:")
        print(f"   Found {len(found_markers)}/{len(enhanced_markers)} markers")

        if found_markers:
            print("\n✅ Found enhancement markers:")
            for marker in found_markers:
                print(f"   • {marker}")

        assert len(found_markers) >= len(enhanced_markers) * 0.8, \
            f"At least 80% of enhancement markers should be present"

        # Check that the import statement is correct
        assert "from src.infrastructure.tools.run_ncu import (" in agent_loop_content, \
            "Should import from run_ncu module"
        assert "mark_ncu_unavailable," in agent_loop_content, \
            "Should import mark_ncu_unavailable function"

        print("\n✅ Agent loop enhancement test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Agent loop enhancement test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("\n" + "="*70)
    print("GPU Profiling System - Phase 2 Optimization Verification")
    print("="*70)
    print("\nThis script verifies the following Phase 2 enhancements:")
    print("1. NCU Permission Pre-check Mechanism (OPT-001)")
    print("2. GPUFeatureDB Integration Logging")
    print("3. Backward Compatibility")
    print("4. Agent Loop NCU Error Handling Enhancement")

    results = []

    # Run all tests
    results.append(("NCU Permission Cache", test_ncu_permission_cache()))
    results.append(("GPUFeatureDB Logging", test_gpu_featuredb_logging()))
    results.append(("Backward Compatibility", test_backward_compatibility()))
    results.append(("Agent Loop Enhancement", test_agent_loop_enhancement()))

    # Print summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Phase 2 optimization is working correctly.")
        print("\nNext steps:")
        print("1. Run the system in a Kaggle-like environment to verify NCU caching")
        print("2. Check execution.log for '[CodeGen] 📊 GPUFeatureDB detected:' messages")
        print("3. Monitor MetricAnalysis time (should be <30s vs previous 135s)")
        return 0
    else:
        print("\n⚠️ SOME TESTS FAILED! Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
