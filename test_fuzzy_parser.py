"""Test for enhanced FuzzyToolCallParser."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.application.tool_call_parser import FuzzyToolCallParser, ToolRegistry

def test_fuzzy_parser():
    """Test enhanced fuzzy tool call parser."""
    print("\n" + "="*60)
    print("TEST: Enhanced FuzzyToolCallParser")
    print("="*60)
    
    parser = FuzzyToolCallParser()
    
    # Create a mock tool registry
    class MockRegistry:
        def list_tools(self):
            return ["compile_cuda", "execute_binary", "read_file", "write_file", "run_ncu"]
    
    registry = MockRegistry()
    
    test_cases = [
        # (input_text, expected_tool, description)
        ("I'll call compile_cuda with source=...", "compile_cuda", "Original pattern 1"),
        ("compile_cuda(source='...', flags=[...])", "compile_cuda", "Original pattern 2"),
        ("Let me use the run_ncu tool on ...", "run_ncu", "Original pattern 3"),
        ("compile_cuda with source='...'", "compile_cuda", "Original pattern 4"),
        ("Now I will compile using compile_cuda", "compile_cuda", "New pattern 1"),
        ("I will now call compile_cuda", "compile_cuda", "New pattern 2"),
        ("Let's execute using execute_binary", "execute_binary", "New pattern 3"),
        ("I shall now compile with compile_cuda", "compile_cuda", "New pattern 4"),
        ("We need to execute via execute_binary", "execute_binary", "New pattern 5"),
        ("Let me compile using compile_cuda to build the binary", "compile_cuda", "New pattern 6"),
        ("I will run compile_cuda to compile the code", "compile_cuda", "New pattern 7"),
        ("Now I'll use execute_binary to run the benchmark", "execute_binary", "New pattern 8"),
        ("I must call compile_cuda to proceed", "compile_cuda", "New pattern 9"),
        ("Let's run execute_binary on the compiled binary", "execute_binary", "New pattern 10"),
    ]
    
    all_passed = True
    for text, expected, description in test_cases:
        result = parser.parse(text, registry)
        if result:
            passed = result.name == expected
            status = "✅" if passed else "❌"
            print(f"{status} {description}")
            print(f"   Input:    {text[:60]}...")
            print(f"   Expected: {expected}")
            print(f"   Got:      {result.name}")
            
            if not passed:
                print(f"   ❌ MISMATCH!")
                all_passed = False
        else:
            print(f"❌ {description}")
            print(f"   Input:    {text[:60]}...")
            print(f"   Expected: {expected}")
            print(f"   Got:      None (no tool call detected)")
            all_passed = False
        print()
    
    print("="*60)
    if all_passed:
        print("✅ ALL FUZZY PARSER TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(test_fuzzy_parser())
