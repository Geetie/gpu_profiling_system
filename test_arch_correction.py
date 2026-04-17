"""Test for extended architecture flag correction in compile_cuda.py."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.infrastructure.tools.compile_cuda import _correct_arch_flag


def test_arch_correction():
    """Test all supported architecture flag formats."""
    print("\n" + "="*60)
    print("TEST: Extended Architecture Flag Correction")
    print("="*60)
    
    test_cases = [
        # (input_flag, expected_output, description)
        ("-arch=sm_50", "-arch=sm_75", "sm_50 -> sm_75"),
        ("-arch=sm_60", "-arch=sm_75", "sm_60 -> sm_75"),
        ("-arch=sm_70", "-arch=sm_75", "sm_70 -> sm_75"),
        ("-arch=sm_75", "-arch=sm_75", "sm_75 stays sm_75"),
        ("-arch=sm_80", "-arch=sm_80", "sm_80 stays sm_80"),
        ("-arch=sm_86", "-arch=sm_86", "sm_86 stays sm_86"),
        
        ("-gencode=arch=compute_60,code=sm_60", "-gencode=arch=compute_75,code=sm_75", "gencode sm_60 -> sm_75"),
        ("-gencode=arch=compute_70,code=sm_70", "-gencode=arch=compute_75,code=sm_75", "gencode sm_70 -> sm_75"),
        ("-gencode=arch=compute_75,code=sm_75", "-gencode=arch=compute_75,code=sm_75", "gencode sm_75 stays"),
        ("-gencode=arch=compute_80,code=sm_80", "-gencode=arch=compute_80,code=sm_80", "gencode sm_80 stays"),
        
        ("--gpu-architecture=compute_60", "--gpu-architecture=compute_75", "compute_60 -> compute_75"),
        ("--gpu-architecture=sm_60", "--gpu-architecture=sm_75", "sm_60 -> sm_75"),
        ("--gpu-architecture=compute_75", "--gpu-architecture=compute_75", "compute_75 stays"),
        ("--gpu-architecture=sm_80", "--gpu-architecture=sm_80", "sm_80 stays"),
        
        ("-code=sm_50", "-code=sm_75", "code sm_50 -> sm_75"),
        ("-code=sm_60", "-code=sm_75", "code sm_60 -> sm_75"),
        ("-code=sm_75", "-code=sm_75", "code sm_75 stays"),
        ("-code=sm_80", "-code=sm_80", "code sm_80 stays"),
        
        ("-O3", "-O3", "Non-arch flag unchanged"),
        ("-I/usr/local/cuda/include", "-I/usr/local/cuda/include", "Include path unchanged"),
    ]
    
    all_passed = True
    for input_flag, expected, description in test_cases:
        result = _correct_arch_flag(input_flag)
        passed = result == expected
        status = "✅" if passed else "❌"
        
        print(f"{status} {description}")
        print(f"   Input:    {input_flag}")
        print(f"   Expected: {expected}")
        print(f"   Got:      {result}")
        
        if not passed:
            print(f"   ❌ MISMATCH!")
            all_passed = False
        print()
    
    print("="*60)
    if all_passed:
        print("✅ ALL ARCHITECTURE CORRECTION TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(test_arch_correction())
