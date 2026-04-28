import sys
sys.path.insert(0, '/workspace/mlsys-project-main/agent/src')
from src.infrastructure.tools.compile_cuda import _detect_kernel_quality_issues, _fix_cuda_enum_names

test_source = '''
#include <cuda_runtime.h>
__global__ void kernel(double* sink) {
    uint64_t start = clock64();
    double val = 0.0;
    for(int i=0; i<1000; i++) {
        val = val * 1.0001 + 0.001;
    }
    sink[threadIdx.x] = val;
}
int main() {
    int clock;
    cudaDeviceGetAttribute(&clock, cudaDevAttrClockRate, 0);
    return 0;
}
'''

print('=== Testing detection BEFORE enum replacement ===')
warnings, errors = _detect_kernel_quality_issues(test_source, 'sm__throughput.avg.pct_of_peak_sustained_elapsed')
print(f'Warnings: {len(warnings)}')
print(f'Errors: {len(errors)}')
for e in errors:
    print(f'  ERROR: {e}')

print()
print('=== Testing detection AFTER enum replacement ===')
patched = _fix_cuda_enum_names(test_source)
enum_replacement = '(enum cudaDeviceAttr)13'
print(f'Patched source contains enum replacement: {enum_replacement in patched}')
warnings2, errors2 = _detect_kernel_quality_issues(patched, 'sm__throughput.avg.pct_of_peak_sustained_elapsed')
print(f'Warnings: {len(warnings2)}')
print(f'Errors: {len(errors2)}')

print()
print('=== Verification ===')
if len(errors) > 0 and len(errors2) == 0:
    print('SUCCESS: Detection works BEFORE enum replacement but fails AFTER')
    print('This confirms the order fix is correct!')
else:
    print('UNEXPECTED: Please check the logic')
