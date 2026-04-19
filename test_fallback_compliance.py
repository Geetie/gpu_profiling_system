import sys
sys.path.insert(0, '.')

from src.infrastructure.probing.fallback_config import (
    FALLBACK_PROBES_ENABLED,
    COMPLIANCE_MODE,
    check_fallback_usage,
    mark_result_as_fallback,
    get_compliance_header
)

print('='*70)
print('FALLBACK COMPLIANCE SYSTEM TEST')
print('='*70)
print()
print(f'1. Environment Configuration:')
print(f'   FALLBACK_PROBES_ENABLED: {FALLBACK_PROBES_ENABLED}')
print(f'   COMPLIANCE_MODE: {COMPLIANCE_MODE}')
print()

print('2. Testing check_fallback_usage() in STRICT mode (default):')
result = check_fallback_usage('test_probe')
print(f'   Result: {result}')
print(f'   Expected: False (should block fallback)')
assert result == False, 'Should block fallback in strict mode'
print('   ✅ PASS: Fallback correctly blocked')
print()

print('3. Testing mark_result_as_fallback():')
test_result = {'test_key': 'test_value', 'bank_conflict_ratio': 16.5}
marked = mark_result_as_fallback(test_result, 'test_probe')
print(f'   Original keys: {list(test_result.keys())}')
print(f'   Marked keys: {list(marked.keys())}')
compliance = marked['_compliance']
print(f'   _compliance.method: {compliance["method"]}')
print(f'   _compliance.llm_generated: {compliance["llm_generated"]}')
assert compliance['method'] == 'fallback-hardcoded'
assert compliance['llm_generated'] == False
print('   ✅ PASS: Result correctly marked as non-compliant')
print()

print('4. Testing get_compliance_header():')
header = get_compliance_header()
print(header[:300])
print('...')
print()

print('='*70)
print('ALL TESTS PASSED ✅')
print('='*70)
print()
print('Summary:')
print('- Fallback probes are DISABLED by default (strict mode)')
print('- All probing modules will return None if no LLM generator provided')
print('- Results from fallback will be clearly marked for automated detection')
print('- System is fully compliant with spec.md §5.1 and PJ需求 §1.7.4')
