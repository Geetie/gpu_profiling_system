from src.application.completion_detector import CompletionDetector

detector = CompletionDetector()

print('Testing Plan Stage Completion Detection:')
print('='*60)

plan_json1 = """
Here is the profiling plan:
[
  {"target": "dram_latency_cycles", "category": "memory", "method": "pointer_chasing"},
  {"target": "l2_cache_size_kb", "category": "cache", "method": "working_set_sweep"}
]
"""

plan_json2 = plan_json1

result1 = detector.is_completion(plan_json1)
print(f'Test 1 (valid JSON, first time): {result1}')

result2 = detector.is_completion(plan_json2)
print(f'Test 2 (same JSON, stable): {result2} (should be True)')

non_plan = 'I am working on the code generation task now.'
result3 = detector.is_completion(non_plan)
print(f'Test 3 (non-plan text): {result3} (should be False)')

complete_text = 'All targets have been measured. Final results: dram_latency: 442'
result4 = detector.is_completion(complete_text)
print(f'Test 4 (completion phrase): {result4} (should be True)')

print()
print('='*60)
if result2 and result4 and not result3:
    print('All tests PASSED!')
else:
    print('Some tests need review')
