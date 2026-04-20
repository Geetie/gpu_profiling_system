"""Debug script to check why measurements aren't being parsed."""
from src.domain.stage_executor import StageExecutor
from src.domain.enums import SubAgentStatus

# Test case: 83% (5/6)
target_spec = {'targets': [f'target_{i}' for i in range(6)]}
measured_lines = '\n'.join([f'target_{i}: {i * 10.0}' for i in range(5)])
tool_results = [{'tool': 'execute_binary', 'stdout': measured_lines, 'return_code': 0}]
data = {}

status = StageExecutor._codegen_status(
    final_text='test',
    tool_results=tool_results,
    data=data,
    target_spec=target_spec
)

print(f'Status: {status}')
print(f'Data keys: {list(data.keys())}')
print(f'Measurements: {data.get("measurements", {})}')
print(f'Completion rate: {data.get("completion_rate")}')
print(f'Error detail: {data.get("error_detail", "N/A")}')
