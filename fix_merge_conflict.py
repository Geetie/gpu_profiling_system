#!/usr/bin/env python3
"""Fix Git merge conflict in agent_loop.py"""

filepath = r'e:\GPU_Profiling_System\src\application\agent_loop.py'

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Check if conflict markers exist
if '<<<<<<< HEAD' in content:
    print('❌ Merge conflict detected! Fixing...')
    
    # Define the conflict block to replace (exact match from file)
    old_text = """<<<<<<< HEAD

                # P0 FIX: Inject current_target into compile_cuda arguments for target-specific binary names
                if tool_call.name == "compile_cuda" and current_target:
                    tool_call.arguments["target"] = current_target
=======
>>>>>>> dae6ef7088db669b5b620a870a5972fd4695633d"""
    
    # Define the replacement (keep our P0 FIX code, remove markers)
    new_text = """# P0 FIX: Inject current_target into compile_cuda arguments for target-specific binary names
                if tool_call.name == "compile_cuda" and current_target:
                    tool_call.arguments["target"] = current_target"""
    
    if old_text in content:
        content = content.replace(old_text, new_text)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print('✅ Conflict resolved successfully!')
    else:
        print('⚠️ Conflict pattern not found exactly. Attempting line-by-line fix...')
        # Fallback: remove lines with conflict markers
        lines = content.split('\n')
        cleaned_lines = []
        skip_mode = False
        for line in lines:
            if '<<<<<<< HEAD' in line or '=======' in line or '>>>>>>>' in line:
                continue
            cleaned_lines.append(line)
        
        content = '\n'.join(cleaned_lines)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print('✅ Cleaned conflict markers (line-by-line mode)')
else:
    print('✅ No merge conflicts found')

# Verify syntax
import py_compile
try:
    py_compile.compile(filepath, doraise=True)
    print('✅ Python syntax check passed!')
except py_compile.PyCompileError as e:
    print(f'❌ Syntax error: {e}')
