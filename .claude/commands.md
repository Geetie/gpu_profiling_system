# GPU Profiling System — Verified Commands

## Git Operations
```bash
# Push (may need retry — proxy at 127.0.0.1:7897 is flaky)
git push origin master

# If push fails, retry — it usually works on second try
git push origin master

# Switch to SSH (fails — no SSH key configured)
# git remote set-url origin git@github.com:Geetie/gpu_profiling_system.git

# Verify push succeeded
git log --oneline -5
git status
```

## Local Testing
```bash
# E2E test harness (rule-based simulation, no GPU needed)
python test_e2e_harness.py

# Syntax check all Python files
python -c "
import py_compile, glob, sys
errors = []
for f in glob.glob('src/**/*.py', recursive=True):
    try: py_compile.compile(f, doraise=True)
    except py_compile.PyCompileError as e: errors.append(str(e))
for f in glob.glob('test_*.py'):
    try: py_compile.compile(f, doraise=True)
    except py_compile.PyCompileError as e: errors.append(str(e))
if errors:
    for e in errors: print(e)
    sys.exit(1)
print('All files pass syntax check')
"

# Provider manager test
python src/infrastructure/provider_manager.py
```

## Kaggle Deployment
```bash
# The Kaggle notebook was already created on Kaggle platform
# Reuse existing notebook — update with latest kaggle_kernel.py content
# Notebook URL: check Kaggle dashboard for "GPU Profiling System" kernel

# On Kaggle notebook: paste kaggle_kernel.py content into a single cell
# GPU must be enabled in notebook settings (T4 or P100)
```

## Kaggle Secrets Setup (in notebook or Kaggle Settings → Secrets)
```
LONGCAT_API_KEY=<your_longcat_key>
DASHSCOPE_API_KEY=<your_dashscope_key>
ANTHROPIC_API_KEY=<your_anthropic_key>
```

## Kaggle Execution (inside notebook cell)
```bash
# Run the kernel script
python kaggle_kernel.py

# Or if running from working dir after clone:
cd /kaggle/working/gpu_profiling_system
python -m src.main --probes-only --no-docker --output-dir /kaggle/working
```

## Viewing Results on Kaggle
```bash
# Check results
cat /kaggle/working/results.json
cat /kaggle/working/execution_summary.json
cat /kaggle/working/execution.log

# List all output files
ls -la /kaggle/working/
ls -la /kaggle/working/audit/ 2>/dev/null
ls -la /kaggle/working/.state/ 2>/dev/null
```

## API Config Generation (local, for testing)
```bash
# Generate api_config.json from environment variables
python config/setup_provider.py

# Or manually write config
python -c "
import json, os
cfg = {
    'env': {
        'ANTHROPIC_API_KEY': os.environ.get('ANTHROPIC_API_KEY', ''),
        'ANTHROPIC_BASE_URL': os.environ.get('ANTHROPIC_BASE_URL', 'https://api.anthropic.com'),
        'ANTHROPIC_MODEL': 'claude-sonnet-4-5-20250514',
    },
    'includeCoAuthoredBy': False,
    'effortLevel': 'high'
}
with open('config/api_config.json', 'w') as f:
    json.dump(cfg, f, indent=2)
"
```

## Pipeline Manual Run (local, if GPU available)
```bash
python -m src.main "Profile GPU" --pipeline --target-spec config/target_spec.json --no-docker --mode high_autonomy --max-turns 50 --max-tokens 16000
```

## Probes-Only Run (local, if GPU available)
```bash
python -m src.main --probes-only --no-docker --output-dir . --state-dir .state
```

## Architecture Reference
- 7 design principles (P1-P7) in CLAUDE.md
- 4 mechanical safety invariants (M1-M4) in CLAUDE.md
- 4-layer architecture: Presentation → Application → Domain → Infrastructure
- 4-agent pipeline: Planner → CodeGen → MetricAnalysis → Verification
- P7: Verification must have empty context at run() start — never inherits generator context
- Tool registry: per-agent tool isolation (P2 fail-closed)
- State persistence: all critical state written to .state/ directory
