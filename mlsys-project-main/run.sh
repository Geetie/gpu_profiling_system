#!/bin/bash
set -e

echo "=========================================="
echo "  GPU Profiling Agent - Starting..."
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Set Python path
export PYTHONPATH="/workspace:$PYTHONPATH"
export PATH="/usr/local/cuda/bin:$PATH"

# Log environment info
echo "[INFO] Python: $(which python3)"
echo "[INFO] Python version: $(python3 --version)"
echo "[INFO] CUDA version: $(nvcc --version 2>/dev/null | head -n1 || echo 'N/A')"
echo "[INFO] GPU info: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Check API key
if [ -z "$API_KEY" ]; then
    echo "⚠️ WARNING: API_KEY not set!"
else
    echo "✅ API_KEY is set (length: ${#API_KEY})"
    echo "✅ BASE_MODEL: ${BASE_MODEL:-'not set'}"
    echo "✅ BASE_URL: ${BASE_URL:-'not set'}"
fi

# Check target spec
TARGET_SPEC="/target/target_spec.json"
if [ -f "$TARGET_SPEC" ]; then
    echo ""
    echo "📋 Target Specification:"
    cat "$TARGET_SPEC"
    echo ""
else
    echo "⚠️ Target spec not found at $TARGET_SPEC"
fi

# Create necessary directories
mkdir -p /workspace/benchmarks /workspace/build

# Run the agent
echo ""
echo "🚀 Launching GPU Profiling Agent..."
echo "=========================================="

cd /workspace
python3 -m agent.agent 2>&1 | tee /workspace/results.log

EXIT_CODE=${?}

echo ""
echo "=========================================="
echo "  Agent Execution Completed!"
echo "  Exit code: $EXIT_CODE"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Check output file
if [ -f "/workspace/output.json" ]; then
    echo "✅ Output file generated at /workspace/output.json"
    cp /workspace/output.json /workspace/output_results.json
else
    echo "⚠️ No output.json found"
fi

exit $EXIT_CODE
