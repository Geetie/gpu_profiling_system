#!/bin/bash
set -e

echo "=========================================="
echo "  GPU Profiling Agent - Starting..."
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

export PYTHONPATH="/workspace:$PYTHONPATH"
export PATH="/usr/local/cuda/bin:$PATH"

echo "[INFO] Python: $(which python3)"
echo "[INFO] Python version: $(python3 --version)"
echo "[INFO] CUDA version: $(nvcc --version 2>/dev/null | head -n1 || echo 'N/A')"
echo "[INFO] GPU info: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

if [ -n "$API_KEY" ]; then
    echo "API_KEY is set (length: ${#API_KEY})"
    echo "BASE_MODEL: ${BASE_MODEL:-'not set'}"
    echo "BASE_URL: ${BASE_URL:-'not set'}"
else
    echo "WARNING: API_KEY not set!"
fi

TARGET_SPEC="/target/target_spec.json"

if [ ! -f "$TARGET_SPEC" ]; then
    echo "ERROR: Target specification not found at $TARGET_SPEC"
    echo "Trying config/target_spec.json as fallback..."
    TARGET_SPEC="/workspace/config/target_spec.json"
    if [ ! -f "$TARGET_SPEC" ]; then
        echo "ERROR: No target_spec.json found anywhere!"
        exit 1
    fi
fi

echo ""
echo "Target Specification:"
cat "$TARGET_SPEC"
echo ""

mkdir -p /workspace

cd /workspace

echo ""
echo "Launching GPU Profiling Pipeline..."
echo "=========================================="

python3 -m src.main \
    --pipeline \
    --target-spec "$TARGET_SPEC" \
    --output-dir /workspace \
    --max-turns 30 \
    --mode high_autonomy \
    "Profile GPU metrics according to target specification" \
    2>&1 | tee /workspace/results.log

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
echo "  Agent Execution Completed!"
echo "  Exit code: $EXIT_CODE"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

if [ -f "/workspace/results.json" ]; then
    echo ""
    echo "Converting results.json to output_results.json..."
    cp /workspace/results.json /workspace/output_results.json
    echo "Output file created at /workspace/output_results.json"
fi

OUTPUT_FILE="/workspace/output_results.json"

if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "SUCCESS: Output file generated at $OUTPUT_FILE"
    echo "Output preview (first 500 chars):"
    head -c 500 "$OUTPUT_FILE"
    echo ""
    echo "..."
else
    echo ""
    echo "WARNING: Output file not found at $OUTPUT_FILE"
    echo "Checking for alternative output files..."
    OUTPUT_FILES=$(find /workspace -maxdepth 1 -name "output.*" -type f 2>/dev/null || true)
    if [ -n "$OUTPUT_FILES" ]; then
        echo "Found output files:"
        echo "$OUTPUT_FILES"
    else
        echo "No output files found. Check /workspace/results.log for errors."
    fi
fi

echo ""
echo "Files in /workspace:"
ls -lh /workspace/*.json /workspace/*.log /workspace/report.* 2>/dev/null || true

exit $EXIT_CODE
