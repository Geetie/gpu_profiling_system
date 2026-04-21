#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo "=========================================="
echo "  GPU Profiling Agent - Starting..."
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Set Python path to include workspace
export PYTHONPATH="/workspace:$PYTHONPATH"
export PATH="/usr/local/cuda/bin:$PATH"

# Log environment info (for debugging)
echo "[INFO] Python: $(which python3)"
echo "[INFO] Python version: $(python3 --version)"
echo "[INFO] CUDA version: $(nvcc --version 2>/dev/null | head -n1 || echo 'N/A')"
echo "[INFO] GPU info: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Check required environment variables for API access
if [ -z "$API_KEY" ] && [ -z "$ANTHROPIC_AUTH_TOKEN" ]; then
    echo "⚠️ WARNING: No API key found in environment variables!"
    echo "   Expected: API_KEY or ANTHROPIC_AUTH_TOKEN"
    echo "   The agent will fail without API credentials."
fi

if [ -n "$API_KEY" ]; then
    echo "✅ API_KEY environment variable is set (length: ${#API_KEY})"
    echo "✅ BASE_MODEL: ${BASE_MODEL:-'not set'}"
    echo "✅ BASE_URL: ${BASE_URL:-'not set'}"
fi

# Target specification path (provided by evaluation environment)
TARGET_SPEC="/target/target_spec.json"

# Validate target specification exists
if [ ! -f "$TARGET_SPEC" ]; then
    echo "❌ ERROR: Target specification not found at $TARGET_SPEC"
    echo "   The evaluation environment should provide this file."
    exit 1
fi

echo ""
echo "📋 Target Specification:"
cat "$TARGET_SPEC"
echo ""

# Create output directory if it doesn't exist
mkdir -p /workspace

# Run the GPU profiling agent
echo ""
echo "🚀 Launching GPU Profiling Pipeline..."
echo "=========================================="

cd /workspace

python3 -m src.main \
    --pipeline \
    --target-spec "$TARGET_SPEC" \
    --output-dir /workspace \
    --max-turns 30 \
    --mode high_autonomy \
    "Profile GPU metrics according to target specification" \
    2>&1 | tee /workspace/results.log

EXIT_CODE=${?}

echo ""
echo "=========================================="
echo "  Agent Execution Completed!"
echo "  Exit code: $EXIT_CODE"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Convert results.json to output.* format for evaluation system
if [ -f "/workspace/results.json" ]; then
    echo ""
    echo "✅ Converting results.json to output_results.json for evaluation system..."
    cp /workspace/results.json /workspace/output_results.json
    echo "✅ Output file created at /workspace/output_results.json"
fi

# Verify output file was generated
OUTPUT_FILE="/workspace/output_results.json"

if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "✅ SUCCESS: Output file generated at $OUTPUT_FILE"
    echo "📊 Output preview (first 500 chars):"
    head -c 500 "$OUTPUT_FILE"
    echo ""
    echo "..."
else
    echo ""
    echo "❌ WARNING: Output file not found at $OUTPUT_FILE"
    echo "   Checking for alternative output files..."
    
    # Look for any output.* files
    OUTPUT_FILES=$(find /workspace -maxdepth 1 -name "output.*" -type f 2>/dev/null || true)
    
    if [ -n "$OUTPUT_FILES" ]; then
        echo "   Found output files:"
        echo "$OUTPUT_FILES"
    else
        echo "   No output files found. Check /workspace/results.log for errors."
    fi
fi

# Final summary
echo ""
echo "📁 Files in /workspace:"
ls -lh /workspace/*.json /workspace/*.log /workspace/report.* 2>/dev/null || true

exit $EXIT_CODE
