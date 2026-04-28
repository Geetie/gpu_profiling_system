#!/bin/bash
# deploy.sh - 清理服务器并上传新代码进行稳定性测试
# 使用方法: ./deploy.sh

set -e

SERVER="10.176.37.31"
PORT="39333"
REMOTE_DIR="/workspace/mlsys-project-main"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "======================================"
echo "GPU Profiling System - 部署脚本"
echo "======================================"
echo "服务器: ${SERVER}:${PORT}"
echo "本地目录: ${LOCAL_DIR}"
echo "远程目录: ${REMOTE_DIR}"
echo ""

# Step 1: 清理服务器上的缓存和快照
echo "[Step 1/5] 清理服务器缓存和旧代码..."
ssh -o StrictHostKeyChecking=no -p ${PORT} root@${SERVER} << 'EOF'
echo "  - 清理旧构建产物..."
rm -rf /workspace/mlsys-project-main/build/*
rm -rf /workspace/mlsys-project-main/.state/*
rm -rf /workspace/mlsys-project-main/benchmarks/*
rm -rf /workspace/mlsys-project-main/output.json
rm -rf /workspace/mlsys-project-main/results.log
echo "  - 清理Python缓存..."
find /workspace/mlsys-project-main -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find /workspace/mlsys-project-main -type f -name "*.pyc" -delete 2>/dev/null || true
echo "  - 清理完成"
EOF

# Step 2: 上传核心代码
echo "[Step 2/5] 上传核心代码..."
echo "  - 上传 agent/..."
scp -o StrictHostKeyChecking=no -P ${PORT} -r "${LOCAL_DIR}/agent/" root@${SERVER}:"${REMOTE_DIR}/agent/"
echo "  - 上传 runner/..."
scp -o StrictHostKeyChecking=no -P ${PORT} -r "${LOCAL_DIR}/runner/" root@${SERVER}:"${REMOTE_DIR}/runner/"
echo "  - 上传 llm/..."
scp -o StrictHostKeyChecking=no -P ${PORT} -r "${LOCAL_DIR}/llm/" root@${SERVER}:"${REMOTE_DIR}/llm/"

# Step 3: 上传配置文件
echo "[Step 3/5] 上传配置文件..."
scp -o StrictHostKeyChecking=no -P ${PORT} "${LOCAL_DIR}/run.sh" root@${SERVER}:"${REMOTE_DIR}/run.sh"
scp -o StrictHostKeyChecking=no -P ${PORT} "${LOCAL_DIR}/target_spec_sample.json" root@${SERVER}:"${REMOTE_DIR}/target_spec_sample.json"

# Step 4: 创建必要目录
echo "[Step 4/5] 创建远程目录..."
ssh -o StrictHostKeyChecking=no -p ${PORT} root@${SERVER} << 'EOF'
mkdir -p /workspace/mlsys-project-main/build
mkdir -p /workspace/mlsys-project-main/build/profiles
mkdir -p /workspace/mlsys-project-main/benchmarks
mkdir -p /workspace/mlsys-project-main/.state
chmod +x /workspace/mlsys-project-main/run.sh
echo "  - 目录创建完成"
EOF

# Step 5: 验证部署
echo "[Step 5/5] 验证部署..."
ssh -o StrictHostKeyChecking=no -p ${PORT} root@${SERVER} << 'EOF'
echo "  - 检查文件结构..."
test -f /workspace/mlsys-project-main/run.sh && echo "    ✓ run.sh" || echo "    ✗ run.sh 缺失!"
test -f /workspace/mlsys-project-main/agent/agent_framework.py && echo "    ✓ agent_framework.py" || echo "    ✗ agent_framework.py 缺失!"
test -f /workspace/mlsys-project-main/agent/agent.py && echo "    ✓ agent.py" || echo "    ✗ agent.py 缺失!"
test -f /workspace/mlsys-project-main/agent/prompts/generate_benchmark.txt && echo "    ✓ generate_benchmark.txt" || echo "    ✗ generate_benchmark.txt 缺失!"
test -f /workspace/mlsys-project-main/agent/prompts/analyze_metrics.txt && echo "    ✓ analyze_metrics.txt" || echo "    ✗ analyze_metrics.txt 缺失!"
test -f /workspace/mlsys-project-main/runner/run.py && echo "    ✓ run.py" || echo "    ✗ run.py 缺失!"
test -f /workspace/mlsys-project-main/llm/openai_client.py && echo "    ✓ openai_client.py" || echo "    ✗ openai_client.py 缺失!"
echo "  - 验证完成"
EOF

echo ""
echo "======================================"
echo "部署完成！"
echo "======================================"
echo ""
echo "要启动稳定性测试，请运行:"
echo "  ssh -p ${PORT} root@${SERVER}"
echo "  cd ${REMOTE_DIR}"
echo "  bash run.sh"
echo ""
echo "或者使用 submit 接口提交测试"
