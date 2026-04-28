cd /workspace

# 设置 CUDA 环境变量
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 设置 Python 路径
export PYTHONPATH=/workspace:/workspace/mlsys-project-main:${PYTHONPATH}

echo "[run.sh] CUDA_HOME: ${CUDA_HOME}"
echo "[run.sh] nvcc: $(which nvcc)"
echo "[run.sh] ncu: $(which ncu)"
echo "[run.sh] PYTHONPATH: ${PYTHONPATH}"

python -m agent.agent_framework
