# Kaggle 测试目录

| 文件 | 说明 |
| ---- | ---- |
| `KAGGLE_TEST_GUIDE.md` | 完整测试指南（环境配置、LLM API 设置、Pipeline 流程、预期结果、故障排查） |
| `run_kaggle_test.py` | 一键执行脚本 -- 支持 pipeline / probes 两种模式，含 PJ 评分自查 |
| `quick_start_notebook.py` | 快速启动模板 -- 按 Cell 分隔，适合 Kaggle Notebook 逐格运行 |

## 最快上手（Pipeline 模式）

在 Kaggle Notebook 中运行：

```python
import os, sys
os.chdir("/kaggle/working/gpu_profiling_system")
sys.path.insert(0, "/kaggle/working/gpu_profiling_system")

# 1. 先配置 LLM API（见 KAGGLE_TEST_GUIDE.md 第 3 节）
# 2. 运行管线
exec(open("kaggle_test/run_kaggle_test.py").read())
```

脚本会自动：

1. 检查 nvcc、GPU、CUDA 运行时
2. 验证 LLM API 配置
3. 运行多智能体管线（Planner -> CodeGen -> MetricAnalysis -> Verification）
4. 生成 `/kaggle/working/results.json`
5. 打印测量摘要、19 项交叉验证、PJ 评分自查报告
