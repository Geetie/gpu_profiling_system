"""Kaggle Notebook 快速启动模板（多智能体模式）。

逐格复制到 Kaggle Notebook 的独立单元格中，或整体运行。
"""

# ============================================================
# Cell 1: 环境准备
# ============================================================
import os, shutil, sys

# 检测项目根目录
for candidate in [
    "/kaggle/working/gpu_profiling_system",
    "/kaggle/input/gpu-profiling-system",
    os.getcwd(),
]:
    if os.path.isfile(os.path.join(candidate, "src", "main.py")):
        PROJECT_ROOT = candidate
        break
else:
    print("ERROR: Cannot find project root. Upload dataset first.")
    sys.exit(1)

os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
print(f"Project: {PROJECT_ROOT}")

# 检查 nvcc / GPU
import subprocess
print(f"nvcc: {'OK' if shutil.which('nvcc') else 'MISSING'}")
r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
print(f"GPU: {r.stdout.strip()}")


# ============================================================
# Cell 2: 配置 LLM API（Pipeline 模式必需）
# ============================================================
# 取消下方注释并填入真实 API Key：
#
# import json
# config = {
#     "env": {
#         "ANTHROPIC_BASE_URL": "https://api.longcat.com/openaicompatible/api/v1/",
#         "ANTHROPIC_AUTH_TOKEN": "YOUR_API_KEY_HERE",
#         "ANTHROPIC_MODEL": "longcat-flash-chat",
#         "ANTHROPIC_REASONING_MODEL": "longcat-flash-chat",
#         "ANTHROPIC_DEFAULT_HAIKU_MODEL": "longcat-flash-chat",
#         "ANTHROPIC_DEFAULT_SONNET_MODEL": "longcat-flash-chat",
#         "ANTHROPIC_DEFAULT_OPUS_MODEL": "longcat-flash-chat"
#     },
#     "includeCoAuthoredBy": False,
#     "effortLevel": "high"
# }
# with open("config/api_config.json", "w") as f:
#     json.dump(config, f, indent=2)
# print("API configured.")


# ============================================================
# Cell 3: 运行多智能体管线
# ============================================================
# 方式 A：使用一键脚本（编辑顶部 RUN_MODE = "pipeline"）
# exec(open("kaggle_test/run_kaggle_test.py").read())

# 方式 B：直接调用 main
from src.main import main

exit_code = main([
    "Profile GPU hardware parameters for the given targets",
    "--pipeline",
    "--target-spec", "config/target_spec.json",
    "--output-dir", "/kaggle/working",
    "--no-docker",
    "--mode", "high_autonomy",
    "--max-turns", "50",
])
print(f"Pipeline exit code: {exit_code}")


# ============================================================
# Cell 4: 查看结果
# ============================================================
import json

with open("/kaggle/working/results.json") as f:
    results = json.load(f)

m = results.get("measurements", {})
print("=== Measurements ===")
for k, v in m.items():
    if not k.startswith("_"):
        print(f"  {k}: {v}")

cv = results.get("cross_validation", {})
passed = sum(1 for v in cv.values() if v is True)
print(f"\nCross-validation: {passed}/{len(cv)} passed")
for check, ok in cv.items():
    mark = "PASS" if ok else ("FAIL" if ok is False else "SKIP")
    print(f"  [{mark}] {check}")

print(f"\nresults.json: {os.path.getsize('/kaggle/working/results.json')} bytes")
