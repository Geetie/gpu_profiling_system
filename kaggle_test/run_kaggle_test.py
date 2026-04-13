#!/usr/bin/env python3
"""Kaggle 多智能体完整测试脚本。

支持两种运行模式：
  1. --mode pipeline  : 完整多智能体管线 (Planner → CodeGen → MetricAnalysis → Verification)
  2. --mode probes     : 仅硬件探针 (无 LLM，跳过智能体框架)

在 Kaggle Notebook 中执行：
    exec(open('/kaggle/working/gpu_profiling_system/kaggle_test/run_kaggle_test.py').read())

或在本地 GPU 机器上：
    python kaggle_test/run_kaggle_test.py --mode pipeline
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time


# ============================================================
# 配置区 — 修改此处选择运行模式
# ============================================================
RUN_MODE = "pipeline"  # "pipeline" = 多智能体完整管线 | "probes" = 仅探针
OUTPUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
STATE_DIR = os.path.join(OUTPUT_DIR, ".state")
NO_DOCKER = True  # Kaggle 不需要 Docker


def banner(title: str) -> None:
    width = 60
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def section(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


# ============================================================
# 阶段 0：项目路径设置
# ============================================================
banner("GPU Profiling System — 多智能体 Kaggle 测试")
print(f"运行模式: {RUN_MODE}")
print(f"输出目录: {OUTPUT_DIR}")

# 自动检测项目根目录
for candidate in [
    os.getcwd(),
    "/kaggle/working/gpu_profiling_system",
    "/kaggle/input/gpu-profiling-system",
]:
    if os.path.isfile(os.path.join(candidate, "src", "main.py")):
        PROJECT_ROOT = candidate
        break
else:
    print("[错误] 找不到项目根目录，请先上传数据集")
    sys.exit(1)

os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
print(f"项目路径: {PROJECT_ROOT}\n")


# ============================================================
# 阶段 1：环境检查
# ============================================================
section("阶段 1：环境检查")

# Python
print(f"Python: {sys.version.split()[0]}")

# nvcc (必须)
nvcc = shutil.which("nvcc")
if nvcc:
    print(f"nvcc:   已找到 ✓")
    try:
        r = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=10)
        for line in r.stdout.splitlines():
            if "release" in line.lower():
                print(f"        {line.strip()}")
                break
    except Exception:
        pass
else:
    print("nvcc:   未找到 ✗ — 无法编译 CUDA 内核")
    print("        Kaggle 通常预装了 CUDA Toolkit")
    print("        尝试: !apt-get install -y cuda-toolkit-12-0")
    sys.exit(1)

# ncu (可选)
ncu = shutil.which("ncu")
if ncu:
    print(f"ncu:    已找到 ✓ — 将使用 Nsight Compute 精确分析")
else:
    print("ncu:    未找到 — 将使用 cudaEventElapsedTime 回退方案")
    print("        (多智能体管线仍可运行，仅无法做 ncu 级别的深度分析)")

# GPU
try:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10,
    )
    if r.returncode == 0:
        gpu_info = r.stdout.strip()
        print(f"GPU:    {gpu_info}")
    else:
        r2 = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        if r2.returncode == 0:
            gpu_info = r2.stdout.strip()
            print(f"GPU:    {gpu_info}")
        else:
            print("GPU:    未检测到 GPU ✗")
            sys.exit(1)
except FileNotFoundError:
    print("nvidia-smi: 未找到 ✗")
    sys.exit(1)

# CUDA Runtime
try:
    import ctypes
    ctypes.CDLL("libcudart.so")
    print("CUDA:   运行时可用 ✓")
except Exception:
    try:
        ctypes.CDLL("libcudart.so.12")
        print("CUDA:   运行时可用 (v12) ✓")
    except Exception:
        try:
            ctypes.CDLL("libcudart.so.11")
            print("CUDA:   运行时可用 (v11) ✓")
        except Exception:
            print("CUDA:   运行时不可用 ✗")
            sys.exit(1)


# ============================================================
# 阶段 2：API 配置检查
# ============================================================
section("阶段 2：API 配置检查")

config_path = os.path.join(PROJECT_ROOT, "config", "api_config.json")
api_configured = False

if os.path.isfile(config_path):
    with open(config_path, "r") as f:
        api_config = json.load(f)

    env = api_config.get("env", {})
    base_url = env.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    auth_token = env.get("ANTHROPIC_AUTH_TOKEN", "")
    model = env.get("ANTHROPIC_MODEL", "")

    # 检查 token 是否为有效值（不是占位符）
    token_valid = bool(auth_token) and not auth_token.startswith("sk-<") and not auth_token.startswith("在此填入") and len(auth_token) > 10

    print(f"配置文件: config/api_config.json")
    print(f"API 端点: {base_url}")
    print(f"主模型:   {model or '未设置'}")
    print(f"Token:    {'已配置 ✓' if token_valid else '占位符/未配置 ✗'}")

    if token_valid:
        api_configured = True
        print(f"\n→ LLM API 已配置，多智能体管线可用")
    else:
        print(f"\n→ LLM API 未配置，请编辑 config/api_config.json 填入真实 API Key")
        print(f"  LongCat API 配置示例:")
        print(f'    "ANTHROPIC_BASE_URL": "https://api.longcat.com"')
        print(f'    "ANTHROPIC_AUTH_TOKEN": "你的LongCat API Key"')
        print(f'    "ANTHROPIC_MODEL": "longcat-flash-chat"')
else:
    print(f"[警告] config/api_config.json 不存在")
    print(f"        多智能体管线需要 LLM API，请先创建配置文件")


# ============================================================
# 阶段 3：执行
# ============================================================
if RUN_MODE == "pipeline":
    # ========== 多智能体完整管线模式 ==========
    section("阶段 3：多智能体管线 (Planner → CodeGen → MetricAnalysis → Verification)")

    if not api_configured:
        print("[错误] 多智能体管线需要 LLM API 配置，但当前未设置")
        print("")
        print("解决方案:")
        print("  1. 编辑 config/api_config.json，填入真实的 API Key")
        print("  2. 或将 RUN_MODE 改为 'probes' 仅运行硬件探针")
        print("")
        print("LongCat API 配置方法:")
        print("  在 config/api_config.json 中修改:")
        print('    "ANTHROPIC_BASE_URL": "https://api.longcat.com/openaicompatible/api/v1/"')
        print('    "ANTHROPIC_AUTH_TOKEN": "你的实际API Key"')
        print('    "ANTHROPIC_MODEL": "longcat-flash-chat"')
        print("")
        print("是否切换到 probes 模式继续? (y/n)")
        try:
            resp = input("> ").strip().lower()
            if resp in ("y", "yes"):
                RUN_MODE = "probes"
                print("切换到 probes 模式...")
            else:
                sys.exit(1)
        except (EOFError, KeyboardInterrupt):
            print("\n交互输入不可用，自动切换到 probes 模式")
            RUN_MODE = "probes"

    if RUN_MODE == "pipeline":
        start_time = time.time()

        # 确保 target_spec.json 存在
        target_spec_path = os.path.join(PROJECT_ROOT, "config", "target_spec.json")
        if not os.path.isfile(target_spec_path):
            # 创建默认 target spec
            default_spec = {
                "targets": [
                    "dram_latency_cycles",
                    "l2_cache_size_mb",
                    "actual_boost_clock_mhz",
                    "max_shmem_per_block_kb",
                    "dram_bandwidth_gbps",
                    "shmem_bandwidth_gbps",
                    "bank_conflict_penalty_ratio",
                    "sm_count",
                ]
            }
            os.makedirs(os.path.dirname(target_spec_path), exist_ok=True)
            with open(target_spec_path, "w") as f:
                json.dump(default_spec, f, indent=2)
            print(f"创建默认 target_spec.json: {target_spec_path}")

        print(f"Target spec: {json.load(open(target_spec_path))}")
        print(f"State dir:   {STATE_DIR}")
        print(f"Output dir:  {OUTPUT_DIR}")
        print("")
        print("启动多智能体管线...")
        print("  Planner      → 分析目标，制定探针策略")
        print("  CodeGen      → 生成 CUDA 内核代码")
        print("  MetricAnalysis → 分析 ncu/微基准测试结果")
        print("  Verification → 独立验证测量结果的合理性")
        print("")

        try:
            from src.main import main

            exit_code = main([
                "Profile GPU hardware parameters for the given targets",
                "--pipeline",
                "--target-spec", target_spec_path,
                "--output-dir", OUTPUT_DIR,
                "--state-dir", STATE_DIR,
                "--no-docker",
                "--mode", "high_autonomy",  # 高自治模式，减少人工审批
                "--max-turns", "50",
                "--max-tokens", "16000",
            ])

            elapsed = time.time() - start_time
            print(f"\n管线执行完成，耗时: {elapsed:.1f}s，退出码: {exit_code}")

        except Exception as e:
            print(f"\n[错误] 管线执行异常: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

elif RUN_MODE == "probes":
    # ========== 仅硬件探针模式 ==========
    section("阶段 3：硬件探针 (无 LLM 智能体)")
    start_time = time.time()

    try:
        from src.main import main

        exit_code = main([
            "--probes-only",
            "--output-dir", OUTPUT_DIR,
            "--state-dir", STATE_DIR,
            "--no-docker",
        ])

        elapsed = time.time() - start_time
        print(f"\n探针执行完成，耗时: {elapsed:.1f}s，退出码: {exit_code}")

        if exit_code != 0:
            print("\n[错误] 探针执行失败")
            sys.exit(1)

    except Exception as e:
        print(f"\n[错误] 探针执行异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================
# 阶段 4：结果验证
# ============================================================
section("阶段 4：结果验证")

results_path = os.path.join(OUTPUT_DIR, "results.json")
if not os.path.isfile(results_path):
    print(f"[错误] results.json 未生成: {results_path}")

    # 列出可能相关的文件
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.startswith("evidence_") or f.endswith(".json") or f.endswith(".log"):
            print(f"  发现: {f} ({os.path.getsize(os.path.join(OUTPUT_DIR, f))} bytes)")
    sys.exit(1)

with open(results_path, "r") as f:
    results = json.load(f)


# ---- 4a. 测量结果摘要 ----
section("测量结果")
measurements = results.get("measurements", {})

display_table = [
    ("sm_count", "", "SM 数量"),
    ("likely_gpu_family", "", "GPU 家族"),
    ("actual_boost_clock_mhz", "MHz", "实际 Boost 频率"),
    ("dram_latency_cycles", "cycles", "DRAM 延迟"),
    ("l2_latency_cycles", "cycles", "L2 缓存延迟"),
    ("l1_latency_cycles", "cycles", "L1 缓存延迟"),
    ("l2_cache_size_mb", "MB", "L2 缓存大小"),
    ("dram_bandwidth_gbps", "GB/s", "DRAM 带宽"),
    ("shmem_bandwidth_gbps", "GB/s", "共享内存带宽"),
    ("max_shmem_per_block_kb", "KB", "每块最大共享内存"),
    ("bank_conflict_penalty_ratio", "x", "银行冲突惩罚比"),
    ("warp_size", "", "Warp 大小"),
    ("max_threads_per_block", "", "每块最大线程数"),
    ("theoretical_max_concurrent_blocks", "", "理论并发块数"),
]

for key, unit, label in display_table:
    if key in measurements:
        val = measurements[key]
        if unit:
            print(f"  {label:<20} {val:>10} {unit}")
        else:
            print(f"  {label:<20} {val:>10}")

# 置信度
conf_keys = sorted(k for k in measurements if k.startswith("_confidence_"))
if conf_keys:
    print(f"\n  {'置信度':}")
    for ck in conf_keys:
        metric = ck.replace("_confidence_", "")
        print(f"    {metric:<25} {measurements[ck]}")


# ---- 4b. 交叉验证 ----
section("交叉验证 (19 项检查)")
cv = results.get("cross_validation", {})
passed = sum(1 for v in cv.values() if v is True)
failed = sum(1 for v in cv.values() if v is False)
skipped = len(cv) - passed - failed

print(f"  通过: {passed}  失败: {failed}  跳过: {skipped}  总计: {len(cv)}")
if failed > 0:
    print(f"\n  {'失败项:'}")
    for check in cv:
        if cv[check] is False:
            print(f"    [FAIL] {check}")


# ---- 4c. PJ 评分自查 ----
section("PJ 评分自查")

# 数值一致性 (70 分) — 检查关键指标是否存在且合理
score_numerical = 0
checks_numerical = []

# 检查必要指标
required_metrics = {
    "dram_latency_cycles": (100, 600, "DRAM 延迟"),
    "l2_cache_size_mb": (0.5, 128, "L2 缓存大小"),
    "actual_boost_clock_mhz": (200, 3000, "GPU 频率"),
    "sm_count": (16, 200, "SM 数量"),
}

for metric, (low, high, label) in required_metrics.items():
    val = measurements.get(metric, 0)
    if isinstance(val, (int, float)) and low <= val <= high:
        score_numerical += 17.5
        checks_numerical.append(f"  [OK] {label}: {val}")
    elif val:
        checks_numerical.append(f"  [WARN] {label}: {val} (超出预期范围 {low}-{high})")
        score_numerical += 8
    else:
        checks_numerical.append(f"  [MISS] {label}: 未测量")

# 方法论完整性 (30 分)
score_reasoning = 0
checks_reasoning = []

# methodology 字段
methodology = results.get("methodology", "")
if methodology and len(methodology) > 200:
    score_reasoning += 10
    checks_reasoning.append("  [OK] 方法论描述完整")
else:
    checks_reasoning.append("  [WARN] 方法论描述过短或缺失")

# methodology 是否包含关键方法关键词
key_methods = ["pointer chasing", "Knuth shuffle", "clock()", "micro-benchmark", "cross-validation"]
found_methods = [m for m in key_methods if m.lower() in methodology.lower()]
if len(found_methods) >= 3:
    score_reasoning += 10
    checks_reasoning.append(f"  [OK] 方法多样性: {', '.join(found_methods)}")
else:
    checks_reasoning.append(f"  [WARN] 方法单一: 仅提及 {found_methods}")

# 交叉验证存在
if cv and len(cv) >= 15:
    score_reasoning += 10
    checks_reasoning.append(f"  [OK] 交叉验证: {len(cv)} 项检查")
else:
    checks_reasoning.append(f"  [WARN] 交叉验证不足: {len(cv)} 项")

total_score = score_numerical + score_reasoning

print(f"\n  数值一致性: {score_numerical:.0f}/70 分")
for c in checks_numerical:
    print(c)

print(f"\n  工程推理与方法论: {score_reasoning:.0f}/30 分")
for c in checks_reasoning:
    print(c)

print(f"\n  {'=' * 40}")
print(f"  预估总分: {total_score:.0f}/100 分")
if total_score >= 80:
    print(f"  评级: 优秀")
elif total_score >= 60:
    print(f"  评级: 良好")
elif total_score >= 40:
    print(f"  评级: 及格")
else:
    print(f"  评级: 需改进")


# ---- 4d. 证据文件 ----
evidence = results.get("evidence", [])
if evidence:
    section(f"证据文件 ({len(evidence)} 个)")
    for ef in evidence:
        basename = os.path.basename(ef)
        size = ""
        full_path = os.path.join(OUTPUT_DIR, basename)
        if os.path.isfile(full_path):
            size = f" ({os.path.getsize(full_path)} bytes)"
        print(f"  - {basename}{size}")


# ---- 4e. 探针状态 ----
probe_status = results.get("probe_status", {})
if probe_status:
    section("探针状态")
    for probe, status in probe_status.items():
        if status == "success":
            print(f"  [OK] {probe}")
        elif status == "no_data":
            print(f"  [N/A] {probe}")
        else:
            print(f"  [ERR] {probe}: {status}")


# ============================================================
# 阶段 5：提交清单
# ============================================================
section("提交前检查清单")

checklist = [
    (os.path.isfile(results_path), "results.json 存在"),
    (bool(measurements), "measurements 字段非空"),
    ("dram_latency_cycles" in measurements, "包含 dram_latency_cycles"),
    ("l2_cache_size_mb" in measurements, "包含 l2_cache_size_mb"),
    ("actual_boost_clock_mhz" in measurements, "包含 actual_boost_clock_mhz"),
    ("max_shmem_per_block_kb" in measurements, "包含 max_shmem_per_block_kb"),
    ("methodology" in results, "包含 methodology 字段"),
    (len(methodology) > 200, "methodology 内容充实 (>200 字符)"),
    (bool(cv), "包含 cross_validation 字段"),
    (passed >= len(cv) * 0.6 if cv else False, "交叉验证通过率 >= 60%"),
    (bool(evidence), "包含证据文件"),
]

all_pass = True
for ok, desc in checklist:
    mark = "✓" if ok else "✗"
    if not ok:
        all_pass = False
    print(f"  [{mark}] {desc}")

print(f"\n  {'全部通过' if all_pass else '存在未通过项'}")

print(f"\nresults.json: {results_path}")
print(f"文件大小:     {os.path.getsize(results_path)} bytes")

banner("测试完成")
