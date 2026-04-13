# Kaggle 多智能体完整测试指南

> GPU Profiling System -- AI 驱动的多智能体框架用于 GPU 性能分析、CUDA 内核自动调优和 LLM 基础设施生成

---

## 1. 两种运行模式

| 模式 | 说明 | 需要 LLM API | PJ 得分预期 |
| ---- | ---- | ------------ | ----------- |
| **probes** | 仅硬件微基准测试，跳过智能体 | 否 | ~40/100（只有数值，无 reasoning） |
| **pipeline** | 完整多智能体管线（Planner -> CodeGen -> MetricAnalysis -> Verification） | **是** | ~80-100/100（数值 + 方法论 + 推理） |

### PJ 评分维度（LLM-as-a-Judge）

| 维度 | 分值 | 要求 |
| ---- | ---- | ---- |
| 数值一致性 | 70 分 | 测量值与真值在工程容差内（延迟 +/-5%，带宽 +/-2%） |
| 工程推理与方法论 | 30 分 | 正确识别频率锁定/SM 屏蔽、生成有效 CUDA 内核、执行交叉验证 |

**关键**：方法论（30 分）需要完整的 reasoning 日志，只有 pipeline 模式能提供。

---

## 2. 环境准备

### 2.1 Kaggle Notebook 设置

1. `Settings` -> `Accelerator` -> 选择 `GPU T4 x2`
2. `Settings` -> `Internet` -> 开启 `On`（需要调用 LLM API）

### 2.2 上传项目代码

#### Windows 右键压缩（推荐）

1. 在文件资源管理器进入 `GPU_Profiling_System` 目录
2. 选中以下文件夹和文件：`src/`、`config/`、`tests/`、`kaggle_test/`、`CLAUDE.md`、`kaggle_notebook.py`
3. 右键 -> `发送到` -> `压缩(zipped)文件夹` -> `gpu_profiling_system.zip`
4. Kaggle Notebook -> `Add Data` -> `Upload` -> 上传 zip

#### Windows PowerShell

```powershell
cd e:\GPU_Profiling_System
Compress-Archive -Path @(
    "src", "config", "tests", "kaggle_test",
    "CLAUDE.md", "kaggle_notebook.py"
) -DestinationPath .\gpu_profiling_system.zip -Force
```

#### Git Bash / WSL

```bash
cd /e/GPU_Profiling_System
zip -r gpu_profiling_system.zip \
    src/ config/ tests/ kaggle_test/ \
    CLAUDE.md kaggle_notebook.py \
    -x "*.pyc" "__pycache__/*" ".sandbox/*" ".docker_sandbox/*" ".state/*"
```

### 2.3 解压（Kaggle Notebook 单元格）

```python
import os, shutil, sys

src = "/kaggle/input/gpu-profiling-system/gpu_profiling_system"
dst = "/kaggle/working/gpu_profiling_system"

if not os.path.exists(dst):
    shutil.copytree(src, dst)

os.chdir(dst)
sys.path.insert(0, dst)
print(f"Ready: {os.getcwd()}")
```

---

## 3. LLM API 配置（Pipeline 模式必需）

### 3.1 支持的 API 提供商

系统通过 **Anthropic 兼容协议** 调用 LLM，任何提供该协议的服务商都可以：

| 提供商 | base_url | 说明 |
| ------ | -------- | ---- |
| **LongCat** | `https://api.longcat.com/openaicompatible/api/v1/` | 推荐，国内可用 |
| **DashScope** | `https://dashscope.aliyuncs.com/apps/anthropic` | 当前已配置 |
| **Anthropic 官方** | `https://api.anthropic.com` | 需要海外网络 |
| **其他兼容服务** | 你的端点 | 只要支持 Anthropic `/v1/messages` 格式 |

### 3.2 使用 LongCat

```python
import json, os

os.chdir("/kaggle/working/gpu_profiling_system")

config = {
    "env": {
        "ANTHROPIC_BASE_URL": "https://api.longcat.com/openaicompatible/api/v1/",
        "ANTHROPIC_AUTH_TOKEN": "你的LongCat API Key",
        "ANTHROPIC_MODEL": "longcat-flash-chat",
        "ANTHROPIC_REASONING_MODEL": "longcat-flash-chat",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "longcat-flash-chat",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "longcat-flash-chat",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "longcat-flash-chat"
    },
    "includeCoAuthoredBy": False,
    "effortLevel": "high"
}

with open("config/api_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("API config written.")
```

> 项目自带 `config/api_config_longcat.json` 模板，也可以直接 `cp` 然后修改 token。

### 3.3 检查当前配置

```python
import json, os
os.chdir("/kaggle/working/gpu_profiling_system")
with open("config/api_config.json") as f:
    cfg = json.load(f)
token = cfg["env"]["ANTHROPIC_AUTH_TOKEN"]
if token and not token.startswith("sk-<") and len(token) > 10:
    print(f"API OK -> {cfg['env']['ANTHROPIC_BASE_URL']}")
    print(f"Model: {cfg['env']['ANTHROPIC_MODEL']}")
else:
    print("API NOT configured. Set ANTHROPIC_AUTH_TOKEN to a real key.")
```

---

## 4. 运行测试

### 4.1 Pipeline 模式（推荐）

编辑 `kaggle_test/run_kaggle_test.py` 顶部，设置 `RUN_MODE = "pipeline"`：

```python
import os, sys
os.chdir("/kaggle/working/gpu_profiling_system")
sys.path.insert(0, "/kaggle/working/gpu_profiling_system")
exec(open("kaggle_test/run_kaggle_test.py").read())
```

自动执行流程：

1. **环境检查** -- nvcc、GPU、CUDA 运行时
2. **API 检查** -- 验证 LLM API 配置
3. **多智能体管线**：
   - `Planner` -- 分析 target_spec.json 中的目标，制定探针策略
   - `CodeGen` -- 生成/选择 CUDA 微基准测试内核
   - `MetricAnalysis` -- 分析 ncu 输出和微基准测试结果
   - `Verification` -- 独立验证测量结果的合理性（P7 原则：不继承生成上下文）
4. **结果验证** -- 输出测量摘要、交叉验证、PJ 评分自查

### 4.2 Probes 模式（无智能体）

将 `RUN_MODE` 改为 `"probes"`，或直接调用：

```python
from src.main import main
main(["--probes-only", "--output-dir", "/kaggle/working", "--no-docker"])
```

> 此模式只跑硬件微基准测试，不经过智能体框架，PJ 的方法论（30 分）可能不完整。

---

## 5. 预期结果

### 5.1 Tesla T4 参考值

| 指标 | 预期值 | 容差 |
| ---- | ------ | ---- |
| `sm_count` | 40 | 精确匹配 |
| `likely_gpu_family` | `turing_t4` | -- |
| `actual_boost_clock_mhz` | ~1590 | +/-5% |
| `dram_latency_cycles` | 300-500 | +/-5% |
| `l2_latency_cycles` | 30-60 | +/-10% |
| `l1_latency_cycles` | 5-20 | +/-10% |
| `l2_cache_size_mb` | 4.0 | 精确匹配 |
| `dram_bandwidth_gbps` | 250-350 | +/-2% |
| `shmem_bandwidth_gbps` | 800-2000 | +/-5% |
| `max_shmem_per_block_kb` | 64 或 96 | 精确匹配 |
| `bank_conflict_penalty_ratio` | 2.0-8.0 | +/-15% |

### 5.2 输出文件

```text
/kaggle/working/
  results.json                    <-- 主输出，提交评估
  .state/
    session_log.jsonl             <-- 多智能体会话日志（含 reasoning 记录）
  evidence_clock_frequency.json
  evidence_dram_latency.json
  evidence_l2_cache_capacity.json
  evidence_dram_bandwidth.json
  evidence_shmem_capacity.json
  evidence_bank_conflict.json
  evidence_shmem_bandwidth.json
  evidence_sm_detection.json
```

---

## 6. 验证输出是否符合 PJ 标准

测试脚本会自动输出 **PJ 评分自查报告**：

### 6.1 数值一致性（70 分）

- `dram_latency_cycles` 是否在 100-600 范围内
- `l2_cache_size_mb` 是否在 0.5-128 范围内
- `actual_boost_clock_mhz` 是否在 200-3000 范围内
- `sm_count` 是否在 16-200 范围内

### 6.2 工程推理与方法论（30 分）

- `methodology` 字段是否存在且内容充实（>200 字符）
- 是否提及关键方法：pointer chasing、Knuth shuffle、clock()、micro-benchmark、cross-validation
- 交叉验证检查数是否 >= 15 项
- 交叉验证通过率是否 >= 60%

### 6.3 提交检查清单

脚本末尾输出 `[OK]` / `[WARN]` 清单，全部通过即可提交。

---

## 7. 需要发送给我审查的内容

测试完成后，请发送以下信息：

### 必须发送

1. **完整终端输出** -- 从 `阶段 1` 到 `测试完成` 的全部 print 输出
2. **results.json 内容** -- 直接粘贴或发文件
3. **交叉验证结果** -- 哪些检查通过/失败

### 建议发送

4. **session_log.jsonl 前 20 行** -- 多智能体的 reasoning 记录（仅 pipeline 模式）
5. **ncu 是否可用** -- 影响分析深度

### 快捷方式

测试脚本会自动总结。只需**截图或复制**从 `=== 测量结果 ===` 到 `测试完成` 之间的全部输出即可。

---

## 8. 故障排查

### 8.1 nvcc 不可用

```bash
apt-get update && apt-get install -y cuda-toolkit-12-0
```

### 8.2 LLM API 调用失败

- `ConnectionError` -- 检查 Kaggle Internet 是否开启
- `401 Unauthorized` -- API Key 无效
- `404 Not Found` -- base_url 路径不正确
- `429 Too Many Requests` -- 请求频率超限，稍后重试

### 8.3 编译 CUDA 内核失败

系统会自动尝试多个架构（sm_75 -> sm_70 -> sm_60 -> sm_50），如果全部失败说明 CUDA 环境异常。

### 8.4 Pipeline 模式超时

Kaggle T4 有 9 小时限制，通常足够。如果超时：

- 减少 `--max-turns` 到 30
- 或将 RUN_MODE 改为 `probes`

---

Sources:

- [LongCat API Open Platform](https://blog.poixe.com/939/)
- [LongCat GitHub](https://github.com/JessonChan/longcat-web-api)
