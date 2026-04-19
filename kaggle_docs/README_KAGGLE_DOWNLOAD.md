# 📥 Kaggle 文件下载完整指南

**日期**: 2026-04-17  
**状态**: ✅ 完成

---

## 🎯 快速下载清单（6 个核心文件）

### 必须下载的文件（总计 ~70 KB）

| # | 文件名 | Kaggle 路径 | 大小 | 重要性 |
|---|--------|-------------|------|--------|
| 1 | `execution.log` | `/kaggle/working/gpu_profiling_system/` | ~56 KB | ⭐⭐⭐⭐⭐ |
| 2 | `pipeline_log.jsonl` | `/kaggle/working/gpu_profiling_system/.state/` | ~1 KB | ⭐⭐⭐⭐⭐ |
| 3 | `session_log.jsonl` | `/kaggle/working/gpu_profiling_system/.state/` | ~3 KB | ⭐⭐⭐⭐ |
| 4 | `audit_report.md` | `/kaggle/working/gpu_profiling_system/audit/` | ~0.25 KB | ⭐⭐⭐⭐ |
| 5 | `audit_report.json` | `/kaggle/working/gpu_profiling_system/audit/` | ~10 KB | ⭐⭐⭐ |
| 6 | `results.json` | `/kaggle/working/gpu_profiling_system/` | ~1 KB | ⭐⭐⭐⭐⭐ |

---

## 🗺️ Kaggle 目录结构

```
/kaggle/working/gpu_profiling_system/
├── .state/
│   ├── session_log.jsonl          ← AgentLoop 状态日志
│   └── pipeline_log.jsonl         ← Pipeline 状态日志
├── audit/
│   ├── audit_report.md            ← 审计报告（Markdown）
│   └── audit_report.json          ← 审计报告（JSON）
├── execution.log                   ← 完整执行日志
└── results.json                    ← 最终测量结果
```

---

## 📊 文件生成位置详解

### 1. execution.log

**生成代码**: [`main.py:26-31`](file:///e:/GPU_Profiling_System/src/main.py#L26-L31)

**生成方式**: 
- ❌ 不是主动写入的文件
- ✅ 所有日志输出到 `sys.stdout`
- ✅ Kaggle 自动捕获 stdout 到 execution.log

**内容**:
- 所有 `print()` 输出
- 所有 `logging.info/warning/error()` 输出
- Sandbox 路径配置
- 编译错误详情
- Pipeline 执行状态

---

### 2. session_log.jsonl

**生成代码**: [`state_persist.py:18-21`](file:///e:/GPU_Profiling_System/src/infrastructure/state_persist.py#L18-L21)

**生成方式**: 
- ✅ StatePersister 主动写入
- ✅ JSONL 格式（每行一个 JSON）

**内容**:
- 工具执行日志
- 权限决策日志
- 错误日志
- 不变量违规日志

---

### 3. pipeline_log.jsonl

**生成代码**: [`pipeline.py:67`](file:///e:/GPU_Profiling_System/src/domain/pipeline.py#L67)

**生成方式**: 
- ✅ Pipeline 初始化时创建
- ✅ JSONL 格式

**内容**:
- Pipeline 启动/完成日志
- Stage 开始/结束日志
- 重试日志
- Metric 反馈日志

---

### 4. audit_report.md / audit_report.json

**生成代码**: [`audit_report.py:222-238`](file:///e:/GPU_Profiling_System/src/application/audit_report.py#L222-L238)

**生成方式**: 
- ✅ Pipeline 完成后主动写入
- ✅ 同时生成 Markdown 和 JSON 格式

**内容**:
- Stage 执行时间线
- Handoff 验证结果
- Circuit Breaker 状态
- 工具执行摘要
- P7 合规性审计
- 最终测量结果

---

### 5. results.json

**生成代码**: [`main.py:401-446`](file:///e:/GPU_Profiling_System/src/main.py#L401-L446)

**生成方式**: 
- ✅ Pipeline 完成后主动写入
- ✅ JSON 格式

**内容**:
- 所有测量结果
- 交叉验证数据
- 证据文件列表
- 方法论说明
- 目标列表

---

## ❌ 不需要下载的文件

| 文件 | 原因 |
|------|------|
| `source.cu` | 临时文件，每次编译重新生成 |
| `freq_probe` | 编译产物（~1 MB），占用空间大 |
| `freq_event_probe` | 编译产物（~1 MB），占用空间大 |
| `freq_event_timed` | 编译产物（~1 MB），占用空间大 |
| `cmd_*.log` | 冗余命令日志 |
| `debug_messages_*.json` | 冗余调试文件 |

**清理工具**: [`cleanup.py`](file:///e:/GPU_Profiling_System/cleanup.py)

---

## 📥 下载方法

### 方法 1: Kaggle UI 下载

1. 打开 Kaggle Notebook
2. 点击 "Output" 标签
3. 导航到对应目录下载文件

### 方法 2: Kaggle API 下载

```bash
# 下载所有核心文件
kaggle kernels output <username>/<kernel-id> -p execution.log
kaggle kernels output <username>/<kernel-id> -p .state/session_log.jsonl
kaggle kernels output <username>/<kernel-id> -p .state/pipeline_log.jsonl
kaggle kernels output <username>/<kernel-id> -p audit/audit_report.md
kaggle kernels output <username>/<kernel-id> -p audit/audit_report.json
kaggle kernels output <username>/<kernel-id> -p results.json
```

### 方法 3: Notebook 内下载

```python
from IPython.display import FileLink

# 创建下载链接
FileLink('execution.log')
FileLink('.state/session_log.jsonl')
FileLink('.state/pipeline_log.jsonl')
FileLink('audit/audit_report.md')
FileLink('audit/audit_report.json')
FileLink('results.json')
```

---

## 🔍 文件分析技巧

### execution.log

```bash
# 查看 Sandbox 路径配置
grep -n "Sandbox" execution.log | head -20

# 查看编译错误
grep -n "nvcc\|compilation\|error" execution.log

# 查看文件写入位置
grep -n "Writing\|Creating\|path" execution.log
```

### pipeline_log.jsonl

```bash
# 查看每个阶段的状态
cat pipeline_log.jsonl | jq '.stage, .status, .error'

# 查看工具调用
cat pipeline_log.jsonl | jq 'select(.tool_calls > 0) | .tool_calls'
```

### session_log.jsonl

```bash
# 查看工具执行历史
cat session_log.jsonl | jq '.payload | select(.tool_name != "__loop_state__")'
```

### audit_report.md

```bash
# 查看不变量违规
grep -A 5 "Invariant Violation" audit_report.md

# 查看测量结果
grep -A 10 "Final Measurements" audit_report.md
```

---

## 📊 文件大小估算

| 文件 | 大小估算 | 影响因素 |
|------|----------|----------|
| `execution.log` | 50-200 KB | Pipeline 复杂度、错误数量 |
| `session_log.jsonl` | 3-10 KB | 工具调用次数 |
| `pipeline_log.jsonl` | 1-5 KB | Stage 数量、重试次数 |
| `audit_report.md` | 5-20 KB | 测量结果数量 |
| `audit_report.json` | 10-50 KB | 详细数据结构 |
| `results.json` | 1-5 KB | 测量目标数量 |

**总计**: ~70-290 KB（非常小，适合下载）

---

## 🎯 下载优先级

### 🔴 优先级 1（必须下载）- 61 KB

```
execution.log          (56 KB)
pipeline_log.jsonl     (1 KB)
session_log.jsonl      (3 KB)
results.json           (1 KB)
```

**用途**: 90% 的错误诊断

### 🟡 优先级 2（建议下载）- 10 KB

```
audit_report.md        (0.25 KB)
audit_report.json      (10 KB)
```

**用途**: 深度分析 Pipeline 执行质量

### 🟢 优先级 3（无需下载）

```
source.cu              # 临时文件
freq_*                 # 编译产物
cmd_*.log              # 冗余日志
debug_messages_*.json  # 冗余调试
```

**原因**: 临时生成、占用空间大、无分析价值

---

## 📚 相关文档

| 文档 | 用途 |
|------|------|
| [`KAGGLE 文件生成位置详解.md`](file:///e:/GPU_Profiling_System/KAGGLE 文件生成位置详解.md) | 详细的文件生成位置分析 |
| [`KAGGLE 文件下载指南.md`](file:///e:/GPU_Profiling_System/KAGGLE 文件下载指南.md) | 详细的下载建议和场景分析 |
| [`仓库整理报告.md`](file:///e:/GPU_Profiling_System/仓库整理报告.md) | 仓库整理详细报告 |
| [`cleanup.py`](file:///e:/GPU_Profiling_System/cleanup.py) | 自动化清理脚本 |

---

## ✅ 总结

### 核心要点

1. ✅ **6 个核心文件** 总计 ~70 KB，包含所有必要信息
2. ✅ **execution.log** 是 stdout 捕获，不是主动写入
3. ✅ **.state/** 目录包含 JSONL 状态日志
4. ✅ **audit/** 目录包含审计报告
5. ❌ **不要下载** 编译产物和临时文件（~3 MB）

### 下载清单

- ✅ `execution.log` - 完整执行日志
- ✅ `pipeline_log.jsonl` - Pipeline 状态
- ✅ `session_log.jsonl` - AgentLoop 状态
- ✅ `audit_report.md` - 审计报告（Markdown）
- ✅ `audit_report.json` - 审计报告（JSON）
- ✅ `results.json` - 最终测量结果

### 维护建议

- ✅ 每次 Kaggle 测试后只下载这 6 个文件
- ✅ 运行 `python cleanup.py` 清理临时文件
- ✅ 参考文档分析错误

---

**整理人**: AI Assistant  
**整理日期**: 2026-04-17  
**状态**: ✅ 完成
